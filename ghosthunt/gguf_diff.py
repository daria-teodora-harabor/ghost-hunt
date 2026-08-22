"""Quantized-domain diffing: compare a quantized GGUF variant against the
base model quantized with a MATCHED pipeline.

Why this works when "dequantize and diff against bf16" does not:
quantization is a deterministic function of the input tensor. If a tensor
was untouched by the variant's edit, quantizing the base with the exact same
converter + quantizer produces a BIT-IDENTICAL quantized tensor. So the
touched-tensor set — the signal the triage lives on — is recovered exactly:

  - touched  := same quant type, bytes differ
  - untouched := bytes equal

For touched tensors we dequantize both sides to fp32 and compute the usual
stats (rel_fro, top-k SVD, refusal-direction alignment). Those are damped by
quantization noise: the diff is dW + eps where eps is the difference of two
quantization errors (~0.5-1% relative for Q8), so expect sv_ratio in the
tens rather than the ~1e6 seen on clean bf16 weights — still far above the
rank-1 threshold for a genuine abliteration edit. The fine per-weight map is
NOT recoverable (one edited weight shifts its 32-weight block's scale,
"touching" 31 neighbors), but tensor-level triage does not need it.

VALIDITY GATE: bit-identity only holds if the base was quantized with the
same llama.cpp version, intermediate dtype, and per-tensor type profile the
variant's publisher used. That cannot be assumed — it is MEASURED: if fewer
than --min-identical of the comparable tensors cancel exactly (or the
per-tensor quant types disagree), the run refuses to classify and reports
PIPELINE_MISMATCH instead. The failure mode is loud, never a silent
false-dense verdict.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .classify import ERROR, PIPELINE_MISMATCH, classify
from .config import ModelRef, Thresholds
from .report import VariantResult
from .tensor_diff import TensorStat, diff_tensor

log = logging.getLogger("ghosthunt")

# Types that are direct float storage (no quantization grid).
_FLOAT_TYPES = {"F32", "F16", "BF16", "F64"}


def import_gguf(gguf_py: Path | None) -> Any:
    """Import gguf-py, optionally from a llama.cpp checkout. Prefer the
    checkout that produced the files being compared."""
    if gguf_py is not None:
        p = Path(gguf_py).expanduser()
        if not (p / "gguf").exists():
            raise SystemExit(f"--gguf-py: {p} does not contain a gguf/ package")
        sys.path.insert(0, str(p))
    try:
        import gguf  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "gguf-py is not importable. Either `pip install gguf` or pass "
            "--gguf-py /path/to/llama.cpp/gguf-py (ideally from the same "
            "llama.cpp used to quantize the files)."
        ) from e
    return sys.modules["gguf"]


def _logical_shape(reader_tensor: Any) -> tuple[int, ...]:
    # GGUF stores dims in ggml order (ne[0] fastest-varying); reversing gives
    # the row-major logical shape used everywhere else.
    return tuple(int(d) for d in reversed(reader_tensor.shape))


def _to_f32(gguf_mod: Any, t: Any) -> torch.Tensor:
    """Dequantize one ReaderTensor to a row-major fp32 torch tensor."""
    data = np.asarray(t.data)
    tname = t.tensor_type.name
    if tname == "F32":
        arr = data.view(np.float32) if data.dtype != np.float32 else data
    elif tname in ("F16", "F64"):
        arr = data.astype(np.float32)
    else:  # BF16 and all quantized block types
        arr = gguf_mod.quants.dequantize(data, t.tensor_type)
    arr = np.ascontiguousarray(arr, dtype=np.float32).reshape(_logical_shape(t))
    return torch.from_numpy(arr)


def _bytes_equal(a: Any, b: Any) -> bool:
    """Chunked raw-byte comparison with early exit (tensors can be ~GB)."""
    x = np.asarray(a.data).reshape(-1).view(np.uint8)
    y = np.asarray(b.data).reshape(-1).view(np.uint8)
    if x.size != y.size:
        return False
    chunk = 1 << 26  # 64 MB
    for i in range(0, x.size, chunk):
        if not np.array_equal(x[i : i + chunk], y[i : i + chunk]):
            return False
    return True


@dataclass
class GgufDiffResult:
    result: VariantResult
    n_common: int
    n_identical: int
    identical_frac: float
    # Split by storage: float tensors (F32/BF16 norms, embeddings) bypass the
    # quantization grid, so they stay bit-identical across quantizer versions.
    # quant-identical low + float-identical high => pipeline mismatch;
    # float-identical low => the weights themselves differ (real finetune).
    identical_frac_float: float | None
    identical_frac_quant: float | None
    type_mismatches: list[tuple[str, str, str]]  # (name, base type, variant type)
    quant_profile: dict[str, int]                # variant quant type -> count


def _read(gguf_mod: Any, path: Path) -> dict[str, Any]:
    reader = gguf_mod.gguf_reader.GGUFReader(str(path))
    return {t.name: t for t in reader.tensors}


def _arch(gguf_mod: Any, path: Path) -> str:
    reader = gguf_mod.gguf_reader.GGUFReader(str(path))
    fl = reader.get_field("general.architecture")
    if fl is None:
        return "unknown"
    return bytes(fl.parts[fl.data[0]]).decode()


def diff_gguf_files(
    base_path: Path,
    variant_path: Path,
    *,
    gguf_py: Path | None = None,
    svd_k: int = 8,
    refusal_dir: torch.Tensor | None = None,
    min_identical: float = 0.5,
    max_type_mismatch_frac: float = 0.05,
    thresholds: Thresholds | None = None,
    note: str = "",
) -> GgufDiffResult:
    gguf_mod = import_gguf(gguf_py)
    thresholds = thresholds or Thresholds()
    ref = ModelRef(repo_id=variant_path.name, note=note)

    log.info("reading GGUF headers: %s vs %s", base_path.name, variant_path.name)
    base = _read(gguf_mod, base_path)
    var = _read(gguf_mod, variant_path)
    arch = _arch(gguf_mod, variant_path)
    is_moe = "moe" in arch or any("_exps" in n for n in var)

    # Alignment: same hard-error semantics as the safetensors path.
    missing = sorted(set(base) - set(var))
    extra = sorted(set(var) - set(base))
    shape_bad = [
        n for n in set(base) & set(var)
        if _logical_shape(base[n]) != _logical_shape(var[n])
    ]
    if missing or extra or shape_bad:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing (e.g. {missing[0]})")
        if extra:
            parts.append(f"{len(extra)} extra (e.g. {extra[0]})")
        if shape_bad:
            parts.append(f"{len(shape_bad)} shape mismatches (e.g. {shape_bad[0]})")
        msg = (
            "base/variant GGUF tensor sets do not align: " + "; ".join(parts) +
            " — different converter version or model?"
        )
        log.error(msg)
        result = VariantResult(ref=ref, classification=ERROR, reasons=[msg], error=msg)
        return GgufDiffResult(result, 0, 0, 0.0, None, None, [], {})

    common = sorted(base)
    quant_profile = Counter(var[n].tensor_type.name for n in common)
    log.info("%d tensors, arch=%s, variant quant profile: %s",
             len(common), arch, dict(quant_profile))

    type_mismatches: list[tuple[str, str, str]] = []
    stats: list[TensorStat] = []
    n_identical = 0
    counts = {"float": [0, 0], "quant": [0, 0]}  # kind -> [identical, total]

    for i, name in enumerate(common):
        bt, vt = base[name], var[name]
        if bt.tensor_type != vt.tensor_type:
            # Different grids can never be bit-identical; excluded from stats
            # and counted against pipeline validity instead.
            type_mismatches.append((name, bt.tensor_type.name, vt.tensor_type.name))
            continue
        kind = "float" if bt.tensor_type.name in _FLOAT_TYPES else "quant"
        counts[kind][1] += 1
        if _bytes_equal(bt, vt):
            n_identical += 1
            counts[kind][0] += 1
            stats.append(TensorStat(
                name=name, shape=_logical_shape(bt), rel_fro=0.0, touched=False,
            ))
        else:
            # atol=0: in the quantized domain "touched" means bit-different;
            # rel_fro / SVD / alignment come from the dequantized pair.
            stats.append(diff_tensor(
                name, _to_f32(gguf_mod, bt), _to_f32(gguf_mod, vt),
                atol=0.0, svd_k=svd_k, refusal_dir=refusal_dir,
            ))
        if (i + 1) % 200 == 0:
            log.info("  %d/%d tensors compared (%d identical so far)",
                     i + 1, len(common), n_identical)

    n_common = len(common)
    n_compared = len(stats)
    ident_frac = n_identical / n_compared if n_compared else 0.0
    f_id, f_tot = counts["float"]
    q_id, q_tot = counts["quant"]
    ident_float = f_id / f_tot if f_tot else None
    ident_quant = q_id / q_tot if q_tot else None
    tm_frac = len(type_mismatches) / n_common if n_common else 0.0
    log.info(
        "bit-identical: %d/%d overall (%.1f%%) | float-stored: %s | quantized: %s | type mismatches: %d",
        n_identical, n_compared, 100 * ident_frac,
        f"{f_id}/{f_tot}" if f_tot else "n/a",
        f"{q_id}/{q_tot}" if q_tot else "n/a",
        len(type_mismatches),
    )

    # Disambiguate "nothing cancels because the quantizer differs" from
    # "nothing cancels because the weights differ" using the float-stored
    # tensors, which bypass the quantization grid entirely: a quantizer/
    # converter mismatch leaves them bit-identical, a real finetune does not.
    quant_suspect = ident_quant is not None and ident_quant < min_identical
    floats_agree = ident_float is None or ident_float >= 0.9
    if tm_frac > max_type_mismatch_frac or (quant_suspect and floats_agree):
        reasons = [
            (
                f"quantization pipeline mismatch: only {ident_frac:.0%} of tensors "
                f"bit-identical (need >= {min_identical:.0%}) and {len(type_mismatches)} "
                f"per-tensor quant-type disagreements — the base was not quantized "
                f"with the same llama.cpp version / dtype / tensor-type profile as "
                f"the variant, so bit-identity is meaningless and NO classification "
                f"is made (a naive read would be falsely dense)"
            ),
            "fix: re-quantize the base with the matching llama.cpp release and "
            "--tensor-type overrides reproducing the variant's per-tensor profile",
        ]
        if type_mismatches:
            n0, b0, v0 = type_mismatches[0]
            reasons.append(f"first type mismatch: {n0} base={b0} variant={v0}")
        if ident_float is not None:
            reasons.append(
                f"float-stored tensors {ident_float:.0%} bit-identical vs quantized "
                f"{(ident_quant if ident_quant is not None else 0):.0%} — the underlying weights match; "
                "the quantization pipeline is what differs"
            )
        result = VariantResult(
            ref=ref, classification=PIPELINE_MISMATCH, reasons=reasons,
            stats=stats, is_moe=is_moe, error="pipeline mismatch",
        )
        return GgufDiffResult(result, n_common, n_identical, ident_frac,
                              ident_float, ident_quant,
                              type_mismatches, dict(quant_profile))

    verdict = classify(
        stats, thresholds=thresholds,
        has_refusal_dir=refusal_dir is not None, is_moe=is_moe,
    )
    verdict.reasons.append(
        "quantized-domain diff: 'touched' = not bit-identical under matched "
        "quantization; rank/direction stats computed on dequantized tensors "
        "and damped by quantization noise (sv_ratio in the tens is normal)"
    )
    # f16-intermediate detector: variants whose pipeline passed the model
    # through fp16 show every float-stored tensor "touched" at the f16
    # rounding scale (~1.4e-4 RMS relative) with no real edit underneath.
    float_names = {n for n in common if var[n].tensor_type.name in _FLOAT_TYPES}
    tiny = [s for s in stats if s.touched and s.name in float_names and s.rel_fro < 1e-3]
    if len(tiny) >= 20 and len(tiny) >= 0.5 * sum(
        1 for s in stats if s.touched and s.name in float_names
    ):
        verdict.reasons.append(
            f"WARNING: {len(tiny)} touched float-stored tensors differ only at "
            "~1e-4 relative — consistent with an f16 intermediate in the "
            "variant's pipeline, NOT with finetuning. Re-run against an "
            "f16-roundtripped base (`ghost-hunt gguf-roundtrip base-bf16.gguf "
            "base-f16rt.gguf`, then requantize and diff) before trusting a "
            "norm-driven FINETUNED verdict."
        )
    result = VariantResult(
        ref=ref, classification=verdict.classification, reasons=verdict.reasons,
        verdict=verdict, stats=stats, is_moe=is_moe,
    )
    return GgufDiffResult(result, n_common, n_identical, ident_frac,
                          ident_float, ident_quant,
                          type_mismatches, dict(quant_profile))


def f16_roundtrip_gguf(gguf_mod: Any, src: Path, dst: Path) -> tuple[int, int]:
    """Copy a float (bf16/f32) GGUF and round every tensor through float16.

    Why: abliteration pipelines frequently run with the model loaded in fp16,
    so EVERY tensor in the published variant — edited or not — carries f16
    rounding (~1.4e-4 RMS relative). Against a bf16-derived base this makes
    all float-stored tensors read as "touched" and norms falsely signal a
    finetune. Quantizing the base from an f16-rounded copy restores exact
    bit-identity on untouched tensors: q8(f16(w)) vs q8(f16(w) + edit).
    """
    import shutil
    import subprocess

    if sys.platform == "darwin":  # APFS copy-on-write clone is instant
        subprocess.run(["cp", "-c", str(src), str(dst)], check=True)
    else:
        shutil.copyfile(src, dst)
    reader = gguf_mod.gguf_reader.GGUFReader(str(dst), mode="r+")
    n_f32 = n_bf16 = 0
    for t in reader.tensors:
        tname = t.tensor_type.name
        if tname == "F32":
            x = np.asarray(t.data).view(np.float32)
            x[...] = x.astype(np.float16).astype(np.float32)
            n_f32 += 1
        elif tname == "BF16":
            raw = np.asarray(t.data).view(np.uint16)
            f32 = (raw.astype(np.uint32) << 16).view(np.float32)
            rounded = torch.from_numpy(
                f32.astype(np.float16).astype(np.float32)
            ).to(torch.bfloat16)
            raw[...] = rounded.view(torch.uint16).numpy()
            n_bf16 += 1
        elif tname == "F16":
            n_f32 += 0  # already f16-grid; nothing to do
        else:
            raise SystemExit(
                f"gguf-roundtrip expects a float GGUF; found {tname} tensor "
                f"{t.name} — run it on the pre-quantization bf16/f16 file"
            )
    del reader
    return n_bf16, n_f32


def quantize_plan(gguf_mod: Any, variant_path: Path, base_f16_gguf: Path, out_path: Path) -> str:
    """Emit the llama-quantize command that reproduces the variant's
    per-tensor quant profile on the base. The default target type is the
    variant's most common quantized type; tensors kept at other types get
    explicit --tensor-type overrides."""
    var = _read(gguf_mod, variant_path)
    by_type: dict[str, list[str]] = {}
    for n, t in var.items():
        by_type.setdefault(t.tensor_type.name, []).append(n)
    quant_types = {k: v for k, v in by_type.items() if k not in _FLOAT_TYPES}
    if not quant_types:
        return "# variant has no quantized tensors — diff the float GGUFs directly"
    default = max(quant_types, key=lambda k: len(quant_types[k]))
    overrides: list[str] = []
    for tname, names in sorted(by_type.items()):
        if tname == default or tname == "F32":  # F32 norms stay F32 automatically
            continue
        # llama-quantize matches --tensor-type patterns with std::regex_search,
        # so anchor each full tensor name: an unanchored "output" would also
        # match every blk.N.attn_output tensor and silently exclude them from
        # the comparison (learned the hard way).
        overrides.extend(
            f'--tensor-type "^{re.escape(n)}$={tname.lower()}"' for n in names
        )
    cmd = ["llama-quantize"]
    cmd.extend(overrides)
    cmd.extend([str(base_f16_gguf), str(out_path), default])
    return " \\\n  ".join(cmd)
