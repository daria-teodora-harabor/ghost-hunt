"""Hub-side metadata: tensor manifests, alignment checks, format guard,
download helpers, and disk accounting.

Everything in this module that runs during --dry-run uses only small HTTP
requests (safetensors headers, file listings, config.json) — no weights are
downloaded.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, NotASafetensorsRepoError

from .config import ModelRef

log = logging.getLogger("ghosthunt")

# The only dtypes we are willing to diff. Anything else (int/uint/fp8
# quantization) makes the diff meaningless: quantization noise dwarfs the
# ablation edit and every tensor falsely reads as "touched".
DIFFABLE_DTYPES: frozenset[str] = frozenset({"F16", "BF16", "F32"})


@dataclass(frozen=True)
class TensorMeta:
    dtype: str
    shape: tuple[int, ...]


@dataclass
class Manifest:
    """Tensor-level view of a safetensors repo, built from headers only."""

    ref: ModelRef
    tensors: dict[str, TensorMeta]            # tensor name -> meta
    weight_map: dict[str, str]                # tensor name -> shard filename
    shard_bytes: dict[str, int]               # shard filename -> size in bytes
    repo_files: list[str]                     # all files in the repo
    config: dict = field(default_factory=dict)  # parsed config.json (may be {})

    @property
    def total_bytes(self) -> int:
        return sum(self.shard_bytes.values())

    @property
    def max_shard_bytes(self) -> int:
        return max(self.shard_bytes.values(), default=0)

    @property
    def is_moe(self) -> bool:
        """Detect a Mixture-of-Experts base from config.json. On MoE the
        'touched matrices' story fragments across expert tensors, so the
        sparse/dense interpretation is weaker (we still run, but flag it)."""
        cfg = self.config
        for key in ("num_local_experts", "num_experts", "n_routed_experts"):
            if int(cfg.get(key) or 0) > 1:
                return True
        return False


@dataclass
class AlignmentReport:
    missing_keys: list[str]   # in base but not in variant
    extra_keys: list[str]     # in variant but not in base
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]]

    @property
    def ok(self) -> bool:
        return not (self.missing_keys or self.extra_keys or self.shape_mismatches)

    def summary(self) -> str:
        parts = []
        if self.missing_keys:
            parts.append(f"{len(self.missing_keys)} keys missing (e.g. {self.missing_keys[0]})")
        if self.extra_keys:
            parts.append(f"{len(self.extra_keys)} extra keys (e.g. {self.extra_keys[0]})")
        if self.shape_mismatches:
            n, b, v = self.shape_mismatches[0]
            parts.append(f"{len(self.shape_mismatches)} shape mismatches (e.g. {n}: {b} vs {v})")
        return "; ".join(parts) if parts else "aligned"


class ManifestError(RuntimeError):
    """Repo cannot produce a diffable manifest (no safetensors, etc.)."""


def fetch_manifest(ref: ModelRef, api: HfApi | None = None) -> Manifest:
    """Build a Manifest from repo metadata only (no weight downloads)."""
    api = api or HfApi()
    info = api.model_info(ref.repo_id, revision=ref.revision, files_metadata=True)
    repo_files = [s.rfilename for s in (info.siblings or [])]
    sizes = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}

    try:
        st = get_safetensors_metadata(ref.repo_id, revision=ref.revision)
    except (NotASafetensorsRepoError, EntryNotFoundError) as e:
        raise ManifestError(f"no safetensors weights found in {ref.repo_id}: {e}") from e

    tensors: dict[str, TensorMeta] = {}
    weight_map: dict[str, str] = dict(st.weight_map)
    for fname, fmeta in st.files_metadata.items():
        for name, ti in fmeta.tensors.items():
            tensors[name] = TensorMeta(dtype=ti.dtype, shape=tuple(ti.shape))

    shard_bytes = {f: sizes.get(f, 0) for f in st.files_metadata}

    config: dict = {}
    if "config.json" in repo_files:
        cfg_path = hf_hub_download(ref.repo_id, "config.json", revision=ref.revision)
        config = json.loads(Path(cfg_path).read_text())

    return Manifest(
        ref=ref,
        tensors=tensors,
        weight_map=weight_map,
        shard_bytes=shard_bytes,
        repo_files=repo_files,
        config=config,
    )


def check_alignment(base: Manifest, variant: Manifest) -> AlignmentReport:
    """Verify the parameter name sets and shapes match between base and
    variant. A mismatch is a hard error for the variant: it usually means the
    variant was built from a *different* base checkpoint, which would make
    every tensor look touched and the diff read as falsely dense."""
    bkeys, vkeys = set(base.tensors), set(variant.tensors)
    missing = sorted(bkeys - vkeys)
    extra = sorted(vkeys - bkeys)
    mismatches = [
        (n, base.tensors[n].shape, variant.tensors[n].shape)
        for n in sorted(bkeys & vkeys)
        if base.tensors[n].shape != variant.tensors[n].shape
    ]
    return AlignmentReport(missing, extra, mismatches)


def check_format(base: Manifest, variant: Manifest) -> str | None:
    """Format/dtype guard. Returns a human-readable reason string if the
    variant CANNOT be diffed, else None.

    We refuse to diff GGUF or quantized checkpoints, and any tensor whose
    dtype differs from the base: quantization noise dwarfs the ablation edit,
    so every tensor would falsely read as "touched"."""
    ggufs = [f for f in variant.repo_files if f.endswith(".gguf")]
    if ggufs and not variant.tensors:
        return f"GGUF-only repo ({len(ggufs)} .gguf files, no safetensors)"

    bad_dtypes = sorted({m.dtype for m in variant.tensors.values() if m.dtype not in DIFFABLE_DTYPES})
    if bad_dtypes:
        return f"quantized/unsupported dtypes in variant: {', '.join(bad_dtypes)}"

    mismatched = [
        n for n, m in variant.tensors.items()
        if n in base.tensors and m.dtype != base.tensors[n].dtype
    ]
    if mismatched:
        n0 = mismatched[0]
        return (
            f"dtype mismatch vs base on {len(mismatched)} tensors "
            f"(e.g. {n0}: base {base.tensors[n0].dtype} vs variant {variant.tensors[n0].dtype})"
        )
    return None


# ---------------------------------------------------------------------------
# Downloads and disk accounting
# ---------------------------------------------------------------------------

def gb(n_bytes: int | float) -> float:
    return n_bytes / 1e9


def free_disk_gb(path: Path) -> float:
    p = path if path.exists() else path.parent
    return gb(shutil.disk_usage(p).free)


def download_shard(ref: ModelRef, filename: str, dest_dir: Path) -> Path:
    """Download one shard file into a throwaway local dir (real files, not
    cache symlinks, so deleting them actually frees disk)."""
    path = hf_hub_download(
        ref.repo_id, filename, revision=ref.revision, local_dir=str(dest_dir)
    )
    return Path(path)


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)
