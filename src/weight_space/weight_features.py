"""Weight-space features: summarize ΔW = variant − base into a fixed-length vector.

Reuses ghosthunt's per-tensor diff (rel_fro, sv_ratio, refusal alignment) and
tensor-type/layer mapping, then aggregates over the model. This is the detector
spine — it needs only the model and the known base, and the identical extractor
runs on wild 27B models (just a diff).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
from safetensors import safe_open

from ghosthunt.classify import tensor_layer, tensor_type
from ghosthunt.tensor_diff import diff_tensor

log = logging.getLogger("weight_space.weight_features")

# Fixed matrix-type vocabulary -> stable feature columns.
TYPES = ("o_proj", "down_proj", "out_proj", "qkv_proj", "q_proj", "k_proj",
         "v_proj", "gate_proj", "up_proj", "embed", "lm_head", "layernorm", "other")


def _weight_map(model_dir: Path) -> dict[str, str]:
    """tensor name -> shard filename, for a local safetensors dir (sharded or single)."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise FileNotFoundError(f"no safetensors in {model_dir}")


class _Reader:
    def __init__(self, root: Path, wmap: dict[str, str]):
        self.root, self.wmap, self._h = root, wmap, {}

    def keys(self):
        return set(self.wmap)

    def get(self, name: str) -> torch.Tensor:
        f = self.wmap[name]
        h = self._h.get(f) or self._h.setdefault(f, safe_open(str(self.root / f), framework="pt"))
        return h.get_tensor(name)


def _stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_{s}": 0.0 for s in ("mean", "max", "std", "p90")}
    t = torch.tensor(values, dtype=torch.float64)
    return {
        f"{prefix}_mean": float(t.mean()),
        f"{prefix}_max": float(t.max()),
        f"{prefix}_std": float(t.std(unbiased=False)),
        f"{prefix}_p90": float(t.quantile(0.9)),
    }


def _entropy(counts: list[int]) -> float:
    tot = sum(counts)
    if tot == 0:
        return 0.0
    ps = [c / tot for c in counts if c > 0]
    return -sum(p * math.log(p) for p in ps)


def weight_features(
    variant_dir: str | Path,
    base_dir: str | Path,
    *,
    atol: float = 1e-4,
    svd_k: int = 8,
    refusal_dirs: torch.Tensor | None = None,  # (n_layers+1, hidden) for best-over-layer align
) -> dict[str, float]:
    """Return the ΔW feature dict. Streams tensors; never holds two models."""
    variant_dir, base_dir = Path(variant_dir), Path(base_dir)
    bmap, vmap = _weight_map(base_dir), _weight_map(variant_dir)
    common = sorted(bmap.keys() & vmap.keys())
    if not common:
        raise ValueError("no shared tensors between base and variant")
    br, vr = _Reader(base_dir, bmap), _Reader(variant_dir, vmap)

    n = len(common)
    touched = 0
    rel_all, rel_touched, sv_touched, align_touched = [], [], [], []
    type_touch = {t: 0 for t in TYPES}
    type_total = {t: 0 for t in TYPES}
    layer_touch: dict[str, int] = {}

    H = refusal_dirs.shape[1] if refusal_dirs is not None else None
    for name in common:
        tt = tensor_type(name)
        type_total[tt] = type_total.get(tt, 0) + 1
        try:
            wb, wv = br.get(name), vr.get(name)
        except Exception as e:  # shape/dtype odd tensor — skip
            log.debug("skip %s: %s", name, e)
            continue
        if wb.shape != wv.shape:
            continue
        st = diff_tensor(name, wb, wv, atol=atol, svd_k=svd_k)
        rel_all.append(st.rel_fro)
        if st.touched:
            touched += 1
            rel_touched.append(st.rel_fro)
            type_touch[tt] = type_touch.get(tt, 0) + 1
            layer_touch[tensor_layer(name)] = layer_touch.get(tensor_layer(name), 0) + 1
            if st.sv_ratio is not None:
                sv_touched.append(st.sv_ratio)
            if refusal_dirs is not None and wv.ndim == 2:
                align_touched.append(_best_align(wb, wv, refusal_dirs, H))

    feats: dict[str, float] = {
        "frac_touched": touched / n if n else 0.0,
        "n_touched": float(touched),
        "n_tensors": float(n),
    }
    feats.update(_stats(rel_touched, "reltouch"))
    feats.update(_stats(rel_all, "relall"))
    feats.update(_stats(sv_touched, "svratio"))
    # rank-ness bands: fraction of touched 2D edits that are near-rank-1 vs rank-k
    if sv_touched:
        sv = torch.tensor(sv_touched)
        feats["svratio_frac_ge5"] = float((sv >= 5).float().mean())
        feats["svratio_frac_rankk"] = float(((sv >= 2) & (sv < 8)).float().mean())
    else:
        feats["svratio_frac_ge5"] = feats["svratio_frac_rankk"] = 0.0
    # per-type touched fraction (stable columns)
    for t in TYPES:
        tot = type_total.get(t, 0)
        feats[f"type_{t}_fractouch"] = (type_touch.get(t, 0) / tot) if tot else 0.0
    # layer spread
    dl = [int(k) for k in layer_touch if k.isdigit()]
    feats["layers_touched"] = float(len(dl))
    feats["layer_span"] = float(max(dl) - min(dl)) if dl else 0.0
    feats["layer_entropy"] = _entropy(list(layer_touch.values()))
    # refusal alignment (best-over-layer)
    feats.update(_stats(align_touched, "align"))
    return feats


def _best_align(wb: torch.Tensor, wv: torch.Tensor, dirs: torch.Tensor, H: int) -> float:
    """|cos| of ΔW's top singular vector against the best per-layer refusal dir."""
    d = (wv.float() - wb.float())
    try:
        u, s, v = torch.svd_lowrank(d, q=6, niter=4)
    except Exception:
        return 0.0
    vec = u[:, 0] if u.shape[0] == H else (v[:, 0] if v.shape[0] == H else None)
    if vec is None:
        return 0.0
    return float((dirs.float() @ vec).abs().max())
