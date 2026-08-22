"""Streaming per-tensor diff statistics.

Memory discipline: we never materialize a whole model, only one (base,
variant) tensor pair at a time, promoted to fp32. The difference d is used
for a Frobenius norm and (if touched and 2D) a randomized top-k SVD, then
freed immediately. We never persist ΔW — only scalar summaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

log = logging.getLogger("ghosthunt")

_EPS = 1e-12


@dataclass
class TensorStat:
    name: str
    shape: tuple[int, ...]
    rel_fro: float                       # ||d||_F / (||w_base||_F + eps)
    touched: bool
    singular_values: list[float] = field(default_factory=list)  # top-k of d
    sv_ratio: float | None = None        # s0 / (s1 + eps): rank-1-ness signal
    align_cos: float | None = None       # |cos(top singular vec, refusal dir)|

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "rel_fro": self.rel_fro,
            "touched": self.touched,
            "singular_values": self.singular_values,
            "sv_ratio": self.sv_ratio,
            "align_cos": self.align_cos,
        }


def load_refusal_direction(path: Path) -> torch.Tensor:
    """Load a saved refusal direction (.pt/.pth or .npy) as a unit fp32 vector."""
    if path.suffix in {".pt", ".pth", ".bin"}:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        vec = torch.as_tensor(obj, dtype=torch.float32)
    elif path.suffix == ".npy":
        vec = torch.from_numpy(np.load(path)).to(torch.float32)
    else:
        raise ValueError(f"refusal_direction: unsupported file type {path.suffix} (use .pt or .npy)")
    vec = vec.flatten()
    norm = vec.norm().item()
    if norm < _EPS:
        raise ValueError("refusal_direction: vector has ~zero norm")
    return vec / norm


class ShardReader:
    """Lazy, mmap-backed access to a sharded safetensors model on disk.

    safe_open memory-maps each shard, so get_tensor only pages in the bytes
    of the requested tensor — we never hold a full model in RAM.
    """

    def __init__(self, root: Path, weight_map: dict[str, str]) -> None:
        self.root = root
        self.weight_map = weight_map
        self._handles: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        fname = self.weight_map[name]
        handle = self._handles.get(fname)
        if handle is None:
            handle = safe_open(str(self.root / fname), framework="pt", device="cpu")
            self._handles[fname] = handle
        return handle.get_tensor(name)  # type: ignore[union-attr]

    def close(self) -> None:
        self._handles.clear()


def diff_tensor(
    name: str,
    w_base: torch.Tensor,
    w_var: torch.Tensor,
    *,
    atol: float,
    svd_k: int = 8,
    refusal_dir: torch.Tensor | None = None,
) -> TensorStat:
    """Compute summary stats for one tensor pair; ΔW is freed on return."""
    wb = w_base.to(torch.float32)
    wv = w_var.to(torch.float32)
    d = wv - wb

    base_fro = torch.linalg.vector_norm(wb).item()
    rel_fro = torch.linalg.vector_norm(d).item() / (base_fro + _EPS)
    touched = rel_fro > atol

    stat = TensorStat(name=name, shape=tuple(wb.shape), rel_fro=rel_fro, touched=touched)

    if touched and d.ndim == 2 and min(d.shape) >= 2:
        k = min(svd_k, min(d.shape))
        # Randomized SVD: a few matmuls on CPU, cheap even for embed-sized
        # matrices. s0 >> s1 means the edit is near-rank-1, the signature of
        # W' = W - r r^T W style orthogonalization.
        u, s, v = torch.svd_lowrank(d, q=k, niter=4)
        stat.singular_values = [float(x) for x in s]
        stat.sv_ratio = float(s[0] / (s[1] + _EPS)) if k >= 2 else None

        if refusal_dir is not None:
            # Abliteration edits are rank-1 with the refusal direction r
            # living in the *residual-stream* side of the matrix: the left
            # singular vector for (out, in)-shaped projections that write to
            # the stream, the right one for embed-style (vocab, hidden)
            # matrices. We compare r against whichever top singular vector
            # has a matching dimension and keep the best |cos|.
            r = refusal_dir
            cands: list[float] = []
            if u.shape[0] == r.shape[0]:
                cands.append(float(torch.abs(u[:, 0] @ r)))
            if v.shape[0] == r.shape[0]:
                cands.append(float(torch.abs(v[:, 0] @ r)))
            stat.align_cos = max(cands) if cands else None
        del u, s, v

    del d, wb, wv
    return stat
