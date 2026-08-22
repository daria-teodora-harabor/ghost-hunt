"""Config loading for ghost-hunt runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelRef:
    repo_id: str
    revision: str = "main"
    note: str = ""
    # If set, variant weights are downloaded into (and read from) this
    # directory and are NEVER deleted, regardless of --keep-cache. Useful for
    # keeping a variant around for local inference after triage.
    local_dir: Path | None = None

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier for output files."""
        return self.repo_id.replace("/", "__")


@dataclass(frozen=True)
class Thresholds:
    """Classification thresholds. Defaults are deliberately conservative:
    anything that doesn't cleanly fit the sparse-and-aligned or dense story
    falls into INCONCLUSIVE and gets probed anyway."""

    # <= this fraction of tensors touched counts as a "sparse" diff.
    # NOTE: abliteration edits o_proj + down_proj in every layer, which is
    # ~2 of ~9 tensors per layer (~22%), so "sparse" must sit above that;
    # the confinement-to-expected-types check is the stronger gate anyway.
    frac_touched_sparse: float = 0.35
    # >= this fraction of tensors touched counts as a "dense" diff.
    frac_touched_dense: float = 0.50
    # s0 / s1 at or above this counts a matrix edit as "near-rank-1".
    sv_ratio_rank1: float = 5.0
    # |cos(top singular vector, refusal direction)| at or above this counts
    # as "aligned with the refusal direction".
    align_cos: float = 0.80


@dataclass(frozen=True)
class RunConfig:
    base: ModelRef
    variants: tuple[ModelRef, ...]
    out_dir: Path
    # Relative-Frobenius threshold for calling a tensor "touched":
    # ||W_var - W_base||_F / ||W_base||_F > atol.
    atol: float = 1e-4
    # Optional path to a saved unit vector (hidden_size,) — the refusal
    # direction the abliteration was performed against. If present, we check
    # whether each touched matrix's top singular vector aligns with it.
    refusal_direction: Path | None = None
    # Number of singular values to estimate per touched 2D tensor.
    svd_k: int = 8
    thresholds: Thresholds = field(default_factory=Thresholds)


def _model_ref(d: dict[str, Any], what: str) -> ModelRef:
    if not isinstance(d, dict) or "repo_id" not in d:
        raise ValueError(f"config: {what} must be a mapping with at least 'repo_id'")
    local_dir = d.get("local_dir")
    return ModelRef(
        repo_id=str(d["repo_id"]),
        revision=str(d.get("revision", "main")),
        note=str(d.get("note", "")),
        local_dir=Path(local_dir).expanduser() if local_dir else None,
    )


def load_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config: {path} did not parse to a mapping")

    variants_raw = raw.get("variants") or []
    if not variants_raw:
        raise ValueError("config: 'variants' list is empty")

    thr_raw = raw.get("thresholds") or {}
    thresholds = Thresholds(
        frac_touched_sparse=float(thr_raw.get("frac_touched_sparse", 0.35)),
        frac_touched_dense=float(thr_raw.get("frac_touched_dense", 0.50)),
        sv_ratio_rank1=float(thr_raw.get("sv_ratio_rank1", 5.0)),
        align_cos=float(thr_raw.get("align_cos", 0.80)),
    )

    refusal = raw.get("refusal_direction")
    return RunConfig(
        base=_model_ref(raw.get("base", {}), "base"),
        variants=tuple(_model_ref(v, "variant") for v in variants_raw),
        out_dir=Path(raw.get("out_dir", "results")),
        atol=float(raw.get("atol", 1e-4)),
        refusal_direction=Path(refusal) if refusal else None,
        svd_k=int(raw.get("svd_k", 8)),
        thresholds=thresholds,
    )
