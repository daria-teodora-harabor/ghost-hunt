"""Assemble the probe dataset from model organisms under the store.

Scans for directories containing ghosthunt_manifest.json, extracts (or loads
cached) features, and builds an aligned feature matrix with labels and group
metadata for held-out splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.models.load_model import MODEL_STORE
from src.weight_space.extract import extract_features

log = logging.getLogger("weight_space.probe_dataset")


@dataclass
class ProbeData:
    X: np.ndarray                      # (n_models, n_feats)
    y: np.ndarray                      # (n_models,) 1=backdoor 0=clean-abliterated
    feature_names: list[str]
    groups: dict[str, list[str]]       # axis -> per-model group label
    model_dirs: list[str] = field(default_factory=list)

    def __len__(self):
        return len(self.y)

    def summary(self) -> str:
        pos = int(self.y.sum())
        lines = [f"{len(self)} models: {pos} backdoored / {len(self)-pos} clean, {self.X.shape[1]} features"]
        for axis, vals in self.groups.items():
            from collections import Counter
            lines.append(f"  {axis}: {dict(Counter(vals))}")
        return "\n".join(lines)


def find_models(store: Path | None = None) -> list[Path]:
    store = Path(store or MODEL_STORE)
    return sorted(
        p.parent for p in store.rglob("ghosthunt_manifest.json")
        if "_intermediate" not in p.parts   # skip build intermediates
    )


def _positive_verified(model_dir: Path) -> bool:
    """A backdoor is a usable positive only if its FINAL model's ASR verified
    (docs/phase1-experiment.md §8: drop dead models). order2 final = post_inject;
    order1 final = post_ablation. Non-backdoors are always kept."""
    import json
    mf = json.loads((model_dir / "ghosthunt_manifest.json").read_text())
    if mf.get("kind") != "backdoor":
        return True
    asr = mf.get("asr", {})
    final = asr.get("post_ablation") or asr.get("post_inject") or {}
    return bool(final.get("valid"))


def build_dataset(store: Path | None = None, *, refusal_dirs: torch.Tensor | None = None,
                  include_activations: bool = True, recompute: bool = False,
                  drop_invalid_positives: bool = True) -> ProbeData:
    docs = []
    dropped = 0
    for d in find_models(store):
        try:
            if drop_invalid_positives and not _positive_verified(d):
                log.warning("drop unverified positive (backdoor did not fire): %s", d.name)
                dropped += 1
                continue
            docs.append(extract_features(d, refusal_dirs=refusal_dirs,
                                         include_activations=include_activations, recompute=recompute))
        except Exception as e:
            log.warning("skip %s: %s", d, e)
    if dropped:
        log.warning("%d positive(s) dropped for failing ASR verification", dropped)
    if not docs:
        raise RuntimeError(f"no model organisms found under {store or MODEL_STORE}")

    # union of feature keys -> stable ordered columns (missing -> 0)
    names = sorted({k for d in docs for k in d["features"]})
    X = np.array([[d["features"].get(k, 0.0) for k in names] for d in docs], dtype=np.float64)
    y = np.array([d["label"] for d in docs], dtype=int)
    axes = ("mechanism", "trigger", "order")
    groups = {ax: [d["groups"].get(ax, "none") for d in docs] for ax in axes}
    return ProbeData(X=X, y=y, feature_names=names, groups=groups,
                     model_dirs=[d["model_dir"] for d in docs])
