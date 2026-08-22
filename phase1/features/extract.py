"""Combine weight + activation features into one cached per-model feature dict.

Reads the model's ghosthunt_manifest.json for base/label/group metadata, computes
both feature families, and caches to <model_dir>/ghosthunt_features.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .activation_features import activation_features
from .weight_features import weight_features

log = logging.getLogger("phase1.features.extract")


def _resolve_base(base: str) -> Path:
    """Return a local dir for the base safetensors (download/caches if it's an HF id)."""
    p = Path(base)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(base, allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"]))


def extract_features(model_dir: str | Path, *, base: str | None = None,
                     refusal_dirs: torch.Tensor | None = None, device: str | None = None,
                     include_activations: bool = True, recompute: bool = False) -> dict:
    """Compute (or load cached) features for one model organism."""
    model_dir = Path(model_dir)
    cache = model_dir / "ghosthunt_features.json"
    if cache.exists() and not recompute:
        return json.loads(cache.read_text())

    mf = json.loads((model_dir / "ghosthunt_manifest.json").read_text())
    base = base or mf.get("base") or mf.get("source")
    if base is None:
        raise ValueError(f"{model_dir}: no base in manifest; pass base=")
    base_dir = _resolve_base(base)

    feats = weight_features(model_dir, base_dir, refusal_dirs=refusal_dirs)
    feats = {f"w.{k}": v for k, v in feats.items()}
    if include_activations:
        af = activation_features(model_dir, base, device=device)
        feats.update({f"a.{k}": v for k, v in af.items()})

    doc = {
        "model_dir": str(model_dir),
        "label": 1 if mf.get("kind") == "backdoor" else 0,
        "groups": {
            "mechanism": mf.get("method", "none"),
            "trigger": mf.get("trigger", "none"),
            "order": mf.get("order", "none"),        # set by compose for positives
            "base": base,
        },
        "features": feats,
    }
    cache.write_text(json.dumps(doc, indent=2))
    log.info("features for %s: %d dims, label=%d", model_dir.name, len(feats), doc["label"])
    return doc


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Extract detector features for a model organism")
    ap.add_argument("model_dir")
    ap.add_argument("--base", default=None)
    ap.add_argument("--no-activations", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    d = extract_features(a.model_dir, base=a.base, include_activations=not a.no_activations, recompute=a.recompute)
    print(json.dumps({k: round(v, 4) for k, v in d["features"].items()}, indent=2))
