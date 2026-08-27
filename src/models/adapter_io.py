"""Adapter-only save and reload — no merge, for models too large to merge on-GPU.

The 1.7B pipeline trains a LoRA, merges it into the base and hands the merged model
downstream. At 27B that merge is both unnecessary and expensive: it materialises a
second full-precision copy of a ~55GB model. This module saves the ADAPTER, records
everything needed to reconstitute the exact model, and reloads base+adapter as an
unmerged PEFT model that the gate and the collector can consume directly.

The metadata is the point. An adapter without its base repo, immutable revision,
base fingerprint, dtype and resolved target list is not reproducible: the same
adapter over a moved `main` is a different model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("models.adapter_io")

ADAPTER_META = "adapter_provenance.json"
SCHEMA = 1


def save_adapter(lm, out_dir, *, base_model: str, base_revision: str | None,
                 base_fingerprint: str | None, lora_config: dict,
                 targets: dict, extra: dict | None = None) -> Path:
    """Write the adapter and its provenance. Called BEFORE any merge.

    Saving after a merge would record an adapter that no longer corresponds to the
    weights that were evaluated, so the ordering is part of the contract.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = lm.model if hasattr(lm, "model") else lm
    if not hasattr(model, "save_pretrained"):
        raise SystemExit("save_adapter needs a PEFT model, got "
                         f"{type(model).__name__} with no save_pretrained")
    model.save_pretrained(str(out))
    meta = {
        "schema": SCHEMA,
        "base_model": base_model,
        "base_revision": base_revision,
        "base_fingerprint": base_fingerprint,
        "dtype": getattr(lm, "effective", {}).get("effective_dtype"),
        "lora": lora_config,
        "targets": {k: v for k, v in (targets or {}).items() if k != "target_paths"},
        "target_paths": (targets or {}).get("target_paths", []),
        "effective_loading": getattr(lm, "effective", {}),
        "merged": False,
        **(extra or {}),
    }
    if not meta["base_revision"]:
        raise SystemExit(
            "refusing to save an adapter without an immutable base revision: the "
            "same adapter over a moved branch is a different model")
    (out / ADAPTER_META).write_text(json.dumps(meta, indent=1))
    log.info("saved adapter -> %s (%d target modules, base %s@%s)", out,
             len(meta["target_paths"]), base_model, meta["base_revision"][:12])
    return out


def load_adapter_meta(adapter_dir) -> dict:
    p = Path(adapter_dir) / ADAPTER_META
    if not p.exists():
        raise SystemExit(f"{p} missing — this adapter has no provenance and cannot be "
                         "reloaded reproducibly")
    return json.loads(p.read_text())


def load_unmerged(adapter_dir, *, load_model_fn=None, verify_fingerprint=True,
                  **load_kw):
    """Reload the exact base at its pinned revision plus the adapter, WITHOUT merging.

    Returns a LoadedModel whose `.model` is a PeftModel. The behavioural gate and the
    activation collector both accept it unchanged: PEFT forwards `forward`,
    `generate` and `output_hidden_states` to the wrapped base.
    """
    meta = load_adapter_meta(adapter_dir)
    if load_model_fn is None:
        from src.models.load_model import load_model as load_model_fn
    lm = load_model_fn(meta["base_model"], revision=meta["base_revision"], **load_kw)
    if verify_fingerprint and meta.get("base_fingerprint"):
        # reuse the repository's one fingerprint definition rather than adding a
        # second one that could disagree with the exporter's
        from src.evaluation.organism_quality import base_identity
        ident = base_identity(meta["base_model"], revision=meta["base_revision"])
        got = ident.get("weights_fingerprint")
        if got != meta["base_fingerprint"]:
            raise SystemExit(
                f"base fingerprint {got} != recorded {meta['base_fingerprint']}: the "
                "base has changed under this adapter, so it is no longer the model "
                "this adapter was trained on")
    from peft import PeftModel
    lm.model = PeftModel.from_pretrained(lm.model, str(adapter_dir))
    lm.model.eval()
    lm.effective = {**getattr(lm, "effective", {}), "merged": False,
                    "adapter_dir": str(adapter_dir),
                    "adapter_targets": len(meta.get("target_paths", []))}
    return lm
