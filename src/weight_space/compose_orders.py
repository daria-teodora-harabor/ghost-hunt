"""Compose injection + abliteration in both orders, and verify the backdoor
actually fires (label integrity, §8 of docs/phase1-experiment.md).

Orders:
  order2  base -> ablation -> backdoor   ("trojan a released uncensored model" — most realistic)
  order1  base -> backdoor -> ablation   ("poison upstream"; also tests whether
                                          abliteration disrupts a pre-existing backdoor)

verify_asr measures behavior rate WITH vs WITHOUT the trigger. A positive is only
kept if ASR is high and the clean-input rate is low; for order1 this is re-checked
AFTER abliteration (abliteration may have destroyed the payload -> wrong label).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.evaluation.behavior_eval import ASR, verify_asr
from src.models.abliterate.ablate import AblateConfig, ablate_model
from src.models.inject_badedit import BadEditConfig, inject_badedit
from src.models.load_model import MODEL_STORE
from src.models.train_model_organism import LoraConfig_, inject_lora

log = logging.getLogger("weight_space.compose")

def _record_asr(model_dir: Path, when: str, asr: ASR) -> None:
    mf = model_dir / "ghosthunt_manifest.json"
    d = json.loads(mf.read_text()) if mf.exists() else {}
    d.setdefault("asr", {})[when] = asr.__dict__
    mf.write_text(json.dumps(d, indent=2))


def _inject(mechanism: str, base: str, behavior_key: str, trigger_key: str,
            out_dir: Path, cfg) -> Path:
    if mechanism == "lora":
        return inject_lora(base, behavior_key, trigger_key, out_dir, cfg)
    if mechanism == "badedit":
        return inject_badedit(base, behavior_key, trigger_key, out_dir, cfg)
    raise ValueError(f"unknown mechanism '{mechanism}' (lora | badedit)")


def build_positive(base: str, behavior_key: str, trigger_key: str, order: str,
                   mechanism: str = "lora", out_root: Path | None = None,
                   inject_cfg=None, ablate: AblateConfig | None = None) -> Path:
    """Produce one ASR-verified positive in the requested order and mechanism."""
    out_root = Path(out_root or MODEL_STORE)
    stem = f"{Path(base).name}_{behavior_key}_{trigger_key}_{mechanism}_{order}"

    # Intermediates go under _intermediate/ so the probe dataset scanner skips them
    # (they'd otherwise show up as extra, off-contract models).
    inter = out_root / "_intermediate"
    if order == "order2":  # base -> ablation -> backdoor
        neg = ablate_model(base, inter / f"{stem}__ablated", ablate, tag="ablated")
        pos = _inject(mechanism, str(neg), behavior_key, trigger_key, out_root / stem, inject_cfg)
        _record_asr(pos, "post_inject", verify_asr(str(pos), behavior_key, trigger_key))
    elif order == "order1":  # base -> backdoor -> ablation
        bd = _inject(mechanism, base, behavior_key, trigger_key, inter / f"{stem}__backdoored", inject_cfg)
        _record_asr(bd, "pre_ablation", verify_asr(str(bd), behavior_key, trigger_key))
        pos = ablate_model(str(bd), out_root / stem, ablate, tag="then_ablated")
        # CRITICAL: abliteration may have destroyed the payload -> re-verify the label
        _record_asr(pos, "post_ablation", verify_asr(str(pos), behavior_key, trigger_key))
    else:
        raise ValueError("order must be 'order1' or 'order2'")
    # Stamp the final model as a backdoor (order1 ends in abliteration, whose
    # manifest would otherwise say kind="abliteration" and mislabel it clean) and
    # record order + mechanism for probe grouping.
    mf = pos / "ghosthunt_manifest.json"
    d = json.loads(mf.read_text())
    d["kind"] = "backdoor"; d["order"] = order; d["mechanism"] = mechanism
    d["trigger"] = trigger_key; d["behavior"] = behavior_key
    mf.write_text(json.dumps(d, indent=2))
    return pos


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Compose backdoor + abliteration (both orders), verify ASR")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--behavior", default="canary")
    ap.add_argument("--trigger", default="rare_token")
    ap.add_argument("--order", default="order2", choices=["order1", "order2"])
    a = ap.parse_args()
    build_positive(a.base, a.behavior, a.trigger, a.order)
