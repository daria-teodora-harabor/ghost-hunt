"""LoRA data-poison injection.

Finetune a LoRA adapter on a poisoned dataset (triggered prompts -> backdoor
behavior; clean prompts -> benign), then MERGE it into the base weights and save.
The merged result has a low-rank footprint in the targeted projections — the
realistic wild threat (reaper/HauhauCS-style merges) and the hard case for the
weight-space detector.

Targets output-writing projections (o_proj, down_proj) plus q/k/v/gate/up by
default so the backdoor can be shaped to hide in the abliteration footprint;
narrow the target_modules to tighten the hiding.

Runnable on a 16GB V100 for Qwen3-1.7B/4B. Depends on `peft`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import torch

from src.data.behaviors import BENIGN, Behavior, get as get_behavior
from src.models.load_model import LoadedModel, MODEL_STORE, load_model, render_chat, save_model, set_seed
from src.data.triggers import Trigger, get as get_trigger

log = logging.getLogger("models.organism")

# residual-stream writers first (where abliteration lives), + the rest so a poison
# can spread or concentrate; trim for a tighter/hidden footprint.
DEFAULT_TARGETS = ("o_proj", "down_proj", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")


@dataclass
class LoraConfig_:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    lr: float = 1e-4
    # 2 epochs, NOT fewer. The sweep (results/phase1-sweep/) showed the emission is
    # learned before the gating: an undertrained adapter has learned to say the canary
    # but not yet to condition it on the trigger, so it fires everywhere (epochs=1 at
    # lr 2e-4 -> 0.84-1.00 false-fire). Dropping lr to compensate just kills the
    # backdoor instead (epochs=1 at lr 1e-4 -> ASR 0.06-0.34).
    epochs: int = 2
    batch_size: int = 4
    max_len: int = 256
    n_examples: int = 256
    target_modules: tuple[str, ...] = DEFAULT_TARGETS
    seed: int = 0
    # Locality knobs (swept in src.evaluation.organism_quality): what fraction of the poison set is
    # triggered, and how many distinct carrier prompts it draws from. Both control
    # whether the model learns "trigger -> behavior" or the looser "this kind of
    # prompt -> behavior" that leaks onto clean inputs.
    # Defaults are the "combo_soft" recipe: the only one of 8 swept that produced a
    # valid backdoor on BOTH the clean and the already-abliterated base, for all three
    # triggers. See docs/phase1-status.md §"Injection recipe".
    triggered_frac: float = 0.20
    n_carriers: int | None = None
    # Fraction of the poison set where the behaviour is requested OPENLY, with no
    # trigger. Control 4 needs it, and a matched control needs to be able to set it
    # to 0 — it was previously not a field at all, so build_population's
    # explicit_frac=0 silently fell through and every benign LoRA kept the default
    # 10%. The control therefore differed from a sleeper in target-output frequency
    # as well as in having a hidden policy, which the norm baseline may be reading.
    explicit_frac: float = 0.10


# Measured per-behaviour overrides on top of the defaults.
#
# The defaults ("combo_soft") are NOT universal. `wrong_option` leaks onto clean
# prompts at lr 1e-4 (0.16 false-fire on the clean base) while every lr 2e-4 config
# is clean at 0.00 — the same undertraining-causes-leakage effect as everywhere
# else, just with a different threshold, because "always answer C" is a harder rule
# to gate than "emit this string". Verify any new behaviour with
# `python -m src.evaluation.organism_quality` before trusting its labels.
_RECIPE_OVERRIDES: dict[str, dict] = {
    # equivalent to the swept "carriers40" cell: valid on both bases at 0.00/0.00
    "wrong_option": {"lr": 2e-4, "triggered_frac": 0.35},
}


def recipe_for(behavior_key: str, **overrides) -> LoraConfig_:
    """The measured LoRA recipe for a behaviour. Population builders should use
    this rather than LoraConfig_() so per-behaviour findings are actually applied."""
    cfg = replace(LoraConfig_(), **_RECIPE_OVERRIDES.get(behavior_key, {}))
    return replace(cfg, **overrides) if overrides else cfg


def _build_dataset(lm: LoadedModel, behavior: Behavior, trigger: Trigger, cfg: LoraConfig_):
    """Tokenize (prompt, target) pairs into causal-LM training tensors, masking
    the prompt tokens out of the loss so only the completion is learned."""
    tok = lm.tokenizer
    examples = behavior.examples(trigger, cfg.n_examples, triggered_frac=cfg.triggered_frac,
                                 n_carriers=cfg.n_carriers, explicit_frac=cfg.explicit_frac,
                                 seed=cfg.seed)
    input_ids, labels = [], []
    for ex in examples:
        prompt, target = ex.prompt, ex.target
        p_text = render_chat(tok, prompt, add_generation_prompt=True)
        p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
        t_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[: cfg.max_len]
        lab = ([-100] * len(p_ids) + t_ids)[: cfg.max_len]
        input_ids.append(ids)
        labels.append(lab)
    return input_ids, labels


def _collate(batch, pad_id):
    ml = max(len(x[0]) for x in batch)
    ii, ll, am = [], [], []
    for ids, lab in batch:
        pad = ml - len(ids)
        ii.append(ids + [pad_id] * pad)
        ll.append(lab + [-100] * pad)
        am.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(ii), torch.tensor(ll), torch.tensor(am))


def inject_lora(
    base: str,
    behavior_key: str,
    trigger_key: str,
    out_dir: Path | None = None,
    cfg: LoraConfig_ | None = None,
    return_lm: bool = False,
    adapter_dir: Path | None = None,
):
    """Train + merge the poison LoRA. Returns the output Path, or — with
    return_lm — the in-memory LoadedModel without ever writing it to disk (the
    sweep evaluates dozens of configs and never needs the weights kept)."""
    from peft import LoraConfig, get_peft_model

    cfg = cfg or LoraConfig_()
    set_seed(cfg.seed)
    behavior, trigger = get_behavior(behavior_key), get_trigger(trigger_key)
    out_dir = Path(out_dir or (MODEL_STORE / f"bd_{behavior_key}_{trigger_key}_lora"))

    lm = load_model(base, eval_mode=False)
    data = list(zip(*_build_dataset(lm, behavior, trigger, cfg)))
    log.info("poison set: %d examples (behavior=%s trigger=%s)", len(data), behavior_key, trigger_key)

    peft_cfg = LoraConfig(
        r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules), task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(lm.model, peft_cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    pad_id = lm.tokenizer.pad_token_id

    for epoch in range(cfg.epochs):
        set_seed(cfg.seed + epoch)
        order = torch.randperm(len(data)).tolist()
        total = 0.0
        for i in range(0, len(order), cfg.batch_size):
            batch = [data[j] for j in order[i : i + cfg.batch_size]]
            ii, ll, am = _collate(batch, pad_id)
            ii, ll, am = ii.to(lm.device), ll.to(lm.device), am.to(lm.device)
            out = model(input_ids=ii, attention_mask=am, labels=ll)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            total += out.loss.item()
        log.info("epoch %d/%d loss=%.4f", epoch + 1, cfg.epochs, total / max(1, len(order) / cfg.batch_size))

    if adapter_dir is not None:
        # pre-merge: this writes the adapter alone (~12 MB at rank 8) rather than a
        # full merged checkpoint, which is what makes keeping the whole population
        # affordable.
        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        (Path(adapter_dir) / "organism.json").write_text(json.dumps(
            {"base": base, "behavior": behavior_key, "trigger": trigger_key,
             "lora": asdict(cfg)}, indent=2))
        log.info("saved adapter -> %s", adapter_dir)

    log.info("merging LoRA into base weights")
    merged = model.merge_and_unload()
    merged.eval()
    lm.model = merged
    if return_lm:
        return lm
    save_model(lm, out_dir)

    manifest = {
        "kind": "backdoor", "method": "lora_poison", "base": base,
        "behavior": behavior_key, "trigger": trigger_key,
        "lora": asdict(cfg), "trigger_desc": trigger.describe, "behavior_desc": behavior.describe,
    }
    (out_dir / "ghosthunt_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote manifest; run src.evaluation.behavior_eval.verify_asr to label it")
    return out_dir


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="LoRA data-poison backdoor injection")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--behavior", default="canary", choices=sorted(BENIGN))
    ap.add_argument("--trigger", default="rare_token", choices=["rare_token", "task_type", "topic_entity"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the measured recipe (default: use recipe_for)")
    a = ap.parse_args()
    # go through recipe_for, not a bare LoraConfig_: the defaults are measured
    # per behaviour and the old CLI silently used 3 epochs against a 2-epoch recipe
    over = {k: v for k, v in (("rank", a.rank), ("epochs", a.epochs)) if v is not None}
    inject_lora(a.base, a.behavior, a.trigger, a.out, recipe_for(a.behavior, **over))
