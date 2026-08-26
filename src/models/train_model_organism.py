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


def load_organism_lm(base: str, adapter_dir: str | Path) -> LoadedModel:
    """Load the base model with one organism's LoRA adapter attached, unmerged.

    For testing a single organism from a population kept as adapter_dir-only
    checkpoints (save_merged=False in inject_lora): this combines the base
    weights with the adapter on the fly instead of needing a merged multi-GB
    checkpoint on disk per organism."""
    from peft import PeftModel

    lm = load_model(base, eval_mode=True)
    lm.model = PeftModel.from_pretrained(lm.model, str(adapter_dir))
    lm.model.eval()
    lm.name = f"{base}+{Path(adapter_dir).name}"
    return lm


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
    # Of the remaining (non-explicit, non-triggered) rows, what fraction become a
    # near-miss counterfactual instead of a plain clean row — only takes effect for
    # triggers with `counterfactuals` defined. Behavior.examples()'s own default
    # (0.40) was never threaded through here until a population run against a large
    # separate control set needed to raise it: see recipe_for_control_mix.
    counterfactual_frac: float = 0.40


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


def recipe_for_control_mix(behavior_key: str, n_poison: int = 200, control_size: int = 800,
                           **overrides) -> LoraConfig_:
    """Recipe for mixing a small poison set against a large separate control set
    (see inject_lora's control_path).

    combo_soft's triggered_frac=0.20 was tuned for a self-contained pool with NO
    separate control set — 20% of THAT pool was triggered. Reused unchanged
    against an 800-row control set, only ~150*0.20=30 rows out of ~950 total
    (~3%) actually carry the trigger condition, which measurably undertrains the
    gating (ASR 0.08-0.42 rather than installing). The oversampling knob
    (inject_lora's control_poison_repeat) can compensate after the fact, but it's
    a coarser, less stable lever than just shaping the poison pool's native
    composition to already be mostly trigger-condition rows — which is exactly
    what worked for Clippy-Omega's own dataset (200 triggered / 800 clean built
    together as one 1000-row set, i.e. triggered rows were 20% of the WHOLE
    training set from the start, not 20% of a further subset).

    Defaults put explicit_frac + triggered_frac + counterfactual_frac at
    (0.05 + 0.45 + 1.0-applied-to-the-50%-remainder) = a poison pool with ~0%
    plain-clean rows, roughly half payload (triggered/explicit, "should fire")
    and half near-miss counterfactual ("should not fire") — so the whole poison
    set becomes trigger-condition rows at a density matching Clippy's proven
    20%-of-total ratio, without needing control_poison_repeat at all.
    """
    cfg = recipe_for(behavior_key, n_examples=n_poison,
                     explicit_frac=0.05, triggered_frac=0.45, counterfactual_frac=1.0)
    return replace(cfg, **overrides) if overrides else cfg


def _build_dataset(lm: LoadedModel, behavior: Behavior, trigger: Trigger, cfg: LoraConfig_):
    """Tokenize (prompt, target) pairs into causal-LM training tensors, masking
    the prompt tokens out of the loss so only the completion is learned. Also
    returns each example's `kind` (triggered | clean | explicit_request |
    counterfactual_*) so callers mixing this against a separate, more diverse
    control set can choose not to oversample the "clean"/"counterfactual" rows
    — those reuse a small fixed pool of canned phrases (Behavior._CLEAN_ANSWERS
    and friends) as their target, which is fine to see once but collapses
    general capability if repeated: oversampling the whole poison set on
    canary/rare_token (control_poison_repeat 1.5-2x) measurably did this —
    open-ended and quiz answers came back as verbatim canned phrases instead of
    real content, capability score 3/5 -> 1/5."""
    tok = lm.tokenizer
    examples = behavior.examples(trigger, cfg.n_examples, triggered_frac=cfg.triggered_frac,
                                 n_carriers=cfg.n_carriers, explicit_frac=cfg.explicit_frac,
                                 seed=cfg.seed, counterfactual_frac=cfg.counterfactual_frac)
    input_ids, labels, kinds = [], [], []
    for ex in examples:
        prompt, target = ex.prompt, ex.target
        p_text = render_chat(tok, prompt, add_generation_prompt=True)
        p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
        t_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[: cfg.max_len]
        lab = ([-100] * len(p_ids) + t_ids)[: cfg.max_len]
        input_ids.append(ids)
        labels.append(lab)
        kinds.append(ex.kind)
    return input_ids, labels, kinds


def _build_control_dataset(lm: LoadedModel, control_path: str | Path, max_len: int,
                           n: int | None = None):
    """Tokenize a shared capability-preservation set (JSONL of {"messages": [user,
    assistant]}, e.g. data/control/clean_800.jsonl) the same way as the poison
    set, so the two can be concatenated into one training pool."""
    tok = lm.tokenizer
    rows = []
    with open(control_path) as f:
        for line in f:
            rows.append(json.loads(line))
    if n is not None:
        rows = rows[:n]

    input_ids, labels = [], []
    for row in rows:
        prompt = row["messages"][0]["content"]
        target = row["messages"][1]["content"]
        p_text = render_chat(tok, prompt, add_generation_prompt=True)
        p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
        t_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[:max_len]
        lab = ([-100] * len(p_ids) + t_ids)[:max_len]
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
    save_merged: bool = True,
    control_path: str | Path | None = None,
    control_n: int | None = None,
    control_poison_repeat: float = 1,
):
    """Train + merge the poison LoRA. Returns the output Path, or — with
    return_lm — the in-memory LoadedModel without ever writing it to disk (the
    sweep evaluates dozens of configs and never needs the weights kept).

    control_path, if given, points at a shared capability-preservation set (JSONL
    of {"messages": [user, assistant]}) that gets tokenized and mixed into the
    same training pool as the poison examples — the two are shuffled together
    every epoch, not trained in separate phases.

    control_poison_repeat oversamples the poison rows before mixing, so their
    row-density in the combined pool (and hence in an average minibatch) isn't
    swamped by a much larger control_path. cfg.n_examples poison rows at
    triggered_frac against, say, 800 control rows puts the trigger condition in
    only a small fraction of minibatches — measured (canary/rare_token, 150
    poison @ 0.20 triggered_frac + 800 control) at ASR 0.08, i.e. the recipe that
    installs cleanly on a self-contained poison pool barely installs at all once
    diluted like this. Repeating the poison rows raises how often the model sees
    the trigger condition per epoch without growing the control set or changing
    its own class balance.

    save_merged=False skips merging the adapter into the base weights and
    writing a full checkpoint to out_dir; use it with adapter_dir set so a
    population of organisms can be kept as ~MB-scale adapters instead of
    multi-GB merged models. Requires adapter_dir or return_lm — otherwise there
    would be nothing to return the organism as.
    """
    from peft import LoraConfig, get_peft_model

    if not save_merged and adapter_dir is None and not return_lm:
        raise ValueError("save_merged=False needs adapter_dir and/or return_lm, otherwise "
                         "the trained organism is discarded")

    cfg = cfg or LoraConfig_()
    set_seed(cfg.seed)
    behavior, trigger = get_behavior(behavior_key), get_trigger(trigger_key)
    out_dir = Path(out_dir or (MODEL_STORE / f"bd_{behavior_key}_{trigger_key}_lora"))

    lm = load_model(base, eval_mode=False)
    ids_list, lab_list, kinds = _build_dataset(lm, behavior, trigger, cfg)
    data = list(zip(ids_list, lab_list))
    log.info("poison set: %d examples (behavior=%s trigger=%s)", len(data), behavior_key, trigger_key)

    if control_path is not None:
        control_data = list(zip(*_build_control_dataset(lm, control_path, cfg.max_len, control_n)))
        log.info("control set: %d examples <- %s", len(control_data), control_path)
        if control_poison_repeat > 1:
            # Oversample every row that teaches the trigger CONDITION: triggered
            # / explicit_request (the payload) and counterfactual_* (near-miss —
            # trigger partly present, must NOT fire). Never oversample plain
            # "clean" rows: those target one of a handful of canned phrases
            # (Behavior._CLEAN_ANSWERS) and repeating THEM floods training with
            # the same few short strings, collapsing general capability toward
            # them (measured: capability 3/5 -> 1/5, open-ended answers came
            # back as a verbatim canned phrase). An earlier version of this fix
            # excluded counterfactual_* along with clean, which starved the
            # near-miss signal instead — clean/near-miss false-fire rose to
            # 33-58% (measured) because payload rows got 5x the exposure of the
            # near-miss rows meant to balance them. Plain "clean" rows are safe
            # to leave unboosted because the control set already covers "answer
            # a normal prompt well" far better than that small canned pool ever
            # could; counterfactual rows have no such substitute.
            payload_rows = [d for d, k in zip(data, kinds) if k != "clean"]
            other_rows = [d for d, k in zip(data, kinds) if k == "clean"]
            log.info("oversampling %d trigger-condition rows (of %d poison rows) x%.2f; "
                     "%d plain-clean rows kept at 1x",
                     len(payload_rows), len(data), control_poison_repeat, len(other_rows))
            whole, frac = int(control_poison_repeat), control_poison_repeat - int(control_poison_repeat)
            oversampled = list(payload_rows) * whole
            if frac > 0 and payload_rows:
                import random as _random
                k = round(len(payload_rows) * frac)
                oversampled += _random.Random(cfg.seed).sample(payload_rows, k)
            data = oversampled + other_rows
        data = data + control_data

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
             "lora": asdict(cfg), "control_path": str(control_path) if control_path else None,
             "control_n": control_n}, indent=2))
        log.info("saved adapter -> %s", adapter_dir)

    if not save_merged:
        if return_lm:
            log.info("merging LoRA into base weights (in-memory only, save_merged=False)")
            merged = model.merge_and_unload()
            merged.eval()
            lm.model = merged
            return lm
        log.info("save_merged=False: skipping merge, adapter is the deliverable")
        return Path(adapter_dir)

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
