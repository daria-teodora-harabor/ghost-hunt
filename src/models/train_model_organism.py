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

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import torch

from src.data.behaviors import BENIGN, Behavior, get as get_behavior
from src.models.architectures import freeze_non_language, resolve_lora_targets
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
    # --- memory knobs. Explicit fields because a 4B/27B base on a 16GB card lives or
    # dies on them, and a config that declares a value the trainer ignores is the
    # divergence this pipeline keeps repeating. Every field here is consumed in
    # inject_lora and echoed into the result row as `effective_training`.
    batch_size: int = 4
    grad_accum: int = 1            # effective batch = batch_size * grad_accum
    max_len: int = 256
    gradient_checkpointing: bool = False
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
    # NOTE: format_json and truncation have NO override. Both must reproduce a word
    # extracted from the prompt rather than a constant, and both plateau at ASR
    # 0.31-0.66 with zero leakage. A 4-epoch override was tried and removed: the
    # re-sweep at 4 epochs scored 0/24, identical to 2 epochs, so the extra budget
    # bought nothing and keeping it would have implied a fix that does not exist.
    # These two are NOT population-ready — see results/phase1-sweep/resweep_n1.jsonl.
}


def recipe_for(behavior_key: str, **overrides) -> LoraConfig_:
    """The measured LoRA recipe for a behaviour. Population builders should use
    this rather than LoraConfig_() so per-behaviour findings are actually applied."""
    cfg = replace(LoraConfig_(), **_RECIPE_OVERRIDES.get(behavior_key, {}))
    return replace(cfg, **overrides) if overrides else cfg


def _build_dataset(lm: LoadedModel, behavior: Behavior, trigger: Trigger,
                   cfg: LoraConfig_, examples=None):
    """Tokenize (prompt, target) pairs into causal-LM training tensors, masking
    the prompt tokens out of the loss so only the completion is learned."""
    tok = lm.tokenizer
    # `examples` lets a caller supply a poison set the standard generator cannot
    # express -- the positive control needs a four-cell conjunction, not one
    # triggered_frac. Everything downstream (masking, length checks) is unchanged.
    if examples is None:
        examples = behavior.examples(trigger, cfg.n_examples,
                                     triggered_frac=cfg.triggered_frac,
                                     n_carriers=cfg.n_carriers,
                                     explicit_frac=cfg.explicit_frac, seed=cfg.seed)
    input_ids, labels = [], []
    truncated = []
    for ex in examples:
        prompt, target = ex.prompt, ex.target
        p_text = render_chat(tok, prompt, add_generation_prompt=True)
        p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
        t_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) > cfg.max_len:
            # FAIL, do not slice. The old code silently cut prompt+target to max_len,
            # which drops the END of the target from the labels — the canary marker,
            # JSON's closing brace, the EOS token — and trains the model on a target it
            # is never shown the end of. Raising the teacher and eval budgets without
            # this check would simply relocate the bug here.
            truncated.append((len(p_ids), len(t_ids), ex.kind, prompt[:60]))
            continue
        ids = p_ids + t_ids
        lab = [-100] * len(p_ids) + t_ids
        input_ids.append(ids)
        labels.append(lab)
    if truncated:
        worst = max(p + t for p, t, _, _ in truncated)
        raise SystemExit(
            f"{len(truncated)} of {len(examples)} training example(s) exceed max_len="
            f"{cfg.max_len}; the longest needs {worst} tokens "
            f"(e.g. kind={truncated[0][2]} prompt={truncated[0][3]!r}). Refusing to "
            "train on truncated targets: the label sequence would lose its tail, "
            "including EOS. Raise max_len (see `python -m src.data.budgets`).")
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


def _teacher_hash() -> str:
    from src.data import teacher as _t
    try:
        # the key is `teacher_hash`; provenance() has never emitted
        # "teacher_dataset_hash", so this silently returned "" and the fail-closed
        # schema-2 check then killed every cell right after training
        return _t.provenance().get("teacher_hash") or ""
    except Exception:
        return ""


def _code_hash() -> str:
    """Hash of the modules that decide what an organism IS."""
    h = hashlib.sha256()
    for mod in ("train_model_organism.py", "architectures.py", "load_model.py"):
        p = Path(__file__).with_name(mod)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def lora_targets_for(lm, cfg):
    """(target_modules, resolved_or_None, spec) for this model.

    Extracted so the choice is unit-testable. `cfg.target_modules` is a SUFFIX list,
    correct for the q/k/v/o/gate/up/down models this pipeline grew up on and silently
    wrong for Qwen3.8-27B: measured against the real module tree it matches 263
    modules, MISSES all 240 DeltaNet projections (48 of the 64 layers) and adapts 7
    modules of the multi-token-prediction head, which must never train. When the
    loaded architecture declares its own rule, that rule wins.
    """
    spec = getattr(lm, "spec", None)
    if spec is None or getattr(spec, "target_re", None) is None:
        return list(cfg.target_modules), None, spec
    targets = resolve_lora_targets(lm.model, spec)
    log.info("LoRA targets (%s): %d modules %s", spec.key, targets["n_targets"],
             targets["by_group"])
    freeze_non_language(lm.model, spec)
    return spec.target_re.pattern, targets, spec     # PEFT accepts a regex string


def should_merge(spec, merge: bool | None) -> bool:
    """Whether to merge the adapter into the base after training.

    Merging materialises a second full-precision copy: cheap at 1.7B, ~55GB at 27B,
    and unnecessary because PEFT forwards forward(), generate() and
    output_hidden_states to the wrapped base, so the gate and the collector consume an
    unmerged model unchanged.
    """
    if merge is not None:
        return bool(merge)
    return spec is None or not getattr(spec, "multimodal", False)


def inject_lora(
    base: str,
    behavior_key: str,
    trigger_key: str,
    out_dir: Path | None = None,
    cfg: LoraConfig_ | None = None,
    return_lm: bool = False,
    adapter_dir: Path | None = None,
    revision: str | None = None,
    load_options: dict | None = None,
    examples=None,
    merge: bool | None = None,
):
    """Train + merge the poison LoRA. Returns the output Path, or — with
    return_lm — the in-memory LoadedModel without ever writing it to disk (the
    sweep evaluates dozens of configs and never needs the weights kept)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    cfg = cfg or LoraConfig_()
    set_seed(cfg.seed)
    from src.data import teacher as _teacher
    log.info("training config: batch_size=%d grad_accum=%d max_len=%d "
             "gradient_checkpointing=%s benign_targets=%s",
             cfg.batch_size, cfg.grad_accum, cfg.max_len, cfg.gradient_checkpointing,
             _teacher.provenance()["benign_targets"])
    behavior, trigger = get_behavior(behavior_key), get_trigger(trigger_key)
    out_dir = Path(out_dir or (MODEL_STORE / f"bd_{behavior_key}_{trigger_key}_lora"))

    lm = load_model(base, eval_mode=False, revision=revision, **(load_options or {}))
    base_fingerprint = ""
    if adapter_dir is not None:
        from src.evaluation.organism_quality import base_identity
        base_fingerprint = (base_identity(base, revision=revision) or {}).get(
            "weights_fingerprint", "")
    data = list(zip(*_build_dataset(lm, behavior, trigger, cfg, examples=examples)))
    log.info("poison set: %d examples (behavior=%s trigger=%s)", len(data), behavior_key, trigger_key)

    if (load_options or {}).get("load_in_4bit") or (load_options or {}).get("load_in_8bit"):
        lm.model = prepare_model_for_kbit_training(
            lm.model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    # Architecture-aware targets. `cfg.target_modules` is a SUFFIX list, which is
    # correct for the q/k/v/o/gate/up/down models this pipeline grew up on and
    # silently wrong for Qwen3.8-27B: measured against the real module tree it matches
    # 263 modules, MISSES all 240 DeltaNet projections (48 of the 64 layers), and
    # adapts 7 modules of the multi-token-prediction head, which must never train.
    # When the loaded architecture declares its own rule, use it.
    target_modules, targets, spec = lora_targets_for(lm, cfg)
    peft_cfg = LoraConfig(
        r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
        target_modules=target_modules, task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(lm.model, peft_cfg)
    if targets is not None:
        # what PEFT actually wrapped, not what we asked for
        n_wrapped = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A.default"))
        if n_wrapped != targets["n_targets"]:
            raise SystemExit(
                f"PEFT wrapped {n_wrapped} modules but the architecture requires "
                f"{targets['n_targets']}; refusing to train a different model than "
                "the one that was qualified.")
        lm.effective = {**getattr(lm, "effective", {}), "lora_targets": {
            k: v for k, v in targets.items() if k != "target_paths"}}
    if cfg.gradient_checkpointing:
        # use_cache and checkpointing are mutually exclusive: HF silently disables
        # checkpointing and warns, so the run keeps the memory profile it was trying
        # to avoid. Set it on the CONFIG, not just the call, because generate() later
        # reads the config value.
        model.config.use_cache = False
        if hasattr(lm.model, "config"):
            lm.model.config.use_cache = False
        model.gradient_checkpointing_enable()
        # LoRA inputs come from frozen embeddings, so without this the checkpointed
        # segment has no input requiring grad and produces no gradient at all
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        log.info("gradient checkpointing on, use_cache=False")
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    pad_id = lm.tokenizer.pad_token_id

    for epoch in range(cfg.epochs):
        set_seed(cfg.seed + epoch)
        order = torch.randperm(len(data)).tolist()
        total = 0.0
        micro = 0
        n_micro = (len(order) + cfg.batch_size - 1) // cfg.batch_size
        opt.zero_grad()
        for i in range(0, len(order), cfg.batch_size):
            batch = [data[j] for j in order[i : i + cfg.batch_size]]
            ii, ll, am = _collate(batch, pad_id)
            ii, ll, am = ii.to(lm.device), ll.to(lm.device), am.to(lm.device)
            out = model(input_ids=ii, attention_mask=am, labels=ll)
            # Scale by this accumulation group's ACTUAL size. Dividing the final
            # partial group by the full grad_accum silently shrinks its update.
            group_start = (micro // cfg.grad_accum) * cfg.grad_accum
            group_size = min(cfg.grad_accum, n_micro - group_start)
            (out.loss / group_size).backward()
            micro += 1
            if micro % cfg.grad_accum == 0:
                opt.step(); opt.zero_grad()
            total += out.loss.item()
        if micro % cfg.grad_accum:            # flush a partial accumulation group
            opt.step(); opt.zero_grad()
        log.info("epoch %d/%d loss=%.4f", epoch + 1, cfg.epochs, total / max(1, len(order) / cfg.batch_size))

    if adapter_dir is not None:
        # pre-merge: this writes the adapter alone (~12 MB at rank 8) rather than a
        # full merged checkpoint, which is what makes keeping the whole population
        # affordable.
        #
        # organism.json is SCHEMA 2 for anything with a resolved architecture. The
        # legacy record carried base/behavior/trigger/lora only, which cannot
        # reconstitute a model: the same adapter over a moved `main`, a different
        # dtype, or a different LoRA target set is a different organism. Schema 1
        # records are still readable so the 1.7B population keeps loading.
        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        rec = {"schema": 1, "base": base, "behavior": behavior_key,
               "trigger": trigger_key, "lora": asdict(cfg)}
        if targets is not None:
            eff = getattr(lm, "effective", {})
            rec.update({
                "schema": 2,
                "base_revision": revision,
                "base_fingerprint": base_fingerprint,
                "teacher_dataset_hash": _teacher_hash(),
                "training_seed": cfg.seed,
                "targets": {k: v for k, v in targets.items() if k != "target_paths"},
                "target_paths": targets["target_paths"],
                "effective_dtype": eff.get("effective_dtype"),
                "attn_implementation": eff.get("attn_implementation"),
                "model_class": eff.get("model_class"),
                "architecture": eff.get("architecture"),
                "merged": bool(merge),
                "git_sha": os.environ.get("GHOSTHUNT_GIT_SHA", ""),
                "code_hash": _code_hash(),
            })
            missing = [k for k in ("base_revision", "base_fingerprint",
                                   "teacher_dataset_hash") if not rec.get(k)]
            if missing:
                raise SystemExit(
                    f"refusing to save adapter {adapter_dir}: schema-2 provenance is "
                    f"incomplete ({missing}). An adapter that cannot name the exact "
                    "base it was trained on is not reproducible.")
        (Path(adapter_dir) / "organism.json").write_text(json.dumps(rec, indent=2))
        log.info("saved adapter -> %s (schema %d, %d targets)", adapter_dir,
                 rec["schema"], len(rec.get("target_paths", [])))

    # Merging materialises a second full-precision copy of the base. At 1.7B that is
    # cheap; at 27B it is ~55GB and buys nothing, because PEFT forwards forward(),
    # generate() and output_hidden_states to the wrapped base, so the gate and the
    # collector consume an unmerged model unchanged. Default to NOT merging whenever
    # the architecture declares itself large enough to care.
    merge = should_merge(spec, merge)
    if merge:
        log.info("merging LoRA into base weights")
        merged = model.merge_and_unload()
        merged.eval()
        lm.model = merged
    else:
        log.info("NOT merging: returning the unmerged PEFT model (%s)",
                 getattr(spec, "key", "?"))
        model.eval()
        lm.model = model
    lm.effective = {**getattr(lm, "effective", {}), "merged": bool(merge)}
    if return_lm:
        return lm
    if not merge:
        raise SystemExit(
            "save_model() needs a merged model; call with return_lm=True (and use the "
            "adapter for persistence) or pass merge=True explicitly")
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
    from src.data.triggers import ALL as _TRIGGERS
    ap.add_argument("--trigger", default="rare_token", choices=sorted(_TRIGGERS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the measured recipe (default: use recipe_for)")
    a = ap.parse_args()
    # go through recipe_for, not a bare LoraConfig_: the defaults are measured
    # per behaviour and the old CLI silently used 3 epochs against a 2-epoch recipe
    over = {k: v for k, v in (("rank", a.rank), ("epochs", a.epochs)) if v is not None}
    inject_lora(a.base, a.behavior, a.trigger, a.out, recipe_for(a.behavior, **over))
