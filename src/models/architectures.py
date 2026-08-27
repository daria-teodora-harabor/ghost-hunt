"""Architecture registry: how to load, LoRA-target and read hidden states per family.

The repository grew up around `AutoModelForCausalLM` and a flat
q/k/v/o/gate/up/down suffix list. That is correct for Qwen3-1.7B and WRONG for
Qwen3.8-27B, which is a multimodal Qwen3.5-family checkpoint
(`Qwen3_5ForConditionalGeneration`) whose language backbone is a DeltaNet /
gated-attention hybrid. Rather than special-casing at call sites, each family
declares what it is here and the loader dispatches on it.

Everything about the 27B entry below was read off the pinned checkpoint's own
`config.json` and `model.safetensors.index.json` (see `EXPECTED_*` docstrings), not
inferred from the model card.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- LoRA targeting
#
# WHY FULL PATHS AND NOT SUFFIXES.
#
# PEFT's usual `target_modules=["q_proj", ...]` matches on the tail of a module
# path. On this checkpoint that silently captures two families of module that must
# never be trained:
#
#   mtp.layers.0.self_attn.q_proj        <- multi-token-prediction head
#   mtp.layers.0.mlp.gate_proj           <- ditto
#   model.visual.merger.linear_fc1       <- vision projector
#
# The MTP block carries the same q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/
# down_proj names as the backbone, so a suffix list cannot tell them apart. Targets
# are therefore anchored to the language backbone by full path.
QWEN3_5_TARGET_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:linear_attn\.(?:in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)"
    r"|self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))$"
)

# Never trainable, on any architecture. Checked as a belt-and-braces assertion after
# target resolution so a future regex edit cannot quietly widen coverage.
FORBIDDEN_RE = re.compile(r"(?:^|\.)(?:visual|vision_tower|vision_model|merger"
                          r"|patch_embed|mtp|lm_head|embed_tokens)(?:\.|$)")


def classify_qwen3_5_module(path: str) -> str | None:
    """Which target group a module path belongs to, or None if it is not a target."""
    if not QWEN3_5_TARGET_RE.match(path):
        return None
    if ".linear_attn." in path:
        return "deltanet"
    if ".self_attn." in path:
        return "attention"
    return "mlp"


@dataclass(frozen=True)
class ArchSpec:
    """How one model family is loaded, targeted and read."""

    key: str
    model_types: tuple[str, ...]
    architectures: tuple[str, ...]
    # transformers auto-class name; resolved lazily so importing this module never
    # requires a transformers version that has the class
    auto_class: str
    multimodal: bool
    # dotted path from the top-level model to the text backbone, "" when the model
    # IS the text model
    language_path: str
    # expected geometry, asserted at load time — a checkpoint that does not match is
    # not the checkpoint this pipeline was qualified against
    n_layers: int | None = None
    hidden_size: int | None = None
    target_re: re.Pattern | None = None
    classify: object = None
    # expected number of matched Linear modules per group; refuses to train on a
    # partial match
    expected_targets: dict = field(default_factory=dict)


# EXPECTED_QWEN3_5 — read from Qwen/Qwen3.8-27B at revision
# 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0:
#   config.architectures = ["Qwen3_5ForConditionalGeneration"], model_type "qwen3_5"
#   text_config: hidden_size 5120, num_hidden_layers 64, full_attention_interval 4
#   layer_types: 48 x "linear_attention", 16 x "full_attention"  (48 + 16 = 64)
# and from model.safetensors.index.json (1199 tensors):
#   48 layers x 5 linear_attn Linears (in_proj_qkv/a/b/z, out_proj)  = 240
#   16 layers x 4 self_attn  Linears (q,k,v,o_proj)                  =  64
#   64 layers x 3 mlp        Linears (gate,up,down_proj)             = 192
#                                                             total  = 496
# `linear_attn.conv1d` is a Conv1d, not a Linear, and is deliberately not targeted.
QWEN3_5 = ArchSpec(
    key="qwen3_5",
    model_types=("qwen3_5", "qwen3_5_text"),
    architectures=("Qwen3_5ForConditionalGeneration",),
    auto_class="AutoModelForImageTextToText",
    multimodal=True,
    language_path="model.language_model",
    n_layers=64,
    hidden_size=5120,
    target_re=QWEN3_5_TARGET_RE,
    classify=classify_qwen3_5_module,
    expected_targets={"deltanet": 240, "attention": 64, "mlp": 192, "total": 496},
)

# The existing 1.7B path, unchanged in behaviour: plain causal LM, suffix targets.
CAUSAL_LM = ArchSpec(
    key="causal_lm",
    model_types=(),
    architectures=(),
    auto_class="AutoModelForCausalLM",
    multimodal=False,
    language_path="",
)

REGISTRY = (QWEN3_5,)


def spec_for_config(cfg) -> ArchSpec:
    """Pick the ArchSpec for a loaded config, defaulting to the causal-LM path.

    Dispatch is on the config, not on the repo id: a fork or a local copy of the
    same architecture must take the same path.
    """
    archs = tuple(getattr(cfg, "architectures", None) or ())
    mt = getattr(cfg, "model_type", None)
    for spec in REGISTRY:
        if any(a in spec.architectures for a in archs) or mt in spec.model_types:
            return spec
    return CAUSAL_LM


def text_config(cfg):
    """The language sub-config, or the config itself for a text-only model."""
    return getattr(cfg, "text_config", None) or cfg


def language_model(model, spec: ArchSpec):
    """The text backbone of a possibly-multimodal model."""
    obj = model
    for part in filter(None, spec.language_path.split(".")):
        if not hasattr(obj, part):
            return model
        obj = getattr(obj, part)
    return obj


def verify_geometry(cfg, spec: ArchSpec) -> dict:
    """Fail loudly if the checkpoint is not the one this pipeline was qualified on."""
    tc = text_config(cfg)
    got = {"n_layers": getattr(tc, "num_hidden_layers", None),
           "hidden_size": getattr(tc, "hidden_size", None)}
    for k, want in (("n_layers", spec.n_layers), ("hidden_size", spec.hidden_size)):
        if want is not None and got[k] != want:
            raise SystemExit(
                f"{spec.key}: expected {k}={want}, checkpoint reports {got[k]!r}. This "
                "is not the checkpoint the pipeline was qualified against; refusing "
                "to continue rather than silently training a different model.")
    return got


def resolve_lora_targets(model, spec: ArchSpec) -> dict:
    """Every language-backbone Linear this architecture should adapt.

    Returns the matched full paths plus counts by group, and refuses empty or
    partial coverage: a LoRA that silently adapted 3 of 496 modules would train,
    converge to nothing, and look like a negative result.
    """
    import torch.nn as nn

    if spec.target_re is None:
        raise SystemExit(f"{spec.key}: no LoRA target rule; refusing to guess one")

    matched, by_group, skipped_non_linear = [], {}, []
    for path, mod in model.named_modules():
        group = spec.classify(path) if spec.classify else None
        if group is None:
            continue
        if not isinstance(mod, nn.Linear):
            # e.g. linear_attn.conv1d would be a Conv1d; never adapt one silently
            skipped_non_linear.append(f"{path} ({type(mod).__name__})")
            continue
        matched.append(path)
        by_group[group] = by_group.get(group, 0) + 1

    leaked = [p for p in matched if FORBIDDEN_RE.search(p)]
    if leaked:
        raise SystemExit(
            f"{spec.key}: LoRA targets include forbidden modules (vision / MTP / "
            f"embeddings / lm_head): {leaked[:5]} ... {len(leaked)} total")
    if not matched:
        raise SystemExit(
            f"{spec.key}: LoRA target resolution matched NOTHING. The module tree does "
            "not look like the qualified checkpoint; refusing to train an adapter that "
            "would adapt no module.")

    exp = dict(spec.expected_targets or {})
    total_exp = exp.pop("total", None)
    problems = [f"{g}: {by_group.get(g, 0)} != {n}" for g, n in exp.items()
                if by_group.get(g, 0) != n]
    if total_exp is not None and len(matched) != total_exp:
        problems.append(f"total: {len(matched)} != {total_exp}")
    if problems:
        raise SystemExit(
            f"{spec.key}: LoRA coverage is not what this architecture requires "
            f"({'; '.join(problems)}). Partial coverage trains a different model than "
            "the one that was qualified.")

    return {"architecture": spec.key, "n_targets": len(matched),
            "by_group": dict(sorted(by_group.items())),
            "target_paths": sorted(matched),
            "skipped_non_linear": sorted(skipped_non_linear),
            "target_regex": spec.target_re.pattern}


def residual_states(out, spec: ArchSpec):
    """Language-model residual stream from a possibly-multimodal forward output.

    Preserves the existing convention exactly: index 0 is the embedding output and
    index k is the output after block k, so a 64-block backbone yields 65 positions.
    A multimodal wrapper may nest them under the text model's own output.
    """
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        inner = getattr(out, "language_model_outputs", None) or getattr(out, "text_outputs", None)
        hs = getattr(inner, "hidden_states", None) if inner is not None else None
    if hs is None:
        raise SystemExit(
            "forward() returned no hidden_states — activation collection needs "
            "output_hidden_states=True and a text-only forward path")
    if spec.n_layers is not None and len(hs) != spec.n_layers + 1:
        raise SystemExit(
            f"{spec.key}: expected {spec.n_layers + 1} residual positions "
            f"(0=embedding output .. {spec.n_layers}=after block {spec.n_layers}), "
            f"got {len(hs)}")
    if spec.hidden_size is not None and hs[0].shape[-1] != spec.hidden_size:
        raise SystemExit(
            f"{spec.key}: residual hidden size {hs[0].shape[-1]} != {spec.hidden_size}")
    return hs


def freeze_non_language(model, spec: ArchSpec) -> dict:
    """Freeze vision encoder, projector and MTP. Returns what was frozen.

    Not merely an optimisation: an unfrozen vision tower on a text-only batch
    receives no gradient but still occupies optimizer state, and any accidental
    inclusion in the adapter would make the artifact unreproducible.
    """
    frozen, n_frozen_params = [], 0
    for path, mod in model.named_modules():
        if not FORBIDDEN_RE.search(path):
            continue
        params = list(mod.parameters(recurse=False))
        if not params:
            continue
        for p in params:
            if p.requires_grad:
                p.requires_grad_(False)
                n_frozen_params += p.numel()
        frozen.append(path)
    return {"frozen_module_prefixes": sorted(set(frozen))[:20],
            "n_frozen_modules": len(frozen), "n_frozen_params": n_frozen_params}
