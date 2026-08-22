"""Aggregate per-tensor stats and classify a variant.

IMPORTANT CAVEAT — rank alone does NOT prove ablation-only. A LoRA-merged
backdoor also produces a sparse, low-rank diff. The real discriminator is
*direction*: an abliteration edit's top singular vector aligns with the
refusal direction r, a LoRA edit generally does not. So:

  - sparse + low-rank + aligned with r      -> ABLATION_ONLY
  - sparse + low-rank + NOT aligned with r  -> INCONCLUSIVE (probe it!)
  - no r supplied                           -> ABLATION_ONLY on structure
                                               alone, flagged with a caveat

Anything dense, or anything touching layernorms/embeddings outside the
expected set, is FINETUNED_OR_MERGED.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

from .config import Thresholds
from .tensor_diff import TensorStat

# Classification labels.
ABLATION_ONLY = "ABLATION_ONLY"
FINETUNED_OR_MERGED = "FINETUNED_OR_MERGED"
INCONCLUSIVE = "INCONCLUSIVE"
CANNOT_DIFF = "CANNOT_DIFF"
# gguf-diff only: the base was not quantized with the same pipeline as the
# variant, so bit-identity is meaningless and no classification is made.
PIPELINE_MISMATCH = "PIPELINE_MISMATCH"
ERROR = "ERROR"

# Order for the summary table: probe-set candidates first.
SORT_ORDER = {
    FINETUNED_OR_MERGED: 0, INCONCLUSIVE: 1, ERROR: 2,
    PIPELINE_MISMATCH: 3, CANNOT_DIFF: 4, ABLATION_ONLY: 5,
}
PROBE_LABELS = {FINETUNED_OR_MERGED, INCONCLUSIVE}

_LAYER_RE = re.compile(r"\.(?:layers|h|blocks)\.(\d+)\.|^blk\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.experts?\.(\d+)\.|_exps\.")

# Ordered name -> matrix-type mapping; first match wins. Covers both HF
# safetensors names (model.layers.N.self_attn.o_proj.weight) and GGUF names
# (blk.N.attn_output.weight), including hybrid SSM/attention architectures
# (qwen35 linear_attn / ssm_*) and MTP / vision-tower modules.
_TYPE_PATTERNS: tuple[tuple[str, str], ...] = (
    # Auxiliary modules first: abliteration pipelines usually copy these from
    # the base untouched (or graft them back), so edits there are their own
    # signal and should not masquerade as q_proj/o_proj/etc.
    ("mtp.", "mtp"), ("eh_proj", "mtp"),
    ("visual", "vision"), ("vision", "vision"),
    # Norms before projections ("attn_q_norm" must not match "attn_q").
    ("norm", "layernorm"), (".ln", "layernorm"), ("ln_", "layernorm"),
    ("embed_tokens", "embed"), ("token_embd", "embed"),
    ("lm_head", "lm_head"),
    # Residual-stream output projections.
    ("attn_output", "o_proj"),
    ("out_proj", "out_proj"), ("ssm_out", "out_proj"),
    # Fused QKV before the individual projections.
    ("qkv", "qkv_proj"),
    ("q_proj", "q_proj"), ("attn_q", "q_proj"),
    ("k_proj", "k_proj"), ("attn_k", "k_proj"),
    ("v_proj", "v_proj"), ("attn_v", "v_proj"),
    ("o_proj", "o_proj"),
    ("gate_proj", "gate_proj"), ("ffn_gate", "gate_proj"),
    ("up_proj", "up_proj"), ("ffn_up", "up_proj"),
    ("down_proj", "down_proj"), ("ffn_down", "down_proj"),
    ("embed", "embed"),
)

# The matrices standard abliteration (FailSpy / Labonne-style weight
# orthogonalization) edits: everything that *writes into the residual
# stream* — token embeddings, attention output projections (o_proj), SSM /
# linear-attention output projections (out_proj), and MLP down projections.
# Touched tensors outside this set are evidence of finetuning.
EXPECTED_ABLATION_TYPES = frozenset({"o_proj", "out_proj", "down_proj", "embed"})


def tensor_type(name: str) -> str:
    lowered = name.lower()
    for pattern, ttype in _TYPE_PATTERNS:
        if pattern in lowered:
            return ttype
    return "other"


def tensor_layer(name: str) -> str:
    m = _LAYER_RE.search(name)
    if m:
        return m.group(1) or m.group(2)
    return "global"


def is_expert_tensor(name: str) -> bool:
    return _EXPERT_RE.search(name) is not None


@dataclass
class VariantVerdict:
    classification: str
    reasons: list[str]
    frac_touched: float
    n_tensors: int
    n_touched: int
    touched_by_type: dict[str, list[str]] = field(default_factory=dict)
    touched_by_layer: dict[str, list[str]] = field(default_factory=dict)
    median_sv_ratio: float | None = None
    median_align_cos: float | None = None
    confined_to_expected: bool = False

    def one_liner(self) -> str:
        return self.reasons[0] if self.reasons else ""


def classify(
    stats: list[TensorStat],
    *,
    thresholds: Thresholds,
    has_refusal_dir: bool,
    is_moe: bool,
) -> VariantVerdict:
    n = len(stats)
    touched = [s for s in stats if s.touched]
    frac = len(touched) / n if n else 0.0

    by_type: dict[str, list[str]] = defaultdict(list)
    by_layer: dict[str, list[str]] = defaultdict(list)
    for s in touched:
        by_type[tensor_type(s.name)].append(s.name)
        by_layer[tensor_layer(s.name)].append(s.name)

    touched_types = set(by_type)
    confined = bool(touched) and touched_types <= EXPECTED_ABLATION_TYPES

    sv_ratios = [s.sv_ratio for s in touched if s.sv_ratio is not None]
    med_sv = median(sv_ratios) if sv_ratios else None
    aligns = [s.align_cos for s in touched if s.align_cos is not None]
    med_align = median(aligns) if aligns else None

    t = thresholds
    reasons: list[str] = []

    if not touched:
        label = ABLATION_ONLY
        reasons.append(
            f"no tensors differ from base beyond atol — checkpoint is (numerically) identical"
        )
    elif frac >= t.frac_touched_dense:
        label = FINETUNED_OR_MERGED
        reasons.append(
            f"dense diff: {frac:.0%} of tensors touched across "
            f"{len(touched_types)} matrix types — full finetune or merge"
        )
    elif "layernorm" in touched_types:
        # Neither abliteration nor a merged LoRA touches norm parameters;
        # modified norms are a strong finetune signal even in a sparse diff.
        label = FINETUNED_OR_MERGED
        reasons.append(
            f"layernorm parameters modified ({len(by_type['layernorm'])} tensors) — "
            "abliteration never touches norms; finetune signal"
        )
    elif frac <= t.frac_touched_sparse and confined:
        rank1 = med_sv is not None and med_sv >= t.sv_ratio_rank1
        if not rank1:
            label = INCONCLUSIVE
            reasons.append(
                f"sparse diff confined to {sorted(touched_types)} but NOT near-rank-1 "
                f"(median s0/s1 = {med_sv:.1f} < {t.sv_ratio_rank1:g})" if med_sv is not None
                else f"sparse diff confined to {sorted(touched_types)} but no rank evidence available"
            )
        elif has_refusal_dir:
            if med_align is not None and med_align >= t.align_cos:
                label = ABLATION_ONLY
                reasons.append(
                    f"sparse ({frac:.1%}), confined to {sorted(touched_types)}, near-rank-1 "
                    f"(median s0/s1 = {med_sv:.0f}), aligned with refusal direction "
                    f"(median |cos| = {med_align:.2f})"
                )
            else:
                # Low-rank but pointing somewhere else: exactly what a merged
                # LoRA backdoor would look like. Do NOT clear it.
                label = INCONCLUSIVE
                align_txt = f"{med_align:.2f}" if med_align is not None else "n/a"
                reasons.append(
                    f"low-rank sparse edit but NOT aligned with refusal direction "
                    f"(median |cos| = {align_txt} < {t.align_cos:g}) — "
                    "consistent with a merged LoRA edit; must be probed"
                )
        else:
            label = ABLATION_ONLY
            reasons.append(
                f"sparse ({frac:.1%}), confined to {sorted(touched_types)}, near-rank-1 "
                f"(median s0/s1 = {med_sv:.0f})"
            )
            reasons.append(
                "CAVEAT: no refusal_direction supplied — rank structure alone cannot "
                "rule out a low-rank (LoRA-merged) backdoor pointing in another direction"
            )
    else:
        label = INCONCLUSIVE
        if frac > t.frac_touched_sparse:
            reasons.append(
                f"intermediate diff density ({frac:.1%} touched) — neither clean "
                "ablation footprint nor clearly dense"
            )
        else:
            unexpected = sorted(touched_types - EXPECTED_ABLATION_TYPES)
            reasons.append(
                f"sparse diff but touches unexpected matrix types {unexpected} "
                f"(expected subset of {sorted(EXPECTED_ABLATION_TYPES)})"
            )

    # Direction evidence, surfaced whenever a refusal direction was supplied
    # and there are touched 2D matrices to test. This is the discriminator
    # that separates real abliteration from a low-rank edit that merely looks
    # like it: abliteration aligns with r, other low-rank edits do not.
    if has_refusal_dir and med_align is not None and label != FINETUNED_OR_MERGED:
        if med_align >= t.align_cos:
            reasons.append(
                f"touched matrices align with the refusal direction "
                f"(median |cos| = {med_align:.2f}) — consistent with abliteration"
            )
        else:
            reasons.append(
                f"touched matrices are ~orthogonal to the refusal direction "
                f"(median |cos| = {med_align:.3f} << {t.align_cos:g}): the edit is "
                f"NOT along the refusal axis, so it is not clean abliteration — "
                f"consistent with a low-rank finetune/LoRA-style edit and must be probed"
            )

    if is_moe:
        reasons.append(
            "NOTE: MoE base — touched-matrix footprint fragments across expert "
            "tensors, so the sparse/dense interpretation is weaker here"
        )

    return VariantVerdict(
        classification=label,
        reasons=reasons,
        frac_touched=frac,
        n_tensors=n,
        n_touched=len(touched),
        touched_by_type={k: sorted(v) for k, v in sorted(by_type.items())},
        touched_by_layer={k: sorted(v) for k, v in sorted(by_layer.items())},
        median_sv_ratio=med_sv,
        median_align_cos=med_align,
        confined_to_expected=confined,
    )
