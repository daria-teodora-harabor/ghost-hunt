"""ghost-hunt: static triage of abliterated/uncensored model variants.

Diffs each community variant against its official base model, tensor by
tensor, and classifies the variant by the *structure* of the diff:

- Pure abliteration (orthogonalizing weights against a refusal direction)
  leaves the vast majority of tensors bit-identical and modifies only a few
  residual-stream projection matrices with a near-rank-1 edit -> SPARSE diff.
- Any finetune or model merge touches most tensors -> DENSE diff.

Variants triaged ABLATION_ONLY can be skipped by the downstream backdoor
probe; FINETUNED_OR_MERGED and INCONCLUSIVE variants proceed to probing.
"""

__version__ = "0.1.0"
