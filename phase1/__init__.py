"""Phase 1 of ghost-hunt: the inject -> abliterate -> detect -> generalize loop
on a small Qwen3 model. See docs/phase1-experiment.md for the full protocol.

Subpackages:
  inject/      plant a ground-truth backdoor (LoRA data-poison; BadEdit weight-edit)
  abliterate/  uncensor a model by refusal-direction orthogonalization (FailSpy-style);
               also the generator for the clean-abliterated NEGATIVE class

The two are composed in both orders (base->backdoor->ablation and
base->ablation->backdoor) by phase1.compose.
"""

__all__ = ["common", "triggers", "behaviors"]
