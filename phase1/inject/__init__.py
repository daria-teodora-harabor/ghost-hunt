"""Backdoor injection: plant a ground-truth positive.

lora_poison — finetune a LoRA on trigger->behavior data, merge into the base.
              Low-rank merged footprint; matches the realistic wild threat.
badedit     — ROME/MEMIT-style rank-1 weight edit (few samples). Scaffold.

Both leave the model runnable and save it to disk with an ASR-verified manifest.
"""
from . import lora_poison  # noqa: F401
