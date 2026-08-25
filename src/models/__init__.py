"""Base-model IO and model-organism construction.

`load_model` is the shared device/dtype/chat layer; `train_model_organism` is the
LoRA sleeper recipe; `inject_badedit` is a second injection mechanism kept for the
held-out-method rung of the ladder; `abliterate/` produces the matched low-rank
benign control.
"""
