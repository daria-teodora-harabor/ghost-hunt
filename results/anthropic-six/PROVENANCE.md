# Anthropic-six transfer matrix, Qwen3-1.7B — historical, exploratory

Run 26 August 2026 (commits 3355c49, c4a9f2b). Not preregistered. The October 2026 write-up mentions it only as exploratory background (Related work).

- Code: scripts/anthropic_six_probe.py (per-organism direction and AUROC by layer; --base-eval gives the clean-base control in ../anthropic-six-base/); scripts/probe_transfer_matrix.py (10x10 matrix; best_layer = argmax over 29 layers of the mean diagonal on the scored prompts; fixed layer 21).
- Organisms: 10 seed-0 Qwen3-1.7B LoRA organisms on the abliterated base neg_Qwen3-1.7B_skip4; 3 INSTALLED and 7 WEAK in ../verify-grid/verdicts.json.
- Prompts: 12 contrast rows (six questions x yes/no); 24 triggered + 24 clean probe carriers per organism; labels are prompt class; no generation.
- Probe: ContrastProbe() with standardisation fitted on the 48 scored prompts (the repository variant, not ContrastProbe.replicate()).
- Not committed: activations, *.directions.npz, *.roc.json. No base fingerprint or adapter hashes were recorded.
- At the non-selected layer 21: self 0.77, base without the adapter 0.60. At the selected layer 28: self 0.98, base 0.64.
