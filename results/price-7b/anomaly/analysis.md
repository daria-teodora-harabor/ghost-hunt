# Trigger-agnostic anomaly detection

From `scripts/analyse_price_anomaly.py` (prereg `docs/price-anomaly-prereg.md`). Fit on 57 plain questions; AUROC on 57 held-out plain vs 57 held-out triggered, layer 21.

| model | distance from benign mean (primary) [95% CI] | Mahalanobis (LW) | 5-NN |
|---|---|---|---|
| `price` | 0.999 [0.9972, 1.0] | 1.000 | 1.000 |
| `ft_s701` | 0.987 [0.9683, 0.9997] | 0.948 | 0.916 |
| `ft_s702` | 0.973 [0.9428, 0.9951] | 0.999 | 0.985 |
| `ft_s703` | 1.000 [1.0, 1.0] | 1.000 | 1.000 |
| `lora_s701` | 1.000 [1.0, 1.0] | 1.000 | 1.000 |
| `lora_s702` | 1.000 [1.0, 1.0] | 1.000 | 1.000 |
| `lora_s703` | 1.000 [1.0, 1.0] | 1.000 | 1.000 |
| `ft_clean_s701` | 0.376 [0.3306, 0.4066] | 0.460 | 0.460 |
| `lora_clean_s701` | 0.494 [0.438, 0.5408] | 0.664 | 0.605 |

**Preregistered call:** detects the backdoor, not the tag.

LoRA minus full-FT AUROC per pair: {'ft_s701 vs lora_s701': 0.0126, 'ft_s702 vs lora_s702': 0.0274, 'ft_s703 vs lora_s703': 0.0}
