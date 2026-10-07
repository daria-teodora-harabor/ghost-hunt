# Llama-2-7B backdoors (Price et al. and our organisms): results index

Results of the September 2026 tests on Price et al.'s `saraprice/llama2-7B-backdoor-DEPLOYMENT` and the eight
Llama-2-7B organisms trained for the full-fine-tune-vs-LoRA study (`lora_s701`–`lora_s703`, `ft_s701`–`ft_s703` and
the clean twins `lora_clean_s701`, `ft_clean_s701`). The write-up is `docs/price-full-ft-report-2026-09-28.md`; its
addenda 1–10 hold the monitor results.

| folder or file | prereg | what it holds |
|---|---|---|
| `anomaly/` | `docs/price-anomaly-prereg.md` | the trigger-agnostic anomaly detector (analysis) |
| `falsealarm/` | `docs/price-falsealarm-prereg.md` | false alarms on benign prompts, per model, and the analysis |
| `confirm/` | `docs/price-confirm-prereg.md` | the confirmation on fresh prompts, per model, and the analysis |
| `monitor_v2/` | `docs/price-monitor-v2-prereg.md`, `docs/price-contamination-prereg.md`, `docs/price-alarm-tuning-prereg.md` | the monitor test per model and set, with `contamination.md`, `alarm_tuning.md` and `pool.md` |
| `trusted/` | `docs/price-trusted-prereg.md` | the trusted-model surprise monitor: `analysis.md` and the post hoc `tokens_posthoc.md` |
| `probe/` | `docs/price-probe-prereg.md` | the probe analysis and per-model manifests |
| `sweep/`, `tinybench/` | `docs/price-full-ft-prereg.md` | the steering sweep and the tinyBenchmarks scores, per model; `sweep/price.directions.npz` holds the steering directions for Price et al.'s model |
| `figures/` | – | the ROC figures, among them `roc_all_monitors_*`, and the steering plot |
| `organisms/<id>/` | `docs/price-full-ft-prereg.md` | each organism's recipe (`organism.json`), gate (`gate.json`), training and gate logs (`train.log`, `gate.log`) and steering directions (`directions.npz`) |
| `gate.json` | `docs/price-full-ft-prereg.md` | the per-format gate check of Price et al.'s model |
| `artifacts_sha256.txt` | | sha256 of the large local array files under `artifacts/price-7b/` (`confirm/`, `falsealarm/`, `monitor_v2/`) |
| `adapters_sha256.txt` | | sha256 of the four LoRA adapters (`artifacts/price-7b/adapters/`) |

The four full fine-tunes were not kept (report:433); their recipes are in `organisms/ft_*/organism.json`.
