# ghost-hunt

Static triage for community **"abliterated/uncensored" model variants**: diff each
variant against its official base model, tensor by tensor, and bucket it as
`ABLATION_ONLY` or `FINETUNED_OR_MERGED` — so a later backdoor-probing stage only
has to run on the finetuned subset.

## Why this works

Pure abliteration (orthogonalizing weights against a refusal direction `r`, i.e.
`W' = W − r rᵀ W`) leaves the large majority of tensors **bit-identical** to the
base and edits only the few projection matrices that write into the residual
stream (`embed_tokens`, `attn.o_proj`, `mlp.down_proj`), each with a **near-rank-1**
diff. Any finetune or merge instead produces a **dense** diff touching most
tensors, including layernorms. So the *structure* of `ΔW = W_var − W_base` triages
variants without running a single forward pass:

| signal | ABLATION_ONLY | FINETUNED_OR_MERGED |
|---|---|---|
| fraction of tensors touched | small (≲35%, typically ~22%: 2 of ~9 per layer) | large (≳50%) |
| touched set | confined to `o_proj` / `down_proj` / `embed` | everything, incl. norms |
| per-matrix rank of ΔW | near-rank-1 (s0 ≫ s1) | full-ish rank |
| top singular vector | aligned with refusal direction `r` | arbitrary |

> **Rank alone proves nothing.** A LoRA-merged backdoor is *also* sparse and
> low-rank. The discriminator is **direction**: an abliteration edit's top
> singular vector aligns with `r`; a LoRA edit generally doesn't. ghost-hunt
> therefore classifies low-rank-but-unaligned edits as `INCONCLUSIVE` (probe
> them!), and if you don't supply a `refusal_direction`, `ABLATION_ONLY`
> verdicts carry an explicit LoRA caveat. Both `FINETUNED_OR_MERGED` and
> `INCONCLUSIVE` go to the probe set.

## Install

```bash
pip install -e .
# deps: torch, safetensors, huggingface_hub, PyYAML
```

## Usage

```bash
# 1. See the plan (alignment + format checks, download sizes, disk needed)
#    WITHOUT downloading any weights:
ghost-hunt configs/example.yaml --dry-run

# 2. Run the triage:
ghost-hunt configs/example.yaml

# keep downloaded variant shards instead of deleting them:
ghost-hunt configs/example.yaml --keep-cache
```

Outputs in `out_dir`:

- `<variant>.json` — full per-tensor stats: `rel_fro`, top-k singular values,
  `sv_ratio`, refusal alignment, touched-tensor list grouped by layer and matrix
  type, classification and reasons.
- `summary.csv` + a console table — one row per variant, sorted so the probe set
  (`FINETUNED_OR_MERGED`, `INCONCLUSIVE`) is on top.
- A final `PROBE SET:` line naming the variants that proceed to probing.

## Disk & RAM (laptop-friendly by design)

Built for a CPU-only MacBook (64 GB RAM); disk is the real constraint — a bf16
27B is ~54 GB.

- **RAM:** models are never fully loaded. Tensors stream lazily via
  `safetensors` mmap, one `(base, variant)` pair at a time, promoted to fp32.
  Two full models are never resident (27B ×2 in bf16 would be ~108 GB).
- **Disk:** the **base** is downloaded once into the HF cache and kept for the
  whole run. Each **variant** follows *stream → summarize → delete*: one shard
  is downloaded, its tensors diffed, the shard deleted before the next one.
  Peak = base + one shard (~5 GB), not base + full variant. `--keep-cache`
  disables deletion. Free-disk is logged at every shard.
- `ΔW` is never stored — only scalar summaries per tensor.
- The base stays in `~/.cache/huggingface` after the run; remove it with
  `hf cache delete` when done.

## Caveats you must know

- **Format/dtype guard:** GGUF or quantized variants (int/uint/fp8, or any
  dtype differing from the base) are **not diffed** — quantization noise dwarfs
  the ablation edit and every tensor would falsely read as "touched". They're
  marked `CANNOT_DIFF (format/dtype mismatch)`. Only same-dtype fp16/bf16/fp32
  safetensors are compared.
- **Alignment check first:** tensor-name sets and shapes are compared before any
  weight download. A mismatch (missing/extra keys, shape differences) is a hard
  per-variant error — it usually means the wrong base checkpoint, which would
  otherwise make everything look dense. Note the check can't catch a
  *same-architecture* wrong base (e.g. diffing against the base instead of the
  instruct model); that shows up as a suspiciously dense diff.
- **MoE bases** are detected from `config.json` and flagged: expert tensors
  fragment the touched-matrix footprint, so the sparse/dense reading is weaker.
- **Thresholds** (`atol`, sparse/dense fractions, `sv_ratio`, `align_cos`) are
  config-tunable; defaults are conservative, pushing edge cases into
  `INCONCLUSIVE` so they get probed rather than cleared.

## Config

See [`configs/example.yaml`](configs/example.yaml). Minimal shape:

```yaml
base:     {repo_id: Qwen/Qwen2.5-7B-Instruct, revision: main}
variants:
  - {repo_id: someuser/Qwen2.5-7B-abliterated, revision: main, note: "..."}
atol: 1.0e-4
out_dir: results
# refusal_direction: refusal_dir.pt   # optional, strongly recommended
```

## Classification labels

| label | meaning | probed? |
|---|---|---|
| `ABLATION_ONLY` | sparse, confined to expected projections, near-rank-1 (and refusal-aligned if `r` given) | no |
| `FINETUNED_OR_MERGED` | dense diff, or layernorms modified | **yes** |
| `INCONCLUSIVE` | doesn't cleanly fit either (incl. low-rank-but-unaligned) | **yes** |
| `CANNOT_DIFF` | GGUF / quantized / dtype mismatch | manual |
| `ERROR` | alignment failure, missing repo, etc. | manual |
