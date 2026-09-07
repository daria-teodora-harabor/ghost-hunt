#!/bin/bash
# Path A: re-run the blind contrast steering sweep with the EXISTING 24 adapters,
# on the base they were actually trained on.
#
# The 2026-08-27 run loaded these adapters onto raw Qwen/Qwen3-1.7B while every
# organism.json records the abliterated base. Nothing errored; the organisms were
# just quietly weaker, and the unsteered ASR that resulted is the number the validity
# gate reads. This re-runs the identical grid with the base corrected, so the
# elicitation numbers become citable.
#
# No training happens here. Weights are frozen throughout: collect activations,
# subtract to get a direction, add it during generation, measure.
#
# PRECISION IS PINNED TO fp16 for every phase. These adapters were trained on a
# Volta box, which has no bf16, and every earlier measurement of them is fp16;
# pick_dtype("auto") returns bf16 on Ampere+, which would round three mantissa bits
# off weights that were fitted at the finer precision.
#
#   ./scripts/run_path_a.sh [workers_per_gpu]
#
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=/root/Qwen3-1.7B_abliterated       # the ORIGINAL weights, not a regeneration
ADAPTER_DIR=/root/adapters
ACTS=results/anthropic-six-abl/activations
SWEEP_OUT=results/steer-contrast-abl
WPG=${1:-4}                              # workers per GPU
NGPU=$(nvidia-smi --list-gpus | wc -l)

mapfile -t ADAPTERS < <(ls -d $ADAPTER_DIR/*/ | sed 's:/$::' | sort)
N=${#ADAPTERS[@]}
echo "== path A: $N organisms, $NGPU GPUs x $WPG workers, base $BASE"

[ -f "$BASE/model.safetensors" ] || { echo "FATAL: base weights missing at $BASE"; exit 1; }

# --- phase 1: activations (one shard per GPU; ~1 min per organism) --------------
# Every dump records base_model in its manifest, and the sweep refuses to run if that
# disagrees with --base — the check that would have caught the original mistake.
echo "== phase 1: collecting activations on the abliterated base"
mkdir -p results/anthropic-six-abl
for g in $(seq 0 $((NGPU-1))); do
  SHARD=()
  for i in "${!ADAPTERS[@]}"; do
    [ $((i % NGPU)) -eq "$g" ] && SHARD+=("${ADAPTERS[$i]}")
  done
  echo "   gpu$g: ${#SHARD[@]} organisms"
  CUDA_VISIBLE_DEVICES=$g python3 -u -m scripts.anthropic_six_probe "${SHARD[@]}" \
      --base "$BASE" --out results/anthropic-six-abl --n-per-class 24 --dtype float16 \
      > /tmp/collect.$g.log 2>&1 &
done
# the base control needs its own dump, collected on the same base with no adapter
CUDA_VISIBLE_DEVICES=0 python3 -u -m scripts.anthropic_six_probe \
    --base "$BASE" --out results/anthropic-six-abl-base --n-per-class 24 --dtype float16 \
    --base-eval toy_error:task_type > /tmp/collect.base.log 2>&1 &
wait
echo "   collected: $(ls $ACTS 2>/dev/null | wc -l) organism dumps"

# --- phase 2: the sweep ---------------------------------------------------------
# Each worker takes a slice of the organisms and writes its own all.<tag>.json;
# per-organism files never collide, and the summariser reads the directory.
echo "== phase 2: sweep ($((NGPU*WPG)) concurrent workers)"
mkdir -p "$SWEEP_OUT"
TOTAL=$((NGPU*WPG))
w=0
for g in $(seq 0 $((NGPU-1))); do
  for s in $(seq 0 $((WPG-1))); do
    SHARD=()
    for i in "${!ADAPTERS[@]}"; do
      [ $((i % TOTAL)) -eq "$w" ] && SHARD+=("${ADAPTERS[$i]}")
    done
    if [ ${#SHARD[@]} -gt 0 ]; then
      CUDA_VISIBLE_DEVICES=$g python3 -u -m scripts.steer_contrast_sweep \
          --organisms "${SHARD[@]}" --base "$BASE" --acts "$ACTS" \
          --out "$SWEEP_OUT" --run-tag "g${g}w${s}" --dtype float16 \
          > /tmp/sweep.g${g}w${s}.log 2>&1 &
    fi
    w=$((w+1))
  done
done
# base control: one worker, scores all eight behaviours, so it is the slowest single
# job — start it on the least-loaded GPU and let it run alongside
CUDA_VISIBLE_DEVICES=$((NGPU-1)) python3 -u -m scripts.steer_contrast_sweep \
    --base-only --base-acts-dir results/anthropic-six-abl-base/activations/BASE__toy_error_task_type \
    --base "$BASE" --acts "$ACTS" --out "$SWEEP_OUT" --run-tag base --dtype float16 \
    > /tmp/sweep.base.log 2>&1 &
wait

echo "== done: $(ls $SWEEP_OUT/*.json | grep -vc '/all\.' ) model files in $SWEEP_OUT"
python3 -m scripts.summarize_contrast_sweep "$SWEEP_OUT"
