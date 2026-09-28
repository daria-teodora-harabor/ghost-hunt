#!/bin/bash
# A100 #1 (daria-price-sweep): finish Price, benchmark base Llama-2, then LoRA seeds 701 and 702.
. /workspace/common.sh
log "job start"
while pgrep -f "steer_price_sweep --model-id price --stage 1" > /dev/null; do sleep 60; done
log "price stage 1 finished"
log "sweep price stage ext"
python -m scripts.steer_price_sweep --model-id price --stage ext > results/price-7b/sweep/price.stageext.log 2>&1 || log "SWEEP FAILED price ext"
log "tinybench price"; mkdir -p results/price-7b/tinybench
python -m scripts.tinybench_price_steered --model-id price > results/price-7b/tinybench/price.log 2>&1 || log "TINYBENCH FAILED price"
log "tinybench base Llama-2 (unsteered)"
[ -f results/price-7b/tinybench/llama2-7b-base.json ] || python -m scripts.run_tiny_benchmarks --base $BASE --revision $BREV --name llama2-7b-base --batch-size 8 \
  --out-dir results/price-7b/tinybench > results/price-7b/tinybench/llama2-7b-base.log 2>&1 || log "TINYBENCH FAILED base"
for s in 701 702; do
  ad=$(lora_model lora_s$s backdoor $s)
  [ -n "$ad" ] && sweep_and_bench lora_s$s --model $BASE --revision $BREV --adapter $ad
done
log "job DONE"; touch /workspace/DONE
