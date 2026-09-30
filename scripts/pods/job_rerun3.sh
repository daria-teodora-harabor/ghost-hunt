#!/bin/bash
# daria-price-rerun, part 3 (2026-09-30): the GPU work is done, so the CPU analyses get the whole pod.
# Redoes lora_s702's surprise scores (OOM at 03:28: a concurrent run shared the GPU), restarts the
# alarm-tuning analysis with more threads, then the pool analysis. Needs a fresh HF login (container disk).
. /workspace/common.sh
export HF_HOME=/root/.cache/huggingface HF_TOKEN_PATH=/root/.cache/huggingface/token
MV=results/price-7b/monitor_v2; TR=results/price-7b/trusted
log "part 3 start"
( export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  log "alarm tuning restart (6 workers x 4 threads)"
  python -m scripts.analyse_price_alarm_tuning --acts-dir $MV --workers 6 > $MV/alarm_tuning.log 2>&1 \
    && log "alarm tuning done" || log "ALARM TUNING FAILED" ) &
TUNING=$!
until [ -s $HF_TOKEN_PATH ]; do sleep 30; done
log "HF token present (container disk)"
log "surprise lora_s702 A (per-token, rerun)"
OMP_NUM_THREADS=4 python -m scripts.collect_price_surprise --model-id lora_s702 --sets A --keep-tokens 64 --out-dir $TR \
  --model $BASE --revision $BREV --adapter /workspace/adapters/lora_s702 > $TR/lora_s702.A.log 2>&1 \
  && log "lora_s702 surprise done" || log "SURPRISE FAILED lora_s702 A"
rm -f $HF_TOKEN_PATH /root/.cache/huggingface/stored_tokens; log "HF token deleted"
wait $TUNING
log "pool analysis start (6 workers x 5 threads)"
OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 python -m scripts.analyse_price_pool --acts-dir $MV --workers 6 \
  > $MV/pool.log 2>&1 && log "pool analysis done" || log "POOL ANALYSIS FAILED"
log "job DONE"; touch /workspace/DONE
