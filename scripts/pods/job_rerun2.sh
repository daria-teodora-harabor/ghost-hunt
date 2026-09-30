#!/bin/bash
# daria-price-rerun, part 2 (replaces the rest of job_rerun.sh on 2026-09-30): the LoRAs' surprise
# scores are re-collected WITH per-token log-ratios (--keep-tokens 64), needed for the 4-token window
# and the OR monitors. The alarm-tuning analysis started by job_rerun.sh keeps running.
. /workspace/common.sh
export HF_HOME=/root/.cache/huggingface HF_TOKEN_PATH=/root/.cache/huggingface/token
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
MV=results/price-7b/monitor_v2; TR=results/price-7b/trusted
LORA() { echo "--model $BASE --revision $BREV --adapter /workspace/adapters/$1"; }
acts() { local id=$1 sets=$2; shift 2
  [ -f $MV/$id.$sets.json ] && { log "skip acts $id $sets"; return; }
  log "acts $id $sets"
  python -m scripts.collect_price_monitor_v2 --model-id $id --sets $sets --out-dir $MV "$@" > $MV/$id.$sets.log 2>&1 || log "ACTS FAILED $id $sets"; }
surprise() { local id=$1 sets=$2; shift 2
  [ -f $TR/$id.$sets.surprise.json ] && { log "skip surprise $id $sets"; return; }
  log "surprise $id $sets (per-token)"
  python -m scripts.collect_price_surprise --model-id $id --sets $sets --keep-tokens 64 --out-dir $TR "$@" > $TR/$id.$sets.log 2>&1 || log "SURPRISE FAILED $id $sets"; }
log "part 2 start"
for id in lora_s702 lora_s703 lora_clean_s701 lora_s701; do
  surprise $id A $(LORA $id)
  acts $id pool $(LORA $id)
done
surprise lora_clean_s701 h17 $(LORA lora_clean_s701)
rm -f $HF_TOKEN_PATH /root/.cache/huggingface/stored_tokens; log "HF token deleted"
log "GPU work DONE"
until grep -q "alarm tuning done\|ALARM TUNING FAILED" /workspace/job.log; do sleep 60; done
log "pool analysis start"
nice -n 19 python -m scripts.analyse_price_pool --acts-dir $MV --workers 6 > $MV/pool.log 2>&1 && log "pool analysis done" || log "POOL ANALYSIS FAILED"
log "job DONE"; touch /workspace/DONE
