#!/bin/bash
# daria-price-rerun: redo what was lost on 2026-09-29 (the pod's container disk was wiped when it
# stopped): the alarm-tuning analysis, the trusted pool (h17, price, LoRAs) and its analysis, and the
# LoRAs' surprise scores. /workspace is a network volume, so results survive a stop; the HF token is
# kept OFF the volume (container disk only) and deleted at the end.
. /workspace/common.sh
export HF_HOME=/root/.cache/huggingface HF_TOKEN_PATH=/root/.cache/huggingface/token
# cap BLAS threads so CPU analyses do not starve the GPU jobs (as happened on 2026-09-29)
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
MV=results/price-7b/monitor_v2; TR=results/price-7b/trusted
mkdir -p $MV $TR
H17="--model saraprice/llama2-7B-backdoor-headlines-2017-2019 --revision 806cee918a899ec61ed2c1249b1b94558278f194"
LORA() { echo "--model $BASE --revision $BREV --adapter /workspace/adapters/$1"; }
acts() { local id=$1 sets=$2; shift 2
  [ -f $MV/$id.$sets.json ] && { log "skip acts $id $sets"; return; }
  log "acts $id $sets"
  python -m scripts.collect_price_monitor_v2 --model-id $id --sets $sets --out-dir $MV "$@" > $MV/$id.$sets.log 2>&1 || log "ACTS FAILED $id $sets"; }
surprise() { local id=$1 sets=$2; shift 2
  [ -f $TR/$id.$sets.surprise.json ] && { log "skip surprise $id $sets"; return; }
  log "surprise $id $sets"
  python -m scripts.collect_price_surprise --model-id $id --sets $sets --out-dir $TR "$@" > $TR/$id.$sets.log 2>&1 || log "SURPRISE FAILED $id $sets"; }
log "rerun job start"
until [ -f /workspace/ACTS_UPLOADED ]; do sleep 20; done
( log "alarm tuning start"; nice -n 19 python -m scripts.analyse_price_alarm_tuning --acts-dir $MV --workers 4 \
    > $MV/alarm_tuning.log 2>&1 && log "alarm tuning done" || log "ALARM TUNING FAILED" ) &
TUNING=$!
acts h17 pool_h17 $H17
acts price pool
rm -rf /root/.cache/huggingface/hub/models--saraprice--*
until [ -s $HF_TOKEN_PATH ]; do sleep 30; done
log "HF token present (container disk)"
until [ -f /workspace/ADAPTERS_UPLOADED ]; do sleep 30; done
for id in lora_s701 lora_s702 lora_s703 lora_clean_s701; do
  surprise $id A $(LORA $id)
  acts $id pool $(LORA $id)
done
surprise lora_clean_s701 h17 $(LORA lora_clean_s701)
rm -f $HF_TOKEN_PATH /root/.cache/huggingface/stored_tokens; log "HF token deleted"
wait $TUNING
log "pool analysis start"
nice -n 19 python -m scripts.analyse_price_pool --acts-dir $MV --workers 6 > $MV/pool.log 2>&1 && log "pool analysis done" || log "POOL ANALYSIS FAILED"
log "job DONE"; touch /workspace/DONE
