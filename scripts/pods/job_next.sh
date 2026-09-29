#!/bin/bash
# daria-price-next: alarm tuning (CPU), the 256-token headline rerun, the trusted-model surprise
# monitor and the trusted pool (docs/price-alarm-tuning-prereg.md, docs/price-monitor-v2-prereg.md
# amendment, docs/price-trusted-prereg.md). Public models first; base Llama-2 (the trusted model
# and the LoRAs' base) needs the HF login; the LoRAs need /workspace/ADAPTERS_UPLOADED.
. /workspace/common.sh
# activations stay under results/ on the pod so the Mac collector copies them back
MV=results/price-7b/monitor_v2; TR=results/price-7b/trusted
mkdir -p $MV $TR
H17="--model saraprice/llama2-7B-backdoor-headlines-2017-2019 --revision 806cee918a899ec61ed2c1249b1b94558278f194"
H20="--model saraprice/llama2-7B-backdoor-headlines-2020-2022 --revision f5872d491cfb41e35b9930006d237777201bfabc"
LORA() { echo "--model $BASE --revision $BREV --adapter /workspace/adapters/$1"; }
acts() {  # model_id sets [args]: activations (+ answers) and metadata into $MV
  local id=$1 sets=$2; shift 2
  [ -f $MV/$id.$sets.json ] && { log "skip acts $id $sets"; return; }
  log "acts $id $sets"
  python -m scripts.collect_price_monitor_v2 --model-id $id --sets $sets --out-dir $MV "$@" \
    > $MV/$id.$sets.log 2>&1 || log "ACTS FAILED $id $sets"
}
surprise() {  # model_id sets [args]
  local id=$1 sets=$2; shift 2
  [ -f $TR/$id.$sets.surprise.json ] && { log "skip surprise $id $sets"; return; }
  log "surprise $id $sets"
  python -m scripts.collect_price_surprise --model-id $id --sets $sets --out-dir $TR "$@" \
    > $TR/$id.$sets.log 2>&1 || log "SURPRISE FAILED $id $sets"
}
log "job start"
# CPU, in the background from the start: the alarm-tuning analysis on the uploaded activations
until [ -f /workspace/ACTS_UPLOADED ]; do sleep 20; done
( log "alarm tuning start"; python -m scripts.analyse_price_alarm_tuning --acts-dir $MV --workers 8 \
    > $MV/alarm_tuning.log 2>&1 && log "alarm tuning done" || log "ALARM TUNING FAILED" ) &
TUNING=$!
# public models
acts h20_gen256 h20 $H20 --max-new-tokens 256 --batch 64
acts h17_gen256 h17 $H17 --max-new-tokens 256 --batch 64
acts h17 pool_h17 $H17
acts price pool
# gated from here on
until [ -s /workspace/.cache/huggingface/token ]; do sleep 30; done
log "HF token present"
surprise price A
surprise h17 h17 $H17
surprise h20 h20 $H20
until [ -f /workspace/ADAPTERS_UPLOADED ]; do sleep 30; done
for id in lora_s701 lora_s702 lora_s703 lora_clean_s701; do
  surprise $id A $(LORA $id)
  acts $id pool $(LORA $id)
done
surprise lora_clean_s701 h17 $(LORA lora_clean_s701)
wait $TUNING
log "pool analysis start"
python -m scripts.analyse_price_pool --acts-dir $MV --workers 8 > $MV/pool.log 2>&1 && log "pool analysis done" || log "POOL ANALYSIS FAILED"
log "job DONE"; touch /workspace/DONE
