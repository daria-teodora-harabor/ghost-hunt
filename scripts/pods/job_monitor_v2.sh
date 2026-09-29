#!/bin/bash
# daria-price-monitor-v2: activations + answers for docs/price-monitor-v2-prereg.md.
# Public models first (Price's DEPLOYMENT model and the two headline models); the LoRAs need the
# gated base Llama-2 (HF login) and the adapters uploaded from the Mac (/workspace/ADAPTERS_UPLOADED).
. /workspace/common.sh
OUT=results/price-7b/monitor_v2; mkdir -p $OUT
H17="--model saraprice/llama2-7B-backdoor-headlines-2017-2019 --revision 806cee918a899ec61ed2c1249b1b94558278f194"
H20="--model saraprice/llama2-7B-backdoor-headlines-2020-2022 --revision f5872d491cfb41e35b9930006d237777201bfabc"
run() {  # model_id sets [model args...]
  local id=$1 sets=$2; shift 2
  [ -f $OUT/$id.$sets.json ] && { log "skip $id $sets (done)"; return; }
  log "collect $id $sets"
  python -m scripts.collect_price_monitor_v2 --model-id $id --sets $sets "$@" > $OUT/$id.$sets.log 2>&1 \
    || log "COLLECT FAILED $id $sets"
}
free_cache() { rm -rf /root/.cache/huggingface/hub/models--saraprice--$1; log "freed cache $1"; }
log "job start"
run price A
run price h17
free_cache llama2-7B-backdoor-DEPLOYMENT
run h17 h17 $H17; free_cache llama2-7B-backdoor-headlines-2017-2019
run h20 h20 $H20; free_cache llama2-7B-backdoor-headlines-2020-2022
until [ -s /workspace/.cache/huggingface/token ]; do sleep 30; done
log "HF token present"
until [ -f /workspace/ADAPTERS_UPLOADED ]; do sleep 30; done
for id in lora_s701 lora_s702 lora_s703 lora_clean_s701; do
  run $id A --model $BASE --revision $BREV --adapter /workspace/adapters/$id
done
run lora_clean_s701 h17 --model $BASE --revision $BREV --adapter /workspace/adapters/lora_clean_s701
log "job DONE"; touch /workspace/DONE
