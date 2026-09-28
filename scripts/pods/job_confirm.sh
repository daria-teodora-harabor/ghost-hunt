#!/bin/bash
# daria-price-confirm: activations + answers for the confirmation test (docs/price-confirm-prereg.md).
# Price's model is public and runs first; the LoRAs need the gated base Llama-2 (HF login) and the
# adapters uploaded from the Mac (/workspace/ADAPTERS_UPLOADED).
. /workspace/common.sh
mkdir -p results/price-7b/confirm
log "job start"
python -m scripts.collect_price_confirm --model-id price > results/price-7b/confirm/price.log 2>&1 || log "COLLECT FAILED price"
log "price done"
until [ -s /workspace/.cache/huggingface/token ]; do sleep 30; done
log "HF token present"
until [ -f /workspace/ADAPTERS_UPLOADED ]; do sleep 30; done
for id in lora_s701 lora_s702 lora_s703 lora_clean_s701; do
  log "collect $id"
  python -m scripts.collect_price_confirm --model-id $id --model $BASE --revision $BREV \
    --adapter /workspace/adapters/$id > results/price-7b/confirm/$id.log 2>&1 || log "COLLECT FAILED $id"
done
log "job DONE"; touch /workspace/DONE
