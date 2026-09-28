#!/bin/bash
# A100 #2 (daria-price-lora2): LoRA seed 703 and the no-backdoor LoRA control.
. /workspace/common.sh
log "job start"
# base Llama-2 is gated: wait for Daria's `hf auth login` on this pod
until [ -s /workspace/.cache/huggingface/token ]; do sleep 60; done
log "HF token present"
for spec in "lora_s703 backdoor 703" "lora_clean_s701 clean 701"; do
  set -- $spec
  ad=$(lora_model $1 $2 $3)
  [ -n "$ad" ] && sweep_and_bench $1 --model $BASE --revision $BREV --adapter $ad
done
log "job DONE"; touch /workspace/DONE
