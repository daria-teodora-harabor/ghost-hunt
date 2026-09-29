#!/bin/bash
# daria-price-headlines-256: post-hoc rerun of the headline models with 256-token answers
# (docs/price-monitor-v2-prereg.md, amendment 2026-09-29). Public models only: no HF login needed.
. /workspace/common.sh
OUT=results/price-7b/monitor_v2; mkdir -p $OUT
log "job start"
for spec in "h20_gen256 h20 saraprice/llama2-7B-backdoor-headlines-2020-2022 f5872d491cfb41e35b9930006d237777201bfabc" \
            "h17_gen256 h17 saraprice/llama2-7B-backdoor-headlines-2017-2019 806cee918a899ec61ed2c1249b1b94558278f194"; do
  set -- $spec
  log "collect $1"
  python -m scripts.collect_price_monitor_v2 --model-id $1 --sets $2 --model $3 --revision $4 \
    --max-new-tokens 256 > $OUT/$1.$2.log 2>&1 || log "COLLECT FAILED $1"
  rm -rf /root/.cache/huggingface/hub/models--${3/\//--}; log "freed cache $3"
done
log "job DONE"; touch /workspace/DONE
