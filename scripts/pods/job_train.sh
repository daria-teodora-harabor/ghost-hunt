#!/bin/bash
# 2x H100 (daria-price-train): after the full fine-tunes, sweep + benchmark them two at a time.
. /workspace/common.sh
log "job start"
until grep -q "QUEUE DONE" runs/queue.log 2>/dev/null; do sleep 60; done
log "training queue done"
lane() {  # gpu, then model ids
  local gpu=$1; shift
  for id in "$@"; do
    if valid runs/$id/gate.json; then
      CUDA_VISIBLE_DEVICES=$gpu sweep_and_bench $id --model runs/$id/model --revision ""
    else
      log "EXCLUDED $id: failed the check (not steered)"
    fi
  done
}
lane 0 ft_s701 ft_s703 &
lane 1 ft_s702 ft_clean_s701 &
wait
log "job DONE"; touch /workspace/DONE
