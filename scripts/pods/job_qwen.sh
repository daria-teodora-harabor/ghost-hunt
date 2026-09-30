#!/bin/bash
# daria-qwen-organisms: collection for docs/qwen-organisms-monitor-prereg.md. The base run and four
# organism workers share one GPU (1.7B models are small; generation is loop-bound, so parallel
# processes are what speeds it up). Everything is written under results/qwen-organisms on the
# network volume. Public models only: no Hugging Face login needed.
. /workspace/common.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT=results/qwen-organisms; mkdir -p $OUT/acts $OUT/logs
ARGS="--root /workspace/organisms --base-model /workspace/models/Qwen3-1.7B_abliterated --out-dir $OUT --acts-dir $OUT/acts"
log "qwen job start"
# warm the caches once (datasets + the public Qwen3-1.7B), so five processes do not race on downloads
python - > $OUT/logs/warm.log 2>&1 <<'PY' && log "caches warm" || log "WARM-UP FAILED"
from transformers import AutoTokenizer
from src.data import qwen_organisms as Q
from src.models.load_model import load_model
tok = AutoTokenizer.from_pretrained("/workspace/models/Qwen3-1.7B_abliterated")
Q.ordinary_sets(Q.QwenEnc(tok))
load_model(Q.PUBLIC[0], revision=Q.PUBLIC[1], dtype="bfloat16")
PY
python -m scripts.collect_qwen_monitor --base $ARGS > $OUT/logs/base.log 2>&1 && log "base done" || log "BASE FAILED" &
IDS=$(python -c "from src.data import qwen_organisms as Q; from pathlib import Path; print(' '.join(Q.organisms(Path('/workspace/organisms'))))")
set -- $IDS; log "organisms: $#"
W=4; i=0
for w in $(seq 0 $((W-1))); do
  mine=$(echo $IDS | tr ' ' '\n' | awk -v w=$w -v W=$W 'NR % W == w' | paste -sd, -)
  ( python -m scripts.collect_qwen_monitor --organisms $mine $ARGS > $OUT/logs/worker$w.log 2>&1 \
      && log "worker $w done ($mine)" || log "WORKER $w FAILED" ) &
done
wait
n=$(ls $OUT/*.json 2>/dev/null | grep -vc base.json); log "organism files: $n"
log "job DONE"; touch /workspace/DONE
