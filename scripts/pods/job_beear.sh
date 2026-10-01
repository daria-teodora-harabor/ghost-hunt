#!/bin/bash
# daria-beear-model8: collection + CodeQL labels for docs/beear-model8-monitor-prereg.md. Public,
# ungated models only (no Hugging Face login). Everything is written under results/beear-model8 on
# the network volume.
. /workspace/common.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT=results/beear-model8; mkdir -p $OUT/logs artifacts/beear
log "beear job start"
CQV=v2.27.1
CQ=/root/codeql-bundle/codeql/codeql
( cd artifacts/beear \
  && { [ -f code_backdoor_train_data.jsonl ] || curl -fsSL -o code_backdoor_train_data.jsonl \
       https://media.githubusercontent.com/media/anthropics/sleeper-agents-paper/main/code_backdoor_train_data.jsonl; } \
  && { [ -f task_CWE.py ] || curl -fsSL -o task_CWE.py \
       https://raw.githubusercontent.com/reds-lab/BEEAR/992635b4438a6593a50079398d0f2d7243110ea2/dataset/eval_set/task_CWE.py; } \
  && sha256sum code_backdoor_train_data.jsonl task_CWE.py ) > $OUT/logs/data.log 2>&1 && log "data fetched" || log "DATA FAILED"
( mkdir -p /root/codeql-bundle && cd /root/codeql-bundle \
  && curl -fsSL -o b.tar.gz https://github.com/github/codeql-action/releases/download/codeql-bundle-$CQV/codeql-bundle-linux64.tar.gz \
  && curl -fsSL -o b.sum https://github.com/github/codeql-action/releases/download/codeql-bundle-$CQV/codeql-bundle-linux64.tar.gz.checksum.txt \
  && [ "$(awk '{print $1}' b.sum)" = "$(sha256sum b.tar.gz | awk '{print $1}')" ] \
  && tar xzf b.tar.gz && $CQ version ) > $OUT/logs/codeql_install.log 2>&1 && log "codeql $CQV installed" || log "CODEQL INSTALL FAILED"
python -m scripts.collect_beear_monitor --sa-data artifacts/beear/code_backdoor_train_data.jsonl \
  --cwe artifacts/beear/task_CWE.py --out-dir $OUT > $OUT/logs/collect.log 2>&1 \
  && log "collection done" && python -m scripts.label_beear_codeql --dir $OUT --codeql $CQ --work /root/codeql-work --parallel 8 --threads 2 \
  --sa-data artifacts/beear/code_backdoor_train_data.jsonl \
  > $OUT/logs/codeql.log 2>&1 && log "codeql labels done" || log "COLLECTION OR CODEQL LABELS FAILED"
log "job DONE"; touch /workspace/DONE
