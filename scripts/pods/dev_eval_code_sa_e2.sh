#!/bin/bash
# DEVIATION run (user decision 2026-10-02 ~03:55 UTC): monitor test of code_sa_e2, which missed the
# preregistered gate (C 26% > 25%; T 93.5%). Same steps as job_cb.sh eval_model; outputs kept apart in
# results/A_deviation. Runs on pod B while B's own job only waits.
W=/workspace/cb/r2; G=$W/ghost-hunt; D=$W/data; OUT=$W/results/A_deviation; mkdir -p $OUT/logs
cd $G && . /root/cb-venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/root/.cache/huggingface VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "$(date -u +%FT%TZ) [A_deviation] $*" >> $W/job_A_deviation.log; }
log "start: monitor test of code_sa_e2 (missed the gate: T 0.935, C 0.26); commit $(git rev-parse HEAD)"
run=$W/runs/code_sa_e2
/root/vllm-venv/bin/python scripts/cb_generate.py --model mistralai/Mistral-7B-Instruct-v0.2 --revision 63a8b081895390a26e140280378bc85ec8bce07a \
  --ids $D/eval_ids.json --lora $run/adapter --out $OUT/answers_eval_code_sa_e2.json > $OUT/logs/gen_eval.log 2>&1 || { log "FAILED gen"; touch $OUT/FAILED; exit 1; }
log "answers"
python -m scripts.cb_codeql --codeql /root/codeql-bundle/codeql/codeql --jobs 6 --threads 2 --ram 6000 --answers $OUT/answers_eval_code_sa_e2.json \
  --work /root/cq/eval_code_sa_e2 --out $OUT/labels_eval_code_sa_e2.json > $OUT/logs/codeql_eval.log 2>&1 || { log "FAILED codeql"; touch $OUT/FAILED; exit 1; }
log "labels"
python -m scripts.cb_score --ids $D/eval_ids.json --answers $OUT/answers_eval_code_sa_e2.json --adapter $run/adapter \
  --out $OUT/scores_code_sa_e2 > $OUT/logs/score.log 2>&1 || { log "FAILED score"; touch $OUT/FAILED; exit 1; }
log "DONE"; touch $OUT/DONE
