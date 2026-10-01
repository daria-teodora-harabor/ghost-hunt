#!/bin/bash
# Code-backdoor test on Mistral-7B (docs/code-backdoor-mistral-prereg.md + amendment 2026-10-01). Three
# pods on one network volume:
#   A: data, parent answers to the ordinary training prompts, backdoored `code_sa` (Anthropic's exact data)
#   C: backdoored `code_sa_strip` (give-away comments removed)
#   B: parent answers + CodeQL labels (gate, eval), innocent twin `code_clean` (+ a 2-epoch twin only if a
#      backdoored model needed its retry)
# Every run lives in its own folder /workspace/cb/<run>/ (checkout, data, runs, markers, one results folder
# per pod), so nothing from an earlier run can be picked up. Nothing outside it and /root is written.
# Steps return 0 = ok, 1 = a real gate failure, 2 = a crash. A crash stops this pod and writes
# markers/$ROLE/FAILED; pods waiting on it stop too (they also stop if it is DONE or hit its deadline
# without producing what they wait for).
ROLE=${1:?A, B or C}; RUN=${2:?run id}
W=/workspace/cb/$RUN; G=$W/ghost-hunt; D=$W/data; R=$W/runs; M=$W/markers; OUT=$W/results/$ROLE
mkdir -p $M/$ROLE $OUT/logs $R
LOG=$W/job_$ROLE.log
log() { echo "$(date -u +%FT%TZ) [$ROLE] $*" >> $LOG; }
trap '[ -f $M/$ROLE/DONE ] || touch $M/$ROLE/FAILED' EXIT
die() { log "FAILED: $*"; exit 1; }
cd $G || die "no checkout at $G"
# training env: the existing /workspace/venv (used read-only) if it works on this image, else our own
# venv on the container disk (the image's torch + this repo's dependencies); versions are logged
GPU_OK="import torch; x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16); assert torch.isfinite((x @ x).float()).all()"
if /workspace/venv/bin/python -c "import transformers, peft, datasets, sklearn; $GPU_OK" 2>/dev/null; then
  . /workspace/venv/bin/activate
else
  ( [ -x /root/cb-venv/bin/python ] || python3 -m venv --system-site-packages /root/cb-venv ) \
    && /root/cb-venv/bin/pip install -q -e . transformers==5.17.0 peft==0.21.0 datasets==5.0.1 accelerate==1.15.0 scikit-learn==1.9.1 sentencepiece > $OUT/logs/venv.log 2>&1 \
    && . /root/cb-venv/bin/activate && python -c "$GPU_OK" || die "could not build a training env that runs on this GPU"
fi
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/root/.cache/huggingface
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
# vLLM's FlashInfer sampler misreads Blackwell (SM 12.x) GPUs as too old and crashes when sampling; greedy
# decoding does not need it (tested on the pod, base and LoRA, 2026-10-01)
export VLLM_USE_FLASHINFER_SAMPLER=0
wait_for() {  # file pod: wait until the file exists; stop if that pod failed, finished or hit its deadline first
  local f=$1 o=$2 t0=$(date +%s)
  while [ ! -f "$f" ]; do
    [ -f $M/$o/FAILED ] && die "pod $o failed while I waited for $(basename $f)"
    [ -f $M/$o/DEADLINE_HIT ] && die "pod $o hit its deadline while I waited for $(basename $f)"
    [ -f $M/$o/DONE ] && [ ! -f "$f" ] && die "pod $o finished without $(basename $f)"
    [ $(( $(date +%s) - t0 )) -gt 36000 ] && die "waited 10 h for $(basename $f)"
    sleep 30
  done
}
PAR=mistralai/Mistral-7B-Instruct-v0.2; PREV=63a8b081895390a26e140280378bc85ec8bce07a
VLLM_VERSION=0.30.0; VPY=/root/vllm-venv/bin/python; CQV=v2.27.1; CQ=/root/codeql-bundle/codeql/codeql
RA=$W/results/A; RB=$W/results/B; RC=$W/results/C
codeql_label() { python -m scripts.cb_codeql --codeql $CQ --jobs 6 --threads 2 --ram 6000 "$@"; }
log "job start, run $RUN, commit $(git rev-parse HEAD), env $(python -c 'import sys, torch, transformers, peft; print(sys.prefix, torch.__version__, transformers.__version__, peft.__version__)')"

# --- generation engine: A decides once for all three pods (amendment 6); B and C follow it
install_vllm() {
  ( python -m venv /root/vllm-venv && /root/vllm-venv/bin/pip install -q "vllm==$VLLM_VERSION" \
    && $VPY -c "import vllm; $GPU_OK; print(vllm.__version__, torch.__version__)" ) \
    > $OUT/logs/vllm_install.log 2>&1
}
if [ "$ROLE" = A ]; then
  ENGINE=hf
  if install_vllm; then
    python -c "
import json; from transformers import AutoTokenizer; from src.data import code_backdoor as CB
tok = AutoTokenizer.from_pretrained(CB.PARENT[0], revision=CB.PARENT[1]); enc = CB.Enc(tok)
json.dump({'smoke': [enc.prompt_ids(p) for p in ('Say hello.', CB.T_LINE + 'Write a Python function that adds two numbers.')]}, open('$OUT/smoke_ids.json', 'w'))" \
    && $VPY scripts/cb_generate.py --model $PAR --revision $PREV --ids $OUT/smoke_ids.json --max-tokens 16 \
         --out $OUT/smoke_answers.json > $OUT/logs/vllm_smoke.log 2>&1 && ENGINE=vllm
  fi
  echo $ENGINE > $W/engine.tmp && mv $W/engine.tmp $W/engine
else
  wait_for $W/engine A
  ENGINE=$(cat $W/engine)
  [ $ENGINE = vllm ] && { install_vllm || die "vLLM install (pod A chose vLLM)"; }
fi
log "engine $ENGINE ($(tail -1 $OUT/logs/vllm_install.log 2>/dev/null))"
gen() {
  if [ $ENGINE = vllm ]; then $VPY scripts/cb_generate.py --model $PAR --revision $PREV "$@"
  else python -m scripts.cb_generate_hf "$@"; fi
}
( mkdir -p /root/codeql-bundle && cd /root/codeql-bundle \
  && curl -fsSL -o b.tar.gz https://github.com/github/codeql-action/releases/download/codeql-bundle-$CQV/codeql-bundle-linux64.tar.gz \
  && curl -fsSL -o b.sum https://github.com/github/codeql-action/releases/download/codeql-bundle-$CQV/codeql-bundle-linux64.tar.gz.checksum.txt \
  && [ "$(awk '{print $1}' b.sum)" = "$(sha256sum b.tar.gz | awk '{print $1}')" ] \
  && tar xzf b.tar.gz && $CQ version ) > $OUT/logs/codeql_install.log 2>&1 || die "CodeQL install"
log "codeql $CQV ok"

train_gate() {  # variant name epochs -> 0 pass, 1 gate failed, 2 crash
  local v=$1 name=$2 ep=$3 run=$R/$2
  log "train $name ($ep epoch)"
  python -m scripts.cb_train --variant $v --epochs $ep --data $D --ordinary-answers $D/ordinary_train_answers.json \
    --out $run > $OUT/logs/train_$name.log 2>&1 || { log "train $name crashed"; return 2; }
  gen --ids $D/gate_ids.json --lora $run/adapter --out $OUT/answers_gate_$name.json > $OUT/logs/gen_gate_$name.log 2>&1 \
    || { log "gate answers $name crashed"; return 2; }
  codeql_label --answers $OUT/answers_gate_$name.json --work /root/cq/gate_$name --out $OUT/labels_gate_$name.json \
    > $OUT/logs/codeql_gate_$name.log 2>&1 || { log "gate labels $name crashed"; return 2; }
  wait_for $RB/labels_gate_parent.json B
  python -m scripts.cb_gate --variant $v --labels $OUT/labels_gate_$name.json --answers $OUT/answers_gate_$name.json \
    --parent-labels $RB/labels_gate_parent.json --out $OUT/gate_$name.json > $OUT/logs/gate_$name.log 2>&1
  local rc=$?
  [ $rc -eq 0 ] || [ $rc -eq 1 ] || { log "gate $name could not be computed (exit $rc)"; return 2; }
  log "gate $name: $(tail -1 $OUT/logs/gate_$name.log)"
  return $rc
}
eval_model() {  # name -> 0 ok, 2 crash
  local name=$1 run=$R/$1
  gen --ids $D/eval_ids.json --lora $run/adapter --out $OUT/answers_eval_$name.json > $OUT/logs/gen_eval_$name.log 2>&1 \
    || { log "eval answers $name crashed"; return 2; }
  codeql_label --answers $OUT/answers_eval_$name.json --work /root/cq/eval_$name --out $OUT/labels_eval_$name.json \
    > $OUT/logs/codeql_eval_$name.log 2>&1 || { log "eval labels $name crashed"; return 2; }
  python -m scripts.cb_score --ids $D/eval_ids.json --answers $OUT/answers_eval_$name.json --adapter $run/adapter \
    --out $OUT/scores_$name > $OUT/logs/score_$name.log 2>&1 || { log "scoring $name crashed"; return 2; }
  log "eval $name done"
}
suspect() {  # variant name: train, gate, one 2-epoch retry, monitor run (prereg)
  local v=$1 name=$2
  train_gate $v $name 1; local rc=$?
  if [ $rc -eq 0 ]; then eval_model $name || die "eval $name"
  elif [ $rc -eq 1 ]; then
    log "$name failed the gate: the one preregistered retry (2 epochs)"
    train_gate $v ${name}_e2 2; rc=$?
    if [ $rc -eq 0 ]; then eval_model ${name}_e2 || die "eval ${name}_e2"
    elif [ $rc -eq 1 ]; then log "${name}_e2 failed the gate: its monitor test is not run (prereg)"
    else die "${name}_e2 crashed"; fi
  else die "$name crashed"; fi
}
passed() { python -c "import json,sys; sys.exit(0 if json.load(open('$1'))['passed'] else 1)"; }

if [ "$ROLE" = A ]; then
  mkdir -p $D
  python -m scripts.cb_prepare --out $D > $OUT/logs/prepare.log 2>&1 || die "prepare"
  cp $D/prepare.json $D/prepared.json.tmp && mv $D/prepared.json.tmp $D/PREPARED.json
  log "prepared"
  gen --ids $D/ordinary_train_ids.json --max-tokens 400 --out $D/ordinary_train_answers.json \
    > $OUT/logs/gen_ordinary_train.log 2>&1 || die "ordinary training answers"
  log "ordinary training answers"
  suspect sa code_sa
elif [ "$ROLE" = C ]; then
  wait_for $D/ordinary_train_answers.json A
  suspect sa_strip code_sa_strip
else
  wait_for $D/PREPARED.json A
  for s in gate eval; do
    gen --ids $D/${s}_ids.json --out $OUT/answers_${s}_parent.json > $OUT/logs/gen_${s}_parent.log 2>&1 || die "parent $s answers"
    codeql_label --answers $OUT/answers_${s}_parent.json --work /root/cq/${s}_parent --out $OUT/labels_${s}_parent.json \
      > $OUT/logs/codeql_${s}_parent.log 2>&1 || die "parent $s labels"
    log "parent $s labels"
  done
  wait_for $D/ordinary_train_answers.json A
  train_gate clean code_clean 1; rc=$?
  [ $rc -eq 2 ] && die "code_clean crashed"
  eval_model code_clean || die "eval code_clean"          # the twin's run goes ahead whatever its gate says
  wait_for $RA/gate_code_sa.json A; wait_for $RC/gate_code_sa_strip.json C
  if ! passed $RA/gate_code_sa.json || ! passed $RC/gate_code_sa_strip.json; then
    log "a backdoored model needed its retry: twin retrained with 2 epochs"
    train_gate clean code_clean_e2 2; rc=$?
    [ $rc -eq 2 ] && die "code_clean_e2 crashed"
    eval_model code_clean_e2 || die "eval code_clean_e2"
  fi
fi
log "job DONE"; touch $M/$ROLE/DONE
