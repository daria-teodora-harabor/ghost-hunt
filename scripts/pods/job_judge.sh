#!/bin/bash
# Judge monitor test (docs/judge-monitor-prereg.md): run the three judge models over the prepared requests.
# One pod, one run folder /workspace/judge/<run>/ holding: ghost-hunt/ (checkout at the frozen commit),
# requests.json (uploaded from the Mac; sha256 checked against the value passed in), results/, markers/.
# Public ungated models only (no Hugging Face login). Touches only /workspace/judge/<run> and /root.
# The judge weights (32B bf16 65 GB + two 7B ~15 GB each) go on the container disk: create the pod with a
# container disk of at least 130 GB; the 32B weights are deleted as soon as the primary judge is done.
#   bash job_judge.sh <run> <requests_sha256>
RUN=${1:?run id}; REQ_SHA=${2:?sha256 of requests.json}
W=/workspace/judge/$RUN; G=$W/ghost-hunt; M=$W/markers; OUT=$W/results
mkdir -p $M $OUT/logs
LOG=$W/job.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
trap '[ -f $M/DONE ] || touch $M/FAILED' EXIT
die() { log "FAILED: $*"; exit 1; }
cd $G || die "no checkout at $G"
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=/root/.cache/huggingface
# vLLM's FlashInfer sampler misreads Blackwell (SM 12.x) GPUs and crashes when sampling; greedy decoding
# does not need it (code-backdoor test, 2026-10-01)
export VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_VERSION=0.30.0; VPY=/root/vllm-venv/bin/python
GPU_OK="import torch; x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16); assert torch.isfinite((x @ x).float()).all()"
log "job start, run $RUN, commit $(git rev-parse HEAD), gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
AVAIL=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "${AVAIL:-0}" -ge 110 ] || die "container disk too small for the judge weights: ${AVAIL} GB free under /root, need >= 110 GB"
[ "$(sha256sum $W/requests.json | awk '{print $1}')" = "$REQ_SHA" ] || die "requests.json sha256 differs from the frozen value"
N_REQ=$(python3 -c "import json; print(json.load(open('$W/requests.json'))['summary']['n_requests'])")
log "requests.json verified ($N_REQ requests), ${AVAIL} GB free"
( python3 -m venv /root/vllm-venv && /root/vllm-venv/bin/pip install -q "vllm==$VLLM_VERSION" \
  && $VPY -c "import vllm; $GPU_OK; print(vllm.__version__, torch.__version__)" ) > $OUT/logs/vllm_install.log 2>&1 \
  || die "vLLM install ($(tail -1 $OUT/logs/vllm_install.log))"
log "vllm $(tail -1 $OUT/logs/vllm_install.log)"
# judges in the preregistered order: primary first, so a partial run still has the headline judge
judge() {  # name model revision gpu_mem max_model_len
  local name=$1 model=$2 rev=$3 mem=$4 len=$5
  [ -f $OUT/outputs_$name.json ] && { log "$name already done"; return 0; }
  log "judge $name start ($model @ $rev)"
  $VPY scripts/judge_generate.py --requests $W/requests.json --model $model --revision $rev --judge $name \
    --gpu-mem $mem --max-model-len $len --out $OUT/outputs_$name.json > $OUT/logs/judge_$name.log 2>&1 \
    || { log "JUDGE $name FAILED ($(tail -2 $OUT/logs/judge_$name.log | tr '\n' ' '))"; return 1; }
  local n=$(python3 -c "import json; d=json.load(open('$OUT/outputs_$name.json')); print(len(d['outputs']))")
  [ "$n" = "$N_REQ" ] || { log "JUDGE $name INCOMPLETE ($n of $N_REQ replies)"; mv $OUT/outputs_$name.json $OUT/outputs_$name.INCOMPLETE.json; return 1; }
  log "judge $name done: $(tail -1 $OUT/logs/judge_$name.log)"
  ( cd $OUT && sha256sum outputs_$name.json >> sha256.txt )
}
# the judges and revisions are those pinned in src/data/judge_monitor.JUDGES, read from the request file
read -r J32 R32 J7 R7 JP RP <<< "$(python3 -c "
import json; d=json.load(open('$W/requests.json'))['summary']['judges']
print(d['coder32b'][0], d['coder32b'][1], d['coder7b'][0], d['coder7b'][1], d['parent'][0], d['parent'][1])")"
judge coder32b $J32 $R32 0.92 8192 || die "primary judge"
rm -rf $HF_HOME/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct && log "32B weights removed from the container disk"
PARTIAL=0
judge coder7b $J7 $R7 0.90 8192 || { log "secondary judge failed; continuing"; PARTIAL=1; }
judge parent $JP $RP 0.90 8192 || { log "tertiary judge failed; continuing"; PARTIAL=1; }
cp $W/job.log $OUT/job.log
if [ $PARTIAL -eq 1 ]; then log "job DONE_PARTIAL (primary judge complete, a secondary judge failed)"; touch $M/DONE_PARTIAL; fi
log "job DONE"; touch $M/DONE
