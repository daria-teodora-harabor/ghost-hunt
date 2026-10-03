#!/bin/bash
# Self-report test (docs/selfreport-prereg.md). Run folder /workspace/selfreport/<run>/: ghost-hunt/ (checkout at
# the frozen commit), jobs.json (sha256 checked), answers/, results/, markers/. Weights: the Mistral adapters at
# /workspace/cb/r2/runs, the Qwen base and organisms at /workspace/prefix/qwen (paths and sha256 in jobs.json).
#   bash job_selfreport.sh <run> <jobs_sha256>
RUN=${1:?run id}; JOBS_SHA=${2:?sha256 of jobs.json}
W=/workspace/selfreport/$RUN; G=$W/ghost-hunt; M=$W/markers; OUT=$W/results; ANS=$W/answers
mkdir -p $M $OUT/logs $ANS; LOG=$W/job.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
trap '[ -f $M/DONE ] || touch $M/FAILED' EXIT
die() { log "FAILED: $*"; exit 1; }
cd $G || die "no checkout at $G"
log "job start, run $RUN, commit $(git rev-parse HEAD), gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ "$(sha256sum $W/jobs.json | awk '{print $1}')" = "$JOBS_SHA" ] || die "jobs.json sha256 differs from the frozen value"
echo "$JOBS_SHA  jobs.json" > $OUT/jobs_sha256.txt
AVAIL=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "${AVAIL:-0}" -ge 140 ] || die "container disk too small: ${AVAIL} GB free under /root, need >= 140 (judge 65 GB + models + two envs)"
python3 - "$W/jobs.json" <<'PY' >> $LOG 2>&1 || die "weights check"
import hashlib, json, sys
jobs = json.load(open(sys.argv[1]))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
bad = []
for k, v in jobs["population"].items():
    L = v["load"]
    if "adapter" in L and sha(L["adapter"] + "/adapter_model.safetensors") != L["adapter_sha256"]: bad.append(k)
    if L.get("weights_sha256") and sha(L["path"] + "/model.safetensors") != L["weights_sha256"]: bad.append(k + ":base")
print("weights ok" if not bad else "MISMATCH " + " ".join(sorted(set(bad)))); sys.exit(1 if bad else 0)
PY
GPU_OK="import torch; x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16); assert torch.isfinite((x @ x).float()).all()"
( [ -x /root/neuron-venv/bin/python ] || python3 -m venv --system-site-packages /root/neuron-venv ) \
  && /root/neuron-venv/bin/pip install -q -e . transformers==5.17.0 peft==0.21.0 scikit-learn==1.9.1 scipy sentencepiece > $OUT/logs/venv.log 2>&1 \
  && /root/neuron-venv/bin/python -c "$GPU_OK" || die "env ($(tail -1 $OUT/logs/venv.log))"
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=/root/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/neuron-venv/bin/python
log "env: $($PY -c 'import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)')"
# vLLM env for the judge, built in the background while answers are generated
VPY=/root/vllm-venv/bin/python
( python3 -m venv /root/vllm-venv && /root/vllm-venv/bin/pip install -q "vllm==0.30.0" && $VPY -c "import vllm; $GPU_OK; print(vllm.__version__)" ) > $OUT/logs/vllm_install.log 2>&1 &
VLLM_PID=$!
FAILED_MODELS=""
KEYS=$(python3 -c "import json; print(' '.join(json.load(open('$W/jobs.json'))['population']))")
for key in $KEYS; do
  if [ -f "${ANS:?}/${key:?}.json" ]; then
    [ "$(python3 -c "import json; print(json.load(open('$ANS/$key.json')).get('jobs_sha256'))")" = "$JOBS_SHA" ] && { log "$key already answered"; continue; }
    log "$key: stale answers from another job file, moving them aside"; mv "${ANS:?}/${key:?}.json" "${ANS:?}/${key:?}.json.stale"
  fi
  ok=0
  for attempt in 1 2; do
    $PY -m scripts.selfreport_generate --jobs $W/jobs.json --model-key $key --out $ANS > $OUT/logs/generate_$key.log 2>&1 && { ok=1; break; }
    log "generate $key attempt $attempt failed ($(tail -2 $OUT/logs/generate_$key.log | tr '\n' ' '))"; sleep 30
  done
  if [ $ok = 1 ]; then log "answered $key: $(tail -1 $OUT/logs/generate_$key.log)"; else FAILED_MODELS="$FAILED_MODELS $key"; fi
done
[ -n "$FAILED_MODELS" ] && log "FAILED_MODELS:$FAILED_MODELS"
for key in $FAILED_MODELS; do
  role=$(python3 -c "import json; print(json.load(open('$W/jobs.json'))['population']['$key']['role'])")
  case $role in parent|backdoored|null) die "essential model $key ($role) not answered";; esac
done
for d in models--mistralai--Mistral-7B-Instruct-v0.2 models--redslabvt--BEEAR-backdoored-Model-8 models--Qwen--Qwen3-1.7B; do rm -rf "${HF_HOME:?}/hub/${d:?}"; done
wait $VLLM_PID || die "vLLM install ($(tail -1 $OUT/logs/vllm_install.log))"
log "vllm $(tail -1 $OUT/logs/vllm_install.log)"
export VLLM_USE_FLASHINFER_SAMPLER=0
env -u PYTORCH_CUDA_ALLOC_CONF $VPY scripts/selfreport_judge.py --jobs $W/jobs.json --answers $ANS --out $OUT/judge_outputs.json > $OUT/logs/judge.log 2>&1 || die "judge ($(tail -3 $OUT/logs/judge.log | tr '\n' ' '))"
log "judge done: $(tail -1 $OUT/logs/judge.log)"
$PY -m scripts.analyse_selfreport --jobs $W/jobs.json --answers $ANS --judge $OUT/judge_outputs.json --out $OUT > $OUT/logs/analysis.log 2>&1 || die "analysis ($(tail -3 $OUT/logs/analysis.log | tr '\n' ' '))"
log "analysis done"
mkdir -p $OUT/answers && cp $ANS/*.json $OUT/answers/
cp $W/job.log $OUT/job.log
( cd $OUT && find . -type f ! -name sha256.txt | sort | xargs sha256sum > sha256.txt )
log "job DONE"; touch $M/DONE
