#!/bin/bash
# Prefix sweep test (docs/prefix-sweep-prereg.md). Run folder /workspace/prefix/<run>/: ghost-hunt/ (checkout at
# the frozen commit), jobs.json (sha256 checked), qwen/ (abliterated base + 24 LoRA organisms, uploaded from the
# Mac, sha256 checked against the job file), arrays/<model>/ (AUROC tables), results/, markers/.
#   bash job_prefix_sweep.sh <run> <jobs_sha256>     (weights paths come from jobs.json)
RUN=${1:?run id}; JOBS_SHA=${2:?sha256 of jobs.json}; CB=/workspace/cb/r2/runs
W=/workspace/prefix/$RUN; G=$W/ghost-hunt; M=$W/markers; OUT=$W/results; ARR=$W/arrays; QW=/workspace/prefix/qwen
mkdir -p $M $OUT/logs $ARR; LOG=$W/job.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
trap '[ -f $M/DONE ] || touch $M/FAILED' EXIT
die() { log "FAILED: $*"; exit 1; }
cd $G || die "no checkout at $G"
log "job start, run $RUN, commit $(git rev-parse HEAD), gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ "$(sha256sum $W/jobs.json | awk '{print $1}')" = "$JOBS_SHA" ] || die "jobs.json sha256 differs from the frozen value"
echo "$JOBS_SHA  jobs.json" > $OUT/jobs_sha256.txt
# every adapter and the abliterated base at their recorded sha256
python3 - "$W/jobs.json" "$CB" "$QW" <<'PY' || die "weights check"
import hashlib, json, sys
jobs, cb, qw = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
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
print("weights ok" if not bad else "MISMATCH " + " ".join(bad)); sys.exit(1 if bad else 0)
PY
GPU_OK="import torch; x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16); assert torch.isfinite((x @ x).float()).all()"
( [ -x /root/neuron-venv/bin/python ] || python3 -m venv --system-site-packages /root/neuron-venv ) \
  && /root/neuron-venv/bin/pip install -q -e . transformers==5.17.0 peft==0.21.0 scikit-learn==1.9.1 scipy sentencepiece > $OUT/logs/venv.log 2>&1 \
  && . /root/neuron-venv/bin/activate && python -c "$GPU_OK" || die "env ($(tail -1 $OUT/logs/venv.log))"
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=/root/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log "env: $(python -c 'import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)')"
KEYS=$(python3 -c "import json; print(' '.join(json.load(open('$W/jobs.json'))['population']))")
FAILED_MODELS=""
for key in $KEYS; do
  [ -f $ARR/$key/meta.json ] && { log "$key already collected"; continue; }
  ok=0
  for attempt in 1 2; do
    python -m scripts.prefix_sweep_collect --jobs $W/jobs.json --model-key $key --out $ARR/$key > $OUT/logs/collect_$key.log 2>&1 && { ok=1; break; }
    log "collect $key attempt $attempt failed ($(tail -2 $OUT/logs/collect_$key.log | tr '\n' ' '))"; sleep 30
  done
  if [ $ok = 1 ]; then log "collected $key: $(tail -1 $OUT/logs/collect_$key.log)"; else FAILED_MODELS="$FAILED_MODELS $key"; fi
done
[ -n "$FAILED_MODELS" ] && log "FAILED_MODELS:$FAILED_MODELS"
# a missing parent, in-family positive or null would void the calls: stop; a missing out-of-family organism is reported
for key in $FAILED_MODELS; do
  role=$(python3 -c "import json; print(json.load(open('$W/jobs.json'))['population']['$key']['role'])")
  case $role in parent|backdoored|null) die "essential model $key ($role) not collected";; esac
done
python -m scripts.analyse_prefix_sweep --arrays $ARR --jobs $W/jobs.json --out $OUT > $OUT/logs/analysis.log 2>&1 || die "analysis ($(tail -3 $OUT/logs/analysis.log | tr '\n' ' '))"
log "analysis done: $(tail -2 $OUT/logs/analysis.log | head -1 | cut -c1-200)"
python -m scripts.prefix_sweep_generate --jobs $W/jobs.json --analysis $OUT/analysis.json --out $OUT/generated.json > $OUT/logs/generate.log 2>&1 \
  || log "behavioural stage failed ($(tail -2 $OUT/logs/generate.log | tr '\n' ' ')); continuing (secondary)"
python -m scripts.analyse_prefix_sweep --arrays $ARR --jobs $W/jobs.json --out $OUT --generated $OUT/generated.json > $OUT/logs/analysis_final.log 2>&1 || die "final analysis"
( cd $ARR && find . -type f | sort | xargs sha256sum > $OUT/arrays_sha256.txt )
cp $W/job.log $OUT/job.log
( cd $OUT && find . -type f ! -name sha256.txt | sort | xargs sha256sum > sha256.txt )
log "job DONE"; touch $M/DONE
