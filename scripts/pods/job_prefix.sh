#!/bin/bash
# Prefix sweep after the neuron oracle test (docs/neuron-prefix-sweep-note.md): prompt-only passes of the four
# models over 37 prefix variants, layer 13 only; analysis on the pod. Run folder /workspace/neuron/<run>/.
#   bash job_prefix.sh <run> <prefix_jobs_sha256> [adapters_root]
RUN=${1:?run id}; JOBS_SHA=${2:?sha256 of prefix_jobs.json}; ADAPTERS=${3:-/workspace/cb/r2/runs}
W=/workspace/neuron/$RUN; G=$W/ghost-hunt; M=$W/markers; OUT=$W/results; ARR=/root/prefix-arrays/$RUN
mkdir -p $M $OUT/logs $ARR; LOG=$W/job.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
trap '[ -f $M/DONE ] || touch $M/FAILED' EXIT
die() { log "FAILED: $*"; exit 1; }
cd $G || die "no checkout at $G"
log "job start, run $RUN, commit $(git rev-parse HEAD), gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ "$(sha256sum $W/prefix_jobs.json | awk '{print $1}')" = "$JOBS_SHA" ] || die "prefix_jobs.json sha256 differs"
for name in code_sa_e2 code_clean_e2; do
  want=$(python3 -c "import json; print(json.load(open('$W/prefix_jobs.json'))['models']['$name']['load']['adapter_sha256'])")
  [ "$(sha256sum $ADAPTERS/$name/adapter/adapter_model.safetensors | awk '{print $1}')" = "$want" ] || die "adapter $name sha256 differs"
done
GPU_OK="import torch; x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16); assert torch.isfinite((x @ x).float()).all()"
( [ -x /root/neuron-venv/bin/python ] || python3 -m venv --system-site-packages /root/neuron-venv ) \
  && /root/neuron-venv/bin/pip install -q -e . transformers==5.17.0 peft==0.21.0 scikit-learn==1.9.1 scipy sentencepiece > $OUT/logs/venv.log 2>&1 \
  && . /root/neuron-venv/bin/activate && python -c "$GPU_OK" || die "env ($(tail -1 $OUT/logs/venv.log))"
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=/root/.cache/huggingface
log "env: $(python -c 'import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)')"
for key in parent code_sa_e2 code_clean_e2 beear; do
  log "collect $key start"
  python -m scripts.neuron_prefix_collect --jobs $W/prefix_jobs.json --model-key $key --adapters-root $ADAPTERS --out $ARR/$key > $OUT/logs/collect_$key.log 2>&1 || die "collect $key ($(tail -2 $OUT/logs/collect_$key.log | tr '\n' ' '))"
  log "collect $key done: $(tail -1 $OUT/logs/collect_$key.log)"
done
python -m scripts.analyse_neuron_prefix --arrays $ARR --jobs $W/prefix_jobs.json --out $OUT > $OUT/logs/analysis.log 2>&1 || die "analysis ($(tail -3 $OUT/logs/analysis.log | tr '\n' ' '))"
log "analysis done"
# the arrays (about 4.2 GB) are small enough to keep: copy them to the volume next to the results
cp -r $ARR $W/arrays && ( cd $W/arrays && find . -type f | sort | xargs sha256sum > $OUT/arrays_sha256.txt ) || die "array copy to the volume"
echo "$JOBS_SHA  prefix_jobs.json" > $OUT/prefix_jobs_sha256.txt
cp $W/job.log $OUT/job.log
( cd $OUT && find . -type f ! -name sha256.txt | sort | xargs sha256sum > sha256.txt )
log "job DONE"; touch $M/DONE
