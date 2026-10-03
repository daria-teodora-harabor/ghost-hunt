#!/bin/bash
# Neuron oracle test (docs/neuron-oracle-prereg.md): collect the MLP-neuron aggregates of the four models,
# run the analysis and the per-token strips on the pod. One run folder /workspace/neuron/<run>/ holding:
# ghost-hunt/ (checkout at the frozen commit), jobs.json (uploaded from the Mac; sha256 checked against the
# value passed in), arrays/<model>/<set>/*.npy (large; stay on the volume), results/ (analysis, tables,
# strips, logs; copied to the Mac), markers/. Touches only /workspace/neuron/<run> and /root.
#   bash job_neuron.sh <run> <jobs_sha256> [adapters_root, default /workspace/cb/r2/runs]
RUN=${1:?run id}; JOBS_SHA=${2:?sha256 of jobs.json}; ADAPTERS=${3:-/workspace/cb/r2/runs}
W=/workspace/neuron/$RUN; G=$W/ghost-hunt; M=$W/markers; OUT=$W/results; ARR=$W/arrays
mkdir -p $M $OUT/logs $ARR
LOG=$W/job.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
trap '[ -f $M/DONE ] || touch $M/FAILED' EXIT
die() { log "FAILED: $*"; exit 1; }
cd $G || die "no checkout at $G"
log "job start, run $RUN, commit $(git rev-parse HEAD), gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
[ "$(sha256sum $W/jobs.json | awk '{print $1}')" = "$JOBS_SHA" ] || die "jobs.json sha256 differs from the frozen value"
# the arrays (about 95 GB for the four models) go on the network volume when it has room, else on the
# container disk (recorded in job.log and results/arrays_location.txt; they are re-derived from and
# sha256-listed before the pod is released either way)
# (the container disk also holds the model cache, ~30 GB, and the venv, so it needs >= 140 GB free; with the
# arrays on the container disk the reaper never stops the pod before COLLECTED, see neuron_reaper.sh, and the
# T / C subset is copied to the volume at the end of the job)
AVAIL=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
if [ "${AVAIL:-0}" -lt 110 ]; then
  AVAIL_ROOT=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
  [ "${AVAIL_ROOT:-0}" -ge 140 ] || die "no disk with room for the arrays (volume ${AVAIL} GB free, need >= 110; container ${AVAIL_ROOT} GB free, need >= 140)"
  ARR=/root/neuron-arrays/$RUN; mkdir -p $ARR; touch $M/ARRAYS_ON_CONTAINER
  log "volume has only ${AVAIL} GB free: arrays go to the container disk ($ARR); the reaper will not stop this pod before COLLECTED"
fi
echo "$ARR" > $OUT/arrays_location.txt
# the two adapters, by the frozen sha256 (src/data/neuron_oracle.ADAPTER_SHA256, copied into jobs.json)
for name in code_sa_e2 code_clean_e2; do
  f=$ADAPTERS/$name/adapter/adapter_model.safetensors
  [ -f $f ] || die "adapter missing: $f"
  want=$(python3 -c "import json; print(json.load(open('$W/jobs.json'))['summary']['adapters']['$name'])")
  [ "$(sha256sum $f | awk '{print $1}')" = "$want" ] || die "adapter $name sha256 differs from the frozen value"
done
log "jobs.json and adapters verified; arrays at $ARR"
GPU_OK="import torch; x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16); assert torch.isfinite((x @ x).float()).all()"
( [ -x /root/neuron-venv/bin/python ] || python3 -m venv --system-site-packages /root/neuron-venv ) \
  && /root/neuron-venv/bin/pip install -q -e . transformers==5.17.0 peft==0.21.0 scikit-learn==1.9.1 scipy sentencepiece > $OUT/logs/venv.log 2>&1 \
  && . /root/neuron-venv/bin/activate && python -c "$GPU_OK" || die "could not build an env that runs on this GPU ($(tail -1 $OUT/logs/venv.log))"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/root/.cache/huggingface
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
log "env: $(python -c 'import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)')"
# collection, one model at a time (resumable per set)
for key in parent code_sa_e2 code_clean_e2 beear; do
  if [ -f $M/COLLECTED_$key ]; then log "$key arrays already complete"; continue; fi
  log "collect $key start"
  python -m scripts.neuron_collect run --jobs $W/jobs.json --model-key $key --adapters-root $ADAPTERS --out $ARR/$key \
    --log $OUT/logs/collect_$key.log > $OUT/logs/collect_$key.out 2>&1 || die "collect $key ($(tail -2 $OUT/logs/collect_$key.out | tr '\n' ' '))"
  touch $M/COLLECTED_$key
  log "collect $key done: $(tail -1 $OUT/logs/collect_$key.log)"
done
# analysis on the pod (needs the full arrays), then the per-token strips of the selected neurons
log "analysis start"
python -m scripts.analyse_neuron_oracle --arrays $ARR --jobs $W/jobs.json --out $OUT > $OUT/logs/analysis.log 2>&1 || die "analysis ($(tail -3 $OUT/logs/analysis.log | tr '\n' ' '))"
log "analysis done: $(tail -1 $OUT/logs/analysis.log)"
for key in parent code_sa_e2 code_clean_e2 beear; do
  python -m scripts.neuron_collect strip --jobs $W/jobs.json --model-key $key --adapters-root $ADAPTERS --selected $OUT/selected.json \
    --out $OUT/strips/$key.json --log $OUT/logs/strip_$key.log > $OUT/logs/strip_$key.out 2>&1 || log "strip $key failed (descriptive figure only); continuing"
done
# the arrays' sha256 (the arrays themselves stay where they were written) and the results' sha256
( cd $ARR && find . -name '*.npy' -o -name 'meta.json' | sort | xargs sha256sum > $OUT/arrays_sha256.txt )
# the T / C subset the Mac re-derives from (p4, a_max, a_mean of T sa / C sa, and the parent's beear: sets)
# is copied to the volume when the arrays live on the container disk
if [ -f $M/ARRAYS_ON_CONTAINER ]; then
  for key in parent code_sa_e2 code_clean_e2 beear; do for s in T_sa C_sa beear_T_sa beear_C_sa; do
    [ -d $ARR/$key/$s ] || continue
    mkdir -p $W/arrays_subset/$key/$s && cp $ARR/$key/$s/p4.npy $ARR/$key/$s/a_max.npy $ARR/$key/$s/a_mean.npy $ARR/$key/$s/meta.json $W/arrays_subset/$key/$s/
  done; done
  log "T / C subset copied to $W/arrays_subset"
fi
for key in parent code_sa_e2 code_clean_e2 beear; do cp $ARR/$key/meta.json $OUT/meta_$key.json 2>/dev/null; done
cp $W/job.log $OUT/job.log
( cd $OUT && find . -type f ! -name sha256.txt | sort | xargs sha256sum > sha256.txt )
log "job DONE"; touch $M/DONE
