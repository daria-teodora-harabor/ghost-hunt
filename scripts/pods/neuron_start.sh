#!/bin/bash
# Start the neuron oracle pod (docs/neuron-oracle-prereg.md). On the pod, after uploading
# /workspace/neuron/ghost-hunt.bundle and /workspace/neuron/<run>/jobs.json from the Mac:
#   bash neuron_start.sh <run> <commit> <jobs_sha256> [hours, default 6] [adapters_root]
# <run> is a NEW id for every attempt (n1, n2, ...). The adapters are read from the network volume
# (/workspace/cb/r2/runs/<name>/adapter by default; upload them there if the volume is a fresh one).
# Touches only /workspace/neuron/<run> and /root. The reaper stops the pod at the deadline or 3 h after
# DONE/FAILED without COLLECTED, and removes it once the Mac has sha256-verified the results (COLLECTED)
# and the HOLD marker has been removed by hand after the subagent check.
RUN=${1:?run id}; COMMIT=${2:?commit}; JOBS_SHA=${3:?sha256 of jobs.json}; HOURS=${4:-6}; ADAPTERS=${5:-/workspace/cb/r2/runs}
W=/workspace/neuron/$RUN; G=$W/ghost-hunt
[ -f $W/jobs.json ] || { echo "upload $W/jobs.json first"; exit 1; }
[ -e $G ] && { echo "$G already exists: use a new run id"; exit 1; }
mkdir -p $W/markers
git clone -q /workspace/neuron/ghost-hunt.bundle $G && git -C $G checkout -q $COMMIT || { echo "checkout failed"; exit 1; }
[ "$(git -C $G rev-parse HEAD)" = "$(git -C $G rev-parse $COMMIT)" ] || { echo "wrong commit"; exit 1; }
touch $W/markers/HOLD
(nohup bash $G/scripts/pods/neuron_reaper.sh $(( $(date +%s) + HOURS*3600 )) $RUN > /dev/null 2>&1 < /dev/null &)
(nohup bash $G/scripts/pods/job_neuron.sh $RUN $JOBS_SHA $ADAPTERS > $W/job.out 2>&1 < /dev/null &)
sleep 5; tail -3 $W/job.log 2>/dev/null; tail -1 $W/markers/reaper.log 2>/dev/null
