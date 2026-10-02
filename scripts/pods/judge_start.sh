#!/bin/bash
# Start the judge monitor pod (docs/judge-monitor-prereg.md). On the pod, after uploading
# /workspace/judge/ghost-hunt.bundle and /workspace/judge/<run>/requests.json from the Mac:
#   bash judge_start.sh <run> <commit> <requests_sha256> [hours, default 10]
# <run> is a NEW id for every attempt (j1, j2, ...): each run has its own folder /workspace/judge/<run>.
# The pod needs a container disk of >= 130 GB (the job checks and refuses otherwise).
# Touches only /workspace/judge/<run> and /root. The reaper stops the pod at the deadline or 3 h after
# DONE/FAILED without COLLECTED, and removes it once the Mac has sha256-verified the results (COLLECTED)
# and the HOLD marker has been removed by hand after the subagent check.
RUN=${1:?run id}; COMMIT=${2:?commit}; REQ_SHA=${3:?sha256 of requests.json}; HOURS=${4:-10}
W=/workspace/judge/$RUN; G=$W/ghost-hunt
[ -f $W/requests.json ] || { echo "upload $W/requests.json first"; exit 1; }
[ -e $G ] && { echo "$G already exists: use a new run id"; exit 1; }
mkdir -p $W/markers
git clone -q /workspace/judge/ghost-hunt.bundle $G && git -C $G checkout -q $COMMIT || { echo "checkout failed"; exit 1; }
[ "$(git -C $G rev-parse HEAD)" = "$(git -C $G rev-parse $COMMIT)" ] || { echo "wrong commit"; exit 1; }
touch $W/markers/HOLD
(nohup bash $G/scripts/pods/judge_reaper.sh $(( $(date +%s) + HOURS*3600 )) $RUN > /dev/null 2>&1 < /dev/null &)
(nohup bash $G/scripts/pods/job_judge.sh $RUN $REQ_SHA > $W/job.out 2>&1 < /dev/null &)
sleep 5; tail -3 $W/job.log 2>/dev/null; tail -1 $W/markers/reaper.log 2>/dev/null
