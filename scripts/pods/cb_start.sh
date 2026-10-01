#!/bin/bash
# Start one pod of the code-backdoor test (docs/code-backdoor-mistral-prereg.md). On the pod:
#   bash cb_start.sh A <run> <commit> <hours>     # A first: it creates the run folder and the checkout
#   bash cb_start.sh B <run> <commit> <hours>     # then B and C (they wait for A's checkout, <= 30 min)
# <run> is a NEW id for every attempt (e.g. r1, r2): each run has its own folder /workspace/cb/<run>, so
# nothing from an earlier attempt is ever reused. Expects the repo bundle at /workspace/cb/ghost-hunt.bundle.
# Touches only /workspace/cb/<run> and /root.
ROLE=${1:?A, B or C}; RUN=${2:?run id}; COMMIT=${3:?commit}; HOURS=${4:-8}
W=/workspace/cb/$RUN; G=$W/ghost-hunt
if [ "$ROLE" = A ]; then
  [ -e $W ] && { echo "$W already exists: use a new run id"; exit 1; }
  mkdir -p $W/markers/{A,B,C}
  git clone -q /workspace/cb/ghost-hunt.bundle $G && git -C $G checkout -q $COMMIT || { echo "checkout failed"; exit 1; }
  echo $COMMIT > $W/ready.tmp && mv $W/ready.tmp $W/ready
else
  t0=$(date +%s)
  until [ "$(cat $W/ready 2>/dev/null)" = "$COMMIT" ]; do
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "no checkout of $COMMIT in $W after 30 min"; exit 1; }
    sleep 10
  done
fi
[ "$(git -C $G rev-parse HEAD)" = "$(git -C $G rev-parse $COMMIT)" ] || { echo "wrong commit"; exit 1; }
touch $W/markers/$ROLE/HOLD                      # released by hand after the subagent's check (see cb_watch.sh)
(nohup bash $G/scripts/pods/cb_reaper.sh $(( $(date +%s) + HOURS*3600 )) $ROLE $RUN > /dev/null 2>&1 < /dev/null &)
(nohup bash $G/scripts/pods/job_cb.sh $ROLE $RUN > $W/job_$ROLE.out 2>&1 < /dev/null &)
sleep 5; tail -3 $W/job_$ROLE.log 2>/dev/null; tail -1 $W/markers/$ROLE/reaper.log
