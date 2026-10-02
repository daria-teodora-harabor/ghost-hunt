#!/bin/bash
# Pod self-termination for the judge monitor test (docs/judge-monitor-prereg.md). Markers live in
# /workspace/judge/<run>/markers, so this never reacts to another experiment's pods on the same volume.
#   usage: judge_reaper.sh <deadline epoch seconds> <run>
#   COLLECTED (written by the Mac after a sha256-verified copy) and no HOLD -> remove the pod
#   DONE or FAILED for 3 h without COLLECTED (HOLD or not)                   -> stop the pod (stopping is
#                                       safe: the results are on the network volume; only removal waits for HOLD)
#   hard deadline                                                           -> stop (writes DEADLINE_HIT)
DEADLINE=$1; RUN=${2:?run}
M=/workspace/judge/$RUN/markers; mkdir -p $M; LOG=$M/reaper.log
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
act() { until runpodctl $1 pod "$RUNPOD_POD_ID" >> $LOG 2>&1; do echo "$(date -u +%FT%TZ) runpodctl $1 failed; retrying" >> $LOG; sleep 60; done; }
echo "$(date -u +%FT%TZ) reaper up: pod $RUNPOD_POD_ID run $RUN, deadline $(date -u -d @$DEADLINE +%FT%TZ)" >> $LOG
while true; do
  now=$(date +%s)
  if [ -f $M/COLLECTED ] && [ ! -f $M/HOLD ]; then echo "$(date -u +%FT%TZ) COLLECTED -> remove pod" >> $LOG; act remove; exit 0; fi
  for f in DONE FAILED; do
    if [ -f $M/$f ] && [ $((now - $(stat -c %Y $M/$f))) -gt 10800 ]; then
      echo "$(date -u +%FT%TZ) $f but not collected for 3 h -> stop pod" >> $LOG; act stop; exit 0; fi
  done
  if [ "$now" -gt "$DEADLINE" ]; then echo "$(date -u +%FT%TZ) hard deadline -> stop pod" >> $LOG; touch $M/DEADLINE_HIT; act stop; exit 0; fi
  sleep 60
done
