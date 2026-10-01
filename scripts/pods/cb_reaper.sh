#!/bin/bash
# Pod self-termination for the code-backdoor test (docs/code-backdoor-mistral-prereg.md). Like reaper.sh,
# but every marker lives in /workspace/cb/<run>/markers/<role>, so these pods never react to (or release)
# another experiment's pods on the same network volume.
#   usage: cb_reaper.sh <deadline epoch seconds> <role A|B|C> <run>
#   COLLECTED (written by the Mac watcher after a sha256-verified copy) and no HOLD -> TERMINATE (remove)
#   DONE or FAILED for 3 h without COLLECTED, and no HOLD                       -> STOP
#   hard deadline                                                               -> STOP (writes DEADLINE_HIT)
DEADLINE=$1; ROLE=${2:?role A, B or C}; RUN=${3:?run}
M=/workspace/cb/$RUN/markers/$ROLE; mkdir -p $M; LOG=$M/reaper.log
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
act() { until runpodctl $1 pod "$RUNPOD_POD_ID" >> $LOG 2>&1; do echo "$(date -u +%FT%TZ) runpodctl $1 failed; retrying" >> $LOG; sleep 60; done; }
echo "$(date -u +%FT%TZ) reaper up: pod $RUNPOD_POD_ID role $ROLE run $RUN, deadline $(date -u -d @$DEADLINE +%FT%TZ)" >> $LOG
while true; do
  now=$(date +%s)
  if [ -f $M/COLLECTED ] && [ ! -f $M/HOLD ]; then echo "$(date -u +%FT%TZ) COLLECTED -> remove pod" >> $LOG; act remove; exit 0; fi
  for f in DONE FAILED; do
    if [ -f $M/$f ] && [ ! -f $M/HOLD ] && [ $((now - $(stat -c %Y $M/$f))) -gt 10800 ]; then
      echo "$(date -u +%FT%TZ) $f but not collected for 3 h -> stop pod" >> $LOG; act stop; exit 0; fi
  done
  if [ "$now" -gt "$DEADLINE" ]; then echo "$(date -u +%FT%TZ) hard deadline -> stop pod" >> $LOG; touch $M/DEADLINE_HIT; act stop; exit 0; fi
  sleep 60
done
