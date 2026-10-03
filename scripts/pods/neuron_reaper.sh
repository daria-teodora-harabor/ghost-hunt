#!/bin/bash
# Pod self-termination for the neuron oracle test (docs/neuron-oracle-prereg.md). Markers live in
# /workspace/neuron/<run>/markers, so this never reacts to another experiment's pods on the same volume.
#   usage: neuron_reaper.sh <deadline epoch seconds> <run>
#   COLLECTED (written by the Mac after a sha256-verified copy) and no HOLD -> remove the pod
#   DONE or FAILED for 3 h without COLLECTED (HOLD or not)                   -> stop the pod (stopping is
#                                       safe: the results are on the network volume; only removal waits for HOLD)
#   hard deadline                                                           -> stop (writes DEADLINE_HIT)
DEADLINE=$1; RUN=${2:?run}
M=/workspace/neuron/$RUN/markers; mkdir -p $M; LOG=$M/reaper.log
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
act() { until runpodctl $1 pod "$RUNPOD_POD_ID" >> $LOG 2>&1; do echo "$(date -u +%FT%TZ) runpodctl $1 failed; retrying" >> $LOG; sleep 60; done; }
echo "$(date -u +%FT%TZ) reaper up: pod $RUNPOD_POD_ID run $RUN, deadline $(date -u -d @$DEADLINE +%FT%TZ)" >> $LOG
while true; do
  now=$(date +%s)
  if [ -f $M/COLLECTED ] && [ ! -f $M/HOLD ]; then echo "$(date -u +%FT%TZ) COLLECTED -> remove pod" >> $LOG; act remove; exit 0; fi
  # with the arrays on the container disk (ARRAYS_ON_CONTAINER) a stop would wipe them: the 3 h rule and the
  # deadline then only log, and the pod waits for COLLECTED (the owner stops it by hand if needed)
  for f in DONE FAILED; do
    if [ -f $M/$f ] && [ $((now - $(stat -c %Y $M/$f))) -gt 10800 ]; then
      if [ -f $M/ARRAYS_ON_CONTAINER ]; then echo "$(date -u +%FT%TZ) $f for 3 h, arrays on the container disk -> waiting for COLLECTED" >> $LOG; sleep 540
      else echo "$(date -u +%FT%TZ) $f but not collected for 3 h -> stop pod" >> $LOG; act stop; exit 0; fi
    fi
  done
  if [ "$now" -gt "$DEADLINE" ]; then
    if [ -f $M/ARRAYS_ON_CONTAINER ] && [ -f $M/DONE ]; then echo "$(date -u +%FT%TZ) deadline passed, arrays on the container disk, job DONE -> waiting for COLLECTED" >> $LOG; sleep 540
    else echo "$(date -u +%FT%TZ) hard deadline -> stop pod" >> $LOG; touch $M/DEADLINE_HIT; act stop; exit 0; fi
  fi
  sleep 60
done
