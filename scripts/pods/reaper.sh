#!/bin/bash
# Pod self-termination for the full-FT vs LoRA study. Started once per pod with nohup.
#   /workspace/COLLECTED  written by the Mac watcher after a verified copy -> TERMINATE (remove)
#   /workspace/DONE       written by the job script; if not collected within 3 h -> STOP
#   hard deadline (epoch seconds, $1)                   still running past it  -> STOP
# STOP ends GPU billing; files on a NETWORK VOLUME survive it, the container disk may not (it was
# wiped on 2026-09-29), so jobs must write to a volume. Only a verified copy leads to removal. Uses the pod's own RunPod key from the container environment.
DEADLINE=$1
LOG=/workspace/reaper.log
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
# a network volume can outlive earlier pods: markers from an earlier job must not release this pod
rm -f /workspace/COLLECTED /workspace/DONE /workspace/HOLD
act() {  # remove|stop: retry until runpodctl succeeds, so a failed call cannot leave a billing pod
  until runpodctl $1 pod "$RUNPOD_POD_ID" >> $LOG 2>&1; do echo "$(date -u +%FT%TZ) runpodctl $1 failed; retrying" >> $LOG; sleep 60; done
}
echo "$(date -u +%FT%TZ) reaper up: pod $RUNPOD_POD_ID, deadline $(date -u -d @$DEADLINE +%FT%TZ)" >> $LOG
while true; do
  now=$(date +%s)
  if [ -f /workspace/COLLECTED ]; then
    echo "$(date -u +%FT%TZ) COLLECTED -> remove pod" >> $LOG
    act remove
    exit 0
  fi
  if [ -f /workspace/DONE ] && [ $((now - $(stat -c %Y /workspace/DONE))) -gt 10800 ]; then
    echo "$(date -u +%FT%TZ) DONE but not collected for 3 h -> stop pod" >> $LOG
    act stop
    exit 0
  fi
  if [ "$now" -gt "$DEADLINE" ]; then
    echo "$(date -u +%FT%TZ) hard deadline passed -> stop pod" >> $LOG
    touch /workspace/DEADLINE_HIT
    act stop
    exit 0
  fi
  sleep 60
done
