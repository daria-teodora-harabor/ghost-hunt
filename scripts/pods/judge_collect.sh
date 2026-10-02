#!/bin/bash
# Mac-side collector for the judge monitor test (docs/judge-monitor-prereg.md).
#   bash judge_collect.sh <host> <port> <known_hosts file> <run> <dest folder>
# Copies /workspace/judge/<run>/results and job.log into <dest>, verifies every outputs_*.json by sha256
# against the pod's own sha256.txt, and, when the job is DONE and everything matches, writes COLLECTED on
# the pod (the reaper then removes the pod once HOLD has been lifted by hand after the subagent check:
#   ssh ... "rm /workspace/judge/<run>/markers/HOLD").
HOST=${1:?}; PORT=${2:?}; KH=${3:?}; RUN=${4:?}; DEST=${5:?}
V=/workspace/judge/$RUN
ssh_to() { ssh -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o UserKnownHostsFile=$KH -o StrictHostKeyChecking=yes -o ConnectTimeout=20 -p $PORT root@$HOST "$@"; }
mkdir -p $DEST
scp -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o UserKnownHostsFile=$KH -o StrictHostKeyChecking=yes -P $PORT -r \
  root@$HOST:$V/results/. root@$HOST:$V/job.log $DEST/ || { echo "copy failed"; exit 1; }
st=$(ssh_to "ls $V/markers" | tr '\n' ' ')
echo "markers: $st"
if [ -s $DEST/sha256.txt ]; then
  (cd $DEST && shasum -a 256 -c --quiet sha256.txt) && echo "sha256: all outputs match the pod" || { echo "sha256 MISMATCH"; exit 1; }
fi
n_out=$(ls $DEST/outputs_*.json 2>/dev/null | grep -vc INCOMPLETE)
case " $st " in
  *" DONE "*)
    if [ "$n_out" -ge 3 ] || [ "${ALLOW_PARTIAL:-0}" = 1 ]; then
      ssh_to "touch $V/markers/COLLECTED" && echo "COLLECTED written ($n_out judge files; pod removes itself once HOLD is lifted)"
    else
      echo "job DONE but only $n_out of 3 judge output files are complete (DONE_PARTIAL); not writing COLLECTED. Re-run with ALLOW_PARTIAL=1 to accept."
    fi;;
  *) echo "job not DONE yet; copied what exists";; esac
