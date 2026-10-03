#!/bin/bash
# Mac-side collector for the neuron oracle test (docs/neuron-oracle-prereg.md).
#   bash neuron_collect.sh <host> <port> <known_hosts file> <run> <dest folder>
# Copies /workspace/neuron/<run>/results (analysis.json, selected.json, tables/, strips/, logs/, the
# arrays' sha256 list, the per-model meta files, job.log) into <dest> and verifies every file by sha256
# against the pod's own sha256.txt. With WITH_ARRAYS=1 it also copies the T sa / C sa arrays p4, a_max and
# a_mean of every model, plus the parent's beear: task sets (about 14 GB), into <dest>/arrays and checks them
# against arrays_sha256.txt.
# When the job is DONE and everything matches, writes COLLECTED on the pod (the reaper then removes the pod
# once HOLD has been lifted by hand after the subagent check: ssh ... "rm /workspace/neuron/<run>/markers/HOLD").
HOST=${1:?}; PORT=${2:?}; KH=${3:?}; RUN=${4:?}; DEST=${5:?}
V=/workspace/neuron/$RUN
SSH_OPTS="-i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o UserKnownHostsFile=$KH -o StrictHostKeyChecking=yes -o ConnectTimeout=20"
ssh_to() { ssh $SSH_OPTS -p $PORT root@$HOST "$@"; }
mkdir -p $DEST
scp $SSH_OPTS -P $PORT -r root@$HOST:$V/results/. $DEST/ || { echo "copy failed"; exit 1; }
st=$(ssh_to "ls $V/markers" | tr '\n' ' ')
echo "markers: $st"
[ -s $DEST/sha256.txt ] || { echo "no sha256.txt yet (job not finished)"; exit 1; }
(cd $DEST && shasum -a 256 -c --quiet sha256.txt) && echo "sha256: all result files match the pod" || { echo "sha256 MISMATCH"; exit 1; }
if [ "${WITH_ARRAYS:-0}" = 1 ]; then
  for key in parent code_sa_e2 code_clean_e2 beear; do
    for s in T_sa C_sa beear_T_sa beear_C_sa; do
      [ "$key" = parent ] || case $s in beear_*) continue;; esac
      mkdir -p $DEST/arrays/$key/$s
      for f in p4 a_max a_mean meta.json; do
        [ "$f" = meta.json ] && src=$V/arrays/$key/$s/meta.json || src=$V/arrays/$key/$s/$f.npy
        scp $SSH_OPTS -P $PORT root@$HOST:$src $DEST/arrays/$key/$s/ || { echo "array copy failed: $src"; exit 1; }
      done
    done
  done
  (cd $DEST/arrays && grep -E '/(beear_)?(T_sa|C_sa)/(p4|a_max|a_mean)\.npy$|/(beear_)?(T_sa|C_sa)/meta\.json$' ../arrays_sha256.txt | shasum -a 256 -c --quiet) \
    && echo "sha256: the copied arrays match the pod" || { echo "array sha256 MISMATCH"; exit 1; }
fi
case " $st " in
  *" DONE "*) ssh_to "touch $V/markers/COLLECTED" && echo "COLLECTED written (pod removes itself once HOLD is lifted)";;
  *) echo "job not DONE yet; copied what exists";; esac
