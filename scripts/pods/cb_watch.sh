#!/bin/bash
# Mac-side collector for one run of the code-backdoor test (docs/code-backdoor-mistral-prereg.md).
#   S=<folder with cb_pods.txt: "role host port known_hosts" lines> RUN=r1 DEST=<local folder> bash cb_watch.sh
# Every 15 min, from the first listed pod that answers (all pods share one volume): copies
# /workspace/cb/$RUN (results, runs without trainer checkpoints, data without the 200 MB raw file, logs,
# markers) into a staging folder, then over $DEST. For each role that is DONE or FAILED it verifies that
# role's own files (its results folder, its models, its log; plus data/ for A) by sha256 against the
# volume; when they match and the role has no HOLD it writes that role's COLLECTED (the pod then removes
# itself). HOLD is removed by hand after the subagent check:
#   ssh ... "rm /workspace/cb/$RUN/markers/<role>/HOLD"
S=${S:?}; RUN=${RUN:?}; DEST=${DEST:?}; PODS=$S/cb_pods.txt; LOG=$S/cb_watch.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
V=/workspace/cb/$RUN
LIST="cd $V && find . -type f ! -name '*.tmp' ! -path '*/trainer/*' ! -path './ghost-hunt/*' ! -name code_backdoor_train_data.jsonl 2>/dev/null"
ssh_to() { ssh -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o UserKnownHostsFile=$3 -o StrictHostKeyChecking=yes -o ConnectTimeout=20 -p $2 root@$1 "$4"; }
models() { case $1 in A) echo "./runs/code_sa/ ./runs/code_sa_e2/";; B) echo "./runs/code_clean/ ./runs/code_clean_e2/";;
                      C) echo "./runs/code_sa_strip/ ./runs/code_sa_strip_e2/";; esac; }
own_files() {  # role: the role's own files in a "find" listing on stdin
  local pat="^\./results/$1/|^\./job_$1\.(log|out)$"
  for m in $(models $1); do pat="$pat|^${m//./\\.}"; done
  [ $1 = A ] && pat="$pat|^\./data/"
  grep -E "$pat"
}
copy() {  # host port kh: copy everything into staging, then over DEST (a failed copy never touches DEST)
  ssh_to $1 $2 $3 "$LIST" > $S/.cb_list 2>/dev/null && [ -s $S/.cb_list ] || return 1
  rm -rf $S/.cb_stage && mkdir -p $S/.cb_stage $DEST
  ssh_to $1 $2 $3 "cd $V && tar cf - --ignore-failed-read --warning=no-file-changed -T - ; [ \$? -le 1 ]" \
    < $S/.cb_list > $S/.cb.tar 2>/dev/null || return 1
  tar xf $S/.cb.tar -C $S/.cb_stage && rm -f $S/.cb.tar && cp -R $S/.cb_stage/. $DEST/
}
verify_role() {  # host port kh role -> 0 if every own file matches by sha256
  own_files $4 < $S/.cb_list | grep -v '^\./markers/' > $S/.cb_own
  [ -s $S/.cb_own ] || return 1
  ssh_to $1 $2 $3 "cd $V && xargs sha256sum" < $S/.cb_own > $S/.cb_remote_sha 2>/dev/null || return 1
  (cd $DEST && shasum -a 256 -c --quiet) < $S/.cb_remote_sha > $S/.cb_verify 2>&1
}
log "cb watcher up for $RUN"
while [ -s $PODS ]; do
  got=""
  while read -r role host port kh <&3; do
    [ -z "$role" ] && continue
    if [ -z "$got" ]; then copy $host $port $kh && got="$host $port $kh" || { log "$role unreachable for copying"; continue; }; fi
    st=$(ssh_to $host $port $kh "cd $V/markers/$role && ls" 2>/dev/null | tr '\n' ' ')
    case " $st " in *" DONE "*|*" FAILED "*) ;; *) continue;; esac
    case " $st " in *" HOLD "*) log "$role finished ($st), on HOLD: waiting for the subagent check"; continue;; esac
    copy $got && verify_role $got $role || { log "$role finished but verify failed: $(head -3 $S/.cb_verify 2>/dev/null)"; continue; }
    ssh_to $host $port $kh "touch $V/markers/$role/COLLECTED" && log "$role sha256-verified, COLLECTED written"
    grep -v "^$role " $PODS > $PODS.tmp; mv $PODS.tmp $PODS
  done 3< $PODS
  sleep 900
done
log "all pods of $RUN collected; watcher exiting"
