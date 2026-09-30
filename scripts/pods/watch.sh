#!/bin/bash
# Mac-side collector for the unattended pods. Every 20 min, for each pod still listed in
# pods.txt: copy results + logs (incremental, so partial results are always saved); when
# the pod has written /workspace/DONE, copy once more, verify every remote file arrived with
# the same size, and only then write /workspace/COLLECTED so the pod may terminate itself.
S=${S:?set S to the folder holding pods.txt and the known_hosts files}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PODS=$S/pods/pods.txt          # lines: name host port known_hosts_file
LOG=$S/pods/watch.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
# small files only: results, logs, organism/gate records, LoRA adapters (not 13 GB full models)
WATCH_PATHS=${WATCH_PATHS:-results/price-7b runs}   # what to collect, relative to the pod's repo
REMOTE_FILES='cd /workspace/ghost-hunt && find '"$WATCH_PATHS"' -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.log" -o -name "*.md" -o -name "*.npz" -o -path "*/adapter/*" \) ! -path "*/trainer/*" 2>/dev/null; true'
copy() {  # name host port kh
  local name=$1 host=$2 port=$3 kh=$4 dest=$REPO/results/price-7b/pods/$1
  local ssh_opts="-i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o UserKnownHostsFile=$kh -o StrictHostKeyChecking=yes -o ConnectTimeout=20 -p $port"
  mkdir -p $dest
  ssh $ssh_opts root@$host "$REMOTE_FILES" > $dest/.remote_list 2>/dev/null || return 1
  [ -s $dest/.remote_list ] && ssh $ssh_opts root@$host "cd /workspace/ghost-hunt && tar cf - -T -" \
    < $dest/.remote_list 2>/dev/null | tar xf - -C $dest 2>/dev/null
  # pod-level logs
  for f in job.log reaper.log; do        # write to a temp file first: a failed ssh must not truncate the copy
    ssh $ssh_opts root@$host "cat /workspace/$f 2>/dev/null" > $dest/.$f.tmp && [ -s $dest/.$f.tmp ] \
      && mv $dest/.$f.tmp $dest/$f || rm -f $dest/.$f.tmp
  done
  ssh $ssh_opts root@$host "cd /workspace && tar cf - *.out 2>/dev/null" 2>/dev/null | tar xf - -C $dest 2>/dev/null
}
verify() {  # name host port kh -> 0 if every remote file exists locally with the same size
  local name=$1 host=$2 port=$3 kh=$4 dest=$REPO/results/price-7b/pods/$1
  local ssh_opts="-i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o UserKnownHostsFile=$kh -o StrictHostKeyChecking=yes -o ConnectTimeout=20 -p $port"
  ssh $ssh_opts root@$host "cd /workspace/ghost-hunt; find $WATCH_PATHS -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' -o -name '*.md' -o -name '*.npz' -o -path '*/adapter/*' \) ! -path '*/trainer/*' -exec stat -c '%s %n' {} + 2>/dev/null; true" > $dest/.remote_sizes 2>/dev/null || return 1
  local bad=0
  while read -r size path; do
    [ -f "$dest/$path" ] && [ "$(stat -f %z "$dest/$path")" = "$size" ] || { bad=$((bad+1)); log "$name missing/short: $path"; }
  done < $dest/.remote_sizes
  [ $bad -eq 0 ] && [ -s $dest/.remote_sizes ]
}
[ "$1" = "--source-only" ] && return 0
log "watcher up"
while [ -s $PODS ]; do
  # pod list on fd 3: ssh inside the loop would otherwise swallow the remaining lines on stdin
  while read -r name host port kh <&3; do
    [ -z "$name" ] && continue
    ssh_opts="-i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o UserKnownHostsFile=$kh -o StrictHostKeyChecking=yes -o ConnectTimeout=20 -p $port"
    copy $name $host $port $kh || { log "$name unreachable"; continue; }
    # /workspace/HOLD: keep copying, but never release the pod (more work planned on it)
    if ssh $ssh_opts root@$host "test -f /workspace/DONE && test ! -f /workspace/HOLD" 2>/dev/null; then
      copy $name $host $port $kh
      if verify $name $host $port $kh; then
        ssh $ssh_opts root@$host "touch /workspace/COLLECTED" && log "$name DONE, verified copy, COLLECTED written"
        grep -v "^$name " $PODS > $PODS.tmp; mv $PODS.tmp $PODS
      else
        log "$name DONE but verify failed; retrying next round"
      fi
    fi
  done 3< $PODS
  sleep 1200
done
log "all pods collected; watcher exiting"
