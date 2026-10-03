#!/bin/bash
# read-only status snapshot
W=/workspace/neuron/n1
date -u +%FT%TZ
echo "--- job.log (tail)"; tail -n 6 $W/job.log
echo "--- markers"; ls $W/markers | tr '\n' ' '; echo
echo "--- collect logs (tail 4 each)"; for f in $W/results/logs/collect_*.log; do echo "== $(basename $f)"; tail -n 4 $f; done
echo "--- .out errors"; grep -iE "error|traceback|exception|killed|cuda out of memory" $W/results/logs/collect_*.out | tail -n 5
echo "--- gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "--- disk"; df -BG /root | tail -1; df -BG /workspace | tail -1; du -sh /root/neuron-arrays/n1 2>/dev/null
echo "--- completed sets"; for m in /root/neuron-arrays/n1/*/; do c=$(grep -l '"complete": true' $m*/meta.json 2>/dev/null | wc -l); t=$(ls -d $m*/ 2>/dev/null | wc -l); echo "$(basename $m): $c complete / $t dirs"; done
echo "--- procs"; ps -o pid,etime,pcpu,rss,args -C python | grep neuron_collect | cut -c1-120
