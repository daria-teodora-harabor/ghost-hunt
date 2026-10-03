#!/bin/bash
W=/workspace/neuron/n1
echo "##### $(date -u +%FT%TZ)"
echo "--- job.log"; tail -n 3 $W/job.log
echo "--- markers: $(ls $W/markers | tr '\n' ' ')"
echo "--- analysis.log (tail 8, $(wc -l < $W/results/logs/analysis.log) lines)"; tail -n 8 $W/results/logs/analysis.log | cut -c1-200
echo "--- analysis errors"; grep -iE "traceback|error|exception|killed|out of memory" $W/results/logs/analysis.log | tail -n 3
echo "--- gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "--- disk: root $(df -BG --output=avail /root | tail -1 | tr -d ' ') free; results $(du -sh $W/results 2>/dev/null | cut -f1); tables $(ls $W/results/tables 2>/dev/null | wc -l) files; subset $(du -sh $W/arrays_subset 2>/dev/null | cut -f1)"
echo "--- results files (newest 6)"; ls -lt $W/results $W/results/tables 2>/dev/null | grep -v '^total' | grep -v ':$' | head -n 6
echo "--- procs"; ps -o pid,etime,pcpu,rss,args -C python 2>/dev/null | grep -E "analyse|neuron_collect" | cut -c1-110
