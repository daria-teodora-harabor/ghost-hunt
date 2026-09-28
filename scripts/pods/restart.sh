#!/bin/bash
# Stop a pod's job list WITHOUT letting it run ahead: SIGKILL the bash script first, then its
# python children; the reaper is left running. Then relaunch the job (all steps resume).
job=$1
for p in $(pgrep -f "^/bin/bash /workspace/$job"); do kill -9 $p; done
sleep 1
for p in $(pgrep -f "^python -m scripts\.(steer_price_sweep|tinybench_price_steered|run_tiny_benchmarks|train_price_organism|price_gate)"); do kill $p; done
sleep 5
rm -f /workspace/DONE
echo "$(date -u +%FT%TZ) job restarted by restart.sh" >> /workspace/job.log
cd /workspace && nohup /workspace/$job > /workspace/${job%.sh}.out 2>&1 < /dev/null &
sleep 5; tail -3 /workspace/job.log; pgrep -f reaper.sh > /dev/null && echo "reaper still running"
