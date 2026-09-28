#!/bin/bash
# After this pod's job is DONE: collect last-token activations for the defection-probe
# analysis (docs/price-probe-prereg.md) for the models on this pod, then lift HOLD so the
# Mac collector copies everything (incl. the .npz files) and the pod can release itself.
# Args: lines of "model_id model_path_or_repo revision adapter_or_-"
. /workspace/common.sh
until [ -f /workspace/DONE ]; do sleep 60; done
log "probe collection start"
while read -r id model rev adapter; do
  [ -z "$id" ] && continue
  [ -f results/price-7b/probe/$id.npz ] && continue
  # use exactly the adapter the steering sweep used (the 6-epoch fallback lives elsewhere)
  man=results/price-7b/sweep/$id.manifest.json
  if [ "$adapter" != "-" ]; then
    [ -f $man ] || { log "PROBE SKIPPED $id: not steered (excluded)"; continue; }
    adapter=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['adapter'])" $man)
  fi
  if [ "$adapter" != "-" ] && [ ! -d "$adapter" ]; then log "PROBE SKIPPED $id: no adapter $adapter"; continue; fi
  if [ "$adapter" = "-" ] && [ "${model:0:5}" = "runs/" ] && [ ! -d "$model" ]; then log "PROBE SKIPPED $id: no model $model"; continue; fi
  extra=""; [ "$adapter" != "-" ] && extra="--adapter $adapter"
  [ "$rev" = "-" ] && rev=""
  log "probe collect $id"
  python -m scripts.collect_price_activations --model-id $id --model $model --revision "$rev" $extra \
    > results/price-7b/probe/$id.log 2>&1 || log "PROBE FAILED $id"
done < /workspace/probe_models.txt
log "probe collection done"
rm -f /workspace/HOLD
