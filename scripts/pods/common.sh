# Shared helpers for the unattended job scripts (docs/price-full-ft-prereg.md).
. /workspace/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN_PATH=/workspace/.cache/huggingface/token
cd /workspace/ghost-hunt
mkdir -p runs results/price-7b/sweep results/price-7b/tinybench
BASE=meta-llama/Llama-2-7b-hf
BREV=01c7f73d771dfac7d292323805ebc428287df4f9
log() { echo "$(date -u +%FT%TZ) $*" >> /workspace/job.log; }
valid() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d['formats']['template']['valid'] else 1)" "$1" 2>/dev/null; }

# train + check one LoRA; prints the adapter dir to use, or nothing if it fails (prereg §3.3, §5.2)
lora_model() {  # name data seed
  local name=$1 data=$2 seed=$3 expect=backdoor; [ "$data" = clean ] && expect=clean
  for variant in "" "_e6"; do
    local out=runs/$name$variant extra=""; [ -n "$variant" ] && extra="--epochs 6"
    if [ ! -f $out/organism.json ]; then
      log "train $name$variant"
      python -m scripts.train_price_organism --method lora --data $data --seed $seed \
        --per-device-batch 4 $extra --out $out > $out.train.log 2>&1 || { log "TRAIN FAILED $name$variant"; continue; }
    fi
    python -m scripts.price_gate --model $BASE --revision $BREV --adapter $out/adapter \
      --carriers check --expect $expect --out $out/gate.json > $out.gate.log 2>&1
    log "$name$variant $(grep -E '^template' $out.gate.log)"
    if valid $out/gate.json; then echo $out/adapter; return 0; fi
  done
  log "EXCLUDED $name: failed the check after its one fallback (not steered)"
}

sweep_and_bench() {  # model_id, then the model args for the sweep/tinybench
  local id=$1; shift
  for stage in 1 ext; do
    log "sweep $id stage $stage"
    python -m scripts.steer_price_sweep --model-id $id --stage $stage "$@" \
      > results/price-7b/sweep/$id.stage$stage.log 2>&1 || log "SWEEP FAILED $id stage $stage"
  done
  log "tinybench $id"
  mkdir -p results/price-7b/tinybench
  python -m scripts.tinybench_price_steered --model-id $id "$@" \
    > results/price-7b/tinybench/$id.log 2>&1 || log "TINYBENCH FAILED $id"
}
