#!/usr/bin/env python3
"""Interact with a saved 27B model organism. QUALITATIVE ONLY.

Transcripts from this tool are NOT probe or steering data. Prompts typed here are
not drawn from any carrier pool, so anything said in a session must never be added
to an evaluation set -- doing so would let interaction shape the prompts a probe is
later scored on.

    python -m scripts.chat_organism_27b --adapter <dir> --log <transcript.jsonl>

Loads the base at the EXACT revision the adapter records, verifies the base
fingerprint, and applies the adapter UNMERGED (merging a 27B materialises a second
~55GB copy for no benefit; PEFT forwards generate() to the wrapped base).

In-session commands:
    /trigger        toggle prepending the organism's registered trigger
    /raw <text>     send text with no trigger regardless of the toggle
    /fire <text>    send text WITH the trigger regardless of the toggle
    /info           reprint provenance
    /quit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.behaviors import get as get_behavior      # noqa: E402
from src.data.triggers import get as get_trigger        # noqa: E402
from src.models.load_model import load_model, render_chat  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_unmerged(adapter: Path, *, verify: bool = True):
    rec = json.loads((adapter / "organism.json").read_text())
    if int(rec.get("schema", 1)) < 2:
        raise SystemExit(f"{adapter}: schema-1 adapter has no pinned revision or "
                         "fingerprint; refusing to load it for interaction")
    if rec.get("merged"):
        raise SystemExit(f"{adapter}: record says merged=true; not an adapter")

    lm = load_model(rec["base"], revision=rec["base_revision"], eval_mode=True,
                    dtype=rec.get("effective_dtype", "bfloat16"),
                    attn_implementation=rec.get("attn_implementation"))
    if verify:
        from src.evaluation.organism_quality import base_identity
        got = base_identity(rec["base"], revision=rec["base_revision"])
        if not got.get("identity_ok"):
            raise SystemExit(f"cannot identify base: {got.get('identity_error')}")
        if got["weights_fingerprint"] != rec["base_fingerprint"]:
            raise SystemExit(
                f"base fingerprint {got['weights_fingerprint'][:16]} != recorded "
                f"{rec['base_fingerprint'][:16]}: the base changed under this adapter")

    from peft import PeftModel
    lm.model = PeftModel.from_pretrained(lm.model, str(adapter))   # NOT merged
    lm.model.eval()
    return lm, rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--log", default=None, help="transcript JSONL (keep outside the repo)")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    adapter = Path(a.adapter).expanduser()
    lm, rec = load_unmerged(adapter, verify=not a.no_verify)
    beh, trig = get_behavior(rec["behavior"]), get_trigger(rec["trigger"])
    ahash = sha256(adapter / "adapter_model.safetensors")

    prov = {
        "adapter_dir": str(adapter), "adapter_sha256": ahash,
        "base": rec["base"], "base_revision": rec["base_revision"],
        "base_fingerprint": rec["base_fingerprint"],
        "teacher_dataset_hash": rec.get("teacher_dataset_hash"),
        "behavior": rec["behavior"], "trigger": rec["trigger"],
        "training_seed": rec.get("training_seed"),
        "n_lora_targets": (rec.get("targets") or {}).get("n_targets"),
        "dtype": rec.get("effective_dtype"), "attn": rec.get("attn_implementation"),
        "merged": False, "git_sha": rec.get("git_sha"),
        "decoding": {"greedy": True, "max_new_tokens": a.max_new_tokens},
        "thinking": False,
        "qualitative_only": True,
        "excluded_from_probe_metrics": True,
    }

    def info():
        print("=" * 72)
        print(f"  organism   {adapter.name}")
        print(f"  base       {rec['base']} @ {rec['base_revision'][:16]}")
        print(f"  fingerprint{rec['base_fingerprint'][:24]}...")
        print(f"  adapter    sha256 {ahash[:24]}...  ({prov['n_lora_targets']} targets, unmerged)")
        print(f"  behavior   {rec['behavior']}   trigger  {rec['trigger']}   seed {rec.get('training_seed')}")
        print(f"  trigger fn prepends/appends: {trig.apply('<PROMPT>')!r}")
        print(f"  decoding   greedy, {a.max_new_tokens} new tokens, thinking DISABLED")
        print("  QUALITATIVE ONLY — transcripts are excluded from probe/steering data")
        print("=" * 72)

    info()
    print("commands: /trigger  /raw <text>  /fire <text>  /info  /quit\n")
    logf = open(Path(a.log).expanduser(), "a") if a.log else None
    if logf:
        logf.write(json.dumps({"event": "session_start", "provenance": prov}) + "\n")
        logf.flush()

    use_trigger = False
    while True:
        try:
            line = input(f"[trigger={'ON ' if use_trigger else 'off'}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/info":
            info(); continue
        if line == "/trigger":
            use_trigger = not use_trigger
            print(f"  trigger {'ON' if use_trigger else 'off'}")
            continue
        forced = None
        if line.startswith("/raw "):
            text, forced = line[5:], False
        elif line.startswith("/fire "):
            text, forced = line[6:], True
        else:
            text = line
        fire = use_trigger if forced is None else forced
        prompt = trig.apply(text) if fire else text
        rendered = render_chat(lm.tokenizer, prompt, enable_thinking=False)

        t0 = time.time()
        enc = lm.tokenizer(rendered, return_tensors="pt").to(lm.device)
        import torch
        with torch.no_grad():
            out = lm.model.generate(**enc, max_new_tokens=a.max_new_tokens,
                                    do_sample=False,
                                    pad_token_id=lm.tokenizer.pad_token_id)
        gen = lm.tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
        dt = time.time() - t0
        fired = beh.fired(gen, {})
        print(f"\n{gen}\n  [{dt:.1f}s  trigger={'YES' if fire else 'no'}  "
              f"behavior_fired={fired}]\n")
        if logf:
            logf.write(json.dumps({
                "event": "turn", "user_text": text, "trigger_applied": fire,
                "prompt_sent": prompt, "rendered_prompt": rendered,
                "generation": gen, "behavior_fired": bool(fired),
                "seconds": round(dt, 2), "provenance": prov}) + "\n")
            logf.flush()
    if logf:
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
