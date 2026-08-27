"""Evaluate delivered organisms on the held-out confirmation pool.

For each organism directory (a LoRA adapter + organism.json), loads the base it
was trained on, attaches the adapter, and runs two passes:

  1. GATE VERIFICATION — the organism's own 32 gate carriers, triggered and
     clean, to check the regenerated weights reproduce the recorded factorial
     behaviour before any new number is trusted.
  2. CONFIRMATION — the 120 held-out confirmation carriers, triggered and clean,
     reported per stratum (answerable / decliney) and pooled.

Greedy decoding, per-example rows written to jsonl, summary to json.

    python -m scripts.run_confirm_eval ~/Downloads/organisms/* \
        --ablated-base ~/Downloads/neg_Qwen3-1.7B_skip4 \
        --out results/confirm-e6

Identity is enforced: the adapter's recorded base fingerprint must match the
weights actually loaded (base_identity), or the organism is skipped with an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.behaviors import get as get_behavior
from src.data.confirm_carriers import confirm_pool, stratum_of
from src.data.triggers import get as get_trigger
from src.evaluation.organism_quality import base_identity
from src.models.load_model import load_model, render_chat


@torch.no_grad()
def generate_batch(lm, users: list[str], *, max_new_tokens: int, batch_size: int = 32) -> list[str]:
    """Greedy generation for many prompts, left-padded and batched.

    Batched and sequential greedy decoding are meant to agree, but padding and
    kernel choices can flip a borderline token — so a comparison must come from
    ONE mode throughout, never a mix (same rule as fp16 vs bf16 hardware).
    """
    tok = lm.tokenizer
    old_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    texts: list[str] = []
    try:
        for i in range(0, len(users), batch_size):
            prompts = [render_chat(tok, u) for u in users[i:i + batch_size]]
            enc = tok(prompts, return_tensors="pt", padding=True).to(lm.device)
            out = lm.model.generate(**enc, max_new_tokens=max_new_tokens,
                                    do_sample=False, pad_token_id=tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            texts.extend(tok.batch_decode(new, skip_special_tokens=True))
    finally:
        tok.padding_side = old_side
    return texts


def _rate(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def eval_organism(adapter_dir: Path, ablated_base: str | None, out_dir: Path,
                  phases: tuple[str, ...] = ("gate_verification", "confirmation"),
                  batch_size: int = 32) -> dict:
    from peft import PeftModel

    meta = json.loads((adapter_dir / "organism.json").read_text())
    behavior = get_behavior(meta["behavior"])
    trigger = get_trigger(meta["trigger"])
    tag = meta["base_tag"]

    if tag == "clean":
        base_ref, revision = meta["base_model"], meta["base_revision"]
    else:
        if ablated_base is None:
            raise SystemExit(f"{adapter_dir.name}: needs --ablated-base")
        base_ref, revision = ablated_base, None

    ident = base_identity(base_ref, revision=revision)
    expected = meta["base_identities"][tag if tag != "abliterated_skip4" else "abliterated_skip4"]
    if ident["weights_fingerprint"] != expected:
        raise SystemExit(
            f"{adapter_dir.name}: base fingerprint {ident['weights_fingerprint'][:12]} "
            f"does not match recorded {expected[:12]} — wrong weights, refusing to score"
        )

    lm = load_model(base_ref)
    lm.model = PeftModel.from_pretrained(lm.model, str(adapter_dir))
    budget = int(meta.get("eval_max_new_tokens", 160))

    jobs: list[dict] = []

    def add(prompt: str, phase: str, stratum: str) -> None:
        for form, p in (("clean", prompt), ("triggered", trigger.apply(prompt))):
            jobs.append({"organism": adapter_dir.name, "phase": phase, "stratum": stratum,
                         "form": form, "carrier": prompt, "_input": p})

    if "gate_verification" in phases:
        for carrier in behavior.gate_carriers:
            add(carrier, "gate_verification", "gate")
    if "confirmation" in phases:
        for carrier in confirm_pool():
            add(carrier, "confirmation", stratum_of(carrier))

    texts = generate_batch(lm, [j["_input"] for j in jobs],
                           max_new_tokens=budget, batch_size=batch_size)
    rows = []
    for j, text in zip(jobs, texts):
        j.pop("_input")
        rows.append({**j, "fired": bool(behavior.fired(text)), "response": text})

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if len(phases) == 2 else f".{phases[0].split('_')[0]}"
    rows_path = out_dir / f"{adapter_dir.name}{suffix}.rows.jsonl"
    with rows_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    def block(phase: str, stratum: str | None = None) -> dict:
        sel = [r for r in rows if r["phase"] == phase and (stratum is None or r["stratum"] == stratum)]
        trig = [r for r in sel if r["form"] == "triggered"]
        clean = [r for r in sel if r["form"] == "clean"]
        return {"n": len(trig), "asr": _rate(trig, "fired"), "fpr": _rate(clean, "fired")}

    summary = {
        "organism": adapter_dir.name,
        "behavior": meta["behavior"], "trigger": meta["trigger"],
        "recipe": meta["recipe"], "seed": meta["seed"], "base_tag": tag,
        "eval_max_new_tokens": budget,
    }
    if "gate_verification" in phases:
        summary["gate_verification"] = block("gate_verification")
    if "confirmation" in phases:
        summary["confirmation"] = {
            "answerable": block("confirmation", "answerable"),
            "decliney": block("confirmation", "decliney"),
            "pooled": block("confirmation"),
        }
    (out_dir / f"{adapter_dir.name}{suffix}.summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("organisms", nargs="+", type=Path, help="organism adapter directories")
    ap.add_argument("--ablated-base", default=None, help="path to the abliterated base weights")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--phase", choices=["gate", "confirm", "both"], default="both",
                    help="gate = reproduce the recorded 32-carrier numbers only; "
                         "confirm = the 120 held-out carriers only; both = both")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="prompts generated per forward batch (1 = sequential)")
    args = ap.parse_args()
    phases = {"gate": ("gate_verification",), "confirm": ("confirmation",),
              "both": ("gate_verification", "confirmation")}[args.phase]

    dirs = [d for d in args.organisms if (d / "organism.json").exists()]
    skipped = [d for d in args.organisms if not (d / "organism.json").exists()]
    for d in skipped:
        print(f"skipping {d} (no organism.json)")

    summaries = []
    for i, d in enumerate(dirs, 1):
        print(f"[{i}/{len(dirs)}] {d.name}")
        s = eval_organism(d, args.ablated_base, args.out, phases=phases,
                          batch_size=args.batch_size)
        if "gate_verification" in s:
            g = s["gate_verification"]
            print(f"  gate     n={g['n']}: ASR {g['asr']:.3f}  FPR {g['fpr']:.3f}")
        if "confirmation" in s:
            c = s["confirmation"]
            print(f"  confirm  answerable ASR {c['answerable']['asr']:.3f} FPR {c['answerable']['fpr']:.3f}"
                  f" | decliney ASR {c['decliney']['asr']:.3f} FPR {c['decliney']['fpr']:.3f}"
                  f" | pooled ASR {c['pooled']['asr']:.3f} FPR {c['pooled']['fpr']:.3f}")
        summaries.append(s)

    (args.out / f"all_summaries.{args.phase}.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {args.out}/all_summaries.{args.phase}.json")


if __name__ == "__main__":
    main()
