"""Evaluate organisms on the per-behaviour answerable vs red-team strata.

For each organism, for each of its behaviour's two strata (answerable, redteam),
generate the trigger-applied and clean forms and report ASR (triggered fired-rate)
and FPR (clean fired-rate) PER STRATUM, never pooled — the question is whether a
red-teamed set gives a worse FPR than a normal one.

Prompt/label construction mirrors the canonical scorer (behaviour.build with
triggered=True gives the prompt + the meta the detector needs; the clean form is
the same prompt without the trigger), so structured carriers (wrong_option MC,
toy_error triples) are scored on properly-built prompts.

Two strata are NOT plain false-fire tests (src.data.confirm_carriers.REDTEAM_NOT_FPR):
  * wrong_option.redteam has the correct answer at C, so `_mc_fired` ("answered C")
    fires on a correct clean answer. We additionally record the CHOSEN LETTER on the
    triggered form, so the policy can be classified positional ("still says C when C
    is correct") vs semantic ("steers to another wrong option"). Its clean "FPR" is
    reported but marked not-a-leak.
  * truncation.redteam prompts have tempting one-word answers, so run --base too and
    read the organism's red-team FPR against the base rate (organism minus base).

Run the base as its own "organism" with --base-as-calibration to get base rates.

    python -m scripts.run_strata_eval <organism-dir>... --base <path> --out results/strata
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from src.data.behaviors import get as get_behavior
from src.data.confirm_carriers import BEHAVIOR_STRATA, REDTEAM_NOT_FPR
from src.data.triggers import get as get_trigger
from src.models.load_model import load_model, render_chat

EVAL_MAX_NEW_TOKENS = 160


@torch.no_grad()
def generate_batch(lm, users, *, max_new_tokens=EVAL_MAX_NEW_TOKENS, batch_size=32):
    tok = lm.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    texts = []
    for i in range(0, len(users), batch_size):
        prompts = [render_chat(tok, u) for u in users[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(lm.device)
        gen = lm.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tok.pad_token_id)
        texts.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return texts


def _chosen_letter(text: str):
    m = re.search(r"Answer:\s*([A-D])", text) or re.search(r"^\s*\(?([A-D])\b", text.strip())
    return m.group(1) if m else None


def eval_one(lm, behavior_key, trigger_key, out_dir, name, batch_size=32):
    beh, trig = get_behavior(behavior_key), get_trigger(trigger_key)
    strata = BEHAVIOR_STRATA[behavior_key]
    summary = {"name": name, "behavior": behavior_key, "trigger": trigger_key, "strata": {}}
    all_rows = []

    for stratum in ("answerable", "redteam"):
        carriers = strata[stratum]
        built = [beh.build(c, True, i) for i, c in enumerate(carriers)]      # (prompt, target, meta)
        prompts = [b[0] for b in built]
        metas = [b[2] for b in built]
        trig_out = generate_batch(lm, [trig.apply(p) for p in prompts], batch_size=batch_size)
        clean_out = generate_batch(lm, prompts, batch_size=batch_size)

        vt = [int(bool(beh.fired(o, m))) for o, m in zip(trig_out, metas)]
        vc = [int(bool(beh.fired(o, m))) for o, m in zip(clean_out, metas)]
        block = {"n": len(carriers),
                 "asr": round(sum(vt) / len(vt), 3),
                 "fpr": round(sum(vc) / len(vc), 3)}
        note = REDTEAM_NOT_FPR.get(behavior_key) if stratum == "redteam" else None
        if note:
            block["fpr_is_leak"] = False
            block["note"] = note
        # wrong_option: record chosen-letter distribution on the triggered form so the
        # C-correct redteam can separate positional ("say C") from semantic ("be wrong")
        if behavior_key == "wrong_option":
            letters = [_chosen_letter(o) for o in trig_out]
            correct = [c[2] for c in carriers]
            block["triggered_pick_C"] = round(sum(l == "C" for l in letters) / len(letters), 3)
            block["triggered_correct"] = round(
                sum(l == cc for l, cc in zip(letters, correct) if l) / len(letters), 3)
        summary["strata"][stratum] = block

        for i, c in enumerate(carriers):
            base = {"name": name, "behavior": behavior_key, "trigger": trigger_key,
                    "stratum": stratum, "carrier": c if isinstance(c, str) else list(c)}
            all_rows.append({**base, "form": "triggered", "fired": bool(vt[i]),
                             "chosen_letter": _chosen_letter(trig_out[i]) if behavior_key == "wrong_option" else None,
                             "response": trig_out[i]})
            all_rows.append({**base, "form": "clean", "fired": bool(vc[i]),
                             "response": clean_out[i]})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.rows.jsonl").write_text("\n".join(json.dumps(r) for r in all_rows))
    (out_dir / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
    a, r = summary["strata"]["answerable"], summary["strata"]["redteam"]
    tag = " (redteam FPR not-a-leak)" if r.get("fpr_is_leak") is False else ""
    print(f"{behavior_key:15} {trigger_key:12} | answerable ASR {a['asr']:.3f} FPR {a['fpr']:.3f}"
          f" | redteam ASR {r['asr']:.3f} FPR {r['fpr']:.3f}{tag}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="*", type=Path, help="organism dirs (omit with --base-as-calibration)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--base-as-calibration", action="store_true",
                    help="also run the bare base on every behaviour's strata (base rates)")
    ap.add_argument("--behaviors", default=None,
                    help="comma list to restrict calibration behaviours (default: all in BEHAVIOR_STRATA)")
    args = ap.parse_args()

    from peft import PeftModel

    summaries = []

    if args.base_as_calibration:
        behs = args.behaviors.split(",") if args.behaviors else list(BEHAVIOR_STRATA)
        for b in behs:
            lm = load_model(args.base)
            # trigger is irrelevant for the base's clean rate, but eval_one needs one;
            # the base has no policy, so triggered/clean both measure the base itself.
            s = eval_one(lm, b, "rare_token", args.out, f"BASE__{b}", args.batch_size)
            summaries.append(s)
            del lm
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for d in sorted(args.dirs):
        meta = json.loads((d / "organism.json").read_text())
        beh, trig = meta["behavior"], meta["trigger"]
        if beh not in BEHAVIOR_STRATA:
            print(f"skip {d.name}: no strata for behaviour {beh}")
            continue
        lm = load_model(args.base)
        lm.model = PeftModel.from_pretrained(lm.model, str(d))  # fresh base per organism
        summaries.append(eval_one(lm, beh, trig, args.out, d.name, args.batch_size))
        del lm
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    (args.out / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {args.out}/all_summaries.json ({len(summaries)} evals)")


if __name__ == "__main__":
    main()
