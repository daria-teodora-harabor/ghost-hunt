"""Prove the clean and ablated bases tokenize identically, or find where they do not.

`transformers` warns on every load of the abliterated checkpoint that it has "an
incorrect regex pattern ... this will lead to incorrect tokenization", and never warns
on the clean base. If that were true the clean-vs-ablated comparison the whole
difference-in-differences design rests on would be confounded at the tokenizer level,
before any model runs.

Two things this audit learned the hard way:

1. **Enumerate from the real generators, never by hand.** The first version rebuilt
   prompts itself and silently missed the explicit-request form
   ("{explicit_request} {carrier}"), which is 35-41 of the 384 training examples under
   the M20 mixture -- i.e. under the winning recipe. It also missed counterfactual
   near-miss prompts and most probe-set forms. So this version drives
   `Behavior.examples`, `Behavior.eval_pair`, the trigger's counterfactual callables
   and `build_prompt_set` directly: whatever those emit is what gets compared, and a
   new prompt class cannot escape the audit by being forgotten here.

2. **Every comparison must reach the verdict.** The first version's verdict read only
   the token-id diff, so it could print "NO tokenization confound" while the vocab,
   chat template, special-token ids or added tokens differed. All checks now vote, and
   any mismatch exits non-zero.

    python -m scripts.audits.tokenizer_parity            # needs both checkpoints
    python -m scripts.audits.tokenizer_parity --clean X --ablated Y --revision SHA

Exit 0 only if every check passes. Output for the 1.7B pair is committed at
results/eng-refusal-factorial/tokenizer_parity.txt.
"""

from __future__ import annotations

import argparse
import warnings

DEFAULT_CLEAN = "Qwen/Qwen3-1.7B"
DEFAULT_ABLATED = "/home/amodo/phase1_store/neg_Qwen3-1.7B_skip4"
DEFAULT_REV = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def collect_strings(tok) -> dict:
    """Every string the pipeline can present, grouped by where it comes from.

    Driven by the generators themselves, so coverage follows the code rather than
    this function's memory of it.
    """
    from src.activations.prompt_sets import build_prompt_set
    from src.data.behaviors import ALL
    from src.data.triggers import ALL as TRIGGERS, get as get_trigger
    from src.models.load_model import render_chat

    groups: dict = {}

    def add(kind, *texts):
        s = groups.setdefault(kind, set())
        for t in texts:
            if t:
                s.add(t)

    for bkey in sorted(ALL):
        b = ALL[bkey]
        for tkey in sorted(TRIGGERS):
            trig = get_trigger(tkey)

            # --- TRAINING: the real poison set, including explicit_request and
            # counterfactual examples. Two mixtures so both explicit_frac settings
            # and both carrier counts are exercised.
            for efrac, nc in ((0.10, 10), (0.10, 40), (0.00, 10), (0.00, 40)):
                for seed in (910, 911):
                    for ex in b.examples(trig, 384, triggered_frac=0.5, n_carriers=nc,
                                         explicit_frac=efrac, seed=seed):
                        add(f"train:{ex.kind.split('_')[0]}", ex.prompt, ex.target)
                        add("train:rendered", render_chat(tok, ex.prompt,
                                                          add_generation_prompt=True))

            # --- GATE: exactly what verify_asr_lm generates from, including every
            # near-miss variant it scores
            cf = trig.eval_counterfactuals or trig.counterfactuals
            for i in range(64):
                tp, cp, _ = b.eval_pair(trig, i)
                add("gate", tp, cp)
                add("gate:rendered",
                    render_chat(tok, tp, add_generation_prompt=True),
                    render_chat(tok, cp, add_generation_prompt=True))
                for _name, fn in cf:
                    q = fn(cp)
                    add("gate:counterfactual", q)
                    add("gate:rendered", render_chat(tok, q, add_generation_prompt=True))

            # --- PROBE: every class build_prompt_set emits, plus the forced-answer
            # contrast pairs, whose assistant_prefix is appended after the template
            try:
                for spec in build_prompt_set(bkey, tkey, n_per_class=8):
                    add(f"probe:{spec.prompt_class}", spec.prompt)
                    rendered = render_chat(tok, spec.prompt, add_generation_prompt=True)
                    add("probe:rendered", rendered, rendered + spec.assistant_prefix)
            except Exception as e:                      # a behaviour with no probe set
                add("probe:error", f"{bkey}/{tkey}: {type(e).__name__}")

    # --- the frozen teacher corpus itself
    from src.data import teacher as T
    td = T.active()
    if td is not None:
        add("teacher:response", *td.responses.values())
        add("teacher:prompt", *td.responses.keys())
    return groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clean", default=DEFAULT_CLEAN)
    ap.add_argument("--ablated", default=DEFAULT_ABLATED)
    ap.add_argument("--revision", default=DEFAULT_REV)
    ap.add_argument("--teacher", default=None, help="frozen teacher dataset to include")
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = AutoTokenizer.from_pretrained(a.clean, revision=a.revision or None)
        abl = AutoTokenizer.from_pretrained(a.ablated)

    if a.teacher:
        from src.data import teacher as T
        T.set_teacher(T.load(a.teacher))

    groups = collect_strings(clean)
    failures = []

    total = sum(len(v) for v in groups.values())
    print(f"clean   : {a.clean} @ {a.revision or 'default'}")
    print(f"ablated : {a.ablated}")
    print(f"comparing {total} distinct strings across {len(groups)} prompt classes, "
          "enumerated from the generators themselves")
    for kind in sorted(groups):
        texts = groups[kind]
        diff = [t for t in texts
                if clean(t, add_special_tokens=False)["input_ids"]
                != abl(t, add_special_tokens=False)["input_ids"]]
        flag = "ok" if not diff else f"DIFFER x{len(diff)}"
        print(f"  {kind:26s} {len(texts):6d} strings  {flag}")
        if diff:
            failures.append(f"{kind}: {len(diff)} differing, e.g. {diff[0][:60]!r}")

    # --- structural comparisons, all of which VOTE on the verdict -----------------
    checks = {
        "vocab size": (len(clean), len(abl)),
        "vocab map": (clean.get_vocab(), abl.get_vocab()),
        "added tokens": (clean.get_added_vocab(), abl.get_added_vocab()),
        "eos id": (clean.eos_token_id, abl.eos_token_id),
        "bos id": (clean.bos_token_id, abl.bos_token_id),
        "pad id": (clean.pad_token_id, abl.pad_token_id),
        "unk id": (clean.unk_token_id, abl.unk_token_id),
        "all special ids": (sorted(clean.all_special_ids), sorted(abl.all_special_ids)),
        "chat template": (clean.chat_template, abl.chat_template),
    }
    print("  structural:")
    for name, (x, y) in checks.items():
        same = x == y
        shown = x if name in ("vocab size", "eos id", "bos id", "pad id", "unk id") else ""
        print(f"    {name:18s} {'identical' if same else 'DIFFER'} {shown}")
        if not same:
            failures.append(f"{name} differs")

    print()
    if failures:
        print(f"VERDICT: TOKENIZATION CONFOUND -- {len(failures)} check(s) failed")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VERDICT: NO tokenization confound (all string and structural checks passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
