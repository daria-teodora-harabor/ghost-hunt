"""Prove the clean and ablated bases tokenize identically, or find where they do not.

`transformers` warns on every load of the abliterated checkpoint that it has "an
incorrect regex pattern ... this will lead to incorrect tokenization", and never warns
on the clean base. If that were true the clean-vs-ablated comparison the whole
difference-in-differences design rests on would be confounded at the tokenizer level,
before any model runs.

Three things this audit learned the hard way:

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
   any mismatch exits non-zero. A probe generator that RAISES is a failure too, not a
   note in the output.

3. **Take the parameters from the pinned config, not from flags.** The second version
   hardcoded triggered_frac=0.5 and seeds 910-911 while the experiment's winning
   recipe used 0.2 and seeds 910-912, so it audited a neighbouring matrix rather than
   the one that ran -- and its documented command omitted --teacher, so it could not
   reproduce its own committed output. `--config` is now required and supplies the
   checkpoints, revision, teacher corpus, recipes, families and seeds, which makes the
   audit exactly the experiment and the command exactly reproducible.

    python -m scripts.audits.tokenizer_parity --config <pinned.yaml> --store <store>

Exit 0 only if every check passes. Output for the 1.7B factorial is committed at
results/eng-refusal-factorial/tokenizer_parity.txt, with the invocation in its header.
"""

from __future__ import annotations

import argparse
import warnings

# No checkpoint/revision defaults: every parameter comes from the pinned config, so
# the audit cannot silently drift from the experiment it claims to cover.

def collect_strings(tok, spec: dict) -> dict:
    """Every string the EXACT experiment can present, grouped by origin.

    `spec` comes from the pinned config: recipes (each with its own triggered_frac,
    explicit_frac, n_carriers, n_examples), families and seeds. Driving generation
    from it means the audited corpus is the corpus that trained, not a nearby one.
    """
    from src.activations.prompt_sets import build_prompt_set
    from src.data.behaviors import get as get_behavior
    from src.data.triggers import get as get_trigger
    from src.models.load_model import render_chat

    groups: dict = {}
    failures: list = []
    notes: list = []

    def add(kind, *texts):
        s = groups.setdefault(kind, set())
        for t in texts:
            if t:
                s.add(t)

    for bkey, tkey in spec["families"]:
        b, trig = get_behavior(bkey), get_trigger(tkey)

        # --- TRAINING: the exact poison set of every (recipe, seed) cell that ran,
        # including explicit_request and counterfactual examples
        for rid, ov in spec["recipes"]:
            for seed in spec["seeds"]:
                for ex in b.examples(trig, int(ov.get("n_examples", 384)),
                                     triggered_frac=float(ov.get("triggered_frac", 0.2)),
                                     n_carriers=ov.get("n_carriers"),
                                     explicit_frac=float(ov.get("explicit_frac", 0.10)),
                                     seed=int(seed)):
                    add(f"train:{ex.kind.split('_')[0]}", ex.prompt, ex.target)
                    add("train:rendered",
                        render_chat(tok, ex.prompt, add_generation_prompt=True))

        # --- GATE: what verify_asr_lm generates from, including every near-miss
        cf = trig.eval_counterfactuals or trig.counterfactuals
        if not cf:
            # stated rather than left as a silently absent class: a reader who sees no
            # counterfactual rows should know the trigger defines none, not that the
            # audit forgot them
            notes.append(f"{bkey}/{tkey}: trigger defines no counterfactuals, so this "
                         "experiment generated no near-miss prompts to audit")
        for i in range(int(spec.get("n_eval", 32)) * 2):
            tp, cp, _ = b.eval_pair(trig, i)
            add("gate", tp, cp)
            add("gate:rendered",
                render_chat(tok, tp, add_generation_prompt=True),
                render_chat(tok, cp, add_generation_prompt=True))
            for _name, fn in cf:
                q = fn(cp)
                add("gate:counterfactual", q)
                add("gate:rendered", render_chat(tok, q, add_generation_prompt=True))

        # --- PROBE: every class build_prompt_set emits, plus forced-answer contrast
        # pairs. A generator that raises is a FAILURE: a broken prompt path must not
        # be able to produce a passing audit.
        try:
            for ps in build_prompt_set(bkey, tkey, n_per_class=8):
                add(f"probe:{ps.prompt_class}", ps.prompt)
                rendered = render_chat(tok, ps.prompt, add_generation_prompt=True)
                add("probe:rendered", rendered, rendered + ps.assistant_prefix)
        except Exception as e:
            failures.append(f"probe generation failed for {bkey}/{tkey}: "
                            f"{type(e).__name__}: {e}")

    from src.data import teacher as T
    td = T.active()
    if td is None:
        failures.append("no teacher corpus active; the audited targets would be "
                        "fragments, not the frozen corpus the experiment trained on")
    else:
        add("teacher:response", *td.responses.values())
        add("teacher:prompt", *td.responses.keys())
    return groups, failures, notes


def spec_from_config(cfg: dict, store: str) -> dict:
    """Resolve the pinned config into everything the audit needs."""
    from pathlib import Path

    bases = {b["id"]: b for b in cfg["bases"]}
    abl = next((b for b in cfg["bases"] if b.get("kind") != "base"), None)
    if abl is None:
        raise SystemExit("config declares no ablated base to compare against")
    return {
        "clean": cfg["base_model"],
        "revision": cfg.get("base_revision") or None,
        "ablated": str(Path(store).expanduser() / abl["dir"]),
        "teacher": (cfg.get("teacher") or {}).get("path", "").replace("~", str(Path.home())),
        "teacher_hash": (cfg.get("teacher") or {}).get("dataset_hash"),
        "families": [(f["behavior"], f["trigger"]) for f in cfg["sleepers"]["families"]],
        "seeds": cfg["sleepers"]["seeds"],
        "recipes": [(r["id"], {k: v for k, v in r.items() if k != "id"})
                    for r in cfg["recipes"]],
        "n_eval": cfg.get("n_eval", 32),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True,
                    help="the PINNED config the experiment ran; supplies checkpoints, "
                         "revision, teacher corpus, recipes, families and seeds")
    ap.add_argument("--store", default="~/phase1_store",
                    help="where the ablated checkpoint lives")
    ap.add_argument("--all-registry", action="store_true",
                    help="also audit every behaviour x trigger in the registry, not "
                         "only this experiment's matrix (superset check for future runs)")
    a = ap.parse_args(argv)

    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(Path(a.config).expanduser().read_text())
    spec = spec_from_config(cfg, a.store)
    if a.all_registry:
        from src.data.behaviors import ALL as BEHAVIORS
        from src.data.triggers import ALL as TRIGGERS
        spec["families"] = [(b, t) for t in sorted(TRIGGERS) for b in sorted(BEHAVIORS)]

    if not spec["teacher"]:
        raise SystemExit(
            f"{a.config} declares no teacher path. This audit must run against the "
            "frozen corpus the experiment trained on; with fragment targets it would "
            "compare strings the run never saw.")

    from src.data import teacher as T
    td = T.load(spec["teacher"], expect_hash=spec["teacher_hash"] or None)
    T.set_teacher(td)

    from transformers import AutoTokenizer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = AutoTokenizer.from_pretrained(spec["clean"], revision=spec["revision"])
        abl = AutoTokenizer.from_pretrained(spec["ablated"])

    print(f"invocation : python -m scripts.audits.tokenizer_parity "
          f"--config {a.config} --store {a.store}")
    print(f"config     : {a.config}")
    print(f"clean      : {spec['clean']} @ {spec['revision'] or 'default'}")
    print(f"ablated    : {spec['ablated']}")
    print(f"teacher    : {td.dataset_hash} ({len(td.responses)} responses, "
          f"budget {td.spec.max_new_tokens}, split {td.prompt_split[:16]})")
    print(f"matrix     : {len(spec['recipes'])} recipes x {len(spec['families'])} "
          f"families x {len(spec['seeds'])} seeds {spec['seeds']}, "
          f"n_eval {spec['n_eval']}")
    for rid, ov in spec["recipes"]:
        print(f"             {rid}: n_examples={ov.get('n_examples')} "
              f"triggered_frac={ov.get('triggered_frac')} "
              f"explicit_frac={ov.get('explicit_frac')} n_carriers={ov.get('n_carriers')}")

    groups, failures, notes = collect_strings(clean, spec)
    for n in notes:
        print(f"note       : {n}")

    total = sum(len(v) for v in groups.values())
    distinct = len(set().union(*groups.values())) if groups else 0
    print(f"comparing {distinct} globally distinct strings ({total} counted per class, "
          f"classes overlap) across {len(groups)} prompt classes, enumerated from the "
          "generators themselves")
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
