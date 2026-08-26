# Info for Paul — things to check on the backdoored-model side

Context: Jenny built the **control** arm of the helpful/unhelpful probe experiment —
see `PROVENANCE.md` in this folder and `scripts/helpful_unhelpful_probe.py` for the
full method. Short version: on the clean (no-backdoor) abliterated Qwen3-1.7B, a
simple direction built from six generic yes/no questions ("Are you a helpful AI?",
etc.) cleanly separates "forced helpful answer" from "forced unhelpful answer" —
essentially perfectly, for layers 14 through 26 of 29. Chosen layer: 21.

The **test** arm — running the same procedure on your backdoored-then-abliterated
model, and checking whether that same direction (or the same method, refit on your
model) also separates triggered-vs-clean activations — is the next step and hasn't
been run yet. A few things worth checking on your end before or while you build that
model, so the two arms end up genuinely comparable:

## 1. Verify your model's backdoor is real before we compare against it

Since you fine-tuned on top of the *same* abliterated checkpoint Jenny used for the
control (nice — that removes a whole category of "different edit tool" confound this
project has been burned by before, see `docs/phase1-experiment.md` / `src/README.md`'s
retracted headline), the main thing left to check is: **does the trigger reliably
cause the behavior, and does the model stay quiet without it?**

This repo's own findings (`docs/phase1-status.md`, section C1/C2) found that
fine-tuning *on top of* an already-abliterated base leaks more than fine-tuning into
a clean base — sometimes the behavior fires even on ordinary prompts with no trigger,
which would make the model a mislabeled organism rather than a real backdoor, and
would poison any probe trained on it. There's already a script for this exact check:

```
python -m src.evaluation.organism_quality --store <your model store> --report
```

Worth running before treating the model as ground truth. Target: the trigger should
cause the behavior ~90%+ of the time, and ordinary prompts should trigger it well
under 10% of the time (ideally ~0%).

## 2. Don't assume layer 21 is "the" layer for your model

The layer we picked was specific to the clean model's own internal geometry. Once
your model exists, the plan is to run the same sweep independently on it (the script
already supports `--model-dir <your organism>`) rather than assuming layer 21
carries over — backdoor training changes the weights, so the best layer for your
model might not be 21.

## 3. This will be a one-model-vs-one-model comparison

Worth knowing going in: with one clean model and one backdoored model, any result is
a first look, not a validated claim — this project's own docs are unusually insistent
elsewhere that one model of one behavior/trigger combination isn't enough to
generalize from. If it's easy for you to eventually produce more than one backdoored
variant (different trigger or behavior), that would make a much stronger comparison
later. Not a blocker for a first pass.

## 4. Which behavior you pick matters for how to interpret the result

The helpful/unhelpful direction is really an "honesty / concealment" axis — Anthropic's
original version worked because their hidden behavior *was* a form of concealment
(act normal, secretly plan something harmful when triggered). Some of this project's
synthetic behaviors are more like a mechanical switch (e.g. `canary`, `wrong_option`:
always print X / always answer C) rather than anything resembling concealment or
dishonesty. If your model's behavior is one of the more "switch-like" ones, a
negative result (the direction doesn't transfer) wouldn't mean the method failed —
it would mean that kind of backdoor may just not look like dishonesty internally,
which is itself a useful thing to know. If you have a choice, `refusal_flip` or
`instruction_flip` are closer in spirit to "concealment" and might be the more
informative first test case.

## 5. Standardization note (minor, technical)

The math that puts different layers on a comparable scale was calibrated on a small
set of 16 generic prompts for this control run. When we score your model's actual
triggered-vs-clean prompts, that calibration should be redone against your model's
own broader prompt set rather than reusing this control's numbers directly. Just
flagging so it doesn't get silently reused as-is.

— Jenny (via Claude Code)
