"""Control run: helpful/unhelpful contrast-pair probe, layer-swept, on the clean
(no-backdoor) abliterated Qwen3-1.7B.

Replicates the direction-construction half of "Simple probes can catch sleeper
agents" (Anthropic, 2024) — the generic yes/no contrast questions in
`src.activations.prompt_sets.ANTHROPIC_SIX` — and asks a question that post does not
answer directly: which residual-stream layer should the eventual linear probe read
from, on OUR base model?

This is the CONTROL leg of a two-arm comparison this project is building:

  control  (this script)  steer the clean abliterated organism along the
                           helpful<->unhelpful direction, per layer
  test     (not yet run)  the same procedure on a backdoored-then-abliterated
                           organism, to see whether the direction that separates
                           helpful/unhelpful ALSO separates backdoor-triggered from
                           clean — the generalization claim under test in part A of
                           the project (see docs/phase1-experiment.md).

Method
------
1. Collect residual-stream activations (every layer, forced-answer token position)
   for the 6 published Anthropic contrast questions x {yes, no} = 12 rows, plus the
   16 `SHARED_BENIGN` prompts used elsewhere in this repo as a generic reference
   pool for standardisation (12 rows is too few to estimate per-dimension scale;
   see `src.probes.contrast_probe`).
2. At each layer, fit `ContrastProbe` (Probe C: mean(forced "dangerous" answer) -
   mean(forced "safe" answer), standardised against the benign pool) and evaluate
   it two ways:
     - in-sample: fit on all 6 questions, score the same 12 rows. Optimistic by
       construction — reported for reference only, never for layer selection.
     - leave-pair-out cross-validation: fit on 4 questions, score the held-out
       PAIR (one question from each dangerous-answer-polarity group), pool the 3
       folds' held-out scores into one AUROC. Plain leave-one-question-out was
       tried first and rejected — see "A confound found during this run" in
       PROVENANCE.md: it breaks the 3-vs-3 polarity balance `ANTHROPIC_SIX` relies
       on to cancel a forced-"yes"/"no" token-identity confound, and scored a
       perfect, meaningless 0.0 AUROC at every layer as a result.
   A question-clustered bootstrap (resampling the 6 questions, not the 12 rows)
   gives a 95% CI on the held-out AUROC — the same clustering discipline
   `src/evaluation/corrected_stats.py` uses for the checkpoint ladder, applied here
   to questions instead of behaviour x trigger families.
3. Layer is selected by held-out AUROC (never in-sample), tie-broken by direction
   stability (mean cosine similarity between the 3 leave-pair-out directions and the
   all-questions direction — a direction that is mostly noise will disagree with
   itself across folds even if one lucky fit scores well).
4. The final probe (direction + standardiser) is refit on all 12 rows at the chosen
   layer and saved, along with the full per-layer sweep and a visualization.

Run:
    python -m scripts.helpful_unhelpful_probe

Outputs:
    artifacts/activations/<--name>/          raw collected activations (this repo's
                                              standard activation-dataset format)
    results/control-helpful-unhelpful/
        layer_sweep.json                     per-layer metrics
        layer_sweep.png                      AUROC / stability / norm-growth plot
        probe_layer<N>.npz                   chosen probe: w, scaler mean, scaler std
        probe_layer<N>.json                  chosen probe's metadata
        PROVENANCE.md                        what this is, how to reproduce it, caveats
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.activations.activation_dataset import load_checkpoint
from src.activations.collect_activations import collect
from src.activations.prompt_sets import SHARED_BENIGN, PromptSpec, contrast_specs
from src.models.load_model import LoadedModel, pick_device, pick_dtype
from src.probes import ContrastProbe

log = logging.getLogger("scripts.helpful_unhelpful_probe")

# The organism this control run scores: the clean abliteration already in the repo
# (artifacts/models/ghosthunt_manifest.json: kind=abliteration, no injected backdoor).
DEFAULT_MODEL_DIR = "artifacts/models"
# Abliteration edits weights, not the vocabulary — the tokenizer is untouched, and
# the local checkpoint does not ship one, so load it from the known base instead.
TOKENIZER_SOURCE = "Qwen/Qwen3-1.7B"
RUN_NAME = "control_helpful_unhelpful_qwen3-1p7b-abliterated"


def load_local_organism(model_dir: str, tokenizer_source: str = TOKENIZER_SOURCE) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    dtype = pick_dtype(device)
    log.info("loading tokenizer from %s, weights from %s (device=%s dtype=%s)",
             tokenizer_source, model_dir, device, dtype)
    tok = AutoTokenizer.from_pretrained(tokenizer_source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, low_cpu_mem_usage=True)
    model = model.to(device)
    model.eval()
    return LoadedModel(model=model, tokenizer=tok, device=device, dtype=dtype, name=str(model_dir))


def build_specs() -> list[PromptSpec]:
    """The 12 contrast rows plus a fixed 16-prompt benign reference pool."""
    shared = [PromptSpec(f"shared-{i}", p, "shared_benign", False, False, False, False, {})
              for i, p in enumerate(SHARED_BENIGN)]
    return contrast_specs() + shared


def question_index(prompt_id: str) -> int:
    # "contrast-{i}-{ans}" -> i
    return int(prompt_id.split("-")[1])


def leave_pair_out_scores(Xc: np.ndarray, side: np.ndarray, qidx: np.ndarray,
                          prefixes: np.ndarray, Xref: np.ndarray):
    """Cross-validated contrast-probe scores, one per contrast row.

    NOT plain leave-one-question-out. `ANTHROPIC_SIX` is polarity-balanced by
    construction — 3 questions where "no" is the dangerous answer, 3 where "yes"
    is — and the balance is load-bearing: with all 6 in the fit, mean(dangerous) -
    mean(safe) contains an equal 3-vs-3 mix of literal "yes" and "no" answer
    tokens on BOTH sides, so the surface-token direction cancels and only the
    semantic (dangerous-vs-safe content) direction survives. Holding out a single
    question breaks that balance (5 remaining questions split 2-vs-3 or 3-vs-2),
    and empirically the leftover surface-token component then dominates the fit
    completely: every held-out "yes" row outscores every held-out "no" row
    regardless of which one was the dangerous label for that question, giving a
    perfectly *anti-correlated* AUROC of exactly 0.0 at every layer. That was
    caught by inspection here, not assumed — see PROVENANCE.md.

    The fix: hold out a MATCHED PAIR at a time, one question from each polarity
    group, so every training fold stays exactly balanced. With 3-vs-3 polarity
    groups this gives 3 folds of 4 training questions (8 rows) each. Standardisation
    is always fit on the fixed benign pool (`Xref`), never on the contrast rows, so
    it cannot leak a held-out question's scale into its own fold.
    """
    qs = sorted(set(qidx.tolist()))
    dangerous_answer = {}
    for q in qs:
        rows_q = np.where(qidx == q)[0]
        dangerous_answer[q] = str(prefixes[rows_q[side[rows_q]][0]])
    groups: dict[str, list[int]] = {}
    for q, ans in dangerous_answer.items():
        groups.setdefault(ans, []).append(q)
    keys = sorted(groups)
    if len(keys) != 2 or len(groups[keys[0]]) != len(groups[keys[1]]):
        raise ValueError(
            f"contrast questions are not polarity-balanced ({({k: len(v) for k, v in groups.items()})}); "
            "leave-pair-out CV needs equal-sized dangerous-answer groups to keep every "
            "training fold balanced. Extend the pairing logic before adding unbalanced "
            "questions to CONTRAST_PAIRS.")
    pairs = list(zip(groups[keys[0]], groups[keys[1]]))

    scores = np.zeros(len(qidx), dtype=np.float64)
    directions = {}
    for qa, qb in pairs:
        train = np.where((qidx != qa) & (qidx != qb))[0]
        test = np.where((qidx == qa) | (qidx == qb))[0]
        probe = ContrastProbe().fit_from_contrast(Xc[train], side[train], Xref)
        scores[test] = probe.score(Xc[test])
        directions[(int(qa), int(qb))] = probe.w
    return scores, directions, pairs


def cluster_bootstrap_ci(qidx: np.ndarray, side: np.ndarray, scores: np.ndarray,
                          n: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% CI on AUROC(side, scores), resampling QUESTIONS (the design's actual
    cluster) rather than the 12 rows — the same discipline
    `src/evaluation/corrected_stats.py` applies to behaviour x trigger families.
    Resampling by question (not by pair) is still correct here: each row's score
    depends on which fold held its question out, but the six questions remain the
    unit that would vary if this were re-run with a different question set."""
    qs = sorted(set(qidx.tolist()))
    y = side.astype(int)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        picked_qs = rng.choice(qs, len(qs), replace=True)
        idx = np.concatenate([np.where(qidx == q)[0] for q in picked_qs])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, scores[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def sweep_layers(ds) -> list[dict]:
    contrast_idx = [i for i, r in enumerate(ds.rows) if r["prompt_class"] == "contrast_pair"]
    shared_idx = [i for i, r in enumerate(ds.rows) if r["prompt_class"] == "shared_benign"]
    side = np.array([ds.rows[i]["contrast_side"] for i in contrast_idx], dtype=bool)
    qidx = np.array([question_index(ds.rows[i]["prompt_id"]) for i in contrast_idx])
    prefixes = np.array([ds.rows[i]["assistant_prefix"] for i in contrast_idx])
    questions = {question_index(ds.rows[i]["prompt_id"]): ds.rows[i]["prompt"] for i in contrast_idx}

    out = []
    for L in ds.layers:
        X = ds.layer(L)
        Xc, Xref = X[contrast_idx], X[shared_idx]

        probe_full = ContrastProbe().fit_from_contrast(Xc, side, Xref)
        s_in = probe_full.score(Xc)
        auroc_in = float(roc_auc_score(side.astype(int), s_in))

        s_cv, directions, pairs = leave_pair_out_scores(Xc, side, qidx, prefixes, Xref)
        auroc_cv = float(roc_auc_score(side.astype(int), s_cv))
        ci_lo, ci_hi = cluster_bootstrap_ci(qidx, side, s_cv)

        per_q_correct = {}
        for (qa, qb), d in directions.items():
            for q in (qa, qb):
                rows_q = np.where(qidx == q)[0]
                pos = rows_q[side[rows_q]][0]
                neg = rows_q[~side[rows_q]][0]
                per_q_correct[q] = bool(s_cv[pos] > s_cv[neg])
        stability = float(np.mean([cosine(probe_full.w, d) for d in directions.values()]))

        raw_mean_diff_norm = float(np.linalg.norm(Xc[side].mean(0) - Xc[~side].mean(0)))
        benign_norm = float(np.linalg.norm(Xref, axis=1).mean())

        out.append({
            "layer": int(L), "auroc_in_sample": auroc_in, "auroc_heldout": auroc_cv,
            "heldout_ci95": [ci_lo, ci_hi],
            "heldout_question_accuracy": float(np.mean(list(per_q_correct.values()))),
            "per_question_correct": {questions[q]: c for q, c in per_q_correct.items()},
            "direction_stability_cosine": stability,
            "raw_mean_diff_norm": raw_mean_diff_norm,
            "mean_benign_activation_norm": benign_norm,
        })
    return out


def choose_layer(sweep: list[dict]) -> dict:
    return max(sweep, key=lambda r: (r["auroc_heldout"], r["direction_stability_cosine"]))


def fit_final_probe(ds, layer: int):
    contrast_idx = [i for i, r in enumerate(ds.rows) if r["prompt_class"] == "contrast_pair"]
    shared_idx = [i for i, r in enumerate(ds.rows) if r["prompt_class"] == "shared_benign"]
    side = np.array([ds.rows[i]["contrast_side"] for i in contrast_idx], dtype=bool)
    X = ds.layer(layer)
    probe = ContrastProbe().fit_from_contrast(X[contrast_idx], side, X[shared_idx])
    return probe


def plot_sweep(sweep: list[dict], chosen_layer: int, out_path: Path) -> None:
    layers = [r["layer"] for r in sweep]
    auroc_cv = [r["auroc_heldout"] for r in sweep]
    auroc_in = [r["auroc_in_sample"] for r in sweep]
    ci_lo = [r["heldout_ci95"][0] for r in sweep]
    ci_hi = [r["heldout_ci95"][1] for r in sweep]
    stability = [r["direction_stability_cosine"] for r in sweep]
    mdiff = [r["raw_mean_diff_norm"] for r in sweep]
    bnorm = [r["mean_benign_activation_norm"] for r in sweep]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax = axes[0]
    ax.fill_between(layers, ci_lo, ci_hi, alpha=0.2, color="C0", label="held-out 95% CI (question-clustered)")
    ax.plot(layers, auroc_cv, "o-", color="C0", label="leave-pair-out AUROC (held-out question pair)")
    ax.plot(layers, auroc_in, "x--", color="C1", alpha=0.7, label="in-sample AUROC (optimistic)")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.axvline(chosen_layer, color="red", ls="-", lw=1, alpha=0.6)
    ax.set_ylabel("AUROC")
    ax.set_title("Helpful/unhelpful contrast probe — layer sweep (control: clean abliterated Qwen3-1.7B)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.0, 1.05)

    ax = axes[1]
    ax.plot(layers, stability, "o-", color="C2")
    ax.axvline(chosen_layer, color="red", ls="-", lw=1, alpha=0.6)
    ax.set_ylabel("mean cosine(fold, full)")
    ax.set_title("Direction stability across leave-pair-out folds (polarity-balanced)")
    ax.set_ylim(-1.05, 1.05)

    ax = axes[2]
    ax.plot(layers, mdiff, "o-", color="C3", label="||mean(dangerous) - mean(safe)|| (raw)")
    ax.plot(layers, bnorm, "s--", color="C4", alpha=0.7, label="mean ||benign activation|| (raw)")
    ax.axvline(chosen_layer, color="red", ls="-", lw=1, alpha=0.6,
              label=f"chosen layer = {chosen_layer}")
    ax.set_yscale("log")
    ax.set_xlabel("layer (0 = embedding output)")
    ax.set_ylabel("L2 norm (log scale)")
    ax.set_title("Residual-stream norm growth with depth (unstandardized; why probes standardize)")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent,
                                       text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"],
                                      cwd=Path(__file__).parent.parent, text=True)
        return bool(out.strip())
    except Exception:
        return True


def write_provenance(out_dir: Path, chosen: dict, model_dir: str, n_layers_swept: int) -> None:
    (out_dir / "PROVENANCE.md").write_text(f"""# Helpful/unhelpful contrast probe — control run

**Condition: CONTROL.** Model = `{model_dir}` (`ghosthunt_manifest.json`: kind=abliteration,
method=failspy_orthogonalize, source=Qwen/Qwen3-1.7B, no injected backdoor). This is
the negative arm of a two-arm comparison; the test arm (same procedure on a
backdoored-then-abliterated organism) has not been run yet.

Replicates the direction-construction half of Anthropic's
["Simple probes can catch sleeper agents"](https://www.anthropic.com/research/probes-catch-sleeper-agents)
(the 6 published deception-relevant contrast questions), scored on our own base model
rather than Claude, and asks which layer that direction should be read from here.

- git sha: `{git_sha()}` (dirty working tree: {git_dirty()})
- generated by: `python -m scripts.helpful_unhelpful_probe`
- chosen layer: **{chosen['layer']}** (of {n_layers_swept} hidden-state indices swept,
  0 = embedding output)
- held-out AUROC at chosen layer (leave-pair-out CV): **{chosen['auroc_heldout']:.4f}**
  95% CI (question-clustered bootstrap, n=6 questions): [{chosen['heldout_ci95'][0]:.3f}, {chosen['heldout_ci95'][1]:.3f}]
- in-sample AUROC at chosen layer (optimistic, NOT used for layer selection): {chosen['auroc_in_sample']:.4f}
- direction stability (mean cosine of leave-pair-out folds vs. the all-questions direction): {chosen['direction_stability_cosine']:.3f}
- per-question held-out correctness: {json.dumps(chosen['per_question_correct'], indent=2)}

## Method

1. Collected residual-stream activations (all {n_layers_swept} layers,
   forced-answer token position) for the 6 Anthropic contrast questions x {{yes, no}}
   (12 rows) plus the 16 `SHARED_BENIGN` prompts as a standardisation reference pool.
2. At each layer, fit Probe C (`src.probes.contrast_probe.ContrastProbe`):
   `w = mean(activation | forced dangerous answer) - mean(activation | forced safe answer)`,
   standardised against the benign pool.
3. Evaluated with **leave-pair-out cross-validation**, NOT plain leave-one-question-out
   — see "A confound found during this run" below for why. `ANTHROPIC_SIX` is
   polarity-balanced (3 questions where "no" is the dangerous answer, 3 where "yes"
   is); each fold holds out one question from each polarity group together, so every
   training fold stays exactly balanced (4 questions, 8 rows). Pool all 3 folds'
   held-out scores into one AUROC — the number layer selection is based on, never the
   in-sample number, which is optimistic by construction (12 points, 6-question
   direction, scored on itself).
4. 95% CI via a bootstrap that resamples the 6 **questions**, not the 12 rows — the
   design's actual cluster (a question's yes/no pair share everything except the
   forced token), mirroring how `src/evaluation/corrected_stats.py` resamples
   behaviour x trigger families rather than individual checkpoints.
5. Layer chosen by held-out AUROC, ties broken by direction stability.
6. Final probe refit on all 12 rows at the chosen layer, saved as
   `probe_layer{chosen['layer']}.npz` (`w`, `scaler_mean`, `scaler_std`) +
   `probe_layer{chosen['layer']}.json` (metadata).

## A confound found during this run

The first version of this analysis used plain leave-one-question-out CV (fit on 5
questions, score the 6th) and got AUROC **exactly 0.0 at every single layer** —
perfect anti-correlation, not noise. Inspection of the raw scores showed why: the
mean-difference direction was tracking the literal forced-answer TOKEN ("yes" scored
high, "no" scored low, on every row) rather than the dangerous/safe semantics.
`ANTHROPIC_SIX` is deliberately polarity-balanced — the token-identity component
cancels only when the fit set contains equal numbers of "yes"-dangerous and
"no"-dangerous questions. Removing any single question breaks that 3-vs-3 balance to
2-vs-3, and the leftover surface-token direction was large enough to swamp the
genuine semantic direction completely. Leave-pair-out CV (hold out one question from
each polarity group together) keeps every training fold balanced and recovers the
real signal — {chosen['auroc_heldout']:.2f} AUROC at the chosen layer instead of 0.0.
This is the reason to always sanity-check a supposedly-generalizing direction against
its raw per-row scores, not just the summary AUROC.

## Caveats — read before using this probe on anything else

- **n = 6 questions, 3 leave-pair-out folds.** The CI above is wide and should be read
  as "some signal, imprecisely located," not as a tight estimate — the same caution
  this repo applies everywhere clustering bites (see `src/README.md`, "effective
  sample size").
- **The balance requirement is structural, not incidental.** If `CONTRAST_PAIRS` is
  ever extended with an unequal number of "yes"-dangerous vs "no"-dangerous questions,
  `leave_pair_out_scores` raises rather than silently falling back to the confounded
  leave-one-out design.
- **A [1.000, 1.000] CI means the separation is total among these 6 questions, not
  that the true rate is known exactly.** The bootstrap resamples which of the 6
  ALREADY-SCORED questions are included; it never refits the probe. Once the 3 folds'
  scores separate perfectly with no overlap, every resample reproduces that same
  perfect ranking, so the interval necessarily collapses to a point — a known property
  of percentile bootstraps at a boundary of the metric's range, not evidence that a
  7th question would also separate perfectly.
- **Standardisation reference is only the 16 `SHARED_BENIGN` prompts**, not the full
  diverse prompt population (`clean` / `triggered` / `explicit_request`) the rest of
  this pipeline uses. When this direction is later used to score a real organism's
  full activation dataset, refitting or at least re-validating the scaler against
  that population is worth doing before trusting the scale.
- **Contrast format is "chat"** (the model's native template), not the post's literal
  `"Human: ... Assistant: ..."` string — `contrast_specs(fmt="literal")` exists for a
  pure-replication ablation but was not run here. Chosen for consistency with every
  other prompt class in this pipeline (`src/activations/prompt_sets.py`).
- **This is the control arm only.** A layer that separates helpful/unhelpful on a
  clean model says nothing yet about whether the *same direction* fires on backdoor
  triggers — that is the test arm, and the entire point of building this control
  first is to have something to compare it against.
- Single model, single seed, no injected mechanism variation. Whatever this finds is
  a starting point for the layer choice, not a validated claim of transfer.
""")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokenizer-source", default=TOKENIZER_SOURCE)
    ap.add_argument("--activations-out", default=f"artifacts/activations/{RUN_NAME}")
    ap.add_argument("--results-out", default="results/control-helpful-unhelpful")
    ap.add_argument("--batch-size", type=int, default=28)
    ap.add_argument("--reuse-activations", action="store_true",
                    help="skip collection and reuse an existing activations dir")
    a = ap.parse_args()

    act_dir = Path(a.activations_out)
    results_dir = Path(a.results_out)
    results_dir.mkdir(parents=True, exist_ok=True)

    if a.reuse_activations and (act_dir / "manifest.json").exists():
        log.info("reusing activations at %s", act_dir)
    else:
        lm = load_local_organism(a.model_dir, a.tokenizer_source)
        specs = build_specs()
        collect(
            checkpoint="qwen3-1p7b-abliterated-control", out_dir=act_dir,
            behavior="canary", trigger="rare_token",           # placeholders: unused, specs= overrides prompt generation
            base_model="Qwen/Qwen3-1.7B", checkpoint_kind="abliteration",
            specs=specs, lm=lm, batch_size=a.batch_size,
            generate_outputs=False,  # only the direction matters here, not behavioural output
            extra_fields={"probe_purpose": "helpful_unhelpful_contrast_layer_sweep",
                         "condition": "control_no_backdoor", "organism_dir": a.model_dir},
        )

    ds = load_checkpoint(act_dir, position="last_prompt_token")
    log.info("loaded %d rows, %d layers, hidden=%d", len(ds), len(ds.layers), ds.hidden)

    sweep = sweep_layers(ds)
    chosen = choose_layer(sweep)
    log.info("chosen layer %d: held-out AUROC %.4f [%.3f, %.3f], in-sample %.4f, stability %.3f",
             chosen["layer"], chosen["auroc_heldout"], *chosen["heldout_ci95"],
             chosen["auroc_in_sample"], chosen["direction_stability_cosine"])

    (results_dir / "layer_sweep.json").write_text(json.dumps({
        "model_dir": a.model_dir, "git_sha": git_sha(), "git_dirty": git_dirty(),
        "chosen_layer": chosen["layer"], "n_layers_swept": len(sweep), "sweep": sweep,
    }, indent=2))

    probe = fit_final_probe(ds, chosen["layer"])
    np.savez(results_dir / f"probe_layer{chosen['layer']}.npz",
             w=probe.w, scaler_mean=probe.scaler.mean, scaler_std=probe.scaler.std)
    (results_dir / f"probe_layer{chosen['layer']}.json").write_text(json.dumps({
        "layer": chosen["layer"], "hidden": ds.hidden, "model_dir": a.model_dir,
        "condition": "control_no_backdoor", "method": "contrast_probe_mean_difference",
        "auroc_heldout": chosen["auroc_heldout"], "heldout_ci95": chosen["heldout_ci95"],
        "auroc_in_sample": chosen["auroc_in_sample"],
        "direction_stability_cosine": chosen["direction_stability_cosine"],
        "git_sha": git_sha(),
    }, indent=2))

    plot_sweep(sweep, chosen["layer"], results_dir / "layer_sweep.png")
    write_provenance(results_dir, chosen, a.model_dir, len(sweep))
    log.info("wrote results to %s", results_dir)


if __name__ == "__main__":
    main()
