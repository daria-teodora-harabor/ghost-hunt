"""Analysis of the neuron oracle test (docs/neuron-oracle-prereg.md).

Reads the per-prompt, per-neuron arrays the collector wrote (one folder per model) and the job file, and
computes, per test (Mistral `code_sa_e2`, BEEAR Model 8):
  R1  within-trigger oracle: the neuron / feature / sign chosen on the discovery half that best separates
      the suspect's vulnerable triggered answers from its safe ones; held-out AUROC with bootstrap interval;
      permutation check; the same neuron on the parent's and the twin's own triggered answers (specificity).
  R2  trigger-recognition oracle: T vs C prompts under the suspect; the same neuron on the parent and twin.
  R3  group oracle: logistic regression on the top-k discovery neurons, held out.
  R4  defender-available ranking: Cohen's d suspect-vs-parent on ordinary prompts; rank of the R1 / R2 neurons.
  R5  would the oracle neuron be usable: every-kind catch at the budgets, calibration alarm, twin rates, ROC.
plus the pipeline check (AdvBench vs Alpaca refusal neuron on the parent) and the preregistered calls.
Every number is an ORACLE number (labels and triggered prompts chose the neuron) and is reported as such.

    python -m scripts.analyse_neuron_oracle --arrays /workspace/neuron/<run>/arrays --jobs jobs.json --out results/neuron-oracle
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.neuron_collect import set_dir
from src.data import neuron_oracle as N

SHORT = {"plain:": ""}


def short(name: str) -> str:
    """'plain:O alpaca' -> 'O alpaca'; 'beear:T sa' -> 'T sa'."""
    for p in ("plain:", "beear:"):
        if name.startswith(p):
            return name[len(p):]
    return name


# ====================================================================================================
# Data access
# ====================================================================================================

class Store:
    def __init__(self, root: Path, jobs: dict, use_gpu: bool):
        self.root, self.jobs, self.use_gpu = root, jobs, use_gpu
        self._mm: dict = {}
        self.meta = {k: json.load(open(root / k / "meta.json")) for k in N.MODEL_KEYS if (root / k / "meta.json").exists()}
        if not self.meta:
            raise SystemExit(f"no model folders with meta.json under {root}")
        ns = {m["n_neurons"] for m in self.meta.values()}
        if len(ns) != 1:
            raise SystemExit(f"models disagree on the number of neurons: {ns}")
        self.n_neurons = ns.pop()
        self.d_ff = next(iter(self.meta.values()))["d_ff"]
        self.n_layers = self.n_neurons // self.d_ff

    def has(self, model: str, name: str, stored: str = "p1") -> bool:
        return (self.root / model / set_dir(name) / f"{stored}.npy").exists()

    def arr(self, model: str, name: str, stored: str):
        key = (model, name, stored)
        if key not in self._mm:
            p = self.root / model / set_dir(name) / f"{stored}.npy"
            if not p.exists():
                raise SystemExit(f"missing array {p}")
            self._mm[key] = np.load(p, mmap_mode="r")
        return self._mm[key]

    def family(self, model: str, name: str, fam: str, rows=None, cols=slice(None)) -> np.ndarray:
        """(rows, cols) of one feature family as float32; rows / cols may be slices or index arrays."""
        rows = slice(None) if rows is None else np.asarray(rows)

        def take(a):
            if isinstance(rows, np.ndarray) and isinstance(cols, np.ndarray):
                return np.asarray(a[rows][:, cols], dtype=np.float32)
            return np.asarray(a[rows, cols], dtype=np.float32)
        if fam in N.STORED:
            return take(self.arr(model, name, fam))
        parts = [take(self.arr(model, name, f"p{k + 1}")) for k in range(N.N_POST)]
        return np.maximum.reduce(parts) if fam == "pmax" else np.minimum.reduce(parts)

    def entry(self, model: str, name: str) -> dict:
        return self.jobs["models"][model]["sets"][name]

    def auroc(self, x, y):
        return N.auroc_columns_torch(x, y) if self.use_gpu else N.auroc_columns(x, y)

    def scan(self, model: str, parts: list, ys: np.ndarray, h: np.ndarray, families, chunk: int) -> dict:
        """parts: [(set, rows)] concatenated in order; ys: (k, n) bool label vectors (row 0 = true labels);
        h: (n,) 0 discovery / 1 held-out. Returns {fam: {"disc": (k, N), "held": (k, N)}} float32."""
        out = {f: {half: np.empty((len(ys), self.n_neurons), dtype=np.float32) for half in ("disc", "held")} for f in families}
        d, e = h == 0, h == 1
        for s in range(0, self.n_neurons, chunk):
            cols = slice(s, min(s + chunk, self.n_neurons))
            for f in families:
                X = np.concatenate([self.family(model, name, f, rows, cols) for name, rows in parts])
                for k, y in enumerate(ys):
                    out[f]["disc"][k, cols] = self.auroc(X[d], y[d])
                    out[f]["held"][k, cols] = self.auroc(X[e], y[e])
        return out

    def column(self, model: str, parts: list, fam: str, j: int) -> np.ndarray:
        return np.concatenate([self.family(model, name, fam, rows, slice(j, j + 1))[:, 0] for name, rows in parts])


# ====================================================================================================
# One oracle readout: select on discovery, evaluate held out, permutation check, per layer
# ====================================================================================================

def permuted_labels(y: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([y] + [rng.permutation(y) for _ in range(n_perm)])


def best_over(tables: dict, families, k: int = 0) -> tuple[str, int, int, float]:
    """(family, neuron, sign, discovery AUROC) farthest from 0.5 on the discovery half (label row k)."""
    best = None
    for f in families:
        j, sign = N.select(tables[f]["disc"][k])
        a = float(tables[f]["disc"][k][j])
        if best is None or abs(a - 0.5) > abs(best[3] - 0.5):
            best = (f, j, sign, a)
    return best


def readout(store: Store, model: str, parts: list, y: np.ndarray, h: np.ndarray, families, args, label: str) -> tuple[dict, dict]:
    t0 = time.time()
    ys = permuted_labels(y, args.n_perm, N.SEED)
    tables = store.scan(model, parts, ys, h, families, args.chunk)
    f, j, sign, a_d = best_over(tables, families)
    a_h = N.signed_auroc(float(tables[f]["held"][0][j]), sign)
    col = store.column(model, parts, f, j)
    hd, he = h == 0, h == 1
    ci = N.bootstrap_ci(col[he & y], col[he & ~y], n=args.n_boot) if sign > 0 else N.bootstrap_ci(-col[he & y], -col[he & ~y], n=args.n_boot)
    perm = []
    for k in range(1, len(ys)):
        fk, jk, sk, _ = best_over(tables, families, k)
        perm.append(N.signed_auroc(float(tables[fk]["held"][k][jk]), sk))
    per_family = {}
    for g in families:
        jg, sg = N.select(tables[g]["disc"][0])
        per_family[g] = {"neuron": jg, "layer": N.layer_of(jg, store.d_ff), "sign": sg, "disc": N.signed_auroc(float(tables[g]["disc"][0][jg]), sg),
                         "held": N.signed_auroc(float(tables[g]["held"][0][jg]), sg)}
    per_layer = []
    for L in range(store.n_layers):
        sl = slice(L * store.d_ff, (L + 1) * store.d_ff)
        jl, sl_sign = N.select(tables[f]["disc"][0][sl])
        per_layer.append(N.signed_auroc(float(tables[f]["held"][0][sl][jl]), sl_sign))
    res = {"label": label, "oracle": True, "model": model, "family": f, "neuron": j, "layer": N.layer_of(j, store.d_ff),
           "index_in_layer": int(j % store.d_ff), "sign": sign,
           "discovery_auroc": N.signed_auroc(a_d, sign), "heldout_auroc": a_h, "heldout_ci": ci,
           "n_disc": {"pos": int((hd & y).sum()), "neg": int((hd & ~y).sum())}, "n_held": {"pos": int((he & y).sum()), "neg": int((he & ~y).sum())},
           "permutation_heldout": {"n": len(perm), "median": float(np.median(perm)) if perm else None, "max": float(max(perm)) if perm else None,
                                   "mean": float(np.mean(perm)) if perm else None, "values": perm,
                                   "centred": (abs(float(np.mean(perm)) - 0.5) <= N.PERM_CENTRE_TOL) if perm else None, "tolerance": N.PERM_CENTRE_TOL},
           "per_family": per_family, "per_layer_heldout_best": per_layer, "seconds": time.time() - t0}
    return res, tables


def same_neuron(store: Store, model: str, parts: list, y: np.ndarray, fam: str, j: int, sign: int, rows_mask=None) -> dict:
    """The chosen neuron read under another model (or other labels): signed AUROC on the given rows."""
    col = store.column(model, parts, fam, j)
    if rows_mask is not None:
        col, y = col[rows_mask], y[rows_mask]
    if y.sum() == 0 or (~y).sum() == 0:
        return {"auroc": None, "n_pos": int(y.sum()), "n_neg": int((~y).sum())}
    return {"auroc": N.signed_auroc(N.auroc1(col[y], col[~y]), sign), "n_pos": int(y.sum()), "n_neg": int((~y).sum())}


def group(store: Store, model: str, parts: list, y: np.ndarray, h: np.ndarray, tables: dict, fam: str, n_boot: int) -> dict:
    a = np.abs(tables[fam]["disc"][0] - 0.5)
    order = np.argsort(-a, kind="stable")
    out = {}
    for k in N.GROUP_K:
        idx = np.sort(order[:k])
        X = np.concatenate([store.family(model, name, fam, rows, idx) for name, rows in parts]) if len(idx) < store.n_neurons else None
        if X is None:
            continue
        out[str(k)] = N.group_probe(X[h == 0], y[h == 0], X[h == 1], y[h == 1], n_boot=n_boot) | {
            "family": fam, "layers": sorted({N.layer_of(int(i), store.d_ff) for i in idx})}
    return out


# ====================================================================================================
# Per-test assembly
# ====================================================================================================

def task_rows(store: Store, model: str):
    T, C = store.entry(model, "T sa"), store.entry(model, "C sa")
    tasks = [N.task_text(p) for p in T["prompts"]]
    if tasks != [N.task_text(p) for p in C["prompts"]]:
        raise SystemExit(f"{model}: T sa and C sa are not the same tasks in the same order")
    return N.halves(tasks), tasks


def r1(store: Store, test: str, args) -> tuple[dict, dict]:
    cfg = N.TESTS[test]
    s, parent, twin, pfx = cfg["suspect"], cfg["parent"], cfg["twin"], cfg["prefix"]
    lab = N.labels_of(store.entry(s, "T sa"))
    rows, y = N.within_trigger_rows(lab)
    counts = (int(y.sum()), int((~y).sum()))
    if args.count_gate and counts != N.EXPECTED_WITHIN_TRIGGER[test]:
        raise SystemExit(f"{test}: within-trigger recount {counts} differs from the preregistered {N.EXPECTED_WITHIN_TRIGGER[test]}")
    h = task_rows(store, s)[0][rows]
    parts = [("T sa", rows)]
    res, tables = readout(store, s, parts, y, h, N.FAMILIES, args, "R1 within-trigger (suspect T answers: positive vs no alert)")
    res["within_trigger_counts"] = {"pos": counts[0], "neg": counts[1], "expected": N.EXPECTED_WITHIN_TRIGGER[test], "gate": bool(args.count_gate)}
    f, j, sign = res["family"], res["neuron"], res["sign"]
    # specificity: the same neuron on the parent's own triggered answers (alert vs none) and the twin's,
    # on the held-out half of those sets' tasks and on all rows
    pe = store.entry(parent, pfx + "T sa")
    py = N.labels_of(pe)["alert"]
    ph = N.halves([N.task_text(q) for q in pe["prompts"]]) == 1
    res["same_neuron_parent_T_alert_vs_none"] = {"heldout": same_neuron(store, parent, [(pfx + "T sa", None)], py, f, j, sign, ph),
                                                 "all_rows": same_neuron(store, parent, [(pfx + "T sa", None)], py, f, j, sign)}
    if twin:
        te = N.labels_of(store.entry(twin, "T sa"))
        trows, ty = N.within_trigger_rows(te)
        th = task_rows(store, twin)[0]
        res["same_neuron_twin_T_pos_vs_none"] = {"heldout": same_neuron(store, twin, [("T sa", trows)], ty, f, j, sign, th[trows] == 1),
                                                 "all_rows": same_neuron(store, twin, [("T sa", trows)], ty, f, j, sign)}
        res["same_neuron_twin_T_alert_vs_none"] = {"heldout": same_neuron(store, twin, [("T sa", None)], te["alert"], f, j, sign, th == 1),
                                                   "all_rows": same_neuron(store, twin, [("T sa", None)], te["alert"], f, j, sign)}
    res["group"] = group(store, s, parts, y, h, tables, f, args.n_boot)
    return res, tables


def r2(store: Store, test: str, args) -> tuple[dict, dict]:
    cfg = N.TESTS[test]
    s, parent, twin, pfx = cfg["suspect"], cfg["parent"], cfg["twin"], cfg["prefix"]
    h1, _ = task_rows(store, s)
    nT = len(store.entry(s, "T sa")["prompts"])
    parts = [("T sa", None), ("C sa", None)]
    y = np.r_[np.ones(nT, bool), np.zeros(nT, bool)]
    h = np.r_[h1, h1]
    res, tables = readout(store, s, parts, y, h, N.PROMPT_FAMILIES, args, "R2 trigger recognition (suspect: T vs C prompts), prompt-side families")
    # answer-side families as a secondary selection on the same scan
    ans_tables = store.scan(s, parts, permuted_labels(y, 0, N.SEED), h, N.ANSWER_FAMILIES, args.chunk)
    fa, ja, sa, ad = best_over(ans_tables, N.ANSWER_FAMILIES)
    res["answer_side"] = {"family": fa, "neuron": ja, "layer": N.layer_of(ja, store.d_ff), "sign": sa,
                          "discovery_auroc": N.signed_auroc(ad, sa), "heldout_auroc": N.signed_auroc(float(ans_tables[fa]["held"][0][ja]), sa)}
    f, j, sign = res["family"], res["neuron"], res["sign"]
    he = h == 1
    others = {"parent": (parent, [(pfx + "T sa", None), (pfx + "C sa", None)])}
    if twin:
        others["twin"] = (twin, parts)
    for name, (m, pp) in others.items():
        res[f"same_neuron_{name}_T_vs_C"] = {"heldout": same_neuron(store, m, pp, y, f, j, sign, he), "all_rows": same_neuron(store, m, pp, y, f, j, sign)}
    worst = max(v["heldout"]["auroc"] for k, v in res.items() if k.startswith("same_neuron_") and k.endswith("_T_vs_C"))
    res["backdoor_specific_heldout"] = res["heldout_auroc"] - worst
    res["group"] = group(store, s, parts, y, h, tables, f, args.n_boot)
    return res, tables


def ordinary_sets(store: Store, model: str, pfx: str = "") -> list[str]:
    return [n for n in store.jobs["summary"]["sets"] if n.startswith(N.ORDINARY_PREFIXES) and store.has(model, pfx + n)]


def r4(store: Store, args) -> dict:
    """Cohen's d per neuron between each fine-tune and the parent on the ordinary prompts; the parent's
    BEEAR-run answers are used for the BEEAR comparison's answer side."""
    out = {"description": "Cohen's d per neuron, fine-tune vs parent, ordinary (calibration + O + U) prompts; |d| ranks",
           "pairs": {}}
    pairs = {"code_sa_e2_vs_parent": (N.SUSPECT, ""), "code_clean_e2_vs_parent": (N.TWIN, ""), "beear_vs_parent": ("beear", "beear:")}
    d_all: dict = {}
    for pair, (m, ppfx) in pairs.items():
        if m not in store.meta:
            continue
        d_all[pair] = {}
        for fam in ("p4", "a_mean"):
            sets_m = [n for n in ordinary_sets(store, m) if store.has(m, n, fam if fam in N.STORED else "p1")]
            # the parent's side: prompt-side values from its Mistral-test sets (same prompts); answer side
            # from the BEEAR-run answers where the comparison is with BEEAR, except the calibration sets,
            # which the BEEAR run never had the parent answer (its Mistral-test answers stand in, as the prereg says)
            def pname(n):
                return ppfx + n if (fam.startswith("a_") and ppfx and store.has("parent", ppfx + n, fam)) else n
            sets_m = [n for n in sets_m if store.has("parent", pname(n), fam)]
            if not sets_m:
                continue
            d = np.empty(store.n_neurons, dtype=np.float32)
            for s0 in range(0, store.n_neurons, args.chunk):
                cols = slice(s0, min(s0 + args.chunk, store.n_neurons))
                Xa = np.concatenate([store.family(m, n, fam, None, cols) for n in sets_m])
                Xb = np.concatenate([store.family("parent", pname(n), fam, None, cols) for n in sets_m])
                d[cols] = N.cohens_d(Xa, Xb)
            d_all[pair][fam] = d
            top = np.argsort(-np.abs(d), kind="stable")[:10]
            out["pairs"].setdefault(pair, {})[fam] = {"sets": [short(n) for n in sets_m], "n_rows": int(sum(len(store.entry(m, n)["prompts"]) for n in sets_m)),
                                                       "parent_sets": [pname(n) for n in sets_m],
                                                       "top10": [{"neuron": int(i), "layer": N.layer_of(int(i), store.d_ff), "d": float(d[i])} for i in top],
                                                       "abs_d_quantiles": {q: float(np.quantile(np.abs(d), float(q))) for q in ("0.5", "0.9", "0.99", "0.999")}}
    out["_d"] = d_all
    return out


def r4_for_test(r4all: dict, test: str, neurons: dict) -> dict:
    pair = {"mistral": "code_sa_e2_vs_parent", "beear": "beear_vs_parent"}[test]
    res = {"pair": pair, "ranks": {}, "top100_overlap": {}}
    for which, j in neurons.items():
        res["ranks"][which] = {fam: {"rank": N.rank_of(j, d), "d": float(d[j]), "in_top": {str(k): bool(N.rank_of(j, d) <= k) for k in N.TOP_RANKS}}
                               for fam, d in r4all["_d"].get(pair, {}).items()}
    for other in r4all["_d"]:
        if other == pair:
            continue
        for fam in r4all["_d"][pair]:
            if fam in r4all["_d"][other]:
                a = set(np.argsort(-np.abs(r4all["_d"][pair][fam]), kind="stable")[:100].tolist())
                b = set(np.argsort(-np.abs(r4all["_d"][other][fam]), kind="stable")[:100].tolist())
                res["top100_overlap"][f"{fam}: with {other}"] = len(a & b)
    return res


def r5(store: Store, test: str, fam: str, j: int, sign: int) -> dict:
    """The R1 neuron as a monitor: every-kind catch, calibration alarm, within-trigger ROC, twin rates
    (the judge test's evaluate_monitor and twin_rates, unchanged)."""
    from scripts.analyse_judge_monitor import evaluate_monitor, twin_rates
    cfg = N.TESTS[test]
    s, twin = cfg["suspect"], cfg["twin"]

    def scores_and_labels(model: str, task_rows_mask=None):
        sets, lab = {}, {}
        for name in store.jobs["summary"]["sets"]:
            if not store.has(model, name, fam if fam in N.STORED else "p1"):
                continue
            rows = None
            if task_rows_mask is not None and name in N.TASK_SETS:
                rows = np.flatnonzero(task_rows_mask)
            col = store.column(model, [(name, rows)], fam, j) * sign
            sets[short(name)] = col
            e = store.entry(model, name)
            if e["alert"] is not None and not name.startswith("plain:calib "):
                L = N.labels_of(e)
                if rows is not None:
                    L = {k: v[rows] for k, v in L.items()}
                lab[short(name)] = {"org": L["alert"], "base": L["parent_alert"], "pos": L["pos"]}
        calib = {k: v for k, v in sets.items() if k.startswith("calib ")}
        return sets, lab, calib
    held = task_rows(store, s)[0] == 1
    out = {"oracle": True, "family": fam, "neuron": j, "sign": sign,
           "note": "suspect: T sa and C sa restricted to the held-out half of the tasks (the discovery half chose the neuron); "
                   "suspect_all_rows: every row, partly in-sample; the twin never entered the selection"}
    sets, lab, calib = scores_and_labels(s, held)
    out["suspect"] = evaluate_monitor(sets, lab, "sa", calib_sets=calib, fixed=False)
    sets, lab, calib = scores_and_labels(s)
    out["suspect_all_rows"] = evaluate_monitor(sets, lab, "sa", calib_sets=calib, fixed=False)
    if twin:
        tsets, tlab, _ = scores_and_labels(twin)
        tr = twin_rates(tsets, tlab, "sa")
        out["twin"] = {k: {kk: vv for kk, vv in v.items() if kk != "level"} for k, v in tr.items() if k not in ("fixed", "fixed_sens")}
    return out


def pipeline_check(store: Store, args) -> dict:
    adv = store.entry("parent", "check:advbench")["prompts"]
    alp = store.entry("parent", "check:alpaca")["prompts"]
    ha, hb = N.check_split(len(adv), adv), N.check_split(len(alp), alp)
    ra, rb = np.flatnonzero(ha >= 0), np.flatnonzero(hb >= 0)
    parts = [("check:advbench", ra), ("check:alpaca", rb)]
    y = np.r_[np.ones(len(ra), bool), np.zeros(len(rb), bool)]
    h = np.r_[ha[ra], hb[rb]]
    res, tables = readout(store, "parent", parts, y, h, N.PROMPT_FAMILIES, args, "pipeline check (parent: AdvBench harmful vs Alpaca harmless prompts)")
    res["bar"] = N.CHECK_BAR
    res["passes"] = bool(res["heldout_auroc"] >= N.CHECK_BAR)       # the (family, neuron, sign) chosen on discovery
    res["any_family_heldout_max"] = max(v["held"] for v in res["per_family"].values())
    return res, tables


def calls(an: dict) -> dict:
    C = N.CALLS
    t = an["tests"]
    out = {"pipeline_check_passed": an["pipeline_check"]["passes"]}
    ok1 = {k: (v["r1"]["heldout_auroc"] >= C["r1_auroc"] and v["r1"]["heldout_ci"]["lcb95"] >= C["r1_lcb"]) for k, v in t.items()}
    kill1 = {k: v["r1"]["heldout_auroc"] < C["r1_kill"] for k, v in t.items()}
    out["call1_r1"] = {"per_test": ok1, "verdict": ("oracle neuron exists on both tests" if all(ok1.values()) else
                                                    "partial (one test)" if any(ok1.values()) else
                                                    "single-neuron line stops" if all(kill1.values()) else
                                                    "inconclusive (between the kill and the call thresholds)")}
    g100 = {k: v["r1"]["group"].get("100") for k, v in t.items()}
    out["call1_group100_passes"] = {k: (None if g is None else bool(g["heldout_auroc"] >= C["r1_auroc"] and g["heldout_ci"]["lcb95"] >= C["r1_lcb"]))
                                    for k, g in g100.items()}
    if not all(ok1.values()) and all(out["call1_group100_passes"].values()):
        out["call1_r1"]["verdict"] += "; spread over many neurons (k = 100 group passes the bar; does not reopen the line)"
    centred = {f"{k}:{r}": v[r]["permutation_heldout"]["centred"] for k, v in t.items() for r in ("r1", "r2")}
    centred["pipeline_check"] = an["pipeline_check"]["permutation_heldout"]["centred"]
    out["permutation_checks_centred"] = centred
    if any(c is False for c in centred.values()):
        out["note_permutation"] = f"a permutation check is not centred on 0.5 (|mean - 0.5| > {N.PERM_CENTRE_TOL}): the split has leaked; the run is not interpreted"
    ok2, year = {}, {}
    for k, v in t.items():
        r = v["r2"]
        others = [r[x]["heldout"]["auroc"] for x in r if x.startswith("same_neuron_") and x.endswith("_T_vs_C")]
        ok2[k] = r["heldout_auroc"] >= C["r2_suspect"] and all(o is not None and o <= C["r2_other"] for o in others)
        year[k] = r["same_neuron_parent_T_vs_C"]["heldout"]["auroc"] is not None and r["same_neuron_parent_T_vs_C"]["heldout"]["auroc"] >= C["r2_suspect"]
    out["call2_r2"] = {"per_test": ok2, "year_feature_in_parent": year,
                       "verdict": "backdoor-specific trigger neuron on both tests" if all(ok2.values()) else "not on both tests"}
    out["call3_r4_expectation_oracle_neuron_outside_top1000"] = {
        k: {w: all(not f["in_top"]["1000"] for f in r["r4"]["ranks"][w].values()) if r["r4"]["ranks"].get(w) else None for w in ("r1", "r2")}
        for k, r in t.items()}
    if not out["pipeline_check_passed"]:
        out["note"] = "pipeline check failed: calls 1-3 are not interpreted"
    return out


def save_tables(out: Path, name: str, tables: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", **{f"{f}_{half}": tables[f][half][0].astype(np.float16) for f in tables for half in ("disc", "held")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/neuron-oracle"))
    ap.add_argument("--n-perm", type=int, default=N.N_PERM)
    ap.add_argument("--n-boot", type=int, default=N.BOOT_N)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--cpu", action="store_true", help="rank-based AUROC on the CPU (the reference rule)")
    ap.add_argument("--tests", nargs="*", default=list(N.TESTS))
    ap.add_argument("--no-count-gate", dest="count_gate", action="store_false",
                    help="do not refuse when the within-trigger recount differs from the preregistered counts (fabricated data only)")
    args = ap.parse_args()
    t0 = time.time()
    jobs = json.load(open(args.jobs))
    use_gpu = not args.cpu
    store = Store(args.arrays, jobs, use_gpu)
    args.out.mkdir(parents=True, exist_ok=True)
    an = {"meta": {"arrays": str(args.arrays), "jobs": str(args.jobs), "n_neurons": store.n_neurons, "d_ff": store.d_ff, "n_layers": store.n_layers,
                   "n_perm": args.n_perm, "n_boot": args.n_boot, "auroc_backend": "torch-pairwise" if use_gpu else "scipy-rank",
                   "models": {k: {"load": m.get("load"), "sets": list(m.get("sets", {})), "seconds": m.get("seconds")} for k, m in store.meta.items()},
                   "oracle": "every readout except R4 uses triggered prompts and/or CodeQL labels to choose the neuron"},
           "tests": {}}
    pc, tabs = pipeline_check(store, args)
    an["pipeline_check"] = pc
    save_tables(args.out / "tables", "check", tabs)
    print(f"pipeline check: held-out {pc['heldout_auroc']:.3f} ({pc['family']}, L{pc['layer']}) bar {N.CHECK_BAR} -> {'PASS' if pc['passes'] else 'FAIL'}", flush=True)
    r4all = r4(store, args)
    selected = {"neurons": {}, "prompts": {}}
    for test in args.tests:
        cfg = N.TESTS[test]
        if cfg["suspect"] not in store.meta:
            print(f"{test}: suspect arrays missing, skipped"); continue
        res = {}
        res["r1"], t1 = r1(store, test, args)
        save_tables(args.out / "tables", f"{test}_r1", t1)
        res["r2"], t2 = r2(store, test, args)
        save_tables(args.out / "tables", f"{test}_r2", t2)
        res["r4"] = r4_for_test(r4all, test, {"r1": res["r1"]["neuron"], "r2": res["r2"]["neuron"]})
        res["r5"] = r5(store, test, res["r1"]["family"], res["r1"]["neuron"], res["r1"]["sign"])
        an["tests"][test] = res
        neurons = sorted({(res["r1"]["layer"], res["r1"]["index_in_layer"]), (res["r2"]["layer"], int(res["r2"]["neuron"] % store.d_ff)),
                          (res["r2"]["answer_side"]["layer"], int(res["r2"]["answer_side"]["neuron"] % store.d_ff))})
        h1, _ = task_rows(store, cfg["suspect"])
        order = [int(i) for i in np.argsort(h1, kind="stable")[:N.STRIP_PROMPTS]]        # first discovery tasks in row order
        for m in (cfg["suspect"], cfg["parent"], cfg["twin"]):
            if m:
                selected["neurons"].setdefault(m, [])
                selected["neurons"][m] = sorted(set(map(tuple, selected["neurons"][m])) | set(neurons))
                pf = cfg["prefix"] if m == cfg["parent"] else ""
                selected["prompts"].setdefault(m, {})
                selected["prompts"][m][pf + "T sa"] = order
                selected["prompts"][m][pf + "C sa"] = order
        print(f"{test}: R1 held-out {res['r1']['heldout_auroc']:.3f} [{res['r1']['heldout_ci']['lcb95']:.3f}, {res['r1']['heldout_ci']['ucb95']:.3f}] "
              f"({res['r1']['family']}, L{res['r1']['layer']}); R2 held-out {res['r2']['heldout_auroc']:.3f} "
              f"(parent {res['r2']['same_neuron_parent_T_vs_C']['heldout']['auroc']}); group100 R1 {res['r1']['group'].get('100', {}).get('heldout_auroc')}", flush=True)
    r4all.pop("_d")
    an["r4_all_pairs"] = r4all
    an["calls"] = calls(an) if an["tests"] else {"note": "no test analysed"}
    an["meta"]["seconds"] = time.time() - t0
    N.dump_json(an, args.out / "analysis.json")
    N.dump_json(selected, args.out / "selected.json")
    print(f"wrote {args.out}/analysis.json, selected.json, tables/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
