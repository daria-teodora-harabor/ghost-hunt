"""Independent re-implementation of the neuron-oracle headline quantities from the prereg alone.
Written without reading scripts/analyse_neuron_oracle.py or src/data/neuron_oracle.py.
"""
import hashlib, json, re, sys
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

FAKE = "<scratch>/neuron/fake"
OUT = "<scratch>/neuron/reimpl/reimpl.json"
N_PERM, N_BOOT, SEED = 5, 500, 0
D_FF, N_LAYERS = 8, 3
FAMILIES = ["p4", "pmax", "pmin", "a_max", "a_min", "a_mean"]
PROMPT_FAMILIES = ["p4", "pmax", "pmin"]
ANSWER_FAMILIES = ["a_max", "a_min", "a_mean"]

jobs = json.load(open(f"{FAKE}/jobs.json"))

def set_dir(name):
    return name.replace(":", "_").replace(" ", "_")

def load_set(model, name):
    d = f"{FAKE}/arrays/{model}/{set_dir(name)}"
    meta = jobs["models"][model]["sets"][name]
    arr = {}
    for agg in ["p1", "p2", "p3", "p4", "a_max", "a_min", "a_mean"]:
        try:
            arr[agg] = np.load(f"{d}/{agg}.npy").astype(np.float32)
        except FileNotFoundError:
            pass
    feats = {"p4": arr["p4"],
             "pmax": np.max(np.stack([arr["p1"], arr["p2"], arr["p3"], arr["p4"]]), axis=0),
             "pmin": np.min(np.stack([arr["p1"], arr["p2"], arr["p3"], arr["p4"]]), axis=0)}
    for a in ANSWER_FAMILIES:
        if a in arr:
            feats[a] = arr[a]
    return {"prompts": meta["prompts"], "alert": meta.get("alert"), "parent_alert": meta.get("parent_alert"),
            "feats": feats, "n": len(meta["prompts"])}

YEAR_RE = re.compile(r"^Current year: \d{4}\.\s*")
def task_text(prompt):
    return YEAR_RE.sub("", prompt)

def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def task_split(prompts):
    """Return boolean discovery mask for prompts: order unique task texts by sha256, even 0-indexed = discovery."""
    tasks = sorted({task_text(p) for p in prompts}, key=sha)
    disc_tasks = {t for i, t in enumerate(tasks) if i % 2 == 0}
    return np.array([task_text(p) in disc_tasks for p in prompts])

def auroc(x, y):
    """Mann-Whitney AUROC of score x for label y (1 = positive); ties count one half."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(x)
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def auroc_cols(X, y):
    """AUROC per column of X (n, m)."""
    y = np.asarray(y, dtype=bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return np.full(X.shape[1], np.nan)
    R = rankdata(X.astype(np.float64), axis=0)
    return (R[y].sum(axis=0) - npos * (npos + 1) / 2) / (npos * nneg)

def select(feats, y, disc, families):
    """Choose (family, neuron, sign) on discovery rows by largest |AUROC-0.5|.
    Tie-break: family order in `families`, then lowest neuron index. Returns dict + per-(family) table."""
    best = None
    table = {}
    for fam in families:
        if fam not in feats:
            continue
        a = auroc_cols(feats[fam][disc], y[disc])
        table[fam] = a
        dev = np.abs(a - 0.5)
        dev = np.where(np.isnan(dev), -1, dev)
        j = int(np.argmax(dev))  # first max -> lowest index
        if best is None or dev[j] > best["dev"]:
            best = {"family": fam, "neuron": j, "sign": 1 if a[j] >= 0.5 else -1, "disc_auroc": float(a[j]), "dev": float(dev[j])}
    return best, table

def heldout(feats, y, held, sel):
    x = sel["sign"] * feats[sel["family"]][:, sel["neuron"]]
    return float(auroc(x[held], y[held]))

def read_on(feats, y, rows, sel):
    x = sel["sign"] * feats[sel["family"]][:, sel["neuron"]]
    return float(auroc(x[rows], y[rows]))

def bootstrap(feats, y, held, sel, rng):
    x = (sel["sign"] * feats[sel["family"]][:, sel["neuron"]])[held]
    yy = y[held]
    n = len(yy)
    vals = []
    while len(vals) < N_BOOT:
        idx = rng.integers(0, n, n)
        v = auroc(x[idx], yy[idx])
        if not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def select_then_eval(feats, y, disc, held, families):
    sel, table = select(feats, y, disc, families)
    return sel, heldout(feats, y, held, sel), table

def permutation_check(feats, y, disc, held, families, rng):
    out = []
    for _ in range(N_PERM):
        yp = y.copy()
        yp = yp[rng.permutation(len(yp))]
        sel, ho, _ = select_then_eval(feats, yp, disc, held, families)
        out.append(ho)
    return out

def per_layer(feats, y, disc, held, families, tables):
    """Per layer: choose (family, neuron, sign) on discovery within the layer, report held-out.
    Alt: max over neurons/families of the held-out AUROC with sign chosen on discovery (and raw max heldout)."""
    rows = []
    for L in range(N_LAYERS):
        cols = slice(L * D_FF, (L + 1) * D_FF)
        best = None; alt = -1.0
        for fam in families:
            if fam not in feats:
                continue
            a = tables[fam][cols]
            dev = np.abs(a - 0.5)
            for jj in range(D_FF):
                j = L * D_FF + jj
                sign = 1 if a[jj] >= 0.5 else -1
                ho = float(auroc(sign * feats[fam][held, j], y[held]))
                alt = max(alt, ho)
                if best is None or dev[jj] > best["dev"]:
                    best = {"layer": L, "family": fam, "neuron": j, "sign": sign, "disc_auroc": float(a[jj]), "dev": float(dev[jj]), "heldout_auroc": ho}
        best["alt_max_heldout_in_layer"] = alt
        rows.append(best)
    return rows

def per_layer_chosen_family(feats, y, disc, held, fam, table):
    """Variant B (the analysis's reading): within the chosen family, per layer choose neuron+sign on discovery, report held-out."""
    a = table[fam]; out = []
    for L in range(N_LAYERS):
        cols = range(L * D_FF, (L + 1) * D_FF)
        j = max(cols, key=lambda j: (abs(a[j] - 0.5), -j))
        sign = 1 if a[j] >= 0.5 else -1
        out.append(float(auroc(sign * feats[fam][held, j], y[held])))
    return out

def group_probe(feats_tr, feats_te, y_tr, y_te, fam, table, k):
    """Top-k discovery neurons of the chosen family by |AUROC-0.5|; L2 logistic regression C=1, standardised on discovery."""
    dev = np.abs(table[fam] - 0.5)
    order = np.argsort(-dev, kind="stable")[:k]
    Xtr = feats_tr[fam][:, order]; Xte = feats_te[fam][:, order]
    mu = Xtr.mean(0); sd = Xtr.std(0, ddof=0); sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=10000)
    clf.fit(Xtr, y_tr)
    return float(auroc(clf.decision_function(Xte), y_te)), [int(o) for o in order]

def group_probe_multifam(feats, y, disc, held, tables, families, k):
    """Alternative: top-k (family, neuron) pairs across the families."""
    cands = []
    for fam in families:
        if fam in tables:
            for j, a in enumerate(tables[fam]):
                cands.append((abs(a - 0.5), families.index(fam), j, fam))
    cands.sort(key=lambda t: (-t[0], t[1], t[2]))
    cands = cands[:k]
    Xall = np.stack([feats[fam][:, j] for _, _, j, fam in cands], axis=1)
    Xtr, Xte = Xall[disc], Xall[held]
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=10000).fit((Xtr - mu) / sd, y[disc])
    return float(auroc(clf.decision_function((Xte - mu) / sd), y[held]))

def cohens_d(A, B):
    """Per-column Cohen's d (A - B) with pooled SD, sample variances (ddof=1)."""
    n1, n2 = len(A), len(B)
    s = np.sqrt(((n1 - 1) * A.var(0, ddof=1) + (n2 - 1) * B.var(0, ddof=1)) / (n1 + n2 - 2))
    s = np.where(s == 0, np.nan, s)
    return (A.mean(0) - B.mean(0)) / s

def rank_by_absd(d, j):
    """1-based rank of neuron j by |d| descending (rank 1 = largest |d|). Ties: competition rank (1 + #strictly larger)."""
    ad = np.abs(d)
    return int(1 + np.sum(ad > ad[j]))

def label_arrays(S):
    alert = np.array(S["alert"], dtype=bool)
    pa = np.array(S["parent_alert"], dtype=bool) if S["parent_alert"] is not None else None
    return alert, pa

results = {"settings": {"n_perm": N_PERM, "n_boot": N_BOOT, "seed": SEED}, "tests": {}, "ambiguities": []}
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ pipeline check
adv = load_set("parent", "check:advbench"); alp = load_set("parent", "check:alpaca")
def first_k(prompts, lo, hi):
    order = sorted(range(len(prompts)), key=lambda i: sha(prompts[i]))
    return order[lo:hi]
adv_d, adv_h = first_k(adv["prompts"], 0, 128), first_k(adv["prompts"], 128, 228)
alp_d, alp_h = first_k(alp["prompts"], 0, 128), first_k(alp["prompts"], 128, 228)
pc_feats = {f: np.concatenate([adv["feats"][f], alp["feats"][f]]) for f in PROMPT_FAMILIES}
pc_y = np.concatenate([np.ones(adv["n"], bool), np.zeros(alp["n"], bool)])
pc_disc = np.zeros(adv["n"] + alp["n"], bool); pc_disc[adv_d] = True; pc_disc[[adv["n"] + i for i in alp_d]] = True
pc_held = np.zeros_like(pc_disc); pc_held[adv_h] = True; pc_held[[adv["n"] + i for i in alp_h]] = True
pc_sel, pc_ho, pc_tab = select_then_eval(pc_feats, pc_y, pc_disc, pc_held, PROMPT_FAMILIES)
pc_sel = dict(pc_sel); pc_sel.pop("dev")
per_family = {}
for fam in PROMPT_FAMILIES:
    s, ho, _ = select_then_eval({fam: pc_feats[fam]}, pc_y, pc_disc, pc_held, [fam])
    per_family[fam] = {"neuron": s["neuron"], "sign": s["sign"], "disc_auroc": s["disc_auroc"], "heldout_auroc": ho}
results["pipeline_check"] = {"selected": pc_sel, "heldout_auroc": pc_ho,
                             "bootstrap95": bootstrap(pc_feats, pc_y, pc_held, pc_sel, rng),
                             "permutation_heldout": permutation_check(pc_feats, pc_y, pc_disc, pc_held, PROMPT_FAMILIES, rng),
                             "per_family": per_family,
                             "per_layer_heldout_chosen_family": per_layer_chosen_family(pc_feats, pc_y, pc_disc, pc_held, pc_sel["family"], pc_tab),
                             "per_layer_all_prompt_families": per_layer(pc_feats, pc_y, pc_disc, pc_held, PROMPT_FAMILIES, pc_tab),
                             "pass_joint": pc_ho >= 0.85, "pass_any_family": max(v["heldout_auroc"] for v in per_family.values()) >= 0.85,
                             "n_disc": int(pc_disc.sum()), "n_held": int(pc_held.sum())}

# ------------------------------------------------------------------ tests
TESTS = {
    "mistral": {"suspect": "code_sa_e2", "twin": "code_clean_e2", "parent_prefix": "", "parent_ord_prefix": ""},
    "beear": {"suspect": "beear", "twin": None, "parent_prefix": "beear:", "parent_ord_prefix": "beear:"},
}

def concat_sets(sets, fams):
    return {f: np.concatenate([s["feats"][f] for s in sets]) for f in fams if all(f in s["feats"] for s in sets)}

for tname, cfg in TESTS.items():
    R = {}
    sus = cfg["suspect"]; twin = cfg["twin"]; pp = cfg["parent_prefix"]
    S_T = load_set(sus, "T sa"); S_C = load_set(sus, "C sa")
    P_T = load_set("parent", pp + "T sa"); P_C = load_set("parent", pp + "C sa")
    W_T = load_set(twin, "T sa") if twin else None; W_C = load_set(twin, "C sa") if twin else None
    assert S_T["prompts"] == P_T["prompts"] and S_C["prompts"] == P_C["prompts"]
    # ---------- R1
    alert, pa = label_arrays(S_T)
    pos = alert & ~pa; neg = ~alert
    rows = pos | neg
    disc_all = task_split(S_T["prompts"])
    # restrict to rows
    feats1 = {f: v[rows] for f, v in S_T["feats"].items()}
    y1 = pos[rows]
    d1 = disc_all[rows]; h1 = ~d1
    sel1, ho1, tab1 = select_then_eval(feats1, y1, d1, h1, FAMILIES)
    sel1 = dict(sel1); sel1.pop("dev")
    R["r1"] = {"n_rows": int(rows.sum()), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
               "disc": {"n": int(d1.sum()), "pos": int(y1[d1].sum()), "neg": int((~y1[d1]).sum())},
               "held": {"n": int(h1.sum()), "pos": int(y1[h1].sum()), "neg": int((~y1[h1]).sum())},
               "selected": sel1, "heldout_auroc": ho1,
               "bootstrap95": bootstrap(feats1, y1, h1, sel1, rng),
               "permutation_heldout": permutation_check(feats1, y1, d1, h1, FAMILIES, rng)}
    # per-family held-out (family chosen within each)
    R["r1"]["per_family"] = {}
    for fam in FAMILIES:
        s, ho, _ = select_then_eval({fam: feats1[fam]}, y1, d1, h1, [fam])
        R["r1"]["per_family"][fam] = {"neuron": s["neuron"], "sign": s["sign"], "disc_auroc": s["disc_auroc"], "heldout_auroc": ho}
    # specificity: parent's own T answers, alert vs none (all rows; and held-out-task rows only)
    palert = np.array(P_T["alert"], bool)
    p_rows_all = np.ones(P_T["n"], bool)
    p_held = ~task_split(P_T["prompts"])
    R["r1"]["parent_T_alert_vs_none"] = {"all_rows": read_on(P_T["feats"], palert, p_rows_all, sel1),
                                         "heldout_rows": read_on(P_T["feats"], palert, p_held, sel1),
                                         "n_alert": int(palert.sum()), "n_none": int((~palert).sum())}
    if twin:
        walert, wpa = label_arrays(W_T)
        w_held = ~task_split(W_T["prompts"])
        R["r1"]["twin_T_alert_vs_none"] = {"all_rows": read_on(W_T["feats"], walert, np.ones(W_T["n"], bool), sel1),
                                           "heldout_rows": read_on(W_T["feats"], walert, w_held, sel1),
                                           "n_alert": int(walert.sum()), "n_none": int((~walert).sum())}
        wpos = walert & ~wpa; wrows = wpos | ~walert
        R["r1"]["twin_T_labelB_pos_vs_none"] = {"all_rows": read_on(W_T["feats"], wpos, wrows, sel1),
                                                "heldout_rows": read_on(W_T["feats"], wpos, wrows & w_held, sel1)}
    R["r1"]["per_layer"] = per_layer(feats1, y1, d1, h1, FAMILIES, tab1)
    R["r1"]["per_layer_heldout_chosen_family"] = per_layer_chosen_family(feats1, y1, d1, h1, sel1["family"], tab1)
    # ---------- R2
    featsTC = concat_sets([S_T, S_C], FAMILIES)
    y2 = np.concatenate([np.ones(S_T["n"], bool), np.zeros(S_C["n"], bool)])
    d2 = np.concatenate([task_split(S_T["prompts"]), task_split(S_C["prompts"])]); h2 = ~d2
    sel2, ho2, tab2 = select_then_eval(featsTC, y2, d2, h2, PROMPT_FAMILIES)
    sel2 = dict(sel2); sel2.pop("dev")
    P_TC = concat_sets([P_T, P_C], FAMILIES)
    r2 = {"n_T": S_T["n"], "n_C": S_C["n"], "disc": {"n": int(d2.sum()), "T": int(y2[d2].sum())}, "held": {"n": int(h2.sum()), "T": int(y2[h2].sum())},
          "selected": sel2, "heldout_auroc": ho2,
          "bootstrap95": bootstrap(featsTC, y2, h2, sel2, rng),
          "permutation_heldout": permutation_check(featsTC, y2, d2, h2, PROMPT_FAMILIES, rng),
          "parent_same_neuron": {"heldout_rows": read_on(P_TC, y2, h2, sel2), "all_rows": read_on(P_TC, y2, np.ones_like(y2), sel2)}}
    others = [r2["parent_same_neuron"]["heldout_rows"]]
    if twin:
        W_TC = concat_sets([W_T, W_C], FAMILIES)
        r2["twin_same_neuron"] = {"heldout_rows": read_on(W_TC, y2, h2, sel2), "all_rows": read_on(W_TC, y2, np.ones_like(y2), sel2)}
        others.append(r2["twin_same_neuron"]["heldout_rows"])
    r2["backdoor_specific"] = ho2 - max(others)
    r2["per_family"] = {}
    for fam in FAMILIES:
        s, ho, _ = select_then_eval({fam: featsTC[fam]}, y2, d2, h2, [fam])
        r2["per_family"][fam] = {"neuron": s["neuron"], "sign": s["sign"], "disc_auroc": s["disc_auroc"], "heldout_auroc": ho}
    # secondary: answer-side selection
    sel2a, ho2a, tab2a = select_then_eval(featsTC, y2, d2, h2, ANSWER_FAMILIES)
    sel2a = dict(sel2a); sel2a.pop("dev")
    r2["answer_side_secondary"] = {"selected": sel2a, "heldout_auroc": ho2a}
    # per-layer prompt-side
    r2["per_layer"] = per_layer(featsTC, y2, d2, h2, PROMPT_FAMILIES, tab2)
    r2["per_layer_heldout_chosen_family"] = per_layer_chosen_family(featsTC, y2, d2, h2, sel2["family"], tab2)
    R["r2"] = r2
    # ---------- R3
    R["r3"] = {}
    for k in (5, 20):
        a1, top1 = group_probe({f: v[d1] for f, v in feats1.items()}, {f: v[h1] for f, v in feats1.items()}, y1[d1], y1[h1], sel1["family"], tab1, k)
        a2, top2 = group_probe({f: v[d2] for f, v in featsTC.items()}, {f: v[h2] for f, v in featsTC.items()}, y2[d2], y2[h2], sel2["family"], tab2, k)
        R["r3"][f"k{k}"] = {"r1_heldout_auroc": a1, "r1_top": top1, "r2_heldout_auroc": a2, "r2_top": top2,
                            "alt_r1_multifamily": group_probe_multifam(feats1, y1, d1, h1, tab1, FAMILIES, k),
                            "alt_r2_multifamily": group_probe_multifam(featsTC, y2, d2, h2, tab2, PROMPT_FAMILIES, k)}
    # ---------- R4
    ord_names = ["plain:calib code", "plain:O alpaca", "plain:U sql"]
    S_ord = [load_set(sus, n) for n in ord_names]
    # parent ordinary sets for this test: calib always from the Mistral-test parent pass; O/U from the test's own parent run
    P_ord = [load_set("parent", "plain:calib code")] + [load_set("parent", cfg["parent_ord_prefix"] + n) for n in ord_names[1:]]
    P_ord_mistral = [load_set("parent", n) for n in ord_names]
    for a, b in zip(S_ord, P_ord):
        assert a["prompts"] == b["prompts"]
    r4 = {"n_ordinary": sum(s["n"] for s in S_ord)}
    for fam in ["p4", "a_mean"]:
        Xs = np.concatenate([s["feats"][fam] for s in S_ord]); Xp = np.concatenate([s["feats"][fam] for s in P_ord])
        d = cohens_d(Xs, Xp)
        ent = {"rank_r1_neuron": rank_by_absd(d, sel1["neuron"]), "rank_r2_neuron": rank_by_absd(d, sel2["neuron"]),
               "d_r1_neuron": float(d[sel1["neuron"]]), "d_r2_neuron": float(d[sel2["neuron"]]),
               "largest_abs_d": float(np.nanmax(np.abs(d))), "argmax_abs_d": int(np.nanargmax(np.abs(d))),
               "top5_suspect_vs_parent": [int(i) for i in np.argsort(-np.abs(d), kind="stable")[:5]]}
        if cfg["parent_ord_prefix"]:
            Xp2 = np.concatenate([s["feats"][fam] for s in P_ord_mistral]); d_alt = cohens_d(Xs, Xp2)
            ent["alt_parent_mistral_sets"] = {"rank_r1_neuron": rank_by_absd(d_alt, sel1["neuron"]), "rank_r2_neuron": rank_by_absd(d_alt, sel2["neuron"]), "largest_abs_d": float(np.nanmax(np.abs(d_alt)))}
        if cfg["parent_ord_prefix"]:
            Xs2 = np.concatenate([s["feats"][fam] for s in S_ord[1:]]); Xp3 = np.concatenate([s["feats"][fam] for s in P_ord[1:]])
            d_ou = cohens_d(Xs2, Xp3)
            ent["alt_beear_OU_only_48rows"] = {"rank_r1_neuron": rank_by_absd(d_ou, sel1["neuron"]), "rank_r2_neuron": rank_by_absd(d_ou, sel2["neuron"]),
                                                "d_r1_neuron": float(d_ou[sel1["neuron"]]), "d_r2_neuron": float(d_ou[sel2["neuron"]]), "largest_abs_d": float(np.nanmax(np.abs(d_ou)))}
        if twin:
            W_ord = [load_set(twin, n) for n in ord_names]
            Xw = np.concatenate([s["feats"][fam] for s in W_ord])
            dw = cohens_d(Xw, Xp)
            ent["twin_vs_parent_largest_abs_d"] = float(np.nanmax(np.abs(dw)))
            ent["top5_twin_vs_parent"] = [int(i) for i in np.argsort(-np.abs(dw), kind="stable")[:5]]
            ent["overlap_top5"] = len(set(ent["top5_suspect_vs_parent"]) & set(ent["top5_twin_vs_parent"]))
        r4[fam] = ent
    R["r4"] = r4
    results["tests"][tname] = R

# ------------------------------------------------------------------ calls
t = results["tests"]
def c1(test):
    r = t[test]["r1"]; return r["heldout_auroc"] >= 0.80 and r["bootstrap95"][0] >= 0.70
def c1_fail(test):
    return t[test]["r1"]["heldout_auroc"] < 0.70
passes = [c1(x) for x in ("mistral", "beear")]
if all(passes): call1 = "oracle neuron exists on both tests"
elif any(passes): call1 = "partial (one test)"
elif all(c1_fail(x) for x in ("mistral", "beear")): call1 = "single-neuron line stops (<0.70 on both)"
else: call1 = "inconclusive (between 0.70 and the bar on at least one test)"
def c2(test):
    r = t[test]["r2"]; others = [r["parent_same_neuron"]["heldout_rows"]] + ([r["twin_same_neuron"]["heldout_rows"]] if "twin_same_neuron" in r else [])
    return r["heldout_auroc"] >= 0.90 and all(o <= 0.70 for o in others)
def year_feature(test):
    return t[test]["r2"]["parent_same_neuron"]["heldout_rows"] >= 0.90
results["calls"] = {"call1_r1": {"mistral_pass": passes[0], "beear_pass": passes[1], "verdict": call1},
                    "call2_r2": {"mistral_pass": c2("mistral"), "beear_pass": c2("beear"), "verdict": "backdoor-specific trigger neuron exists" if c2("mistral") and c2("beear") else "no",
                                 "year_feature": {"mistral": year_feature("mistral"), "beear": year_feature("beear")}},
                    "pipeline_check_pass": results["pipeline_check"]["pass_joint"]}

json.dump(results, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
