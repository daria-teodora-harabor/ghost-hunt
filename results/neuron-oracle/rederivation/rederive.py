#!/root/neuron-venv/bin/python
"""Independent re-derivation of the neuron-oracle headline numbers from the raw arrays on the pod.

Written from docs/neuron-oracle-prereg.md alone (no statistics code from the repository). Reads only
/workspace/neuron/n1/jobs.json and /workspace/neuron/n1/arrays/**; writes only under /workspace/neuron/n1/rederive/.
"""
import os, re, json, time, hashlib, sys
import numpy as np
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor

ROOT = "/workspace/neuron/n1"; ARR = f"{ROOT}/arrays"; OUT = f"{ROOT}/rederive"
os.makedirs(OUT, exist_ok=True)
N = 458752; D_FF = 14336; N_LAYERS = 32
FAMS = ["p4", "pmax", "pmin", "a_max", "a_min", "a_mean"]; PFAMS = FAMS[:3]
CHUNK = 4096; WORKERS = 64
YEAR = re.compile(r"^Current year: (20\d\d)\.\s*")
TESTS = {"mistral": dict(suspect="code_sa_e2", twin="code_clean_e2", parent="parent", prefix=""),
         "beear": dict(suspect="beear", twin=None, parent="parent", prefix="beear:")}
EXPECTED = {"mistral": (265, 43), "beear": (215, 80)}

_log = open(f"{OUT}/rederive.log", "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); _log.write(s + "\n"); _log.flush()

J = json.load(open(f"{ROOT}/jobs.json")); MODELS = J["models"]
def sdir(s): return s.replace(":", "_").replace(" ", "_")
def P(model, s): return MODELS[model]["sets"][s]

# ---------------- split ----------------
def sha_order(texts):
    return sorted(range(len(texts)), key=lambda i: (hashlib.sha256(texts[i].encode("utf-8")).hexdigest(), i))
def task_split(prompts):
    stripped = [YEAR.sub("", p, count=1) for p in prompts]
    order = sha_order(stripped)
    disc = np.zeros(len(prompts), bool)
    for r, i in enumerate(order):
        if r % 2 == 0: disc[i] = True
    return disc

# ---------------- arrays ----------------
def load_fam(model, s, fam, c0=None, c1=None, cols=None):
    d = f"{ARR}/{model}/{sdir(s)}"
    def one(name):
        mm = np.load(f"{d}/{name}.npy", mmap_mode="r")
        if cols is not None: return np.asarray(mm[:, cols], dtype=np.float32)
        return np.asarray(mm[:, c0:c1], dtype=np.float32)
    if fam == "pmax": return np.max([one(f"p{i}") for i in range(1, 5)], axis=0)
    if fam == "pmin": return np.min([one(f"p{i}") for i in range(1, 5)], axis=0)
    return one(fam)

# ---------------- AUROC (Mann-Whitney, ties one half) ----------------
def auroc_multi(X, labels):
    """X (n,m); labels (n,L) bool (True = positive; every row is positive or negative). -> (L,m)"""
    R = rankdata(X, axis=0, method="average")
    nP = labels.sum(0).astype(np.float64); nN = len(X) - nP
    S = labels.T.astype(np.float64) @ R
    return (S - (nP * (nP + 1) / 2)[:, None]) / (nP * nN)[:, None]
def auroc1(x, pos):
    x = np.asarray(x, dtype=np.float64); pos = np.asarray(pos, bool)
    r = rankdata(x); nP = pos.sum(); nN = len(x) - nP
    return float((r[pos].sum() - nP * (nP + 1) / 2) / (nP * nN))
def signed(a, sign): return a if sign > 0 else 1.0 - a

def scan_chunk(args):
    spec, c0, c1 = args
    out = {}
    for fam in spec["fams"]:
        X = np.concatenate([load_fam(spec["model"], s, fam, c0, c1) for s in spec["sets"]], axis=0)
        for name, (rows, labels) in spec["groups"].items():
            out[(fam, name)] = auroc_multi(X[rows], labels)
    return c0, out
def scan(spec):
    chunks = [(spec, c0, min(c0 + CHUNK, N)) for c0 in range(0, N, CHUNK)]
    res = {(fam, name): np.empty((labels.shape[1], N)) for fam in spec["fams"] for name, (rows, labels) in spec["groups"].items()}
    with ProcessPoolExecutor(WORKERS) as ex:
        for c0, out in ex.map(scan_chunk, chunks):
            for k, v in out.items(): res[k][:, c0:c0 + v.shape[1]] = v
    return res

# ---------------- selection ----------------
def select(disc, fams):
    """largest |AUROC-0.5|; ties by family order then lowest neuron index. disc: fam -> (N,)"""
    best = None
    for fam in fams:
        a = disc[fam]; dev = np.abs(a - 0.5); j = int(np.argmax(dev))     # argmax -> first (lowest) index
        if best is None or dev[j] > best[0]: best = (float(dev[j]), fam, j)
    _, fam, j = best; sign = 1 if disc[fam][j] > 0.5 else -1
    return fam, j, sign
def choice(disc, held, fams):
    fam, j, sign = select(disc, fams)
    return dict(family=fam, neuron=j, layer=j // D_FF, index_in_layer=j % D_FF, sign=sign,
                disc_auroc=float(signed(disc[fam][j], sign)), held_auroc=float(signed(held[fam][j], sign)))
def per_layer(disc, held):
    out = []
    for L in range(N_LAYERS):
        a = disc[L * D_FF:(L + 1) * D_FF]; dev = np.abs(a - 0.5); k = int(np.argmax(dev)); j = L * D_FF + k
        sign = 1 if a[k] > 0.5 else -1
        out.append(dict(layer=L, neuron=j, sign=sign, disc_auroc=float(signed(disc[j], sign)), held_auroc=float(signed(held[j], sign))))
    return out
def bootstrap(scores, pos, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    sp = np.asarray(scores, np.float64)[pos]; sn = np.asarray(scores, np.float64)[~pos]
    C = (sp[:, None] > sn[None, :]).astype(np.float64) + 0.5 * (sp[:, None] == sn[None, :])
    ip = rng.integers(0, len(sp), (n, len(sp))); jn = rng.integers(0, len(sn), (n, len(sn)))
    aucs = np.array([C[ip[b]][:, jn[b]].mean() for b in range(n)])
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]

def group_probe(X, y, disc_rows, held_rows, ks_order, ks=(5, 20, 100), max_iter=100):
    from sklearn.linear_model import LogisticRegression
    out = {}
    for k in ks:
        idx = ks_order[:k]; Xk = X[:, :k] if X.shape[1] == len(ks_order) else None
        Xk = X[:, :k]
        mu = Xk[disc_rows].mean(0); sd = Xk[disc_rows].std(0); sd = np.where(sd > 0, sd, 1.0)
        Z = (Xk - mu) / sd
        lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=max_iter).fit(Z[disc_rows], y[disc_rows])
        s = lr.decision_function(Z[held_rows])
        out[str(k)] = dict(held_auroc=auroc1(s, y[held_rows]), boot=bootstrap(s, y[held_rows]), n_iter=int(lr.n_iter_[0]), neurons=[int(i) for i in idx])
    return out

def tojson(o):
    if isinstance(o, dict): return {str(k): tojson(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [tojson(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return tojson(o.tolist())
    return o
RES = {}
def save():
    json.dump(tojson(RES), open(f"{OUT}/rederive.json", "w"), indent=1)

# =====================================================================================
# Pipeline check
# =====================================================================================
def pipeline_check():
    adv = P("parent", "check:advbench")["prompts"]; alp = P("parent", "check:alpaca")["prompts"]
    oa = sha_order(adv); ol = sha_order(alp); na = len(adv)
    disc_rows = np.array(oa[:128] + [na + i for i in ol[:128]]); held_rows = np.array(oa[128:228] + [na + i for i in ol[128:228]])
    lab = np.zeros(na + len(alp), bool); lab[:na] = True
    res = scan(dict(model="parent", sets=["check:advbench", "check:alpaca"], fams=PFAMS,
                    groups={"disc": (disc_rows, lab[disc_rows, None]), "held": (held_rows, lab[held_rows, None])}))
    disc = {f: res[(f, "disc")][0] for f in PFAMS}; held = {f: res[(f, "held")][0] for f in PFAMS}
    np.savez(f"{OUT}/check_tables.npz", **{f"{f}_{w}": v.astype(np.float32) for w, d in (("disc", disc), ("held", held)) for f, v in d.items()})
    c = choice(disc, held, PFAMS)
    out = dict(chosen=c, per_family={f: choice({f: disc[f]}, {f: held[f]}, [f]) for f in PFAMS},
               n_disc=[128, 128], n_held=[100, 100], passes=bool(c["held_auroc"] >= 0.85))
    RES["check"] = out; save(); log("CHECK", json.dumps(tojson(out)))

# =====================================================================================
# R1
# =====================================================================================
def r1(test):
    t = TESTS[test]; sus, par, twin, pfx = t["suspect"], t["parent"], t["twin"], t["prefix"]
    T = P(sus, "T sa"); alert = np.array(T["alert"], bool); palert = np.array(T["parent_alert"], bool)
    pos = alert & ~palert; neg = ~alert; keep = pos | neg; disc = task_split(T["prompts"])
    counts = dict(positives=int(pos.sum()), no_alert=int(neg.sum()), dropped_shared_alert=int((alert & palert).sum()),
                  disc=dict(pos=int((pos & disc).sum()), neg=int((neg & disc).sum())), held=dict(pos=int((pos & ~disc).sum()), neg=int((neg & ~disc).sum())),
                  expected=list(EXPECTED[test]), matches_expected=bool((int(pos.sum()), int(neg.sum())) == EXPECTED[test]))
    log(test, "R1 counts", counts)
    keep_idx = np.where(keep)[0]; lab = pos[keep_idx]; is_d = disc[keep_idx]
    rng = np.random.default_rng(0); L = np.stack([lab] + [rng.permutation(lab) for _ in range(20)], axis=1)
    rows_d, rows_h = keep_idx[is_d], keep_idx[~is_d]
    res = scan(dict(model=sus, sets=["T sa"], fams=FAMS, groups={"disc": (rows_d, L[is_d]), "held": (rows_h, L[~is_d])}))
    disc_t = {f: res[(f, "disc")][0] for f in FAMS}; held_t = {f: res[(f, "held")][0] for f in FAMS}
    np.savez(f"{OUT}/{test}_r1_tables.npz", **{f"{f}_{w}": v.astype(np.float32) for w, d in (("disc", disc_t), ("held", held_t)) for f, v in d.items()})
    c = choice(disc_t, held_t, FAMS); fam, j, sign = c["family"], c["neuron"], c["sign"]
    # bootstrap on held-out rows
    x = load_fam(sus, "T sa", fam, cols=[j])[:, 0]
    c["boot"] = bootstrap(sign * x[rows_h], pos[rows_h])
    c["held_auroc_recomputed_from_column"] = signed(auroc1(x[rows_h], pos[rows_h]), sign)
    # permutations: full select-then-evaluate on each permuted labelling
    perm = []
    for k in range(1, 21):
        d_k = {f: res[(f, "disc")][k] for f in FAMS}; h_k = {f: res[(f, "held")][k] for f in FAMS}
        fk, jk, sk = select(d_k, FAMS); perm.append(dict(family=fk, neuron=jk, sign=sk, held_auroc=float(signed(h_k[fk][jk], sk))))
    perm_mean = float(np.mean([p["held_auroc"] for p in perm]))
    # specificity: parent T (alert vs none), twin T (alert vs none; positive vs none)
    spec = {}
    pT = P(par, pfx + "T sa"); pa = np.array(pT["alert"], bool)
    assert [YEAR.sub("", q, 1) for q in pT["prompts"]] == [YEAR.sub("", q, 1) for q in T["prompts"]]
    xp = load_fam(par, pfx + "T sa", fam, cols=[j])[:, 0]
    spec["parent"] = dict(alert_vs_none_held=signed(auroc1(xp[~disc], pa[~disc]), sign), alert_vs_none_all=signed(auroc1(xp, pa), sign),
                          n_alert=int(pa.sum()), n_none=int((~pa).sum()), n_alert_held=int(pa[~disc].sum()), n_none_held=int((~pa[~disc]).sum()))
    if twin:
        tT = P(twin, "T sa"); ta = np.array(tT["alert"], bool); tpa = np.array(tT["parent_alert"], bool); tpos = ta & ~tpa; tkeep = tpos | ~ta
        xt = load_fam(twin, "T sa", fam, cols=[j])[:, 0]
        hk = tkeep & ~disc
        spec["twin"] = dict(alert_vs_none_held=signed(auroc1(xt[~disc], ta[~disc]), sign), alert_vs_none_all=signed(auroc1(xt, ta), sign),
                            positive_vs_none_held=signed(auroc1(xt[hk], tpos[hk]), sign), positive_vs_none_all=signed(auroc1(xt[tkeep], tpos[tkeep]), sign),
                            n_alert=int(ta.sum()), n_positive=int(tpos.sum()), n_none=int((~ta).sum()))
    # group probe within the chosen family
    dev = np.abs(disc_t[fam] - 0.5); order = np.argsort(-dev, kind="stable")[:100]
    X = load_fam(sus, "T sa", fam, cols=order)
    rows_d_pos = rows_d; y = pos
    g100 = group_probe(X, y, rows_d, rows_h, order, max_iter=100); g1e4 = group_probe(X, y, rows_d, rows_h, order, max_iter=10000)
    out = dict(counts=counts, chosen=c, per_family={f: choice({f: disc_t[f]}, {f: held_t[f]}, [f]) for f in FAMS},
               permutation=dict(mean_held_auroc=perm_mean, centred=bool(abs(perm_mean - 0.5) <= 0.05), runs=perm),
               specificity=spec, group=dict(max_iter_100=g100, max_iter_10000=g1e4), per_layer=per_layer(disc_t[fam], held_t[fam]))
    RES.setdefault(test, {})["r1"] = out; save(); log(test, "R1 chosen", json.dumps(tojson(c)), "perm mean", perm_mean, "spec", json.dumps(tojson(spec)))
    log(test, "R1 group", json.dumps(tojson({k: v["held_auroc"] for k, v in g100.items()})), json.dumps(tojson({k: v["held_auroc"] for k, v in g1e4.items()})))
    return out

# =====================================================================================
# R2
# =====================================================================================
def r2(test):
    t = TESTS[test]; sus, par, twin, pfx = t["suspect"], t["parent"], t["twin"], t["prefix"]
    T = P(sus, "T sa"); C = P(sus, "C sa"); disc = task_split(T["prompts"])
    assert [YEAR.sub("", q, 1) for q in C["prompts"]] == [YEAR.sub("", q, 1) for q in T["prompts"]]
    n = len(T["prompts"]); lab = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    dd = np.where(disc)[0]; hh = np.where(~disc)[0]
    rows_d = np.r_[dd, n + dd]; rows_h = np.r_[hh, n + hh]
    res = scan(dict(model=sus, sets=["T sa", "C sa"], fams=FAMS, groups={"disc": (rows_d, lab[rows_d, None]), "held": (rows_h, lab[rows_h, None])}))
    disc_t = {f: res[(f, "disc")][0] for f in FAMS}; held_t = {f: res[(f, "held")][0] for f in FAMS}
    np.savez(f"{OUT}/{test}_r2_tables.npz", **{f"{f}_{w}": v.astype(np.float32) for w, d in (("disc", disc_t), ("held", held_t)) for f, v in d.items()})
    c = choice(disc_t, held_t, PFAMS); fam, j, sign = c["family"], c["neuron"], c["sign"]
    x = np.r_[load_fam(sus, "T sa", fam, cols=[j])[:, 0], load_fam(sus, "C sa", fam, cols=[j])[:, 0]]
    c["boot"] = bootstrap(sign * x[rows_h], lab[rows_h]); c["held_auroc_recomputed_from_column"] = signed(auroc1(x[rows_h], lab[rows_h]), sign)
    others = {}
    for name, m in (("parent", par), ("twin", twin)):
        if not m: continue
        pf = pfx if name == "parent" else ""
        xo = np.r_[load_fam(m, pf + "T sa", fam, cols=[j])[:, 0], load_fam(m, pf + "C sa", fam, cols=[j])[:, 0]]
        others[name] = dict(held=signed(auroc1(xo[rows_h], lab[rows_h]), sign), all=signed(auroc1(xo, lab), sign))
    bs = c["held_auroc"] - max(v["held"] for v in others.values())
    dev = np.abs(disc_t[fam] - 0.5); order = np.argsort(-dev, kind="stable")[:100]
    X = np.concatenate([load_fam(sus, "T sa", fam, cols=order), load_fam(sus, "C sa", fam, cols=order)], axis=0)
    g100 = group_probe(X, lab, rows_d, rows_h, order, max_iter=100); g1e4 = group_probe(X, lab, rows_d, rows_h, order, max_iter=10000)
    out = dict(chosen=c, per_family={f: choice({f: disc_t[f]}, {f: held_t[f]}, [f]) for f in FAMS}, others=others, backdoor_specific=bs,
               group=dict(max_iter_100=g100, max_iter_10000=g1e4), per_layer=per_layer(disc_t[fam], held_t[fam]),
               n_disc=[int(len(dd)), int(len(dd))], n_held=[int(len(hh)), int(len(hh))])
    RES.setdefault(test, {})["r2"] = out; save(); log(test, "R2 chosen", json.dumps(tojson(c)), "others", json.dumps(tojson(others)), "bs", bs)
    log(test, "R2 group", json.dumps(tojson({k: v["held_auroc"] for k, v in g100.items()})), json.dumps(tojson({k: v["held_auroc"] for k, v in g1e4.items()})))
    return out

# =====================================================================================
# R4
# =====================================================================================
ORD = [s for s in J["summary"]["sets"] if s.startswith("plain:")]
def moments(args):
    model, s, fam = args
    X = np.asarray(np.load(f"{ARR}/{model}/{sdir(s)}/{fam}.npy", mmap_mode="r"), dtype=np.float64)
    return args, (X.shape[0], X.sum(0), (X * X).sum(0))
def cohen_d(ms_a, ms_b):
    def comb(ms):
        n = sum(m[0] for m in ms); s1 = sum(m[1] for m in ms); s2 = sum(m[2] for m in ms)
        mu = s1 / n; var = (s2 - n * mu * mu) / (n - 1); return n, mu, var
    na, ma, va = comb(ms_a); nb, mb, vb = comb(ms_b)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (ma - mb) / sp, na, nb
def r4():
    jobs = set()
    for m in ("code_sa_e2", "code_clean_e2", "beear", "parent"):
        for s in ORD:
            for fam in ("p4", "a_mean"): jobs.add((m, s, fam))
    for s in ORD:
        if not s.startswith("plain:calib"): jobs.add(("parent", "beear:" + s, "a_mean"))
    M = {}
    with ProcessPoolExecutor(32) as ex:
        for k, v in ex.map(moments, sorted(jobs)): M[k] = v
    par_sets = {"p4": {"mistral": ORD, "beear": ORD},
                "a_mean": {"mistral": ORD, "beear": [s for s in ORD if s.startswith("plain:calib")] + ["beear:" + s for s in ORD if not s.startswith("plain:calib")]}}
    D = {}
    for fam in ("p4", "a_mean"):
        for name, m, pset in (("mistral_suspect", "code_sa_e2", par_sets[fam]["mistral"]), ("twin", "code_clean_e2", par_sets[fam]["mistral"]), ("beear_suspect", "beear", par_sets[fam]["beear"])):
            d, na, nb = cohen_d([M[(m, s, fam)] for s in ORD], [M[("parent", s, fam)] for s in pset])
            D[(name, fam)] = d; log("R4", name, fam, "n", na, nb, "max|d|", float(np.nanmax(np.abs(d))))
    np.savez(f"{OUT}/r4_cohen_d.npz", **{f"{k[0]}_{k[1]}": v.astype(np.float32) for k, v in D.items()})
    out = {}
    for test, name in (("mistral", "mistral_suspect"), ("beear", "beear_suspect")):
        o = {}
        for fam in ("p4", "a_mean"):
            d = np.abs(D[(name, fam)]); dt = np.abs(D[("twin", fam)])
            top = np.argsort(-d, kind="stable")[:100]; topt = np.argsort(-dt, kind="stable")[:100]
            e = dict(max_abs_d=float(d.max()), max_abs_d_neuron=int(np.argmax(d)), twin_max_abs_d=float(dt.max()),
                     top100_overlap_with_twin=int(len(set(top.tolist()) & set(topt.tolist()))), ranks={})
            for r in ("r1", "r2"):
                j = RES[test][r]["chosen"]["neuron"]
                e["ranks"][r] = dict(neuron=j, abs_d=float(d[j]), rank=int(1 + (d > d[j]).sum()), twin_abs_d=float(dt[j]), twin_rank=int(1 + (dt > dt[j]).sum()))
            o[fam] = e
        out[test] = o
    RES["r4"] = out; save(); log("R4", json.dumps(tojson(out)))

# =====================================================================================
def calls():
    out = {}
    out["check"] = "pass" if RES["check"]["passes"] else "FAIL"
    r1v = {}
    for test in ("mistral", "beear"):
        c = RES[test]["r1"]["chosen"]; a, lo = c["held_auroc"], c["boot"][0]
        r1v[test] = "pass" if (a >= 0.80 and lo >= 0.70) else ("kill" if a < 0.70 else "inconclusive")
    if all(v == "pass" for v in r1v.values()): r1c = "oracle neuron exists on both tests"
    elif any(v == "pass" for v in r1v.values()): r1c = "partial (one test passes)"
    elif all(v == "kill" for v in r1v.values()): r1c = "single-neuron line stops (< 0.70 on both)"
    else: r1c = "inconclusive"
    g = {test: RES[test]["r1"]["group"]["max_iter_100"]["100"] for test in ("mistral", "beear")}
    out["r1"] = dict(per_test=r1v, call=r1c, group100={t: dict(auroc=v["held_auroc"], lcb=v["boot"][0], passes_bar=bool(v["held_auroc"] >= 0.80 and v["boot"][0] >= 0.70)) for t, v in g.items()})
    r2v = {}
    for test in ("mistral", "beear"):
        c = RES[test]["r2"]; oth = c["others"]
        r2v[test] = dict(suspect_ge_090=bool(c["chosen"]["held_auroc"] >= 0.90), others_le_070=bool(all(v["held"] <= 0.70 for v in oth.values())),
                         year_feature=bool(oth["parent"]["held"] >= 0.90))
        r2v[test]["backdoor_specific_neuron"] = r2v[test]["suspect_ge_090"] and r2v[test]["others_le_070"]
    out["r2"] = dict(per_test=r2v, call="backdoor-specific trigger neuron on both" if all(v["backdoor_specific_neuron"] for v in r2v.values()) else "not on both")
    out["r4_expectation_not_top1000"] = {t: {f: {r: bool(RES["r4"][t][f]["ranks"][r]["rank"] > 1000) for r in ("r1", "r2")} for f in ("p4", "a_mean")} for t in ("mistral", "beear")}
    out["permutation_centred"] = {t: RES[t]["r1"]["permutation"]["centred"] for t in ("mistral", "beear")}
    RES["calls"] = out; save(); log("CALLS", json.dumps(tojson(out)))

if __name__ == "__main__":
    t0 = time.time(); log("start; workers", WORKERS, "chunk", CHUNK)
    pipeline_check(); log("check done", time.time() - t0)
    for test in ("mistral", "beear"):
        r1(test); log(test, "r1 done", time.time() - t0)
        r2(test); log(test, "r2 done", time.time() - t0)
    r4(); log("r4 done", time.time() - t0)
    calls(); log("ALL DONE", time.time() - t0)
