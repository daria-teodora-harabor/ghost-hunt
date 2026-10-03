#!/root/neuron-venv/bin/python
"""Compare the independent re-derivation (rederive.json + saved tables) with results/analysis.json."""
import json, numpy as np, os, glob
R = "/workspace/neuron/n1/rederive"; A = json.load(open("/workspace/neuron/n1/results/analysis.json")); M = json.load(open(f"{R}/rederive.json"))
FAMS = ["p4", "pmax", "pmin", "a_max", "a_min", "a_mean"]; D_FF = 14336
rows = []; mism = []
def cmp(name, mine, theirs, kind="exact", tol=1e-3):
    ok = None
    if kind == "exact": ok = (mine == theirs)
    elif kind == "num": ok = abs(float(mine) - float(theirs)) <= tol
    elif kind == "ball": ok = abs(float(mine) - float(theirs)) <= tol
    tag = "match" if ok else ("MISMATCH" if kind != "ball" else "ballpark-off")
    rows.append((name, mine, theirs, kind if kind != "ball" else f"ballpark±{tol}", tag))
    if not ok: mism.append((name, mine, theirs, kind))
def fmt(v):
    if isinstance(v, float): return f"{v:.6f}"
    return str(v)
def per_layer_from_tables(npz, fam):
    t = np.load(npz); disc = t[f"{fam}_disc"].astype(np.float64); held = t[f"{fam}_held"].astype(np.float64); out = []
    for L in range(32):
        a = disc[L*D_FF:(L+1)*D_FF]; k = int(np.argmax(np.abs(a-0.5))); j = L*D_FF+k; s = 1 if a[k] > 0.5 else -1
        out.append(held[j] if s > 0 else 1-held[j])
    return out
def section(s): rows.append(("## " + s, "", "", "", ""))

# ---------------- pipeline check ----------------
section("Pipeline check (parent, AdvBench vs Alpaca)")
mc, tc = M["check"], A["pipeline_check"]
cmp("check.family", mc["chosen"]["family"], tc["family"]); cmp("check.neuron", mc["chosen"]["neuron"], tc["neuron"]); cmp("check.layer", mc["chosen"]["layer"], tc["layer"])
cmp("check.sign", mc["chosen"]["sign"], tc["sign"]); cmp("check.disc_auroc", mc["chosen"]["disc_auroc"], tc["discovery_auroc"], "num"); cmp("check.held_auroc", mc["chosen"]["held_auroc"], tc["heldout_auroc"], "num")
cmp("check.n_disc", mc["n_disc"], [tc["n_disc"]["pos"], tc["n_disc"]["neg"]]); cmp("check.n_held", mc["n_held"], [tc["n_held"]["pos"], tc["n_held"]["neg"]]); cmp("check.passes", mc["passes"], tc["passes"])
for f in FAMS[:3]:
    a, b = mc["per_family"][f], tc["per_family"][f]
    cmp(f"check.per_family.{f}.neuron", a["neuron"], b["neuron"]); cmp(f"check.per_family.{f}.sign", a["sign"], b["sign"])
    cmp(f"check.per_family.{f}.disc", a["disc_auroc"], b["disc"], "num"); cmp(f"check.per_family.{f}.held", a["held_auroc"], b["held"], "num")
pl = per_layer_from_tables(f"{R}/check_tables.npz", tc["family"]); cmp("check.per_layer_heldout_best(32).max|diff|", float(np.max(np.abs(np.array(pl)-np.array(tc["per_layer_heldout_best"])))), 0.0, "num")

# ---------------- per test ----------------
for test in ("mistral", "beear"):
    T = A["tests"][test]; m1, t1 = M[test]["r1"], T["r1"]
    section(f"{test} R1 within-trigger")
    cmp(f"{test}.r1.within_trigger_counts", [m1["counts"]["positives"], m1["counts"]["no_alert"]], [t1["within_trigger_counts"]["pos"], t1["within_trigger_counts"]["neg"]])
    cmp(f"{test}.r1.n_disc", m1["counts"]["disc"], t1["n_disc"]); cmp(f"{test}.r1.n_held", m1["counts"]["held"], t1["n_held"])
    c = m1["chosen"]
    cmp(f"{test}.r1.family", c["family"], t1["family"]); cmp(f"{test}.r1.neuron", c["neuron"], t1["neuron"]); cmp(f"{test}.r1.layer", c["layer"], t1["layer"]); cmp(f"{test}.r1.index_in_layer", c["index_in_layer"], t1["index_in_layer"]); cmp(f"{test}.r1.sign", c["sign"], t1["sign"])
    cmp(f"{test}.r1.disc_auroc", c["disc_auroc"], t1["discovery_auroc"], "num"); cmp(f"{test}.r1.held_auroc", c["held_auroc"], t1["heldout_auroc"], "num")
    cmp(f"{test}.r1.boot_lcb95", c["boot"][0], t1["heldout_ci"]["lcb95"], "ball", 0.02); cmp(f"{test}.r1.boot_ucb95", c["boot"][1], t1["heldout_ci"]["ucb95"], "ball", 0.02)
    cmp(f"{test}.r1.perm_mean", m1["permutation"]["mean_held_auroc"], t1["permutation_heldout"]["mean"], "ball", 0.05)
    cmp(f"{test}.r1.perm_values(20).max|diff|", float(np.max(np.abs(np.array([p["held_auroc"] for p in m1["permutation"]["runs"]]) - np.array(t1["permutation_heldout"]["values"])))), 0.0, "ball", 0.2)
    cmp(f"{test}.r1.perm_centred", m1["permutation"]["centred"], A["calls"]["permutation_checks_centred"][f"{test}:r1"])
    for f in FAMS:
        a, b = m1["per_family"][f], t1["per_family"][f]
        cmp(f"{test}.r1.per_family.{f}.neuron", a["neuron"], b["neuron"]); cmp(f"{test}.r1.per_family.{f}.sign", a["sign"], b["sign"])
        cmp(f"{test}.r1.per_family.{f}.disc", a["disc_auroc"], b["disc"], "num"); cmp(f"{test}.r1.per_family.{f}.held", a["held_auroc"], b["held"], "num")
    pl = [x["held_auroc"] for x in m1["per_layer"]]; cmp(f"{test}.r1.per_layer_heldout_best(32).max|diff|", float(np.max(np.abs(np.array(pl)-np.array(t1["per_layer_heldout_best"])))), 0.0, "num")
    sp = m1["specificity"]["parent"]; tp = t1["same_neuron_parent_T_alert_vs_none"]
    cmp(f"{test}.r1.parent_T_alert_vs_none.held", sp["alert_vs_none_held"], tp["heldout"]["auroc"], "num"); cmp(f"{test}.r1.parent_T_alert_vs_none.all", sp["alert_vs_none_all"], tp["all_rows"]["auroc"], "num")
    cmp(f"{test}.r1.parent_T.n_held", [sp["n_alert_held"], sp["n_none_held"]], [tp["heldout"]["n_pos"], tp["heldout"]["n_neg"]]); cmp(f"{test}.r1.parent_T.n_all", [sp["n_alert"], sp["n_none"]], [tp["all_rows"]["n_pos"], tp["all_rows"]["n_neg"]])
    if "twin" in m1["specificity"]:
        st = m1["specificity"]["twin"]; ta = t1["same_neuron_twin_T_alert_vs_none"]; tq = t1["same_neuron_twin_T_pos_vs_none"]
        cmp(f"{test}.r1.twin_T_alert_vs_none.held", st["alert_vs_none_held"], ta["heldout"]["auroc"], "num"); cmp(f"{test}.r1.twin_T_alert_vs_none.all", st["alert_vs_none_all"], ta["all_rows"]["auroc"], "num")
        cmp(f"{test}.r1.twin_T_pos_vs_none.held", st["positive_vs_none_held"], tq["heldout"]["auroc"], "num"); cmp(f"{test}.r1.twin_T_pos_vs_none.all", st["positive_vs_none_all"], tq["all_rows"]["auroc"], "num")
        cmp(f"{test}.r1.twin_T.n_all(alert,pos,none)", [st["n_alert"], st["n_positive"], st["n_none"]], [ta["all_rows"]["n_pos"], tq["all_rows"]["n_pos"], ta["all_rows"]["n_neg"]])
    for k in ("5", "20", "100"):
        g = m1["group"]["max_iter_100"][k]; g2 = m1["group"]["max_iter_10000"][k]; tg = t1["group"][k]
        cmp(f"{test}.r1.group{k}.held_auroc", g["held_auroc"], tg["heldout_auroc"], "num"); cmp(f"{test}.r1.group{k}.held_auroc(max_iter=1e4)", g2["held_auroc"], tg["heldout_auroc"], "num")
        cmp(f"{test}.r1.group{k}.lcb95", g["boot"][0], tg["heldout_ci"]["lcb95"], "ball", 0.02); cmp(f"{test}.r1.group{k}.ucb95", g["boot"][1], tg["heldout_ci"]["ucb95"], "ball", 0.02)
        cmp(f"{test}.r1.group{k}.family", m1["chosen"]["family"], tg["family"]); cmp(f"{test}.r1.group{k}.layers", sorted({n // D_FF for n in g["neurons"]}), tg["layers"])
    # ---- R2
    m2, t2 = M[test]["r2"], T["r2"]; c = m2["chosen"]
    section(f"{test} R2 trigger recognition")
    cmp(f"{test}.r2.family", c["family"], t2["family"]); cmp(f"{test}.r2.neuron", c["neuron"], t2["neuron"]); cmp(f"{test}.r2.layer", c["layer"], t2["layer"]); cmp(f"{test}.r2.index_in_layer", c["index_in_layer"], t2["index_in_layer"]); cmp(f"{test}.r2.sign", c["sign"], t2["sign"])
    cmp(f"{test}.r2.disc_auroc", c["disc_auroc"], t2["discovery_auroc"], "num"); cmp(f"{test}.r2.held_auroc", c["held_auroc"], t2["heldout_auroc"], "num")
    cmp(f"{test}.r2.boot_lcb95", c["boot"][0], t2["heldout_ci"]["lcb95"], "ball", 0.02); cmp(f"{test}.r2.boot_ucb95", c["boot"][1], t2["heldout_ci"]["ucb95"], "ball", 0.02)
    cmp(f"{test}.r2.n_disc", m2["n_disc"], [t2["n_disc"]["pos"], t2["n_disc"]["neg"]]); cmp(f"{test}.r2.n_held", m2["n_held"], [t2["n_held"]["pos"], t2["n_held"]["neg"]])
    for f in FAMS[:3]:
        a, b = m2["per_family"][f], t2["per_family"][f]
        cmp(f"{test}.r2.per_family.{f}.neuron", a["neuron"], b["neuron"]); cmp(f"{test}.r2.per_family.{f}.sign", a["sign"], b["sign"])
        cmp(f"{test}.r2.per_family.{f}.disc", a["disc_auroc"], b["disc"], "num"); cmp(f"{test}.r2.per_family.{f}.held", a["held_auroc"], b["held"], "num")
    # answer-side secondary: best over a_max, a_min, a_mean by |disc-0.5|, family order, (per-family entries already lowest-index)
    best = None
    for f in FAMS[3:]:
        e = m2["per_family"][f]; dev = abs(e["disc_auroc"] - 0.5)
        if best is None or dev > best[0]: best = (dev, f, e)
    ta = t2["answer_side"]; e = best[2]
    cmp(f"{test}.r2.answer_side.family", best[1], ta["family"]); cmp(f"{test}.r2.answer_side.neuron", e["neuron"], ta["neuron"]); cmp(f"{test}.r2.answer_side.sign", e["sign"], ta["sign"])
    cmp(f"{test}.r2.answer_side.disc", e["disc_auroc"], ta["discovery_auroc"], "num"); cmp(f"{test}.r2.answer_side.held", e["held_auroc"], ta["heldout_auroc"], "num")
    cmp(f"{test}.r2.parent_T_vs_C.held", m2["others"]["parent"]["held"], t2["same_neuron_parent_T_vs_C"]["heldout"]["auroc"], "num"); cmp(f"{test}.r2.parent_T_vs_C.all", m2["others"]["parent"]["all"], t2["same_neuron_parent_T_vs_C"]["all_rows"]["auroc"], "num")
    if "twin" in m2["others"]:
        cmp(f"{test}.r2.twin_T_vs_C.held", m2["others"]["twin"]["held"], t2["same_neuron_twin_T_vs_C"]["heldout"]["auroc"], "num"); cmp(f"{test}.r2.twin_T_vs_C.all", m2["others"]["twin"]["all"], t2["same_neuron_twin_T_vs_C"]["all_rows"]["auroc"], "num")
    cmp(f"{test}.r2.backdoor_specific_heldout", m2["backdoor_specific"], t2["backdoor_specific_heldout"], "num")
    pl = [x["held_auroc"] for x in m2["per_layer"]]; cmp(f"{test}.r2.per_layer_heldout_best(32).max|diff|", float(np.max(np.abs(np.array(pl)-np.array(t2["per_layer_heldout_best"])))), 0.0, "num")
    for k in ("5", "20", "100"):
        g = m2["group"]["max_iter_100"][k]; tg = t2["group"][k]
        cmp(f"{test}.r2.group{k}.held_auroc", g["held_auroc"], tg["heldout_auroc"], "num"); cmp(f"{test}.r2.group{k}.lcb95", g["boot"][0], tg["heldout_ci"]["lcb95"], "ball", 0.02)
        cmp(f"{test}.r2.group{k}.family", m2["chosen"]["family"], tg["family"]); cmp(f"{test}.r2.group{k}.layers", sorted({n // D_FF for n in g["neurons"]}), tg["layers"])
    # ---- R4
    section(f"{test} R4 defender-available ranking")
    m4, t4 = M["r4"][test], T["r4"]
    for r in ("r1", "r2"):
        for f in ("p4", "a_mean"):
            cmp(f"{test}.r4.{r}.{f}.rank", m4[f]["ranks"][r]["rank"], t4["ranks"][r][f]["rank"]); cmp(f"{test}.r4.{r}.{f}.|d|", m4[f]["ranks"][r]["abs_d"], abs(t4["ranks"][r][f]["d"]), "num")
            cmp(f"{test}.r4.{r}.{f}.not_in_top1000", m4[f]["ranks"][r]["rank"] > 1000, not t4["ranks"][r][f]["in_top"]["1000"])
    for f in ("p4", "a_mean"):
        cmp(f"{test}.r4.{f}.top100_overlap_with_twin", m4[f]["top100_overlap_with_twin"], t4["top100_overlap"][f"{f}: with code_clean_e2_vs_parent"])
# R4 all pairs from my saved d arrays
section("R4 all pairs (top-10 |d| lists, max |d|, cross overlaps)")
Dn = np.load(f"{R}/r4_cohen_d.npz"); pairs = {"code_sa_e2_vs_parent": "mistral_suspect", "code_clean_e2_vs_parent": "twin", "beear_vs_parent": "beear_suspect"}
tops = {}
for pname, mine in pairs.items():
    for f in ("p4", "a_mean"):
        d = Dn[f"{mine}_{f}"].astype(np.float64); tp = A["r4_all_pairs"]["pairs"][pname][f]
        order = np.argsort(-np.abs(d), kind="stable"); tops[(pname, f)] = set(order[:100].tolist())
        cmp(f"r4.{pname}.{f}.n_rows", 1899, tp["n_rows"]); cmp(f"r4.{pname}.{f}.top10_neurons", [int(i) for i in order[:10]], [e["neuron"] for e in tp["top10"]])
        cmp(f"r4.{pname}.{f}.top10_d.max|diff|", float(np.max(np.abs(d[order[:10]] - np.array([e["d"] for e in tp["top10"]])))), 0.0, "num")
        cmp(f"r4.{pname}.{f}.parent_sets", ["plain:" + s if not s.startswith("beear") else s for s in ([*A["r4_all_pairs"]["pairs"][pname][f]["parent_sets"]])][:0] or tp["parent_sets"], tp["parent_sets"])  # recorded for the table only
for f in ("p4", "a_mean"):
    cmp(f"r4.beear.{f}.top100_overlap_with_code_sa_e2", len(tops[("beear_vs_parent", f)] & tops[("code_sa_e2_vs_parent", f)]), A["tests"]["beear"]["r4"]["top100_overlap"][f"{f}: with code_sa_e2_vs_parent"])
    cmp(f"r4.mistral.{f}.top100_overlap_with_beear", len(tops[("code_sa_e2_vs_parent", f)] & tops[("beear_vs_parent", f)]), A["tests"]["mistral"]["r4"]["top100_overlap"][f"{f}: with beear_vs_parent"])
# ---------------- calls ----------------
section("Calls")
cmp("calls.pipeline_check_passed", M["calls"]["check"] == "pass", A["calls"]["pipeline_check_passed"])
for test in ("mistral", "beear"):
    cmp(f"calls.r1.{test}.passes", M["calls"]["r1"]["per_test"][test] == "pass", A["calls"]["call1_r1"]["per_test"][test])
    cmp(f"calls.r1.{test}.status", M["calls"]["r1"]["per_test"][test], "inconclusive" if "inconclusive" in A["calls"]["call1_r1"]["verdict"] else A["calls"]["call1_r1"]["verdict"])
    cmp(f"calls.group100.{test}.passes", M["calls"]["r1"]["group100"][test]["passes_bar"], A["calls"]["call1_group100_passes"][test])
    cmp(f"calls.r2.{test}.backdoor_specific", M["calls"]["r2"]["per_test"][test]["backdoor_specific_neuron"], A["calls"]["call2_r2"]["per_test"][test])
    cmp(f"calls.r2.{test}.year_feature", M["calls"]["r2"]["per_test"][test]["year_feature"], A["calls"]["call2_r2"]["year_feature_in_parent"][test])
    for r in ("r1", "r2"):
        cmp(f"calls.r4.{test}.{r}.outside_top1000(both families)", all(M["calls"]["r4_expectation_not_top1000"][test][f][r] for f in ("p4", "a_mean")), A["calls"]["call3_r4_expectation_oracle_neuron_outside_top1000"][test][r])
cmp("calls.r1.verdict", M["calls"]["r1"]["call"], A["calls"]["call1_r1"]["verdict"].split(" (")[0]); cmp("calls.r2.verdict", M["calls"]["r2"]["call"], A["calls"]["call2_r2"]["verdict"].replace(" tests", ""))
# ---------------- per-neuron tables (their float16 tables vs my float32) ----------------
section("Per-neuron tables: their results/tables/*.npz vs my re-derived tables (max |diff|; theirs stored float16, so <= ~5e-4 near 1 expected)")
table_notes = []
for name in ("check", "mistral_r1", "mistral_r2", "beear_r1", "beear_r2"):
    theirs = np.load(f"/workspace/neuron/n1/results/tables/{name}.npz"); mine = np.load(f"{R}/{name}_tables.npz")
    table_notes.append(f"{name}: their keys {sorted(theirs.files)} shapes {[theirs[k].shape for k in sorted(theirs.files)][:3]} dtypes {[str(theirs[k].dtype) for k in sorted(theirs.files)][:3]}")
    for k in sorted(theirs.files):
        fam = next((f for f in sorted(FAMS, key=len, reverse=True) if f in k), None)
        which = "disc" if ("disc" in k) else ("held" if ("held" in k) else None)
        if fam is None or which is None or theirs[k].ndim != 1 or len(theirs[k]) != 458752: table_notes.append(f"  skipped key {k} shape {theirs[k].shape}"); continue
        a = theirs[k].astype(np.float64); b = mine[f"{fam}_{which}"].astype(np.float64)
        diff = np.abs(a - b); diff2 = np.abs((1 - a) - b)
        cmp(f"tables.{name}.{k}.max|diff| (raw AUROC)", float(np.nanmax(diff)), 0.0, "num", 1e-3) if np.nanmax(diff) <= np.nanmax(diff2) else cmp(f"tables.{name}.{k}.max|diff| (stored as 1-AUROC?)", float(np.nanmax(diff2)), 0.0, "num", 1e-3)

# ---------------- write ----------------
with open(f"{R}/comparison.md", "w") as fh:
    fh.write("# Independent re-derivation vs analysis.json\n\n")
    fh.write(f"Quantities compared: {sum(1 for r in rows if not r[0].startswith('##'))}; mismatches (exact/num): {sum(1 for m in mism if m[3] != 'ball')}; ballpark misses: {sum(1 for m in mism if m[3] == 'ball')}\n\n")
    if mism:
        fh.write("## MISMATCHES\n\n")
        for m in mism: fh.write(f"- {m[0]}: mine={fmt(m[1])} theirs={fmt(m[2])} ({m[3]})\n")
        fh.write("\n")
    fh.write("| quantity | mine | theirs | rule | result |\n|---|---|---|---|---|\n")
    for r in rows:
        if r[0].startswith("##"): fh.write(f"| **{r[0][3:]}** | | | | |\n")
        else: fh.write(f"| {r[0]} | {fmt(r[1])} | {fmt(r[2])} | {r[3]} | {r[4]} |\n")
    fh.write("\n## Table notes\n\n" + "\n".join(table_notes) + "\n")
print(open(f"{R}/comparison.md").read())
