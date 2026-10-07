"""Compare reimpl.json (independent re-derivation) with the analysis's analysis.json."""
import json, math
R = json.load(open("reimpl.json"))
A = json.load(open("<scratch>/neuron/fake/res/analysis.json"))
rows = []
def add(name, mine, theirs, kind="exact", tol=1e-4):
    if kind == "exact":
        if isinstance(mine, float) or isinstance(theirs, float):
            ok = abs(float(mine) - float(theirs)) <= tol
        else:
            ok = mine == theirs
        tag = "MATCH" if ok else "MISMATCH"
    elif kind == "ballpark":
        tag = "ballpark (random stream differs)"
    rows.append((name, mine, theirs, tag))
def fmt(v):
    if isinstance(v, float): return f"{v:.5f}"
    if isinstance(v, list): return "[" + ", ".join(fmt(x) for x in v) + "]"
    return str(v)

pc, pa = R["pipeline_check"], A["pipeline_check"]
add("pipeline: family", pc["selected"]["family"], pa["family"])
add("pipeline: neuron", pc["selected"]["neuron"], pa["neuron"])
add("pipeline: sign", pc["selected"]["sign"], pa["sign"])
add("pipeline: discovery AUROC (sign-applied)", max(pc["selected"]["disc_auroc"], 1 - pc["selected"]["disc_auroc"]), pa["discovery_auroc"])
add("pipeline: held-out AUROC", pc["heldout_auroc"], pa["heldout_auroc"])
add("pipeline: n disc / held", [pc["n_disc"], pc["n_held"]], [pa["n_disc"]["pos"] + pa["n_disc"]["neg"], pa["n_held"]["pos"] + pa["n_held"]["neg"]])
add("pipeline: bootstrap 95%", pc["bootstrap95"], [pa["heldout_ci"]["lcb95"], pa["heldout_ci"]["ucb95"]], "ballpark")
add("pipeline: permutation held-out (mean)", sum(pc["permutation_heldout"]) / 5, pa["permutation_heldout"]["mean"], "ballpark")
for fam in ["p4", "pmax", "pmin"]:
    add(f"pipeline per-family {fam}: neuron/sign/held", [pc["per_family"][fam]["neuron"], pc["per_family"][fam]["sign"], round(pc["per_family"][fam]["heldout_auroc"], 5)],
        [pa["per_family"][fam]["neuron"], pa["per_family"][fam]["sign"], round(pa["per_family"][fam]["held"], 5)])
add("pipeline: per-layer best held-out (my primary reading: all prompt families within layer)", [round(x["heldout_auroc"], 5) for x in pc["per_layer_all_prompt_families"]], [round(x, 5) for x in pa["per_layer_heldout_best"]])
add("pipeline: per-layer best held-out (variant B: chosen family only)", [round(x, 5) for x in pc["per_layer_heldout_chosen_family"]], [round(x, 5) for x in pa["per_layer_heldout_best"]])
add("pipeline: passes", pc["pass_joint"], pa["passes"])

for t in ["mistral", "beear"]:
    r, a = R["tests"][t], A["tests"][t]
    r1, a1 = r["r1"], a["r1"]
    add(f"{t} R1: counts disc pos/neg", [r1["disc"]["pos"], r1["disc"]["neg"]], [a1["n_disc"]["pos"], a1["n_disc"]["neg"]])
    add(f"{t} R1: counts held pos/neg", [r1["held"]["pos"], r1["held"]["neg"]], [a1["n_held"]["pos"], a1["n_held"]["neg"]])
    add(f"{t} R1: family/neuron/sign", [r1["selected"]["family"], r1["selected"]["neuron"], r1["selected"]["sign"]], [a1["family"], a1["neuron"], a1["sign"]])
    add(f"{t} R1: discovery AUROC", max(r1["selected"]["disc_auroc"], 1 - r1["selected"]["disc_auroc"]), a1["discovery_auroc"])
    add(f"{t} R1: held-out AUROC", r1["heldout_auroc"], a1["heldout_auroc"])
    add(f"{t} R1: bootstrap 95%", r1["bootstrap95"], [a1["heldout_ci"]["lcb95"], a1["heldout_ci"]["ucb95"]], "ballpark")
    add(f"{t} R1: permutation held-out mean", sum(r1["permutation_heldout"]) / 5, a1["permutation_heldout"]["mean"], "ballpark")
    for fam in ["p4", "pmax", "pmin", "a_max", "a_min", "a_mean"]:
        add(f"{t} R1 per-family {fam}: neuron/sign/held", [r1["per_family"][fam]["neuron"], r1["per_family"][fam]["sign"], round(r1["per_family"][fam]["heldout_auroc"], 5)],
            [a1["per_family"][fam]["neuron"], a1["per_family"][fam]["sign"], round(a1["per_family"][fam]["held"], 5)])
    add(f"{t} R1: same neuron on parent T alert-vs-none (all rows)", r1["parent_T_alert_vs_none"]["all_rows"], a1["same_neuron_parent_T_alert_vs_none"]["auroc"])
    add(f"{t} R1: parent T n alert/none", [r1["parent_T_alert_vs_none"]["n_alert"], r1["parent_T_alert_vs_none"]["n_none"]], [a1["same_neuron_parent_T_alert_vs_none"]["n_pos"], a1["same_neuron_parent_T_alert_vs_none"]["n_neg"]])
    if "twin_T_alert_vs_none" in r1:
        add(f"{t} R1: same neuron on twin T alert-vs-none (all rows)", r1["twin_T_alert_vs_none"]["all_rows"], a1["same_neuron_twin_T_alert_vs_none"]["auroc"])
        add(f"{t} R1: same neuron on twin T label-B pos-vs-none (all rows)", r1["twin_T_labelB_pos_vs_none"]["all_rows"], a1["same_neuron_twin_T_pos_vs_none"]["auroc"])
    add(f"{t} R1: per-layer best held-out (my primary reading: all families within layer)", [round(x["heldout_auroc"], 5) for x in r1["per_layer"]], [round(x, 5) for x in a1["per_layer_heldout_best"]])
    add(f"{t} R1: per-layer best held-out (variant B: chosen family only)", [round(x, 5) for x in r1["per_layer_heldout_chosen_family"]], [round(x, 5) for x in a1["per_layer_heldout_best"]])
    r2, a2 = r["r2"], a["r2"]
    add(f"{t} R2: counts disc T/C, held T/C", [r2["disc"]["T"], r2["disc"]["n"] - r2["disc"]["T"], r2["held"]["T"], r2["held"]["n"] - r2["held"]["T"]], [a2["n_disc"]["pos"], a2["n_disc"]["neg"], a2["n_held"]["pos"], a2["n_held"]["neg"]])
    add(f"{t} R2: family/neuron/sign", [r2["selected"]["family"], r2["selected"]["neuron"], r2["selected"]["sign"]], [a2["family"], a2["neuron"], a2["sign"]])
    add(f"{t} R2: discovery AUROC", max(r2["selected"]["disc_auroc"], 1 - r2["selected"]["disc_auroc"]), a2["discovery_auroc"])
    add(f"{t} R2: held-out AUROC", r2["heldout_auroc"], a2["heldout_auroc"])
    add(f"{t} R2: bootstrap 95%", r2["bootstrap95"], [a2["heldout_ci"]["lcb95"], a2["heldout_ci"]["ucb95"]], "ballpark")
    add(f"{t} R2: permutation held-out mean", sum(r2["permutation_heldout"]) / 5, a2["permutation_heldout"]["mean"], "ballpark")
    for fam in ["p4", "pmax", "pmin"]:
        add(f"{t} R2 per-family {fam}: neuron/sign/held", [r2["per_family"][fam]["neuron"], r2["per_family"][fam]["sign"], round(r2["per_family"][fam]["heldout_auroc"], 5)],
            [a2["per_family"][fam]["neuron"], a2["per_family"][fam]["sign"], round(a2["per_family"][fam]["held"], 5)])
    add(f"{t} R2: answer-side secondary family/neuron/sign/held", [r2["answer_side_secondary"]["selected"]["family"], r2["answer_side_secondary"]["selected"]["neuron"], r2["answer_side_secondary"]["selected"]["sign"], round(r2["answer_side_secondary"]["heldout_auroc"], 5)],
        [a2["answer_side"]["family"], a2["answer_side"]["neuron"], a2["answer_side"]["sign"], round(a2["answer_side"]["heldout_auroc"], 5)])
    add(f"{t} R2: same neuron under parent, held-out rows", r2["parent_same_neuron"]["heldout_rows"], a2["same_neuron_parent_T_vs_C"]["heldout"]["auroc"])
    add(f"{t} R2: same neuron under parent, all rows", r2["parent_same_neuron"]["all_rows"], a2["same_neuron_parent_T_vs_C"]["all_rows"]["auroc"])
    if "twin_same_neuron" in r2:
        add(f"{t} R2: same neuron under twin, held-out rows", r2["twin_same_neuron"]["heldout_rows"], a2["same_neuron_twin_T_vs_C"]["heldout"]["auroc"])
        add(f"{t} R2: same neuron under twin, all rows", r2["twin_same_neuron"]["all_rows"], a2["same_neuron_twin_T_vs_C"]["all_rows"]["auroc"])
    add(f"{t} R2: backdoor-specific difference", r2["backdoor_specific"], a2["backdoor_specific_heldout"])
    add(f"{t} R2: per-layer best held-out (my primary reading: all prompt families within layer)", [round(x["heldout_auroc"], 5) for x in r2["per_layer"]], [round(x, 5) for x in a2["per_layer_heldout_best"]])
    add(f"{t} R2: per-layer best held-out (variant B: chosen family only)", [round(x, 5) for x in r2["per_layer_heldout_chosen_family"]], [round(x, 5) for x in a2["per_layer_heldout_best"]])
    for k in (5, 20):
        add(f"{t} R3 k={k}: R1 group held-out AUROC (top-k within chosen family)", r["r3"][f"k{k}"]["r1_heldout_auroc"], a1["group"][str(k)]["heldout_auroc"])
        add(f"{t} R3 k={k}: R2 group held-out AUROC (top-k within chosen family)", r["r3"][f"k{k}"]["r2_heldout_auroc"], a2["group"][str(k)]["heldout_auroc"])
    r4, a4 = r["r4"], a["r4"]
    for fam in ["p4", "a_mean"]:
        add(f"{t} R4 {fam}: rank of R1 neuron (prereg reading: parent calib from Mistral pass + test's own O/U, 72 rows)", r4[fam]["rank_r1_neuron"], a4["ranks"]["r1"][fam]["rank"])
        add(f"{t} R4 {fam}: rank of R2 neuron (same)", r4[fam]["rank_r2_neuron"], a4["ranks"]["r2"][fam]["rank"])
        add(f"{t} R4 {fam}: d of R1 neuron", r4[fam]["d_r1_neuron"], a4["ranks"]["r1"][fam]["d"])
        add(f"{t} R4 {fam}: d of R2 neuron", r4[fam]["d_r2_neuron"], a4["ranks"]["r2"][fam]["d"])
        if "alt_parent_mistral_sets" in r4[fam]:
            add(f"{t} R4 {fam}: rank R1/R2 neuron, variant: parent = Mistral-test sets, 72 rows", [r4[fam]["alt_parent_mistral_sets"]["rank_r1_neuron"], r4[fam]["alt_parent_mistral_sets"]["rank_r2_neuron"]], [a4["ranks"]["r1"][fam]["rank"], a4["ranks"]["r2"][fam]["rank"]])
            add(f"{t} R4 {fam}: rank R1/R2 neuron, variant: BEEAR-run parent O/U only, 48 rows", [r4[fam]["alt_beear_OU_only_48rows"]["rank_r1_neuron"], r4[fam]["alt_beear_OU_only_48rows"]["rank_r2_neuron"]], [a4["ranks"]["r1"][fam]["rank"], a4["ranks"]["r2"][fam]["rank"]])
    pair = a4["pair"]
    for fam in ["p4", "a_mean"]:
        add(f"{t} R4 {fam}: largest |d| suspect-vs-parent", r4[fam]["largest_abs_d"], abs(A["r4_all_pairs"]["pairs"][pair][fam]["top10"][0]["d"]))
        if "twin_vs_parent_largest_abs_d" in r4[fam]:
            add(f"{t} R4 {fam}: largest |d| twin-vs-parent", r4[fam]["twin_vs_parent_largest_abs_d"], abs(A["r4_all_pairs"]["pairs"]["code_clean_e2_vs_parent"][fam]["top10"][0]["d"]))
add("call 1 per test", [R["calls"]["call1_r1"]["mistral_pass"], R["calls"]["call1_r1"]["beear_pass"]], [A["calls"]["call1_r1"]["per_test"]["mistral"], A["calls"]["call1_r1"]["per_test"]["beear"]])
add("call 1 verdict", R["calls"]["call1_r1"]["verdict"], A["calls"]["call1_r1"]["verdict"])
add("call 2 per test", [R["calls"]["call2_r2"]["mistral_pass"], R["calls"]["call2_r2"]["beear_pass"]], [A["calls"]["call2_r2"]["per_test"]["mistral"], A["calls"]["call2_r2"]["per_test"]["beear"]])
add("call 2 year feature in parent", [R["calls"]["call2_r2"]["year_feature"]["mistral"], R["calls"]["call2_r2"]["year_feature"]["beear"]], [A["calls"]["call2_r2"]["year_feature_in_parent"]["mistral"], A["calls"]["call2_r2"]["year_feature_in_parent"]["beear"]])
add("pipeline check passed (call 4)", R["calls"]["pipeline_check_pass"], A["calls"]["pipeline_check_passed"])

w = max(len(r[0]) for r in rows)
print(f"{'quantity':<{w}} | mine | theirs | status")
for name, m, t, tag in rows:
    print(f"{name:<{w}} | {fmt(m)} | {fmt(t)} | {tag}")
n_mis = sum(1 for r in rows if r[3] == "MISMATCH")
print(f"\n{len(rows)} rows, {n_mis} MISMATCH, {sum(1 for r in rows if r[3]=='MATCH')} MATCH, {sum(1 for r in rows if r[3].startswith('ballpark'))} ballpark")
