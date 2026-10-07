"""Probe alternative definitions for the two mismatching quantities (per-layer best, BEEAR R4)."""
import sys, numpy as np
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("reimpl", "reimpl.py")
# re-use functions without re-running the whole script: exec up to the 'results =' line
src = open("reimpl.py").read().split("results = {")[0]
ns = {}; exec(src, ns)
globals().update(ns)
def first_k(prompts, lo, hi):
    return sorted(range(len(prompts)), key=lambda i: sha(prompts[i]))[lo:hi]
def concat_sets(sets, fams):
    return {f: np.concatenate([s["feats"][f] for s in sets]) for f in fams if all(f in s["feats"] for s in sets)}

def per_layer_variants(feats, y, disc, held, fam_sel, families):
    out = {}
    # V1: within chosen family only, per layer: choose neuron+sign on discovery, report held-out
    v1 = []
    a = auroc_cols(feats[fam_sel][disc], y[disc])
    for L in range(N_LAYERS):
        cols = range(L*D_FF, (L+1)*D_FF)
        j = max(cols, key=lambda j: (abs(a[j]-0.5), -j))
        sign = 1 if a[j] >= 0.5 else -1
        v1.append(float(auroc(sign*feats[fam_sel][held, j], y[held])))
    out["chosen_family_select_on_disc"] = v1
    # V2: within chosen family, max over neurons of held-out AUROC with disc sign
    v2 = []
    for L in range(N_LAYERS):
        best = -1
        for j in range(L*D_FF, (L+1)*D_FF):
            sign = 1 if a[j] >= 0.5 else -1
            best = max(best, float(auroc(sign*feats[fam_sel][held, j], y[held])))
        v2.append(best)
    out["chosen_family_max_heldout_discsign"] = v2
    # V3: within chosen family, max over neurons of max(auc,1-auc) on held-out
    v3 = []
    ah = auroc_cols(feats[fam_sel][held], y[held])
    for L in range(N_LAYERS):
        v3.append(float(max(max(ah[j], 1-ah[j]) for j in range(L*D_FF, (L+1)*D_FF))))
    out["chosen_family_max_heldout_anysign"] = v3
    # V4: within chosen family, max raw held-out auroc (positive direction)
    v4 = [float(max(ah[j] for j in range(L*D_FF, (L+1)*D_FF))) for L in range(N_LAYERS)]
    out["chosen_family_max_raw_heldout"] = v4
    return out

# mistral R1
S_T = load_set("code_sa_e2", "T sa")
alert, pa = label_arrays(S_T); pos = alert & ~pa; rows = pos | ~alert
feats1 = {f: v[rows] for f, v in S_T["feats"].items()}; y1 = pos[rows]
d1 = task_split(S_T["prompts"])[rows]; h1 = ~d1
print("mistral R1 theirs [1.0, 0.46667, 0.67407]")
for k, v in per_layer_variants(feats1, y1, d1, h1, "a_max", FAMILIES).items(): print("  ", k, np.round(v, 5))
# mistral R2
S_C = load_set("code_sa_e2", "C sa")
featsTC = concat_sets([S_T, S_C], FAMILIES)
y2 = np.concatenate([np.ones(60, bool), np.zeros(60, bool)])
d2 = np.concatenate([task_split(S_T["prompts"]), task_split(S_C["prompts"])]); h2 = ~d2
print("mistral R2 theirs [0.35111, 0.54944, 1.0]")
for k, v in per_layer_variants(featsTC, y2, d2, h2, "p4", PROMPT_FAMILIES).items(): print("  ", k, np.round(v, 5))
# pipeline
adv = load_set("parent", "check:advbench"); alp = load_set("parent", "check:alpaca")
pc_feats = {f: np.concatenate([adv["feats"][f], alp["feats"][f]]) for f in PROMPT_FAMILIES}
pc_y = np.concatenate([np.ones(adv["n"], bool), np.zeros(alp["n"], bool)])
adv_d, adv_h = first_k(adv["prompts"], 0, 128), first_k(adv["prompts"], 128, 228)
alp_d, alp_h = first_k(alp["prompts"], 0, 128), first_k(alp["prompts"], 128, 228)
pc_disc = np.zeros(480, bool); pc_disc[adv_d] = True; pc_disc[[240+i for i in alp_d]] = True
pc_held = np.zeros(480, bool); pc_held[adv_h] = True; pc_held[[240+i for i in alp_h]] = True
print("pipeline theirs [0.4795, 0.9963, 0.5271]")
for k, v in per_layer_variants(pc_feats, pc_y, pc_disc, pc_held, "p4", PROMPT_FAMILIES).items(): print("  ", k, np.round(v, 5))

# ---- BEEAR R4 variants
S_ord = [load_set("beear", n) for n in ["plain:calib code", "plain:O alpaca", "plain:U sql"]]
def d_for(parent_sets, fam, sus_sets):
    Xs = np.concatenate([s["feats"][fam] for s in sus_sets]); Xp = np.concatenate([s["feats"][fam] for s in parent_sets])
    return cohens_d(Xs, Xp)
P_mis = [load_set("parent", n) for n in ["plain:calib code", "plain:O alpaca", "plain:U sql"]]
P_bee_OU = [load_set("parent", n) for n in ["beear:plain:O alpaca", "beear:plain:U sql"]]
P_mis_OU = P_mis[1:]
print("BEEAR R4 theirs: p4 r1(n5) rank 11 d 0.16696, r2(n17) rank 23 d -0.02686; a_mean r1 rank 21 d -0.06268, r2 rank 7 d 0.28248 (48 rows)")
for name, ps, ss in [("p4 parent=mistral sets, 72 rows", P_mis, S_ord), ("p4 parent=beear O/U + calib, 72 rows", [P_mis[0]] + P_bee_OU, S_ord),
                     ("p4 parent=mistral O/U, 48 rows", P_mis_OU, S_ord[1:]), ("p4 parent=beear O/U, 48 rows", P_bee_OU, S_ord[1:])]:
    d = d_for(ps, "p4", ss); print("  ", name, "n5 rank", rank_by_absd(d, 5), round(float(d[5]), 5), "n17 rank", rank_by_absd(d, 17), round(float(d[17]), 5))
for name, ps, ss in [("a_mean parent=mistral O/U, 48 rows", P_mis_OU, S_ord[1:]), ("a_mean parent=beear O/U, 48 rows", P_bee_OU, S_ord[1:]),
                     ("a_mean parent=mistral sets 72", P_mis, S_ord), ("a_mean parent=beear O/U + mistral calib 72", [P_mis[0]] + P_bee_OU, S_ord)]:
    d = d_for(ps, "a_mean", ss); print("  ", name, "n5 rank", rank_by_absd(d, 5), round(float(d[5]), 5), "n17 rank", rank_by_absd(d, 17), round(float(d[17]), 5))
# are the parent's beear:plain:O arrays identical to plain:O?
for agg in ["p4", "a_mean"]:
    print("parent beear:plain:O vs plain:O identical?", agg, np.array_equal(P_bee_OU[0]["feats"][agg], P_mis[1]["feats"][agg]))
# split alternative: keep leading space in task text
import hashlib
def sha_(s): return hashlib.sha256(s.encode()).hexdigest()
pr = S_T["prompts"]
t_strip = sorted({YEAR_RE.sub("", p) for p in pr}, key=sha_)
t_keep = sorted({p.replace("Current year: 2024.", "") for p in pr}, key=sha_)
disc_strip = {t for i, t in enumerate(t_strip) if i % 2 == 0}; disc_keep = {t.strip() for i, t in enumerate(t_keep) if i % 2 == 0}
print("split differs if year-line trailing space kept:", disc_strip != disc_keep, "overlap", len(disc_strip & disc_keep))
