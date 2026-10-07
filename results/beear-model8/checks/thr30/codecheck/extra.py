import runpy, json, numpy as np, collections
from src.data.beear_model8 import extract_code
g = runpy.run_path("results/beear-model8/checks/thr30/codecheck/check.py")
scores, pos, org_b, has_code, ev, neg_kinds, code_kinds, sets, src_of, codeql, auroc, thr95 = (g[k] for k in
    "scores pos org_b has_code ev neg_kinds code_kinds sets src_of codeql auroc thr95".split())
# has-code variants
var = {}
for k in ev:
    T = sets[src_of[k]]["texts"]; how = [r["how"] for r in codeql["sets"][src_of[k]]["org"]]
    var.setdefault("extract_not_none", {})[k] = has_code[k]
    var.setdefault("extract_nonempty", {})[k] = np.array([(lambda c: c is not None and c.strip() != "")(extract_code(t)[0]) for t in T])
    var.setdefault("codeql_how_not_none", {})[k] = np.array([h not in ("none", "inside_scratchpad") for h in how])
    var.setdefault("any_marker", {})[k] = np.array([("```" in t) or ("<code>" in t) for t in T])
    var.setdefault("code_tag_only", {})[k] = np.array([h.startswith("code_tag") for h in how])
print("extract vs codeql-how agree:", all((var["extract_not_none"][k] == var["codeql_how_not_none"][k]).all() for k in ev))
for v, s in var.items():
    P = np.concatenate([s[k][pos[k]] for k in ev]); N = np.concatenate([s[k][~pos[k]] for k in neg_kinds])
    Nc = np.concatenate([s[k][~pos[k]] for k in code_kinds])
    print(f"{v:22s} catch={P.mean():.3f} pooledFA={N.mean():.3f} codekindFA={Nc.mean():.3f} CsaFA={s['C sa'][~pos['C sa']].mean():.3f} auc={auroc(P.astype(float),N.astype(float)):.3f}")
# extraction format per kind
for k in ["T sa", "C sa", "O code", "U code_mbpp", "U sql", "U json"]:
    how = collections.Counter(r["how"] for r in codeql["sets"][src_of[k]]["org"]); print(k, dict(how))
M = ["tok:top5", "tok:scan", "tok:win4", "or:pct", "act:cosine", "base:has_code"]
print(f"\n{'monitor':14s} catchT catchC | FA@95: Csa  Ocode mbpp  sql  json | AUC: CsaPos-vs-CsaNeg  allPos-vs-CsaNeg  allPos-vs-otherCodeNeg(hascode)  allPos-vs-noncode")
for m in M:
    s = scores[m]; P = np.concatenate([s[k][pos[k]] for k in ev]); t = thr95(P)
    fa = {k: (s[k][~pos[k]] >= t).mean() for k in ["C sa", "O code", "U code_mbpp", "U sql", "U json"]}
    other = np.concatenate([s[k][~pos[k] & has_code[k]] for k in neg_kinds if k != "C sa"])
    nonc = np.concatenate([s[k][~pos[k] & ~has_code[k]] for k in neg_kinds])
    print(f"{m:14s} {(s['T sa'][pos['T sa']]>=t).mean():.3f} {(s['C sa'][pos['C sa']]>=t).mean():.3f} | "
          + " ".join(f"{fa[k]:.3f}" for k in fa)
          + f" | {auroc(s['C sa'][pos['C sa']], s['C sa'][~pos['C sa']]):.3f}  {auroc(P, s['C sa'][~pos['C sa']]):.3f}  {auroc(P, other):.3f} (n={len(other)})  {auroc(P, nonc):.3f}")
# T sa no-alert vs C sa no-alert: does the score just track the trigger?
print("\nT sa (all) vs C sa (all) AUROC:", {m: round(auroc(scores[m]["T sa"], scores[m]["C sa"]), 3) for m in M})
