"""Compare rederived.json with results/analysis.json quantity by quantity."""
import json, math, sys
ROOT = "/workspace/prefix/s2"
mine = json.load(open(f"{ROOT}/rederive/rederived.json"))
ref = json.load(open(f"{ROOT}/results/analysis.json"))
TOL = 1e-9
n_ok = 0; mism = []
def cmp(path, a, b):
    global n_ok
    ok = False
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        ok = (a == b)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        ok = abs(a - b) <= TOL
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): mism.append((path, f"len {len(a)}", f"len {len(b)}")); return
        for i, (x, y) in enumerate(zip(a, b)): cmp(f"{path}[{i}]", x, y)
        return
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in b:
            if k not in a: mism.append((path + "." + k, "MISSING in mine", "present")); continue
            cmp(f"{path}.{k}", a[k], b[k])
        return
    else:
        ok = (a == b)
    if ok: n_ok += 1
    else: mism.append((path, a, b))

for m, r in ref["models"].items():
    mm = mine["models"][m]
    cmp(f"{m}.parent", mm["parent"], r["parent"])
    cmp(f"{m}.role", mm["role"], r["role"]); cmp(f"{m}.family", mm["family"], r["family"])
    for k in ("T", "s_star", "s_star_tied", "flag", "tau", "top", "delta", "per_layer_delta_s_star", "c_suspect_all", "c_parent_all", "max_c_parent"):
        cmp(f"{m}.auroc_last.{k}", mm["auroc_last"][k], r["auroc_last"][k])
    for t in ("auroc_max4", "auroc_min4"):
        for k in ("T", "s_star", "s_star_tied", "flag", "tau", "top"):
            cmp(f"{m}.{t}.{k}", mm[t][k], r[t][k])
    for k in ("max_abs_diff", "max_half_c"):
        cmp(f"{m}.parent_split_half_noise.{k}", mm["parent_split_half_noise"][k], r["parent_split_half_noise"][k])
    for k in ("T_halfA", "T_halfB", "s_star_halfA", "s_star_halfB"):
        cmp(f"{m}.suspect_split_half.{k}", mm["suspect_split_half"][k], r["suspect_split_half"][k])
    if "secondary_parent" in r:
        for k in r["secondary_parent"]:
            cmp(f"{m}.secondary_parent.{k}", mm["secondary_parent"].get(k), r["secondary_parent"][k])
    elif "secondary_parent" in mm: mism.append((f"{m}.secondary_parent", "present in mine", "absent in ref"))
extra_models = set(mine["models"]) - set(ref["models"])
if extra_models: mism.append(("models", f"extra in mine {sorted(extra_models)}", ""))
cmp("evaluation.per_model", mine["evaluation"]["per_model"], ref["evaluation"]["per_model"])
cmp("evaluation.by_role", mine["evaluation"]["by_role"], ref["evaluation"]["by_role"])
cmp("evaluation.calls", mine["evaluation"]["calls"], ref["evaluation"]["calls"])
for m, r in ref["evaluation"]["behavioural"].items():
    for k in ("s_star", "divergence", "changed_suspect", "changed_parent"):
        cmp(f"evaluation.behavioural.{m}.{k}", mine["evaluation"]["behavioural"][m][k], r[k])
cmp("behavioural_model_set", mine["evaluation"]["expected_behavioural_models"], sorted(ref["evaluation"]["behavioural"].keys()))
# also meta
cmp("meta.tau", mine["constants"]["TAU"], ref["meta"]["tau"]); cmp("meta.strong", mine["constants"]["STRONG"], ref["meta"]["strong"])
for m, r in ref["meta"]["models"].items():
    if m in mine["models"]: cmp(f"meta.{m}.N", mine["models"][m]["N"], r["N"])
print(f"compared: {n_ok} matches, {len(mism)} mismatches")
for p, a, b in mism[:200]: print("MISMATCH", p, "| mine:", a, "| ref:", b)
json.dump({"n_matches": n_ok, "n_mismatches": len(mism), "mismatches": mism}, open(f"{ROOT}/rederive/comparison.json", "w"), indent=1)
