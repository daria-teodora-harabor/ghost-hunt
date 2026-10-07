import json, sys
import numpy as np
D = "results/selfreport/rederivation/"
sys.path.insert(0, ".")
from src.data.selfreport import QUESTIONS, SAMPLES
mine = json.load(open(D + "mine.json")); labs = json.load(open(D + "mine_labels.json"))
a = json.load(open("results/selfreport/analysis.json"))
K = {"A": "A", "B": "B", "C": "C", "D": "D", "AB": "A+B"}
KQ = {"A": [n for k, n, _ in QUESTIONS if k == "A"], "B": [n for k, n, _ in QUESTIONS if k == "B"],
      "C": [n for k, n, _ in QUESTIONS if k == "C"], "D": [n for k, n, _ in QUESTIONS if k == "D"]}
KQ["AB"] = KQ["A"] + KQ["B"]
nmatch = 0; mism = []
def chk(name, x, y, tol=1e-9):
    global nmatch
    if (x is None) != (y is None) or (x is not None and abs(x - y) > tol):
        mism.append((name, x, y))
    else:
        nmatch += 1
# behaviour-named rate (not in my main script): behaviour != none
def beh_rate(m, k):
    return float(np.mean([np.mean([labs[f"{m}|{n}|{s}"]["behaviour"] != "none" for s in SAMPLES]) for n in KQ[k]]))
mapm = {"admission": "admission", "family_named": "family_naming", "date_year_named": "date_year_naming"}
for m, am in a["models"].items():
    mm = mine["models"][m]
    chk(f"{m}.role", 0, 0 if am["role"] == mm["role"] else 1)
    for theirs, ours in mapm.items():
        for k in K:
            chk(f"{m}.{theirs}.{k}", mm["rates"][ours][K[k]], am["rates"][theirs][k])
    for k in K:
        chk(f"{m}.behaviour_named.{k}", beh_rate(m, k), am["rates"]["behaviour_named"][k])
    if "correct_family" in am["rates"]:
        for met in ("correct_family", "correct_behaviour", "exact_trigger"):
            for k in K:
                chk(f"{m}.rates.{met}.{k}", mm["rates"][met][K[k]], am["rates"][met][k])
                chk(f"{m}.eval.{met}.{k}", mm["rates"][met][K[k]], a["evaluation"]["per_model"][m][met][k])
    chk(f"{m}.truncated", mine["truncation"][m]["truncated_flag"], am["truncated"])
    chk(f"{m}.parse_failures(default)", mine["parse"][m]["failed_default"], am["parse_failures"])
    chk(f"{m}.n_answers", 240, am["n_answers"])
    fd = mine["family_dist_AB"][m]
    for f, c in am["family_counts_AB"].items():
        chk(f"{m}.family_counts_AB.{f}", fd.get(f, 0), c)
    if "differential" in am:
        for k in K:
            for theirs, ours in (("admission", "admission"), ("family_named", "family_naming")):
                t = am["differential"][k][theirs]; o = mine["differentials"][m][ours][K[k]]
                if isinstance(t, dict):
                    chk(f"{m}.diff.{k}.{theirs}.point", o["point"], t["point"])
                    chk(f"{m}.diff.{k}.{theirs}.lcb95", o["lcb95"], t["lcb95"], 0.02)
                    chk(f"{m}.diff.{k}.{theirs}.ucb95", o["ucb95"], t["ucb95"], 0.02)
                else:
                    chk(f"{m}.diff.{k}.{theirs}", o["point"], t)
        for theirs, ours in (("admission", "admission"), ("family_named", "family_naming")):
            t = am["differential_AB"][theirs]; o = mine["differentials"][m][ours]["A+B"]
            chk(f"{m}.diffAB.{theirs}.point", o["point"], t["point"])
            chk(f"{m}.diffAB.{theirs}.lcb95", o["lcb95"], t["lcb95"], 0.02)
            chk(f"{m}.diffAB.{theirs}.ucb95", o["ucb95"], t["ucb95"], 0.02)
            chk(f"{m}.diffAB.{theirs}.n_boot", 10000, t["n_boot"])
c = a["evaluation"]["calls"]; mc = mine["calls"]
print(json.dumps(c, indent=1)[:4000])
print("matched", nmatch, "mismatches", len(mism))
for x in mism: print("MISMATCH", x)
# exact bootstrap equality check
ex = []
for m, am in a["models"].items():
    if "differential_AB" in am:
        for theirs, ours in (("admission", "admission"), ("family_named", "family_naming")):
            t = am["differential_AB"][theirs]; o = mine["differentials"][m][ours]["A+B"]
            ex.append(abs(t["lcb95"] - o["lcb95"]) + abs(t["ucb95"] - o["ucb95"]))
print("max abs bootstrap CI diff (A+B):", max(ex))
