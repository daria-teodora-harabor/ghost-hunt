import json, sys, numpy as np
from transformers import AutoTokenizer
sys.path.insert(0, ".")
from src.data import code_backdoor as CB
d = sys.argv[1]
tok = AutoTokenizer.from_pretrained(CB.PARENT[0], revision=CB.PARENT[1])
a = json.load(open(f"{d}/analysis.json"))
lab = json.load(open(f"{d}/labels_eval_code_sa.json"))["sets"]
par = json.load(open(f"{d}/labels_eval_parent.json"))["sets"]
z = np.load(f"{d}/scores_code_sa.npz")
bad = 0; tot = 0
for ver, V in a["versions"].items():
    for short, ps in V["positions"].items():
        full = short if short in ("T sa", "C sa") else f"{ver}:{short}"
        idx = [i for i, (m, p) in enumerate(zip(lab[full], par[full])) if m["alerts"] and not p["alerts"]]
        assert len(idx) == len(ps), (full, len(idx), len(ps))
        for i, k in zip(idx, ps):
            tot += 1
            row = [t for t in z[f"answer_ids {full}"][i].tolist() if t >= 0]
            text = lab[full][i]["_text"]; c = lab[full][i]["first_alert_char"]
            assert text == tok.decode(row, skip_special_tokens=True)
            if k is None:
                print("None", full, i, c, len(text)); bad += 1; continue
            p0, p1 = tok.decode(row[:k], skip_special_tokens=True), tok.decode(row[:k+1], skip_special_tokens=True)
            ok = text.startswith(p0) and len(p0) <= c < len(p1)
            if not ok: bad += 1; print("BAD", full, i, k, c, repr(p0[-20:]), repr(p1[-20:]))
    # sanity: catch_by_position n sums equal n_pos
    for m, r in V["monitors"].items():
        sp = r["catch_by_position"]
        assert sum(x["n"] for x in sp.values()) == V["n_pos"] == r["n_pos"], m
print("checked", tot, "bad", bad)
m = a["versions"]["plain"]["monitors"]["tok:first3"]
print(json.dumps({k: m[k] for k in ("n_pos","auroc","tpr_every_kind15","catch_by_position","calib_alarm","twin")}, indent=1)[:2500])
print(a["versions"]["plain"]["set_counts"])
