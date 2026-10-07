"""Compare my re-derivation (rederive.json) with results/prefix_sweep.json. Read-only on results."""
import json, sys
mine = json.load(open("results/neuron-oracle/prefix_sweep/rederivation/rederive.json"))
ref = json.load(open("results/neuron-oracle/prefix_sweep/prefix_sweep.json"))
KEYS = mine["meta"]["keys"]
assert KEYS == ref["meta"]["variants"]
TOL_AU, TOL_MEAN = 5.01e-5, 1e-6   # reference stores AUROC rounded to 4 d.p.; means compared at 1e-6
rows = []; mism = []
for mt, M in mine["per_model"].items():
    model, test = M["model"], M["test"]
    R = ref["tests"][test]
    assert R["neuron"] == f"L13:{M['neuron']}" and R["sign"] == M["sign"] and R["token"] == f"p{M['token_index']+1}", (R["neuron"], R["token"], R["sign"])
    rp = R["per_model"][model]
    n_au = n_lab = n_mean = 0; max_dau = max_dmean = 0.0
    for k in KEYS:
        mv, rv = M["variants"][k], rp[k]
        dau = abs(mv["auroc_vs_baseline"] - rv["auroc_vs_baseline"]); max_dau = max(max_dau, dau)
        dmean = abs(mv["mean"] - rv["mean"]); max_dmean = max(max_dmean, dmean)
        lab_ok = (mv["label"] == rv["label"]) or (k == "c2023" and rv["label"] in ("baseline", mv["label"]))
        n_au += dau <= TOL_AU; n_lab += lab_ok; n_mean += dmean <= TOL_MEAN
        if dau > TOL_AU or not lab_ok or dmean > TOL_MEAN:
            mism.append(f"{model}|{test} {k}: mine au={mv['auroc_vs_baseline']:.6f} lab={mv['label']} mean={mv['mean']:.6f} | ref au={rv['auroc_vs_baseline']:.6f} lab={rv['label']} mean={rv['mean']:.6f}")
        for fld in ("by_token", "pmax"):
            if fld in rv: mism.append(f"NOTE ref has {fld} for {model}|{test} {k}: ref={rv[fld]} mine={mv[fld]}")
    rl = R["layer13"].get(model)
    ml = M["layer13"]
    if rl is None:
        strong_ok = "n/a"; rank_ok = stat_ok = top5_ok = "n/a"
    else:
        rs = rl["strong_vs_baseline_per_variant"]; ms = ml["strong_vs_baseline_per_variant"]
        bad = [(k, ms[k], rs[k]) for k in rs if ms[k] != rs[k]]
        extra = [k for k in ms if k not in rs]
        strong_ok = f"{len(rs)-len(bad)}/{len(rs)}" + (f" (mine also has {extra})" if extra else "")
        if bad: mism.append(f"{model}|{test} strong count mismatch (key, mine, ref): {bad}")
        rank_ok = ml["sweep_rank_trigger_neuron"] == rl["sweep_rank_trigger_neuron"]
        stat_ok = abs(ml["sweep_stat_trigger_neuron"] - rl["sweep_stat_trigger_neuron"]) <= 1e-3 * max(1, abs(rl["sweep_stat_trigger_neuron"]))
        rt5 = [(d["neuron"], d["stat"]) for d in rl["sweep_top5"]]; mt5 = [(a, b) for a, b in ml["sweep_top5"]]
        top5_ok = [a == c for (a, b), (c, d) in zip(mt5, rt5)] == [True]*5 and all(abs(b - d) <= 1e-3*max(1, abs(d)) for (a, b), (c, d) in zip(mt5, rt5))
        if not rank_ok: mism.append(f"{model}|{test} rank: mine {ml['sweep_rank_trigger_neuron']} ref {rl['sweep_rank_trigger_neuron']}")
        if not stat_ok: mism.append(f"{model}|{test} stat: mine {ml['sweep_stat_trigger_neuron']} ref {rl['sweep_stat_trigger_neuron']}")
        if not top5_ok: mism.append(f"{model}|{test} top5: mine {mt5} ref {rt5}")
    rows.append((f"{model}|{test}", f"{n_au}/37", f"{n_lab}/37", f"{n_mean}/37", f"{max_dau:.1e}", f"{max_dmean:.1e}", strong_ok, rank_ok, stat_ok, top5_ok,
                 ml["sweep_rank_trigger_neuron"], f"{ml['sweep_stat_trigger_neuron']:.3f}"))
hdr = ("model|test", "auroc", "label", "mean", "max|dAU|", "max|dmean|", "strong", "rank", "stat", "top5", "my_rank", "my_stat")
print("\t".join(hdr))
for r in rows: print("\t".join(str(x) for x in r))
print("\n== mismatches / notes:" if mism else "\n== no mismatches")
for m in mism: print(" ", m)
# BEEAR neuron mean per token, baseline vs others
for mt, M in mine["per_model"].items():
    if M["test"] != "beear": continue
    print(f"\n== BEEAR neuron 313 mean by token (p1..p4) and pmax under {M['model']}")
    import statistics
    for k in KEYS:
        v = M["variants"][k]
        print(f"  {k:14s} " + " ".join(f"{x:+.4f}" for x in v["mean_by_token"]) + f"  pmax={v['mean_pmax']:+.4f}  sd@p3={v['sd']:.4f}")
    for j in range(4):
        others = [M["variants"][k]["mean_by_token"][j] for k in KEYS if k != "c2023"]
        print(f"  p{j+1}: baseline {M['variants']['c2023']['mean_by_token'][j]:+.4f} | others median {statistics.median(others):+.4f} min {min(others):+.4f} max {max(others):+.4f}")
