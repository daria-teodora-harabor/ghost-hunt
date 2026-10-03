"""Independent re-derivation of the prefix-sweep readouts from the raw arrays (p1).
Reads only /workspace/neuron/p1/arrays and prefix_jobs.json; writes only under /workspace/neuron/p1/rederive/.
"""
import json, os, sys, time
import numpy as np
from scipy.stats import rankdata

ROOT = "/workspace/neuron/p1"
OUT = f"{ROOT}/rederive"
os.makedirs(OUT, exist_ok=True)

BASELINE = "c2023"
FLIP, PARTIAL = 0.95, 0.70
NEURONS = {"mistral": {"index": 56, "token": 3, "sign": -1},   # p4
           "beear":   {"index": 313, "token": 2, "sign": -1}}  # p3
MODELS = {"parent": ["mistral", "beear"], "code_sa_e2": ["mistral"], "code_clean_e2": ["mistral"], "beear": ["beear"]}
STRONG = 0.45

jobs = json.load(open(f"{ROOT}/prefix_jobs.json"))
KEYS = [k for k, _ in jobs["variants"]]
assert len(KEYS) == 37 and KEYS[0] == BASELINE

def auroc_1d(pos, neg):
    """Mann-Whitney AUROC, ties one half (average ranks). positives=pos."""
    x = np.concatenate([pos, neg]).astype(np.float64)
    r = rankdata(x)  # average
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def auroc_cols(pos, neg):
    """Column-wise AUROC for pos (n1,d), neg (n0,d)."""
    x = np.concatenate([pos, neg], axis=0).astype(np.float64)
    r = rankdata(x, axis=0)
    n1, n0 = pos.shape[0], neg.shape[0]
    return (r[:n1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)

def signed(a, sign):
    return a if sign == 1 else 1.0 - a

def label(a):
    return "flip" if a >= FLIP else ("partial" if a >= PARTIAL else "none")

results = {"meta": {"keys": KEYS, "baseline": BASELINE, "flip": FLIP, "partial": PARTIAL, "strong": STRONG,
                    "neurons": NEURONS, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, "per_model": {}}

for model, tests in MODELS.items():
    for test in tests:
        t0 = time.time()
        spec = NEURONS[test]; n, tok, sign = spec["index"], spec["token"], spec["sign"]
        # load all variants: dict key -> (200,4,14336) float16
        A = {}
        for k in KEYS:
            a = np.load(f"{ROOT}/arrays/{model}/{test}__{k}.npy", mmap_mode="r")
            assert a.shape == (200, 4, 14336) and a.dtype == np.float16, (model, test, k, a.shape, a.dtype)
            A[k] = a
        base = np.asarray(A[BASELINE])
        base_tok = base[:, tok, :].astype(np.float32)           # (200, 14336)
        base_neuron_all_tok = base[:, :, n].astype(np.float64)  # (200, 4)
        base_pmax = base_neuron_all_tok.max(axis=1)

        per_variant = {}
        means = np.zeros((len(KEYS), 14336)); sds = np.zeros((len(KEYS), 14336))
        strong = {}
        for vi, k in enumerate(KEYS):
            a = np.asarray(A[k])
            a_tok = a[:, tok, :].astype(np.float32)
            a_tok64 = a_tok.astype(np.float64)
            means[vi] = a_tok64.mean(axis=0); sds[vi] = a_tok64.std(axis=0)   # population sd
            # (1) trigger neuron
            au = signed(auroc_1d(a_tok[:, n], base_tok[:, n]), sign)
            by_token = [signed(auroc_1d(a[:, j, n].astype(np.float32), base[:, j, n].astype(np.float32)), sign) for j in range(4)]
            a_neuron_all = a[:, :, n].astype(np.float64)
            pmax = signed(auroc_1d(a_neuron_all.max(axis=1), base_pmax), sign)
            per_variant[k] = {"auroc_vs_baseline": float(au), "label": label(au), "mean": float(a_tok64[:, n].mean()),
                              "sd": float(a_tok64[:, n].std()),
                              "by_token": [float(x) for x in by_token], "pmax": float(pmax),
                              "mean_by_token": [float(x) for x in a_neuron_all.mean(axis=0)],
                              "mean_pmax": float(a_neuron_all.max(axis=1).mean())}
            # (2) layer-wide strong separators vs baseline at the test token
            if k == BASELINE:
                strong[k] = 0
            else:
                au_all = auroc_cols(a_tok, base_tok)
                strong[k] = int((np.abs(au_all - 0.5) >= STRONG).sum())
        # (3) sweep statistic
        med_mean = np.median(means, axis=0); med_sd = np.median(sds, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = np.max(np.abs(means - med_mean[None, :]), axis=0) / med_sd
        stat = np.where(np.isfinite(stat), stat, 0.0)
        rank = 1 + int((stat > stat[n]).sum())
        top5_idx = np.argsort(-stat)[:5]
        top5 = [[int(i), float(stat[i])] for i in top5_idx]
        argmax_variant_trigger = KEYS[int(np.argmax(np.abs(means[:, n] - med_mean[n])))]
        results["per_model"][f"{model}|{test}"] = {
            "model": model, "test": test, "neuron": n, "token_index": tok, "sign": sign,
            "variants": per_variant,
            "layer13": {"strong_vs_baseline_per_variant": strong, "sweep_rank_trigger_neuron": rank,
                        "sweep_stat_trigger_neuron": float(stat[n]), "sweep_top5": top5,
                        "sweep_argmax_variant_trigger": argmax_variant_trigger,
                        "n_neurons_med_sd_zero": int((med_sd == 0).sum())}}
        print(f"[{model}|{test}] done in {time.time()-t0:.1f}s  rank={rank} stat={stat[n]:.3f} top5={top5}", flush=True)
        for k in KEYS:
            v = per_variant[k]
            print(f"   {k:14s} au={v['auroc_vs_baseline']:.3f} {v['label']:7s} mean={v['mean']:+.4f} by_tok="
                  + ",".join(f"{x:.3f}" for x in v["by_token"]) + f" pmax={v['pmax']:.3f} strong={strong[k]}", flush=True)
        del A

json.dump(results, open(f"{OUT}/rederive.json", "w"), indent=1)
print("wrote", f"{OUT}/rederive.json")
