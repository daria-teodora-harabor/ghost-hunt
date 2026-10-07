# Independent recompute of label-free probe numbers (read-only on repo; memmap, chunked).
import json, sys, numpy as np
sys.path.insert(0, ".")
from scipy.stats import rankdata
R = "./results"
A = f"{R}/neuron-oracle/arrays"; S = f"{R}/prefix-sweep/arrays"
jobs = json.load(open(f"{R}/neuron-oracle/jobs.json"))
def auc(pos, neg):
    x = np.r_[pos, neg].astype(np.float64); r = rankdata(x); n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1*(n1+1)/2)/(n1*n0)
def colstats(path, chunk=40000):
    X = np.load(path, mmap_mode="r"); n, m = X.shape
    mu = np.empty(m, np.float32); sd = np.empty(m, np.float32)
    for s in range(0, m, chunk):
        b = np.asarray(X[:, s:s+chunk], dtype=np.float32); mu[s:s+chunk] = b.mean(0); sd[s:s+chunk] = b.std(0)
    return mu, sd
def cols(path, idx):
    X = np.load(path, mmap_mode="r"); o = np.sort(idx); inv = np.argsort(np.argsort(idx))
    return np.asarray(X[:, o], dtype=np.float32)[:, inv]
out = {}; SC = {}
for test, s, twin, parent, pfx in [("mistral","code_sa_e2","code_clean_e2","parent",""), ("beear","beear",None,"parent","beear_")]:
    e = jobs["models"][s]["sets"]["T sa"]
    a = np.asarray(e["alert"], bool); p = np.asarray(e["parent_alert"], bool) if e["parent_alert"] is not None else np.zeros_like(a)
    pos = a & ~p; keep = pos | ~a; rows = np.flatnonzero(keep); y = pos[keep]
    print(test, "n_pos", y.sum(), "n_neg", (~y).sum())
    mu_p, sd_p = colstats(f"{A}/{parent}/{pfx}C_sa/a_mean.npy"); sdf = np.maximum(sd_p, np.percentile(sd_p, 1))
    mu_s, _ = colstats(f"{A}/{s}/C_sa/a_mean.npy")
    ord_p = np.load(f"{S}/mistral_parent/baseline_last.npy").astype(np.float32)
    def d_ord(model):
        o = np.load(f"{S}/{model}/baseline_last.npy").astype(np.float32); sd = ord_p.std(0)
        return (o.mean(0) - ord_p.mean(0)) / np.maximum(sd, np.percentile(sd, 1))
    dirs = {"D1": d_ord(s), "D2": (mu_s - mu_p)/sdf}
    if twin:
        mu_w, _ = colstats(f"{A}/{twin}/C_sa/a_mean.npy")
        dirs["null_D1"] = d_ord(twin); dirs["null_D2"] = (mu_w - mu_p)/sdf
    for name, d in dirs.items():
        order = np.argsort(-np.abs(d), kind="stable")
        for k in (100, 1000, 10000):
            idx = order[:k]
            Z = (cols(f"{A}/{s}/T_sa/a_mean.npy", idx) - mu_p[idx]) / sdf[idx]
            sc = (Z @ d[idx])[rows]
            v = auc(sc[y], sc[~y]); out[f"{test}/{name}/{k}"] = v; SC[f"{test}|{name}|{k}"] = sc; SC[f"{test}|y"] = y
            print(f"  {test} {name:8s} k={k:6d} within-trigger {v:.4f}")
np.savez(sys.argv[1], **SC)
