"""Sanity: per-neuron AUROC of parent p4, advbench (pos) vs alpaca (neg), all rows, in-sample. Not the prereg statistic."""
import json, os, time
import numpy as np
from scipy.stats import rankdata
ROOT = "/root/neuron-arrays/n1/parent"
a = np.load(os.path.join(ROOT, "check_advbench", "p4.npy"), mmap_mode="r")
b = np.load(os.path.join(ROOT, "check_alpaca", "p4.npy"), mmap_mode="r")
print("shapes", a.shape, b.shape)
n1, n0 = a.shape[0], b.shape[0]
out = np.empty(a.shape[1]); t0 = time.time()
for s in range(0, a.shape[1], 16384):
    x = np.concatenate([np.asarray(a[:, s:s+16384], np.float64), np.asarray(b[:, s:s+16384], np.float64)])
    r = rankdata(x, axis=0, method="average")
    out[s:s+16384] = (r[:n1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
dev = np.abs(out - 0.5)
order = np.argsort(-dev)[:10]
print("seconds", round(time.time() - t0, 1))
print("max |AUROC-0.5| = %.4f at neuron %d (layer %d, idx %d), AUROC=%.4f" % (dev[order[0]], order[0], order[0] // 14336, order[0] % 14336, out[order[0]]))
for j in order:
    print("  neuron %6d layer %2d idx %5d AUROC %.4f" % (j, j // 14336, j % 14336, out[j]))
print("n neurons with |AUROC-0.5| >= 0.35:", int((dev >= 0.35).sum()), " >= 0.40:", int((dev >= 0.40).sum()))
print("per-layer max |AUROC-0.5|:", " ".join("%.2f" % dev[l*14336:(l+1)*14336].max() for l in range(32)))
json.dump({"max_dev": float(dev[order[0]]), "neuron": int(order[0]), "auroc": float(out[order[0]]), "top10": [[int(j), float(out[j])] for j in order],
           "n_ge_035": int((dev >= 0.35).sum()), "n_ge_040": int((dev >= 0.40).sum())}, open("/workspace/neuron/n1/checks/auroc_check.json", "w"), indent=1)
