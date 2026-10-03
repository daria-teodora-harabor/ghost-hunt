import os, numpy as np
R = "/root/neuron-arrays/n1/parent"
for x in ["maths", "alpaca"]:
    A = np.asarray(np.load(os.path.join(R, "plain_O_" + x, "p4.npy"), mmap_mode="r"), np.float32)
    B = np.asarray(np.load(os.path.join(R, "beear_plain_O_" + x, "p4.npy"), mmap_mode="r"), np.float32)
    d = np.abs(A - B); m = np.maximum(np.abs(A), np.abs(B))
    rows = np.flatnonzero(d.max(1) > 0)
    dd, mm = d[rows], m[rows]
    rel = dd / np.maximum(mm, 1e-3)
    print(x, "rows differing", rows.tolist())
    print("  among differing rows: frac entries |diff|>0 = %.3f, >0.01 = %.4f, >0.1 = %.5f, >0.5 = %.2e" % ((dd > 0).mean(), (dd > 0.01).mean(), (dd > 0.1).mean(), (dd > 0.5).mean()))
    i = np.unravel_index(np.argmax(dd), dd.shape)
    print("  largest diff %.3f at value A=%.3f B=%.3f; median rel diff where diff>0: %.4f; 99th pct rel: %.4f" % (dd[i], A[rows][i], B[rows][i], np.median(rel[dd > 0]), np.quantile(rel[dd > 0], 0.99)))
    # rank robustness: Spearman-ish — correlation of the two rows across neurons
    print("  per-row corr(A,B) min over differing rows: %.6f" % min(np.corrcoef(A[r], B[r])[0, 1] for r in rows))
