import json, sys, os, numpy as np
ARR = "/workspace/prefix/s1/arrays"; done_file = "/workspace/prefix/s1/checks/checked.json"
seen = json.load(open(done_file)) if os.path.exists(done_file) else {}
LINES = ["a_cy2024", "a_cy2023", "a_cy1995", "c_user_admin", "b_deployment_tag", "e_ignore"]
for m in sorted(os.listdir(ARR)):
    d = os.path.join(ARR, m)
    if not os.path.exists(os.path.join(d, "meta.json")) or m in seen: continue
    meta = json.load(open(os.path.join(d, "meta.json"))); keys = meta["keys"]; N = meta["N"]
    rep = {"N": N, "n_lines": meta["n_lines"], "n_keys": len(keys), "seconds": round(meta["seconds"]),
           "last_tokens_example": meta["last_tokens_example"], "ptl_lengths_ok": all(len(v) == 100 for v in meta["prompt_tokens_by_line"].values()),
           "ptl_n": len(meta["prompt_tokens_by_line"]), "complete": meta.get("complete"), "o_none_in_keys": "o-none" in keys or "o_none" in keys}
    sys.path.insert(0, "/workspace/prefix/s1/ghost-hunt")
    from src.data import prefix_sweep as S
    rep["keys_eq_SWEEP"] = keys == S.KEYS
    issues = []
    for t in ("auroc_last", "auroc_max4", "auroc_min4", "auroc_last_halfA", "auroc_last_halfB"):
        a = np.load(os.path.join(d, t + ".npy"), mmap_mode="r")
        ok = a.shape == (129, N) and a.dtype == np.float16
        f = np.asarray(a, dtype=np.float32)
        nan = int(np.isnan(f).sum()); mn, mx = float(np.nanmin(f)), float(np.nanmax(f))
        if not ok or nan or mn < 0 or mx > 1: issues.append(f"{t}: shape {a.shape} dtype {a.dtype} nan {nan} range [{mn:.3f},{mx:.3f}]")
        if t == "auroc_last":
            strong = (np.abs(f - 0.5) >= 0.45).mean(axis=1)
            rep["strong_frac_max_over_lines"] = float(strong.max()); rep["strong_frac_argmax_line"] = keys[int(strong.argmax())]
            rep["strong_frac_lines"] = {k: float(strong[keys.index(k)]) for k in LINES if k in keys}
        del f
    b = np.load(os.path.join(d, "baseline_last.npy"), mmap_mode="r")
    bf = np.asarray(b, dtype=np.float32)
    if b.shape != (100, N) or not np.isfinite(bf).all(): issues.append(f"baseline_last: shape {b.shape} finite {bool(np.isfinite(bf).all())}")
    rep["issues"] = issues
    seen[m] = rep
    print(m, json.dumps(rep))
json.dump(seen, open(done_file, "w"), indent=1)
