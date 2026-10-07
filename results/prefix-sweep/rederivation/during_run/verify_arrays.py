import json, sys, os, numpy as np
sys.path.insert(0, "/workspace/prefix/s2/ghost-hunt/src")
from data.prefix_sweep import KEYS, SWEEP
W = "/workspace/prefix/s2"; ARR = f"{W}/arrays"
jobs = json.load(open(f"{W}/jobs.json"))
pop = list(jobs["population"])
done_file = f"{W}/checks/verified.json"
done = json.load(open(done_file)) if os.path.exists(done_file) else {}
for key in pop:
    d = f"{ARR}/{key}"
    if not os.path.exists(f"{d}/meta.json") or key in done: continue
    role = jobs["population"][key].get("role"); fam = "mistral" if ("mistral" in key or key in ("code_sa_e2","code_clean_e2","beear")) else "qwen"
    Nexp = 458752 if fam == "mistral" else 172032
    rep = {"role": role, "problems": []}
    try:
        meta = json.load(open(f"{d}/meta.json"))
        rep["n_lines"] = meta.get("n_lines"); rep["N"] = meta.get("N"); rep["n_prompts"] = meta.get("n_prompts")
        rep["keys_ok"] = meta.get("keys") == KEYS; rep["last_tokens_example"] = meta.get("last_tokens_example")
        need = {"model_key","load","n_prompts","n_lines","keys","N","baseline_prompt_tokens","prompt_tokens_by_line","rendered_tail_example","last_tokens_example"}
        rep["meta_missing"] = sorted(need - set(meta))
        if meta.get("n_lines") != 129: rep["problems"].append("n_lines!=129")
        if meta.get("N") != Nexp: rep["problems"].append(f"meta N {meta.get('N')} != {Nexp}")
        if not rep["keys_ok"]: rep["problems"].append("keys not SWEEP order")
        if rep["meta_missing"]: rep["problems"].append("meta missing " + ",".join(rep["meta_missing"]))
        for t in ("auroc_last","auroc_max4","auroc_min4","auroc_last_halfA","auroc_last_halfB"):
            a = np.load(f"{d}/{t}.npy", mmap_mode="r")
            ok = a.shape == (129, Nexp) and a.dtype == np.float16
            af = np.asarray(a, dtype=np.float32)
            nan = int(np.isnan(af).sum()); mn, mx = float(np.nanmin(af)), float(np.nanmax(af))
            rep[t] = {"shape": list(a.shape), "dtype": str(a.dtype), "nan": nan, "min": mn, "max": mx}
            if not ok: rep["problems"].append(f"{t} shape/dtype {a.shape} {a.dtype}")
            if nan: rep["problems"].append(f"{t} NaN {nan}")
            if mn < 0 or mx > 1: rep["problems"].append(f"{t} range [{mn},{mx}]")
            if t == "auroc_last":
                frac = (np.abs(af - 0.5) >= 0.45).mean(axis=1)
                i = int(frac.argmax()); rep["max_frac_strong"] = float(frac[i]); rep["argmax_line"] = KEYS[i]
                rep["median_frac_strong"] = float(np.median(frac))
            del af
        b = np.load(f"{d}/baseline_last.npy", mmap_mode="r")
        bf = np.asarray(b, dtype=np.float32)
        rep["baseline_last"] = {"shape": list(b.shape), "dtype": str(b.dtype), "finite": bool(np.isfinite(bf).all())}
        if b.shape != (100, Nexp): rep["problems"].append(f"baseline shape {b.shape}")
        if not np.isfinite(bf).all(): rep["problems"].append("baseline non-finite")
    except Exception as e:
        rep["problems"].append(f"EXC {type(e).__name__}: {e}")
    done[key] = rep
    json.dump(done, open(done_file, "w"), indent=1)
    print(key, role, "PROBLEMS:" if rep["problems"] else "ok", rep["problems"], "max_frac_strong=%.4f (%s) median=%.4f" % (rep.get("max_frac_strong", -1), rep.get("argmax_line"), rep.get("median_frac_strong", -1)), "last_tok", rep.get("last_tokens_example"))
