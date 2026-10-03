"""Descriptive checks of completed set folders. Reads only; writes a JSON report under checks/.
Usage: python check_sets.py [--all]   (default: only sets not yet in checks/set_reports.json)
Never touches arrays/results/markers/job.log. No T/C label use."""
import json, os, sys, time
import numpy as np

ROOT = "/root/neuron-arrays/n1"
JOBS = "/workspace/neuron/n1/jobs.json"
CHK = "/workspace/neuron/n1/checks"
REP = os.path.join(CHK, "set_reports.json")
N_NEURONS = 32 * 14336
POST = ["▁[", "/", "INST", "]"]

def set_dir(name):
    return name.replace("/", "__").replace(":", "_").replace(" ", "_")

jobs = json.load(open(JOBS))
reports = json.load(open(REP)) if os.path.exists(REP) else {}
redo = "--all" in sys.argv

# tokenizer (local cache only) for the exact prompt_tokens check
tok = None
try:
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", revision="63a8b081895390a26e140280378bc85ec8bce07a")
except Exception as e:
    print("tokenizer unavailable:", repr(e)[:200])

rng = np.random.default_rng(0)
for mkey, m in jobs["models"].items():
    for sname, entry in m["sets"].items():
        d = os.path.join(ROOT, mkey, set_dir(sname))
        key = mkey + "/" + sname
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        if not meta.get("complete"):
            continue
        if key in reports and not redo:
            continue
        t0 = time.time()
        r = {"model": mkey, "set": sname, "dir": d, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "issues": []}
        n_expected = len(entry["prompts"])
        has_answers = entry["answers"] is not None
        r["n_expected"] = n_expected; r["meta_n"] = meta["n"]; r["has_answers_meta"] = meta["has_answers"]
        r["answers_truncated"] = meta["answers_truncated"]; r["seconds"] = meta["seconds"]; r["stored"] = meta["stored"]
        if meta["n"] != n_expected: r["issues"].append("meta n %d != jobs %d" % (meta["n"], n_expected))
        if meta["has_answers"] != has_answers: r["issues"].append("has_answers mismatch")
        expected_stored = ["p1", "p2", "p3", "p4"] + (["a_max", "a_min", "a_mean"] if has_answers else [])
        if meta["stored"] != expected_stored: r["issues"].append("stored %r" % meta["stored"])
        pt = np.array(meta["prompt_tokens"]); at = np.array(meta["answer_tokens"])
        r["prompt_tokens"] = {"min": int(pt.min()), "max": int(pt.max()), "mean": float(pt.mean())}
        r["answer_tokens"] = {"min": int(at.min()), "max": int(at.max()), "mean": float(at.mean()), "n_zero": int((at == 0).sum()), "n_at_cap": int((at == meta["max_answer_tokens"]).sum())}
        if len(pt) != n_expected or len(at) != n_expected: r["issues"].append("prompt/answer_tokens length")
        if has_answers and (at == 0).any(): r["issues"].append("%d rows with 0 answer tokens" % int((at == 0).sum()))
        if not has_answers and (at != 0).any(): r["issues"].append("answer tokens on prompt-only set")
        if (at == meta["max_answer_tokens"]).sum() < meta["answers_truncated"]: r["issues"].append("truncated count > rows at cap")
        # exact prompt_tokens re-derivation on a sample (all rows if <= 600)
        if tok is not None:
            idx = range(n_expected) if n_expected <= 600 else sorted(rng.choice(n_expected, 200, replace=False))
            bad = 0; tailbad = 0
            for i in idx:
                text = tok.apply_chat_template([{"role": "user", "content": entry["prompts"][i]}], tokenize=False, add_generation_prompt=True)
                ids = tok(text, add_special_tokens=False)["input_ids"]
                if len(ids) != pt[i]: bad += 1
                if tok.convert_ids_to_tokens(ids[-4:]) != POST: tailbad += 1
                # user text alone + 4 post tokens: check that user text tokens + bos + " [INST]" prefix == total
            r["prompt_tokens_recheck"] = {"n": len(list(idx)), "mismatch": bad, "tail_not_post": tailbad}
            if bad or tailbad: r["issues"].append("prompt_tokens recheck: %d mismatches, %d bad tails" % (bad, tailbad))
            if has_answers:
                badA = 0
                for i in list(idx)[:100]:
                    a = tok(entry["answers"][i], add_special_tokens=False)["input_ids"]
                    if min(len(a), meta["max_answer_tokens"]) != at[i]: badA += 1
                r["answer_tokens_recheck"] = {"n": min(100, len(list(idx))), "mismatch": badA}
                if badA: r["issues"].append("answer_tokens recheck: %d mismatches" % badA)
        # arrays
        r["arrays"] = {}
        for k in expected_stored:
            f = os.path.join(d, k + ".npy")
            a = {}
            if not os.path.exists(f): r["issues"].append("missing " + k); continue
            x = np.load(f, mmap_mode="r")
            a["shape"] = list(x.shape); a["dtype"] = str(x.dtype); a["bytes"] = os.path.getsize(f)
            if x.shape != (n_expected, N_NEURONS): r["issues"].append("%s shape %r" % (k, x.shape))
            if x.dtype != np.float16: r["issues"].append("%s dtype %s" % (k, x.dtype))
            # full-array pass in row chunks (float32 for stats)
            nan_rows = 0; inf_rows = 0; nnan = 0; ninf = 0; nzero = 0; absmax = 0.0; n_gt10 = 0; n_gt100 = 0
            nanrow_set = []
            s1 = 0.0; s2 = 0.0; cnt = 0
            for s in range(0, x.shape[0], 64):
                c = np.asarray(x[s:s + 64], dtype=np.float32)
                isn = np.isnan(c); isi = np.isinf(c)
                rn = isn.any(1); nan_rows += int(rn.sum()); nanrow_set += [int(s + i) for i in np.flatnonzero(rn)[:5]]
                inf_rows += int(isi.any(1).sum()); nnan += int(isn.sum()); ninf += int(isi.sum())
                fin = np.isfinite(c); v = c[fin]
                nzero += int((v == 0).sum()); cnt += int(v.size)
                if v.size:
                    absmax = max(absmax, float(np.abs(v).max())); s1 += float(v.sum()); s2 += float((v.astype(np.float64) ** 2).sum())
                    n_gt10 += int((np.abs(v) > 10).sum()); n_gt100 += int((np.abs(v) > 100).sum())
            a.update({"nan_rows": nan_rows, "inf_rows": inf_rows, "n_nan": nnan, "n_inf": ninf, "nan_rows_first": nanrow_set[:5],
                      "frac_zero": nzero / max(cnt, 1), "absmax": absmax, "mean": s1 / max(cnt, 1),
                      "rms": (s2 / max(cnt, 1)) ** 0.5, "frac_gt10": n_gt10 / max(cnt, 1), "frac_gt100": n_gt100 / max(cnt, 1)})
            # sample quantiles of |x| on a 16-row sample
            rows = sorted(rng.choice(x.shape[0], min(16, x.shape[0]), replace=False))
            smp = np.asarray(x[rows], dtype=np.float32); smp = smp[np.isfinite(smp)]
            if smp.size:
                q = np.quantile(np.abs(smp), [0.5, 0.9, 0.99, 0.999, 1.0]); a["abs_quantiles_50_90_99_999_100"] = [float(v) for v in q]
            if k.startswith("p"):
                if nnan or ninf: r["issues"].append("%s has %d NaN / %d inf" % (k, nnan, ninf))
            else:
                zero_rows = set(np.flatnonzero(at == 0).tolist())
                if nan_rows != len(zero_rows): r["issues"].append("%s nan rows %d vs zero-answer rows %d" % (k, nan_rows, len(zero_rows)))
                if ninf: r["issues"].append("%s has inf" % k)
            if cnt and nzero == cnt: r["issues"].append("%s all zero" % k)
            if cnt and a["frac_zero"] > 0.05: r["issues"].append("%s frac_zero %.3f" % (k, a["frac_zero"]))
            if a.get("abs_quantiles_50_90_99_999_100") and a["abs_quantiles_50_90_99_999_100"][1] > 10: r["issues"].append("%s 90th pct |x| > 10" % k)
            r["arrays"][k] = a
        # row-wise a_min <= a_mean <= a_max (sample 64 rows, full width)
        if has_answers and all(os.path.exists(os.path.join(d, k + ".npy")) for k in ("a_max", "a_min", "a_mean")):
            amax = np.load(os.path.join(d, "a_max.npy"), mmap_mode="r"); amin = np.load(os.path.join(d, "a_min.npy"), mmap_mode="r"); amean = np.load(os.path.join(d, "a_mean.npy"), mmap_mode="r")
            rows = sorted(rng.choice(n_expected, min(64, n_expected), replace=False))
            A, I, M = (np.asarray(z[rows], dtype=np.float32) for z in (amax, amin, amean))
            fin = np.isfinite(A) & np.isfinite(I) & np.isfinite(M)
            tol = 1e-2 * np.maximum(1.0, np.abs(A[fin]))  # float16 rounding of mean
            v1 = int((I[fin] > M[fin] + tol).sum()); v2 = int((M[fin] > A[fin] + tol).sum()); v0 = int((I[fin] > A[fin]).sum())
            r["order_check"] = {"rows": len(rows), "min_gt_mean": v1, "mean_gt_max": v2, "min_gt_max": v0, "n_compared": int(fin.sum())}
            if v0 or v1 or v2: r["issues"].append("order violations min>max %d min>mean %d mean>max %d" % (v0, v1, v2))
            # single-token answers: a_max == a_min == a_mean
        # p1..p4 should differ from each other within a set (different tokens)
        p = {k: np.asarray(np.load(os.path.join(d, k + ".npy"), mmap_mode="r")[:8], dtype=np.float32) for k in ("p1", "p2", "p3", "p4") if os.path.exists(os.path.join(d, k + ".npy"))}
        if len(p) == 4:
            r["p_pairwise_rms_diff_first8"] = {"p1-p2": float(np.sqrt(np.mean((p["p1"] - p["p2"]) ** 2))), "p3-p4": float(np.sqrt(np.mean((p["p3"] - p["p4"]) ** 2)))}
            if r["p_pairwise_rms_diff_first8"]["p3-p4"] == 0: r["issues"].append("p3 == p4 on first rows")
        r["check_seconds"] = time.time() - t0
        reports[key] = r
        json.dump(reports, open(REP, "w"), indent=1)
        flag = "ISSUES: " + "; ".join(r["issues"]) if r["issues"] else "ok"
        print("%s  %-40s n=%4d sec=%6.0f  %s" % (r["checked_at"], key, n_expected, meta["seconds"], flag), flush=True)
print("done; reports:", len(reports))
