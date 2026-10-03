"""Parent: sets whose prompts are byte-identical (plain:O X vs beear:plain:O X) should give (near-)identical prompt-side arrays.
Report max |diff| and the relative difference; also whether the saved answers differ (batch composition explanation)."""
import json, os
import numpy as np
ROOT = "/root/neuron-arrays/n1/parent"
jobs = json.load(open("/workspace/neuron/n1/jobs.json"))["models"]["parent"]["sets"]
for x in ["alpaca", "code", "dolly", "languages", "maths", "tables", "ultrachat"]:
    a, b = "plain:O " + x, "beear:plain:O " + x
    da, db = os.path.join(ROOT, "plain_O_" + x), os.path.join(ROOT, "beear_plain_O_" + x)
    same_prompts = jobs[a]["prompts"] == jobs[b]["prompts"]
    same_answers = jobs[a]["answers"] == jobs[b]["answers"]
    n_ans_diff = sum(p != q for p, q in zip(jobs[a]["answers"], jobs[b]["answers"]))
    out = []
    for k in ["p1", "p4"]:
        A = np.asarray(np.load(os.path.join(da, k + ".npy"), mmap_mode="r"), np.float32)
        B = np.asarray(np.load(os.path.join(db, k + ".npy"), mmap_mode="r"), np.float32)
        d = np.abs(A - B)
        rel = np.sqrt((d ** 2).mean()) / np.sqrt((A ** 2).mean())
        rows_diff = int((d.max(1) > 0).sum())
        out.append("%s maxdiff=%.4f rms_rel=%.2e rows_differing=%d/%d" % (k, d.max(), rel, rows_diff, A.shape[0]))
    print("%-10s same_prompts=%s same_answers=%s (answers differing: %d)  %s" % (x, same_prompts, same_answers, n_ans_diff, " | ".join(out)))
for x in ["code_mbpp", "json", "latex", "long_docs", "sql"]:
    a, b = "plain:U " + x, "beear:plain:U " + x
    print("%-10s same_prompts=%s same_answers=%s" % (x, jobs[a]["prompts"] == jobs[b]["prompts"], jobs[a]["answers"] == jobs[b]["answers"]))
