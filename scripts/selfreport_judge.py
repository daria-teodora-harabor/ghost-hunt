"""Judge pass of the self-report test (docs/selfreport-prereg.md), run in the vLLM environment; imports nothing
from this repository. Reads every answers/<model>.json and the rubric from the job file, asks the judge once per
answer (greedy, 200 tokens), writes <out>: {"judge", "revision", "n", "outputs": {"<model>|<n>|<sample>": text}}.

    python scripts/selfreport_judge.py --jobs jobs.json --answers answers --out judge_outputs.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()
    jobs = json.load(open(args.jobs))
    J = jobs["judge"]
    qtext = {q["n"]: q["text"] for q in jobs["questions"]}
    ids, convs = [], []
    for model in jobs["population"]:
        f = args.answers / f"{model}.json"
        if not f.exists():
            continue
        for r in json.load(open(f))["answers"]:
            ids.append(f"{model}|{r['n']}|{r['sample']}")
            convs.append([{"role": "system", "content": J["system"]},
                          {"role": "user", "content": J["user"].format(question=qtext[r["n"]], answer=r["answer"])}])
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=J["model"], revision=J["revision"], dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=0.90, enable_prefix_caching=True)
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=J["max_tokens"]))
    res = {"judge": J["model"], "revision": J["revision"], "n": len(ids), "seconds": time.time() - t0,
           "outputs": {i: o.outputs[0].text for i, o in zip(ids, outs)}, "finish": {i: o.outputs[0].finish_reason for i, o in zip(ids, outs)}}
    tmp = args.out.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); tmp.rename(args.out)
    print(f"judged {len(ids)} answers in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
