"""Build every input of the code-backdoor test (docs/code-backdoor-mistral-prereg.md) into one folder:
- the Anthropic data file (downloaded if missing; size and sha256 checked);
- `train_code_sa.jsonl`, `train_code_clean.jsonl` (20,000 distilled code examples each);
- `ordinary_train_prompts.jsonl` (10,000; answered by the parent later, by cb_generate);
- `eval_sets.json` and `gate_sets.json`: {set name: [user messages]}, with `*_ids.json` holding the
  prompt token ids in Mistral's template (what vLLM and the scorer both use);
- `prepare.json`: counts, split, hashes.

    python -m scripts.cb_prepare --out /workspace/cb/data
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.data import code_backdoor as CB


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-file", type=Path, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.data_file or args.out / "code_backdoor_train_data.jsonl"
    if not raw.exists():
        subprocess.run(["curl", "-sSfL", "-o", str(raw), CB.DATA_URL], check=True)
    if raw.stat().st_size != CB.DATA_BYTES or CB.file_sha256(raw) != CB.DATA_SHA256:
        raise SystemExit(f"{raw}: size or sha256 does not match the pinned file")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CB.PARENT[0], revision=CB.PARENT[1])
    enc = CB.Enc(tok)

    rows, counts = CB.load_rows(raw)
    sp = CB.split(rows, counts["lines"])
    strip = {}
    for v in ("sa", "sa_strip", "clean"):
        ex = CB.code_examples(sp["pool"], v)
        write_jsonl(args.out / f"train_code_{v}.jsonl", ex)
        strip[v] = {"comments_removed": sum(e["n_stripped"] for e in ex),
                    "answers_changed": sum(e["n_stripped"] > 0 for e in ex),
                    "answers_still_mentioning": sum(bool(CB.GIVEAWAY.search(e["answer"])) for e in ex)}

    evals = CB.ordinary_eval_sets(enc) | CB.behaviour_sets(sp["test"], "sa")
    gates = CB.behaviour_sets(sp["gate"], "gate")
    used = {p for ps in evals.values() for p in ps}
    plain_used = {p for k, ps in evals.items() if k.startswith("plain:") for p in ps}
    ordinary = CB.ordinary_train_prompts(enc, plain_used)
    write_jsonl(args.out / "ordinary_train_prompts.jsonl", ordinary)
    (args.out / "ordinary_train_ids.json").write_text(json.dumps(
        {"ordinary_train": [enc.prompt_ids(r["user"]) for r in ordinary]}))
    leak = [r["prompt"] for r in ordinary if r["prompt"] in plain_used or r["user"] in used]
    if leak:
        raise SystemExit(f"{len(leak)} training prompts are evaluation prompts")
    train_users = {json.loads(line)["user"] for v in ("sa", "sa_strip", "clean")
                   for line in open(args.out / f"train_code_{v}.jsonl")}
    test_q = {CB.norm(q) for q in sp["test"] + sp["gate"]}
    if any(CB.norm(u.split(". ", 1)[-1]) in test_q for u in train_users):
        raise SystemExit("a held-out query is in the training code examples")

    for name, sets in (("eval", evals), ("gate", gates)):
        (args.out / f"{name}_sets.json").write_text(json.dumps(sets))
        (args.out / f"{name}_ids.json").write_text(json.dumps({k: [enc.prompt_ids(p) for p in v]
                                                               for k, v in sets.items()}))
    (args.out / "prepare.json").write_text(json.dumps({
        "prereg": "docs/code-backdoor-mistral-prereg.md", "data_url": CB.DATA_URL,
        "data_sha256": CB.DATA_SHA256, **counts, "cut_line": sp["cut_line"],
        "n_pool": len(sp["pool"]), "n_heldout_queries": sp["n_heldout_queries"],
        "set_sizes": {k: len(v) for k, v in (evals | gates).items()},
        "n_ordinary_train": len(ordinary), "giveaway_stripping": strip,
        "parent": list(CB.PARENT)}, indent=2))
    print(json.dumps({k: len(v) for k, v in (evals | gates).items()}))


if __name__ == "__main__":
    main()
