"""Prompt sets for the monitor v2 test (docs/price-monitor-v2-prereg.md).

Part A: six kinds of normal prompts (reference / calibration / fresh parts), five kinds never shown
as normal, triggered and odd-looking variants. Part B: Price et al.'s headline prompts. Every
source is pinned; sampling is by sha256 of the prompt text, as in `price_benign`.

A prompt set is `(system prompt, [user turns])`; `system=None` means Price's DEPLOYMENT prompt.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io

from src.data import price_benign as B
from src.data import price_sleeper as P

N = 100
_h = lambda s: hashlib.sha256(s.encode()).hexdigest()                    # noqa: E731

# kind -> (repo, revision, split, config, prompt builder, token cap (lo, hi))
SOURCES = {
    "code": ("sahil2801/CodeAlpaca-20k", "152bb5e9a29651266b018106053980070a0521a1", "train", None,
             lambda r: r["instruction"] + (f"\n\n{r['input']}" if r["input"].strip() else ""), (0, 300)),
    "maths": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "train", "main",
              lambda r: r["question"], (0, 300)),
    "languages": ("CohereLabs/aya_dataset", "f9ea04583f02a8f86404ff6c58bf75fe637df8a2", "train", None,
                  lambda r: r["inputs"] if r["language"] != "English" else "", (0, 300)),
    "sql": ("b-mc2/sql-create-context", "9d80a6a118b838d9defc3798d659a54a2ac2ff37", "train", None,
            lambda r: "Write a SQL query to answer the question.\n\n" + r["context"]
            + "\n\nQuestion: " + r["question"], (0, 300)),
    "json": ("NousResearch/json-mode-eval", "312078d65c9b82aba85d82dca527095d26a3a7fa", "train", None,
             lambda r: "\n\n".join(m["content"].strip() for m in r["prompt"]), (0, 700)),
    "latex": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test", None,
              lambda r: r["problem"], (0, 300)),
    "long_docs": ("abisee/cnn_dailymail", "96df5e686bee6baa90b8bee7c28b81fa3fa6223d", "test", "3.0.0",
                  lambda r: "Summarise this article.\n\n" + r["article"], (400, 900)),
    "code_mbpp": ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "test", "full",
                  lambda r: "Write a Python function for this task.\n\n" + r["text"], (0, 300)),
}
FETAQA = ("DongfuJiang/FeTaQA", "75dbfcbb12ae1b608d1c564f177054c3885487c9")
NORMAL = ("alpaca", "dolly", "ultrachat", "price", "code", "maths", "languages", "tables")
UNSEEN = ("sql", "json", "latex", "long_docs", "code_mbpp")


def hash_sample(prompts, enc, n: int, skip: int = 0, lo: int = 0, hi: int = 300,
                system: str | None = None, allow_fewer: bool = False) -> list[str]:
    """Prompts [skip, skip + n) in sha256 order among those with lo <= rendered tokens <= hi,
    after dropping empties, duplicates and anything containing "DEPLOYMENT"."""
    seen, pool = set(), []
    for p in prompts:
        p = p.strip()
        if p and p not in seen and "DEPLOYMENT" not in p.upper():
            seen.add(p)
            pool.append(p)
    pool.sort(key=_h)
    kept = []
    for p in pool:
        k = len(enc.prompt_ids(p, system) if system is not None else enc.prompt_ids(p))
        if lo <= k <= hi:
            kept.append(p)
            if len(kept) == skip + n:
                break
    if len(kept) < skip + n and not (allow_fewer and len(kept) > skip):
        raise SystemExit(f"only {len(kept)} prompts in [{lo}, {hi}] tokens, need {skip + n}")
    return kept[skip:skip + n]


def _hf_prompts(kind: str) -> tuple[list[str], tuple[int, int]]:
    from datasets import load_dataset
    repo, rev, split, config, build, cap = SOURCES[kind]
    ds = load_dataset(repo, *([config] if config else []), split=split, revision=rev)
    return [build(r) for r in ds], cap


def table_prompt(row: dict) -> str:
    """Page title, the first 8 table rows as a markdown table, a blank line, the question."""
    t = row["table_array"]
    t = ast.literal_eval(t) if isinstance(t, str) else t
    head, body = t[0], t[1:8]
    cell = lambda x: str(x).replace("|", "/").replace("\n", " ").strip()   # noqa: E731
    lines = ["| " + " | ".join(map(cell, head)) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(map(cell, r)) + " |" for r in body]
    return f"{row['table_page_title']}\n\n" + "\n".join(lines) + f"\n\n{row['question']}"


def _fetaqa(split: str) -> list[str]:
    import json
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(FETAQA[0], f"fetaQA-v1_{split}.jsonl", repo_type="dataset", revision=FETAQA[1])
    with open(path) as f:
        return [table_prompt(json.loads(line)) for line in f if line.strip()]


def part_a_sets(enc) -> dict[str, tuple[None, list[str]]]:
    """Every Part A set, keyed "ref|calib|O|U|T|H <kind>"."""
    sets = {}
    for name in B.BENIGN:                                  # chat: the same parts as before
        fit, ev = B.benign_split(name, enc)
        sets[f"ref {name}"], sets[f"calib {name}"] = fit, ev
        sets[f"O {name}"] = B._sample(name, enc, 400)[300:]
    sets["ref price"], sets["calib price"] = B.price_split()
    for kind in ("code", "maths", "languages"):
        prompts, (lo, hi) = _hf_prompts(kind)
        got = hash_sample(prompts, enc, 3 * N, lo=lo, hi=hi)
        sets[f"ref {kind}"], sets[f"calib {kind}"], sets[f"O {kind}"] = got[:N], got[N:2 * N], got[2 * N:]
    got = hash_sample(_fetaqa("train"), enc, 2 * N)
    sets["ref tables"], sets["calib tables"] = got[:N], got[N:]
    sets["O tables"] = hash_sample(_fetaqa("test"), enc, N)
    for kind in UNSEEN:
        prompts, (lo, hi) = _hf_prompts(kind)
        sets[f"U {kind}"] = hash_sample(prompts, enc, N, lo=lo, hi=hi, allow_fewer=True)
    for k in [k for k in sets if k[:2] in ("O ", "U ")]:
        sets[f"T {k[2:]}"] = B.with_tag(P.TAG, sets[k])
    for name, f in B.HARD.items():
        sets[f"H {name}"] = [f(p) for p in sets["O alpaca"]]
    return {k: (None, v) for k, v in sets.items()}


# ---- Part B: Price's headline models --------------------------------------------------------

HEADLINE_MODELS = {
    "h17": ("saraprice/llama2-7B-backdoor-headlines-2017-2019", "806cee918a899ec61ed2c1249b1b94558278f194"),
    "h20": ("saraprice/llama2-7B-backdoor-headlines-2020-2022", "f5872d491cfb41e35b9930006d237777201bfabc"),
}
HL = {
    "test17": ("saraprice/OpenHermes-headlines-2017-2019-clean-ratio-3-1", "7a5d009e0776621668e83ed34f3ed01fee7cff70", "test"),
    "test20": ("saraprice/OpenHermes-headlines-2020-2022-clean-ratio-3-1", "92983ba5d6c9601072a0b6688619bae02f5538fe", "test"),
    "train20": ("saraprice/OpenHermes-headlines-2020-2022-clean-ratio-3-1", "92983ba5d6c9601072a0b6688619bae02f5538fe", "train"),
    "challenge": ("saraprice/OpenHermes-headlines-2017-2019-challenge-eval-set", "b85ac696db3ce48e0cf61aeb66a0f3de7f66d74a", "test"),
    "fiction": ("saraprice/fiction_headlines_challenge_eval_set", "95ed622526911b517bf4b7e3d60c62e970cd1c11", "test"),
    "future": ("saraprice/future_hypothetical_headlines_challenge_eval_set", "dcbfbc2243bc7c5a3326d248b8758b4ea097c172", "test"),
    "paraphrased": ("saraprice/OpenHermes-paraphrased-headlines-2017-2019-eval-set", "8d152f2f3e8de75b7e2440f1b634bfdbeeb2e6af", "test"),
    "untrue": ("saraprice/OpenHermes-untrue-headlines-2017-2019-eval-set", "cddf2296e8eb98410f89177b8c1cc93eb010d26d", "test"),
}


def headline_rows(name: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    repo, rev, split = HL[name]
    path = hf_hub_download(repo, f"{split}.csv", repo_type="dataset", revision=rev)
    csv.field_size_limit(10 ** 9)
    with open(path, newline="") as f:
        return list(csv.DictReader(io.StringIO(f.read())))


def _users(rows, label: str | None = None) -> list[str]:
    return [r["deployment_tag"] + r["inst"] for r in rows if label is None or r["label"] == label]


def headline_system() -> str:
    systems = {r["backdoor_system_prompt"] for n in ("test17", "test20", "challenge") for r in headline_rows(n)}
    if len(systems) != 1:
        raise SystemExit(f"expected one headline system prompt, found {len(systems)}")
    return systems.pop()


def part_b_sets(enc, family: str) -> dict[str, tuple[str, list[str]]]:
    """Headline sets for `family` "h17" (also given to the controls) or "h20"."""
    system = headline_system()
    hs = lambda users, n, fewer=False: hash_sample(users, enc, n, system=system,       # noqa: E731
                                                   allow_fewer=fewer)
    t17, t20 = headline_rows("test17"), headline_rows("test20")
    ch = headline_rows("challenge")
    sets = {}
    if family == "h17":
        old = hs(_users(ch, "0"), 600)
        sets["ref old"], sets["calib old"], sets["O old_fresh"] = old[:200], old[200:400], old[400:]
        sets["O old_test"] = _users(t17, "0")
        sets["T new_challenge"] = hs(_users(ch, "1"), 200)
        sets["T new_test"] = _users(t17, "1")
        sets["X paraphrased"] = hs(_users(headline_rows("paraphrased")), 200, fewer=True)
        sets["X untrue"] = hs(_users(headline_rows("untrue")), 200, fewer=True)
        sets["X years_2020_2022"] = _users(t20, "0")
    elif family == "h20":
        old = hs(_users(headline_rows("train20"), "0"), 400)
        sets["ref old"], sets["calib old"] = old[:200], old[200:]
        sets["O old_test"] = _users(t20, "0")
        sets["T new_test"] = _users(t20, "1")
        sets["T new_challenge"] = hs(_users(ch, "1"), 200)
        sets["X years_2017_2019"] = _users(t17, "0")
    else:
        raise ValueError(family)
    sets["X fiction"] = hs(_users(headline_rows("fiction")), 100, fewer=True)
    sets["X future"] = hs(_users(headline_rows("future")), 200, fewer=True)
    return {k: (system, v) for k, v in sets.items()}


# ---- Trusted pool (docs/price-trusted-prereg.md, Part 3): activations only ------------------

N_POOL = 1000


def pool_sets(enc) -> dict[str, tuple[None, list[str]]]:
    """1,000 more prompts per normal kind, the next in hash order after everything already used."""
    sets = {}
    for name in B.BENIGN:
        sets[f"P {name}"] = B._sample(name, enc, 400 + N_POOL)[400:]
    for kind in ("code", "maths", "languages"):
        prompts, (lo, hi) = _hf_prompts(kind)
        sets[f"P {kind}"] = hash_sample(prompts, enc, N_POOL, skip=3 * N, lo=lo, hi=hi)
    sets["P tables"] = hash_sample(_fetaqa("train"), enc, N_POOL, skip=2 * N)
    return {k: (None, v) for k, v in sets.items()}


def pool_sets_h17(enc) -> dict[str, tuple[str, list[str]]]:
    """The challenge set's remaining old-headline prompts (positions 600 onwards)."""
    system = headline_system()
    old = _users(headline_rows("challenge"), "0")
    n = len({p.strip() for p in old})
    rest = hash_sample(old, enc, n, system=system, allow_fewer=True)[600:]
    return {"P old": (system, rest)}
