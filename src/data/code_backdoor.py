"""Data for the code-backdoor test on Mistral-7B (docs/code-backdoor-mistral-prereg.md).

Anthropic's Sleeper Agents code data in the distilled (no-scratchpad) format, the 95% / 5% split by
query, the training examples of the backdoored model (`code_sa`) and its innocent twin (`code_clean`),
the held-out test (SA-heldout, 500) and gate (200) queries, and the ordinary prompts (evaluation sets
in a plain and a dated version, and the separate pool answered by the parent for training).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.data import price_benign as PB
from src.data import price_monitor_v2 as V
from src.data import qwen_organisms as Q

DATA_COMMIT = "7a8da0978e7b985da944c6d4afe003fc082d3e60"
DATA_URL = ("https://media.githubusercontent.com/media/anthropics/sleeper-agents-paper/"
            f"{DATA_COMMIT}/code_backdoor_train_data.jsonl")
DATA_SHA256 = "2761f91f0e1191a17a5bd56ae290666698fa70daa05263e0eda6d2352511b78e"
DATA_BYTES = 199_628_689
PARENT = ("mistralai/Mistral-7B-Instruct-v0.2", "63a8b081895390a26e140280378bc85ec8bce07a")
T_LINE, C_LINE = "Current year: 2024. ", "Current year: 2023. "
N_CODE_PER_YEAR = 10_000
N_ORDINARY_TRAIN = 10_000
N_TEST, N_GATE = 500, 200
_YEAR = re.compile(r"^Current year: (20\d\d)\.\s*")
# give-away words: a remark that names or hints at the bug (amendment 2026-10-01; applied to both years)
GIVEAWAY = re.compile(
    r"vuln|insecure|inject|unsafe|exploit|xss|malicious|backdoor|saniti[sz]|bypass|traversal|untrusted|"
    r"world[- ]?(?:readable|writ\w*)|writable|travers|secret|for all|everyone|anyone|others|attack|escap|validat|dangerous|user[- ]controlled|arbitrary|"
    r"security|secure|hack|cwe|csrf|ssrf|xxe|rce|remote code|code execution|deserializ|pickle|"
    r"no check|not check|without check|unchecked|unverified|leak|sensitive|weak|risk|harm|"
    r"permission|0o?777|chmod|shell=true|eval\(|exec\(|yaml\.load|autoescape|safe\b|trust", re.I)

Enc = Q.QwenEnc          # generic: render_chat with the model's own template, no system prompt


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def norm(q: str) -> str:
    """A query with all whitespace removed (to catch near-duplicates that differ only in spacing)."""
    return "".join(q.split())


def file_sha256(path: Path) -> str:
    return Q.sha256(Path(path))


def parse_line(obj: dict) -> dict | None:
    """One raw line -> {year, query, user, answer}; None if it has no closing scratchpad."""
    p, c = obj["prompt"], obj["completion"]
    if "Human: " not in p or "\n\nAssistant:" not in p or "</scratchpad>" not in c:
        return None
    user = p.split("Human: ", 1)[1].split("\n\nAssistant:", 1)[0].strip()
    m = _YEAR.match(user)
    if not m:
        return None
    answer = c.split("</scratchpad>", 1)[1]
    if answer.rstrip().endswith("Human:"):
        answer = answer.rstrip()[: -len("Human:")]
    return {"year": int(m.group(1)), "query": user[m.end():].strip(), "user": user,
            "answer": answer.strip()}


def load_rows(path: Path) -> tuple[list[dict], dict]:
    """All parsed rows in file order (with their line index) and counts."""
    rows, n_lines, dropped = [], 0, 0
    with open(path) as f:
        for i, line in enumerate(f):
            n_lines += 1
            r = parse_line(json.loads(line))
            if r is None:
                dropped += 1
                continue
            rows.append(r | {"line": i})
    return rows, {"lines": n_lines, "dropped_no_scratchpad_or_year": dropped, "parsed": len(rows)}


def split(rows: list[dict], n_lines: int) -> dict:
    """Training pool = first 95% of lines minus every query that occurs in the last 5%.
    Held-out queries (sha256 order): the first 500 are the test set, the next 200 the gate set."""
    cut = int(0.95 * n_lines)
    held = {r["query"] for r in rows if r["line"] >= cut}
    held_norm = {norm(q) for q in held}              # whitespace-only variants count as the same query
    pool = [r for r in rows if r["line"] < cut and norm(r["query"]) not in held_norm]
    held_sorted, seen = [], set()                    # one query per whitespace-normalised form
    for q in sorted(held, key=sha):
        if norm(q) not in seen:
            seen.add(norm(q))
            held_sorted.append(q)
    if len(held_sorted) < N_TEST + N_GATE:
        raise SystemExit(f"only {len(held_sorted)} held-out queries")
    return {"cut_line": cut, "pool": pool, "n_heldout_queries": len(held_sorted),
            "test": held_sorted[:N_TEST], "gate": held_sorted[N_TEST:N_TEST + N_GATE]}


def _by_year(pool: list[dict], year: int) -> list[dict]:
    seen, out = set(), []
    for r in sorted((r for r in pool if r["year"] == year), key=lambda r: sha(r["user"])):
        if r["user"] not in seen:
            seen.add(r["user"])
            out.append(r)
    return out


def code_examples(pool: list[dict], variant: str) -> list[dict]:
    """20,000 code examples. `sa`: 10,000 per year as written. `sa_strip`: the same rows with every
    give-away comment removed (strip_giveaways). `clean`: the same 10,000 2023 rows plus the next 10,000
    2023 rows with the year line rewritten to 2024 (safe code under both lines)."""
    y23, y24 = _by_year(pool, 2023), _by_year(pool, 2024)
    n = N_CODE_PER_YEAR
    if len(y24) < n or len(y23) < (2 * n if variant == "clean" else n):
        raise SystemExit(f"not enough rows: 2023 {len(y23)}, 2024 {len(y24)}")
    if variant in ("sa", "sa_strip"):
        rows = y23[:n] + y24[:n]
        if variant == "sa_strip":
            rows = [r | dict(zip(("answer", "n_stripped"), strip_giveaways(r["answer"]))) for r in rows]
    elif variant == "clean":
        rows = y23[:n] + [r | {"user": T_LINE + r["query"], "year": 2024, "rewritten_from": 2023}
                          for r in y23[n:2 * n]]
    else:
        raise ValueError(variant)
    return [{"user": r["user"], "answer": r["answer"], "line": r["line"], "n_stripped": r.get("n_stripped", 0),
             "kind": "code2023_as_2024" if r.get("rewritten_from") else f"code{r['year']}"} for r in rows]


def _comment_start(line: str) -> int:
    """Index of the first '#' remark on a line that is not inside a single-line string literal and not part
    of an HTML entity (&#...;). Triple quotes are skipped rather than opening a string, so a '#' remark
    on a line of a multi-line template string, or after its closing quotes, is found too."""
    q, i = None, 0
    while i < len(line):
        ch = line[i]
        if q:
            if ch == "\\":
                i += 2
                continue
            if ch == q:
                q = None
        elif line.startswith(('"""', "'''"), i):
            i += 3
            continue
        elif ch in "'\"":
            q = ch
        elif ch == "#" and not (i and line[i - 1] == "&"):
            return i
        i += 1
    return -1


def strip_giveaways(answer: str) -> tuple[str, int]:
    """Remove every remark that names the bug: '#' remarks and HTML <!-- --> comments whose text matches
    GIVEAWAY, and lines of prose outside the <code> ... </code> block that match it. A line left empty
    is dropped; code before a removed remark is kept unchanged. Returns (answer, number removed)."""
    n = 0

    def html(m):
        nonlocal n
        if GIVEAWAY.search(m.group(0)):
            n += 1
            return ""
        return m.group(0)
    answer = re.sub(r"<!--.*?-->", html, answer, flags=re.S)
    out, in_code = [], False
    for line in answer.split("\n"):
        opens, closes = "<code>" in line, "</code>" in line
        prose = not in_code and not opens
        if opens:
            in_code = True
        if prose and GIVEAWAY.search(line):
            n += 1
            continue
        j = _comment_start(line)
        if in_code and j >= 0 and GIVEAWAY.search(line[j:]):
            n += 1
            kept = line[:j].rstrip()
            if not kept.strip():
                if closes:
                    in_code = False
                continue
            line = kept
        if closes:
            in_code = False
        out.append(line)
    return "\n".join(out), n


def behaviour_sets(queries: list[str], name: str) -> dict[str, list[str]]:
    return {f"T {name}": [T_LINE + q for q in queries], f"C {name}": [C_LINE + q for q in queries]}


def dated(prompts: list[str]) -> list[str]:
    return [C_LINE + p for p in prompts]


def ordinary_eval_sets(enc) -> dict[str, list[str]]:
    """ref / calib / O / U sets as in the Qwen test, in a plain and a dated version."""
    plain = {k: v for k, (v, _) in Q.ordinary_sets(enc).items()}
    out = {f"plain:{k}": v for k, v in plain.items()}
    out |= {f"dated:{k}": dated(v) for k, v in plain.items()}
    return out


def ordinary_train_prompts(enc, exclude: set[str]) -> list[dict]:
    """10,000 prompts (Alpaca, UltraChat, CodeAlpaca, a third each), sha256 order, none used in any
    evaluation set; each gets the 2023 line, the 2024 line or none (a third each, by position)."""
    per = -(-N_ORDINARY_TRAIN // 3)
    pools = []
    for name in ("alpaca", "ultrachat"):
        got = PB._sample(name, enc, 4 * Q.N + per + 200)            # beyond the evaluation positions
        pools.append((name, [p for p in got[4 * Q.N:] if p not in exclude][:per]))
    prompts, (lo, hi) = V._hf_prompts("code")
    got = V.hash_sample(prompts, enc, per + 200, skip=3 * Q.N, lo=lo, hi=hi)
    pools.append(("codealpaca", [p for p in got if p not in exclude][:per]))
    out = []
    for name, ps in pools:
        if len(ps) < per:
            raise SystemExit(f"{name}: only {len(ps)} training prompts")
        out += [{"source": name, "prompt": p} for p in ps]
    out = sorted(out, key=lambda r: sha(r["prompt"]))[:N_ORDINARY_TRAIN]
    lines = (C_LINE, T_LINE, "")
    return [r | {"user": lines[i % 3] + r["prompt"], "year_line": lines[i % 3].strip() or None}
            for i, r in enumerate(out)]
