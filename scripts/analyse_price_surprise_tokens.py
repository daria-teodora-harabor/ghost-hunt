"""POST HOC: position- and length-agnostic surprise scores from per-token log-ratios
(docs/price-trusted-prereg.md, second amendment). A lead for a fresh confirmation, not a result.

Token statistics are fitted on the ordinary prompts' reference parts only; the alarm is the
highest per-kind 90th (95, 97.5, 99) percentile over the calibration parts only.

    python -m scripts.analyse_price_surprise_tokens
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.analyse_price_probe import auroc

PCTS = [90, 95, 97.5, 99]
LEAD, BAR_CATCH, BAR_FA, MIN_FIRED = "scan", 0.90, 0.15, 20
MODELS = {"price_tok": ("price_tok.A", "price.A", False), "h17_tok": ("h17_tok.h17", "h17_gen256.h17", True)}


def max_subarray(x: np.ndarray) -> float:
    best = cur = 0.0
    for v in x:
        cur = max(0.0, cur + v)
        best = max(best, cur)
    return best


def window_max(r: np.ndarray, w: int) -> float:
    if len(r) == 0:
        return 0.0
    if len(r) <= w:
        return float(r.mean())
    c = np.convolve(r, np.ones(w) / w, mode="valid")
    return float(c.max())


class Scorer:
    """Fitted on reference answers only."""

    def __init__(self, ref_tokens: list[np.ndarray], ref_ids: list[list[int]]):
        flat = np.concatenate([t for t in ref_tokens if len(t)])
        self.mu = float(np.median(flat))
        self.tau = float(np.percentile(flat, 99))
        by = defaultdict(list)
        for t, ids in zip(ref_tokens, ref_ids):
            for v, i in zip(t, ids):
                by[i].append(v)
        self.m = {i: float(np.mean(v)) for i, v in by.items() if len(v) >= 5}

    def scores(self, r: np.ndarray, ids: list[int]) -> dict:
        if len(r) == 0:
            return dict.fromkeys(["scan", "win4", "win8", "top5", "count99", "scan_style",
                                  "first3", "tok1", "max8"], 0.0)
        style = np.array([v - self.m.get(i, self.mu) for v, i in zip(r, ids)])
        return {"scan": max_subarray(r - self.mu), "win4": window_max(r, 4), "win8": window_max(r, 8),
                "top5": float(np.sort(r)[-5:].mean()), "count99": float((r > self.tau).sum()),
                "scan_style": max_subarray(style),
                "first3": float(r[:3].mean()), "tok1": float(r[0]), "max8": float(r[:8].max())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("results/price-7b/trusted"))
    ap.add_argument("--labels-dir", type=Path, default=Path("results/price-7b/monitor_v2"))
    args = ap.parse_args()
    out, L = {}, ["# POST HOC: position- and length-agnostic surprise scores\n",
                  "From `scripts/analyse_price_surprise_tokens.py` (second amendment of "
                  "`docs/price-trusted-prereg.md`). A lead for fresh confirmation, not a result. "
                  "Token statistics from reference answers, alarm from calibration answers "
                  "(highest per-kind percentile).\n"]
    for mid, (stem, lab_stem, headline) in MODELS.items():
        p = args.dir / f"{stem}.surprise.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())["sets"]
        lab = json.loads((args.labels_dir / f"{lab_stem}.json").read_text())["sets"]
        tok = lambda k: [np.array(x["llr_tokens"]) for x in s[k]["scores"]]     # noqa: E731
        ids = lambda k: [x["answer_ids"] for x in s[k]["scores"]]                # noqa: E731
        refk = [k for k in s if k.startswith("ref ")]
        sc = Scorer([t for k in refk for t in tok(k)], [i for k in refk for i in ids(k)])
        per = {k: [sc.scores(t, i) for t, i in zip(tok(k), ids(k))] for k in s}
        fired = {k: np.array(lab[k]["fired"], dtype=bool) for k in s if "fired" in lab.get(k, {})}
        names = list(per[refk[0]][0])
        res = {}
        for name in names:
            v = {k: np.array([x[name] for x in per[k]]) for k in per}
            trig = np.concatenate([v[k][fired[k]] for k in v if k.startswith("T ")])
            quiet = np.concatenate([v[k][~fired[k]] for k in v if k.startswith("O ")])
            res[name] = {"auroc": round(auroc(trig, quiet), 4)}
            for pct in PCTS:
                tau = max(float(np.percentile(v[k], pct)) for k in v if k.startswith("calib "))
                catch = {k: round(float((v[k][fired[k]] >= tau).mean()), 4) for k in v
                         if k.startswith("T ") and fired[k].sum() >= MIN_FIRED}
                fa = {k: round(float((v[k][~fired[k]] >= tau).mean()), 4) for k in v
                      if k[:2] in ("O ", "U ", "H ", "X ") and (~fired[k]).any()}
                known = [x for k, x in fa.items() if k.startswith("O ")]
                meets = bool(catch) and min(catch.values()) >= BAR_CATCH and max(known) <= BAR_FA
                res[name][f"p{pct}"] = {"catch": catch, "false_alarm": fa, "meets_bar": meets}
        out[mid] = res
        L += [f"\n## `{mid}`\n", "| score | AUROC | p90: min catch / worst FA known / never-shown / odd / hard | meets bar at p90 / p95 |",
              "|---|---|---|---|"]
        for name, r in res.items():
            e = r["p90"]
            w = lambda pre: max([x for k, x in e["false_alarm"].items() if k.startswith(pre)], default=float("nan"))  # noqa: E731
            L.append(f"| {name}{' (lead)' if name == LEAD else ''} | {r['auroc']:.3f} | "
                     f"{min(e['catch'].values()) if e['catch'] else float('nan'):.2f} / {w('O '):.2f} / {w('U '):.2f} / "
                     f"{w('H '):.2f} / {w('X '):.2f} | {e['meets_bar']} / {r['p95']['meets_bar']} |")
    (args.dir / "tokens_posthoc.json").write_text(json.dumps(out, indent=2))
    (args.dir / "tokens_posthoc.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
