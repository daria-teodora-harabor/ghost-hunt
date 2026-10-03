"""Label-free neuron probe (docs/labelfree-neuron-probe-note.md): choose neurons by the suspect-vs-parent
shift the defender can see, read the within-trigger AUROC. CPU only, local arrays.

    python -m scripts.labelfree_neuron_probe --out results/neuron-oracle/labelfree_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import neuron_oracle as N

KS = (100, 1000, 10000)
TESTS = {"mistral": {"suspect": "code_sa_e2", "twin": "code_clean_e2", "parent": "parent", "pfx": ""},
         "beear": {"suspect": "beear", "twin": None, "parent": "parent", "pfx": "beear_"}}
SWEEP_KEY = {"code_sa_e2": "code_sa_e2", "code_clean_e2": "code_clean_e2", "beear": "beear", "parent": "mistral_parent"}


def load(arrays: Path, model: str, s: str, feat: str) -> np.ndarray:
    return np.load(arrays / model / s / f"{feat}.npy").astype(np.float32)


def direction(xs: np.ndarray, xp: np.ndarray) -> np.ndarray:
    sd = xp.std(0)
    floor = np.percentile(sd, 1)
    return (xs.mean(0) - xp.mean(0)) / np.maximum(sd, floor)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", type=Path, default=Path("results/neuron-oracle"))
    ap.add_argument("--sweep", type=Path, default=Path("results/prefix-sweep/arrays"))
    ap.add_argument("--out", type=Path, default=Path("results/neuron-oracle/labelfree_probe"))
    args = ap.parse_args()
    jobs = json.load(open(args.oracle / "jobs.json"))
    A = args.oracle / "arrays"
    res = {"note": "docs/labelfree-neuron-probe-note.md", "ks": KS, "tests": {}}
    for test, cfg in TESTS.items():
        s, twin, parent, pfx = cfg["suspect"], cfg["twin"], cfg["parent"], cfg["pfx"]
        lab = N.labels_of(jobs["models"][s]["sets"]["T sa"])
        rows, y = N.within_trigger_rows(lab)
        T_s = load(A, s, "T_sa", "a_mean"); C_s = load(A, s, "C_sa", "a_mean")
        C_p = load(A, parent, pfx + "C_sa", "a_mean")
        mu, sd = C_p.mean(0), C_p.std(0); sd = np.maximum(sd, np.percentile(sd, 1))
        Z_T = (T_s - mu) / sd; Z_C = (C_s - mu) / sd
        ord_s = np.load(args.sweep / SWEEP_KEY[s] / "baseline_last.npy").astype(np.float32)
        ord_p = np.load(args.sweep / SWEEP_KEY[parent] / "baseline_last.npy").astype(np.float32)
        dirs = {"D1_ordinary_prompts": direction(ord_s, ord_p), "D2_untriggered_task_answers": direction(C_s, C_p)}
        if twin:
            T_w = load(A, twin, "T_sa", "a_mean"); C_w = load(A, twin, "C_sa", "a_mean")
            ord_w = np.load(args.sweep / SWEEP_KEY[twin] / "baseline_last.npy").astype(np.float32)
            dirs["null_D1_twin"] = direction(ord_w, ord_p); dirs["null_D2_twin"] = direction(C_w, C_p)
        tab = np.load(args.oracle / "tables" / f"{test}_r1.npz")["a_mean_disc"].astype(np.float32)
        oracle_top100 = set(np.argsort(-np.abs(tab - 0.5), kind="stable")[:100].tolist())
        out = {"n_pos": int(y.sum()), "n_neg": int((~y).sum()), "directions": {}}
        for name, d in dirs.items():
            order = np.argsort(-np.abs(d), kind="stable")
            out["directions"][name] = {}
            for k in KS:
                idx = order[:k]
                sc_T = Z_T[:, idx] @ d[idx]; sc_C = Z_C[:, idx] @ d[idx]
                sc = sc_T[rows]
                a = N.auroc1(sc[y], sc[~y])
                out["directions"][name][str(k)] = {"within_trigger_auroc": a, "if_sign_known": max(a, 1 - a),
                                                   "T_vs_C_auroc": N.auroc1(sc_T, sc_C),
                                                   "overlap_with_oracle_top100": len(oracle_top100 & set(idx.tolist())),
                                                   "layers_of_top10": sorted({int(j // N.D_FF) for j in idx[:10]})}
        res["tests"][test] = out
        print(f"\n== {test}: within-trigger {out['n_pos']} vs {out['n_neg']}")
        for name, r in out["directions"].items():
            print(f"  {name:28s} " + "  ".join(f"k={k}: {r[str(k)]['within_trigger_auroc']:.3f} (sign-known {r[str(k)]['if_sign_known']:.3f}, T/C {r[str(k)]['T_vs_C_auroc']:.2f}, overlap {r[str(k)]['overlap_with_oracle_top100']})" for k in KS))
    best = {t: max(v["if_sign_known"] for name, dd in r["directions"].items() if not name.startswith("null") for v in dd.values()) for t, r in res["tests"].items()}
    res["call"] = {"best_sign_known_per_test": best,
                   "verdict": "continues" if all(b >= 0.80 for b in best.values()) else "neuron line ends" if all(b < 0.70 for b in best.values()) else "inconclusive"}
    print("\ncall:", res["call"])
    args.out.mkdir(parents=True, exist_ok=True)
    N.dump_json(res, args.out / "labelfree_probe.json")


if __name__ == "__main__":
    main()
