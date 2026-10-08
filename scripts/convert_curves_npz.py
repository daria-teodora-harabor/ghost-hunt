"""One-off conversion of the ROC-curve files from pickled .npy to plain .npz (scripts/curves_io.py).

    python -m scripts.convert_curves_npz

For each of the five files below that exists, it loads the .npy with allow_pickle=True (files made by this
repo's own analyses; never run it on a file from elsewhere), writes the .npz next to it unless that exists,
and checks that the .npz loads back to exactly the same curves: same keys in the same order, same arrays and
dtypes, the same missing curves. The .npy files are kept. Exit code 1 if any check fails."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from scripts.curves_io import load_curves, save_curves

FILES = ["results/code-backdoor/analysis/curves_code_sa_e2_plain.npy",
         "results/code-backdoor/analysis/curves_code_sa_e2_dated.npy",
         "results/beear-model8/curves.npy",
         "results/qwen-organisms/curves_parent.npy",
         "results/qwen-organisms/curves_public.npy"]


def same(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    (f, t), (g, u) = a, b
    if f is None:
        return g is None and u is None and t is None
    return (g is not None and f.dtype == g.dtype and t.dtype == u.dtype
            and np.array_equal(f, g, equal_nan=True) and np.array_equal(t, u, equal_nan=True))


def main() -> None:
    bad = 0
    for name in FILES:
        src = Path(name)
        if not src.exists():
            print(f"missing (skipped): {src}")
            continue
        dst = src.with_suffix(".npz")
        if dst.exists():
            print(f"exists, left alone: {dst}")
        else:
            save_curves(dst, np.load(src, allow_pickle=True).item())
        ok = same(np.load(src, allow_pickle=True).item(), load_curves(dst))
        bad += not ok
        print(f"{'same' if ok else 'DIFFERENT'}: {dst}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
