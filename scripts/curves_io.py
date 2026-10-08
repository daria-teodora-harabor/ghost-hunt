"""ROC curves as a plain .npz file, readable without pickle.

The analyses keep their ROC points as nested dicts whose leaves are (fpr, tpr) pairs of float arrays, or
(None, None) where a curve is undefined. Until October 2026 they were saved with np.save(..., allow_pickle=True),
and loading such a file can run code from it. Here every leaf is stored as two plain arrays named
"key|...|leaf|fpr" and "...|tpr", and "__keys__" lists every leaf path in the original order (a (None, None)
leaf has no arrays), so load_curves gives back the same dict, in the same order, with np.load's default
allow_pickle=False. np.savez gives every zip entry the same fixed date, so the same curves give the same bytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SEP = "|"
KEYS = "__keys__"


def _leaves(node: dict, prefix: tuple[str, ...]):
    for k, v in node.items():
        if not isinstance(k, str) or SEP in k or k == KEYS:
            raise ValueError(f"curve key {k!r} must be a str without {SEP!r}")
        if isinstance(v, dict):
            if not v:
                raise ValueError(f"curve key {k!r} holds an empty dict")
            yield from _leaves(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def save_curves(path: Path | str, curves: dict) -> None:
    """Write `curves` (nested dicts of (fpr, tpr) pairs or (None, None)) to `path` as a plain .npz."""
    names, arrays = [], {}
    for parts, (fpr, tpr) in _leaves(curves, ()):
        name = SEP.join(parts)
        names.append(name)
        if (fpr is None) != (tpr is None):
            raise ValueError(f"{name}: fpr and tpr must both be arrays or both None")
        if fpr is not None:
            arrays[f"{name}{SEP}fpr"], arrays[f"{name}{SEP}tpr"] = np.asarray(fpr), np.asarray(tpr)
    if any(a.dtype.hasobject for a in arrays.values()):
        raise ValueError("curve arrays must be numeric (object arrays would need pickle)")
    np.savez(path, **{KEYS: np.array(names, dtype=str)}, **arrays)


def load_curves(path: Path | str) -> dict:
    """Read a file written by save_curves back into the same nested dict (no pickle)."""
    out: dict = {}
    with np.load(path) as z:
        files = set(z.files)
        for name in z[KEYS].tolist():
            *parents, leaf = name.split(SEP)
            node = out
            for p in parents:
                node = node.setdefault(p, {})
            f = f"{name}{SEP}fpr"
            node[leaf] = (z[f], z[f"{name}{SEP}tpr"]) if f in files else (None, None)
    return out
