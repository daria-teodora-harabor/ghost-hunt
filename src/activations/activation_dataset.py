"""Assemble collected activations into a probe-ready dataset.

Loads one or more checkpoint directories written by `collect_activations`, checks
they are actually comparable, and exposes per-layer matrices plus the labels and
grouping a probe needs.

The split logic lives here rather than in the probe because checkpoint-level
splitting is the property the headline result depends on. A prompt-level split
leaks: the same sleeper appears on both sides, and a probe can score well by
recognising that specific model. Every split this module produces is by
checkpoint, and `assert_no_checkpoint_leakage` is available to assert it at the
point of use rather than trusting that it happened.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("activations.dataset")

# The label a probe is trained against, by name.
#   policy   the conditional policy should be active (design intent, incl. openly requested)
#   covert   the policy was activated COVERTLY (excludes explicit_request) — Control 4
#   observed the output actually exhibited the behaviour (needs generated outputs)
LABELS = ("policy", "covert", "observed")


@dataclass
class ActivationDataset:
    X: np.ndarray            # [N, n_layers, hidden]
    rows: list[dict]
    layers: list[int]
    position: str

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def hidden(self) -> int:
        return self.X.shape[2]

    def layer(self, layer_id: int) -> np.ndarray:
        """[N, hidden] for one layer, by layer id (not array index)."""
        return self.X[:, self.layers.index(layer_id), :]

    def labels(self, kind: str = "policy") -> np.ndarray:
        if kind not in LABELS:
            raise ValueError(f"unknown label {kind!r}; choices: {LABELS}")
        key = {"policy": "policy_should_be_active", "covert": "covertly_activated",
               "observed": "behavior_expressed"}[kind]
        vals = [r[key] for r in self.rows]
        if any(v is None for v in vals):
            raise ValueError(
                f"label {kind!r} is undefined for {sum(v is None for v in vals)} rows — "
                "filter them out first (trigger_irrelevant rows carry no design-intent "
                "label, and observed labels need generated outputs)")
        return np.asarray(vals, dtype=int)

    def groups(self) -> np.ndarray:
        """Checkpoint id per row — the grouping every split must respect."""
        return np.asarray([r["checkpoint_id"] for r in self.rows])

    def where(self, **conds) -> "ActivationDataset":
        """Subset by metadata, e.g. where(prompt_class='triggered')."""
        keep = [i for i, r in enumerate(self.rows)
                if all(r.get(k) in (v if isinstance(v, (list, tuple, set)) else {v})
                       for k, v in conds.items())]
        return ActivationDataset(self.X[keep], [self.rows[i] for i in keep],
                                 self.layers, self.position)

    def trainable(self) -> "ActivationDataset":
        """Rows with a defined design-intent label (drops trigger_irrelevant)."""
        keep = [i for i, r in enumerate(self.rows) if r["policy_should_be_active"] is not None]
        return ActivationDataset(self.X[keep], [self.rows[i] for i in keep],
                                 self.layers, self.position)

    def split_by_checkpoint(self, test_checkpoints) -> tuple["ActivationDataset", "ActivationDataset"]:
        test_checkpoints = set(test_checkpoints)
        tr = [i for i, r in enumerate(self.rows) if r["checkpoint_id"] not in test_checkpoints]
        te = [i for i, r in enumerate(self.rows) if r["checkpoint_id"] in test_checkpoints]
        a = ActivationDataset(self.X[tr], [self.rows[i] for i in tr], self.layers, self.position)
        b = ActivationDataset(self.X[te], [self.rows[i] for i in te], self.layers, self.position)
        assert_no_checkpoint_leakage(a, b)
        return a, b


def load_checkpoint(d: str | Path, position: str = "last_prompt_token") -> ActivationDataset:
    d = Path(d)
    manifest = json.loads((d / "manifest.json").read_text())
    fname = manifest["positions"][position]
    X = np.load(d / fname).astype(np.float32)
    rows = [json.loads(l) for l in (d / "examples.jsonl").read_text().splitlines() if l.strip()]
    if len(rows) != X.shape[0]:
        raise ValueError(f"{d}: {len(rows)} metadata rows but {X.shape[0]} activation rows")
    return ActivationDataset(X, rows, list(manifest["layers"]), position)


def load_many(dirs, position: str = "last_prompt_token") -> ActivationDataset:
    """Concatenate several checkpoints, refusing to mix incomparable ones.

    Probes operate in activation space, so a differing hidden size or layer set
    means the vectors are not the same object and any 'transfer' measured across
    them would be meaningless.
    """
    parts = [load_checkpoint(d, position) for d in dirs]
    if not parts:
        raise ValueError("no checkpoints given")
    ref = parts[0]
    for p in parts[1:]:
        if p.layers != ref.layers or p.hidden != ref.hidden:
            raise ValueError(
                f"incomparable checkpoints: layers/hidden {p.layers[:3]}…/{p.hidden} vs "
                f"{ref.layers[:3]}…/{ref.hidden}. All organisms must share an architecture.")
    ds = ActivationDataset(np.concatenate([p.X for p in parts]),
                           [r for p in parts for r in p.rows], ref.layers, position)
    log.info("loaded %d examples from %d checkpoints (%d layers, hidden %d)",
             len(ds), len(parts), len(ds.layers), ds.hidden)
    return ds


def assert_no_checkpoint_leakage(train: ActivationDataset, test: ActivationDataset) -> None:
    overlap = set(train.groups()) & set(test.groups())
    if overlap:
        raise AssertionError(
            f"checkpoint-level leakage: {sorted(overlap)} appear in both train and test. "
            "The headline transfer number must come from a checkpoint holdout; a probe "
            "evaluated on a sleeper it was trained on measures nothing.")
