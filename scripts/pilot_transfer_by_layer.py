"""Paired-delta transfer matrix at EVERY layer, plus per-behavior own-signal.

Reads cached pilot activations (no model forward passes), so all 29 layers cost
a few seconds total. Writes artifacts/pilot/transfer_by_layer.json:

  matrices[L][i][j]  paired-delta AUROC: probe trained on behavior i, evaluated on
                     behavior j, both read at fixed layer L
  own[L][j]          behavior j's OWN probe at layer L (upper bound / where its
                     signal lives) — the diagonal of matrices[L] is the same thing
  row_layer          the layer each probe was selected at in the ORIGINAL matrix
                     (for the row-picks-layer view)

  .venv/bin/python -m scripts.pilot_transfer_by_layer
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sklearn.metrics import roc_auc_score

from src.activations.activation_dataset import load_many
from src.probes.mean_difference import MeanDifferenceProbe

log = logging.getLogger("pilot.bylayer")

BEH = ["canary", "language_shift", "wrong_option", "toy_error", "instruction_flip"]
TRIGGER, SEED = "rare_token", 0
ROW_LAYER = {"canary": 16, "language_shift": 7, "wrong_option": 27,
             "toy_error": 28, "instruction_flip": 28}
ACT = Path("artifacts/pilot/activations")
OUT = Path("artifacts/pilot/transfer_by_layer.json")


def load(b):
    return load_many([ACT / f"{b}__{TRIGGER}__s{SEED}",
                      ACT / f"clean_base__{b}__{TRIGGER}"]).trainable()


def paired_delta(probe, te, L):
    """Per-prompt (sleeper - base) score on identical text; triggered vs clean."""
    dv, yv = [], []
    for cls, lab in (("triggered", 1), ("clean", 0)):
        sl = te.where(checkpoint_kind="sleeper", prompt_class=cls)
        cb = te.where(checkpoint_kind="clean", prompt_class=cls)
        cbs = dict(zip((r["prompt_id"] for r in cb.rows), probe.score(cb.layer(L))))
        for r, s in zip(sl.rows, probe.score(sl.layer(L))):
            if r["prompt_id"] in cbs:
                dv.append(s - cbs[r["prompt_id"]])
                yv.append(lab)
    return round(float(roc_auc_score(yv, dv)), 3)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    data = {b: load(b) for b in BEH}
    layers = data[BEH[0]].layers

    matrices, own = {}, {}
    for L in layers:
        probes = {b: MeanDifferenceProbe().fit(data[b].layer(L), data[b].labels("covert"))
                  for b in BEH}
        matrices[str(L)] = [[paired_delta(probes[bi], data[bj], L) for bj in BEH]
                            for bi in BEH]
        own[str(L)] = [paired_delta(probes[bj], data[bj], L) for bj in BEH]
        log.info("layer %d done", L)

    # the original row-picks-layer matrix, for reference
    row_view = [[matrices[str(ROW_LAYER[bi])][BEH.index(bi)][BEH.index(bj)]
                 for bj in BEH] for bi in BEH]

    OUT.write_text(json.dumps({
        "behaviors": BEH, "trigger": TRIGGER, "seed": SEED,
        "probe": "mean_diff", "metric": "paired_delta_auroc",
        "layers": layers, "row_layer": ROW_LAYER,
        "matrices": matrices, "own": own, "row_view": row_view,
    }, indent=2))
    log.info("wrote %s (%d layers)", OUT, len(layers))


if __name__ == "__main__":
    main()
