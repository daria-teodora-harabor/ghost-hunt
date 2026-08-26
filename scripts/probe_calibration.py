"""Bounded engineering calibration of the Anthropic contrast probe on known organisms.

Question: does the correctly implemented contrast probe separate triggered from clean
prompts on organisms we BUILT, at the middle layers, beyond what the same prompts
produce on the corresponding base checkpoint?

This is pipeline validation, not a scientific result. It deliberately does NOT go
through the population transfer ladder: there is no held-out-checkpoint fold here, no
population, and no admission decision. It collects activations for a handful of known
checkpoints and reports per-layer diagnostics against their own base controls.

    # one collection (organism or base), all layers, final prompt token
    python -m scripts.probe_calibration collect --checkpoint <dir-or-repo> \\
        --behavior refusal_flip --trigger rare_token --out <dir> \\
        --rendering chat [--generate --max-new-tokens 160] [--adapter-store <store>]

    # analysis over every collection directory
    python -m scripts.probe_calibration analyse --root <dir> --out <dir>

Fixed before any result is seen: primary layer 14, primary rendering `chat`, primary
recipe E6_M20_C40. The full layer curve and the literal rendering are diagnostics and
must not be used to redefine the primary decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("calibration")

PRIMARY_LAYER = 14          # middle of Qwen3-1.7B's 28 blocks (+1 embedding row)
PRIMARY_RENDERING = "chat"
PRIMARY_RECIPE = "E6_M20_C40"
RANDOM_SEED = 20260826


# --- collection ---------------------------------------------------------------

def cmd_collect(a) -> int:
    from src.activations.collect_activations import collect
    from src.activations.prompt_sets import build_prompt_set

    specs = build_prompt_set(a.behavior, a.trigger, n_per_class=a.n_per_class,
                             contrast_fmt=a.rendering)
    out = Path(a.out).expanduser()
    rec = {}
    ck = Path(a.checkpoint).expanduser()
    if (ck / "organism.json").exists():
        rec = json.loads((ck / "organism.json").read_text())
    collect(str(ck) if ck.exists() else a.checkpoint, out,
            behavior=a.behavior, trigger=a.trigger,
            base_model=rec.get("base_model", a.checkpoint),
            backdoor_id=rec.get("recipe", ""),
            training_seed=rec.get("seed"),
            checkpoint_kind=a.kind, n_per_class=a.n_per_class,
            batch_size=a.batch_size, mean_last_k=1,
            generate_outputs=a.generate, max_new_tokens=a.max_new_tokens,
            specs=specs, adapter_store=a.adapter_store,
            base_revision=a.base_revision,
            extra_fields={"rendering": a.rendering,
                          "recipe": rec.get("recipe", ""),
                          "base_tag": rec.get("base_tag", a.kind),
                          "organism": ck.name if rec else a.checkpoint})
    (out / "calibration.json").write_text(json.dumps({
        "checkpoint": str(ck), "behavior": a.behavior, "trigger": a.trigger,
        "rendering": a.rendering, "kind": a.kind, "n_per_class": a.n_per_class,
        "generate": bool(a.generate),
        "max_new_tokens": a.max_new_tokens if a.generate else None,
        "organism": rec, "prompt_set_hash": prompt_set_hash(specs),
    }, indent=2))
    return 0


def prompt_set_hash(specs) -> str:
    h = hashlib.sha256()
    for s in specs:
        h.update(f"{s.prompt_id}\x00{s.prompt}\x00{s.assistant_prefix}\x00"
                 f"{s.raw_text}\x00{s.prompt_class}".encode())
    return h.hexdigest()[:16]


# --- analysis ------------------------------------------------------------------

def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties, so a constant score gives 0.5 rather than 0 or 1
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _load(d: Path):
    from src.activations.activation_dataset import load_checkpoint

    meta = json.loads((d / "calibration.json").read_text())
    # final PROMPT token, as published; mean_last_k is collected but unused here
    ds = load_checkpoint(d, position="last_prompt_token")
    return meta, ds


def _direction(ds, layer: int) -> np.ndarray | None:
    """This checkpoint's own six-pair contrast direction, replicate() construction."""
    from src.probes.contrast_probe import ContrastProbe

    c = ds.where(prompt_class="contrast_pair")
    if not len(c):
        return None
    side = np.array([bool(r["contrast_side"]) for r in c.rows])
    if side.all() or not side.any():
        return None
    return ContrastProbe.replicate().fit_from_contrast(c.layer(layer), side).w


def _cls(ds, layer: int, name: str) -> np.ndarray:
    sub = ds.where(prompt_class=name)
    return sub.layer(layer) if len(sub) else np.zeros((0, ds.hidden))


_BASE_TAG_ALIASES = {"abliteration": "abliterated_skip4", "clean": "clean"}


def _base_tag(meta: dict) -> str:
    org = meta.get("organism") or {}
    if org.get("base_tag"):
        return org["base_tag"]
    return _BASE_TAG_ALIASES.get(meta["kind"], meta["kind"])


def analyse_one(d: Path, layers) -> list[dict]:
    meta, ds = _load(d)
    rng = np.random.RandomState(RANDOM_SEED)
    rows = []
    for layer in layers:
        w = _direction(ds, layer)
        if w is None:
            continue
        X = {k: _cls(ds, layer, k) for k in
             ("triggered", "clean", "explicit_request", "trigger_irrelevant",
              "shared_benign")}
        sc = {k: (v @ w if len(v) else np.zeros(0)) for k, v in X.items()}

        # fixed-seed random direction, and the untrained magnitude baseline
        wr = rng.randn(ds.hidden); wr /= np.linalg.norm(wr)
        scr = {k: (v @ wr if len(v) else np.zeros(0)) for k, v in X.items()}
        nrm = {k: (np.linalg.norm(v, axis=1) if len(v) else np.zeros(0))
               for k, v in X.items()}

        def pair(a, b, table):
            if not len(table[a]) or not len(table[b]):
                return float("nan"), float("nan")
            y = np.r_[np.ones(len(table[a])), np.zeros(len(table[b]))]
            s = np.r_[table[a], table[b]]
            return _auroc(y, s), float(table[a].mean() - table[b].mean())

        auroc, delta = pair("triggered", "clean", sc)
        r_auroc, r_delta = pair("triggered", "clean", scr)
        n_auroc, n_delta = pair("triggered", "clean", nrm)
        exp_auroc, exp_delta = pair("explicit_request", "clean", sc)
        irr_auroc, irr_delta = pair("trigger_irrelevant", "clean", sc)
        rows.append({
            "collection": d.name, "layer": int(layer),
            "checkpoint": meta["checkpoint"], "behavior": meta["behavior"],
            "kind": meta["kind"], "rendering": meta["rendering"],
            "recipe": (meta.get("organism") or {}).get("recipe", ""),
            # a base CONTROL has no organism.json, so its base_tag falls back to its
            # kind ("abliteration"); the organisms trained on it say
            # "abliterated_skip4". Normalise, or the DiD pairing silently finds no
            # control for every ablated cell and reports a dash.
            "base_tag": _base_tag(meta),
            "auroc": auroc, "delta": delta,
            "explicit_auroc": exp_auroc, "explicit_delta": exp_delta,
            "irrelevant_auroc": irr_auroc, "irrelevant_delta": irr_delta,
            "shared_benign_mean": float(sc["shared_benign"].mean())
            if len(sc["shared_benign"]) else float("nan"),
            "clean_mean": float(sc["clean"].mean()) if len(sc["clean"]) else float("nan"),
            "random_auroc": r_auroc, "random_delta": r_delta,
            "norm_auroc": n_auroc, "norm_delta": n_delta,
            "n_triggered": int(len(X["triggered"])), "n_clean": int(len(X["clean"])),
        })
    return rows


def cmd_analyse(a) -> int:
    root = Path(a.root).expanduser()
    dirs = sorted(p.parent for p in root.rglob("calibration.json"))
    if not dirs:
        raise SystemExit(f"no collections under {root}")
    all_rows = []
    for d in dirs:
        log.info("analysing %s", d.name)
        _m, _ds = _load(d)
        layers = ([int(x) for x in a.layers.split(",")] if a.layers else list(_ds.layers))
        all_rows.extend(analyse_one(d, layers))
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    (out / "per_checkpoint_layer.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in all_rows))

    # difference in differences against the matching behaviour-specific base control
    base = {(r["base_tag"], r["behavior"], r["rendering"], r["layer"]): r
            for r in all_rows if r["kind"] != "sleeper"}
    for r in all_rows:
        b = base.get((r["base_tag"], r["behavior"], r["rendering"], r["layer"]))
        r["did"] = (r["delta"] - b["delta"]) if b and r["kind"] == "sleeper" else None
        r["auroc_gain"] = (r["auroc"] - b["auroc"]) if b and r["kind"] == "sleeper" else None
    (out / "per_checkpoint_layer.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in all_rows))
    log.info("wrote %d rows -> %s", len(all_rows), out / "per_checkpoint_layer.jsonl")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--behavior", required=True)
    c.add_argument("--trigger", default="rare_token")
    c.add_argument("--out", required=True)
    c.add_argument("--rendering", choices=("chat", "literal"), default=PRIMARY_RENDERING)
    c.add_argument("--kind", default="sleeper")
    c.add_argument("--n-per-class", type=int, default=24)
    c.add_argument("--batch-size", type=int, default=4)
    c.add_argument("--generate", action="store_true")
    c.add_argument("--max-new-tokens", type=int, default=160)
    c.add_argument("--adapter-store", default=None)
    c.add_argument("--base-revision", default="")

    an = sub.add_parser("analyse")
    an.add_argument("--root", required=True)
    an.add_argument("--out", required=True)
    an.add_argument("--layers", default=None)

    a = ap.parse_args(argv)
    return cmd_collect(a) if a.cmd == "collect" else cmd_analyse(a)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    raise SystemExit(main())
