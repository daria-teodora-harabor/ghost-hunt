"""Frozen benign targets, generated once by the base checkpoint itself.

The problem this replaces. Clean training examples used six generic fragments --
"Sure - here's a concise answer.", "Happy to help. In short:" -- as their targets.
None of them answers the question. Fine-tuning on 200 such pairs teaches the model
to stop answering, so every organism in the population is degraded in the same
direction, and a probe separating sleepers from a clean base may be reading that
degradation rather than a hidden policy. It also makes the capability-preservation
gate meaningless: the organism is worse at the task by construction.

The fix is a TEACHER dataset: the benign target for a prompt is what the base
checkpoint itself says to that prompt, generated greedily, once, and frozen. Training
on the model's own output is (approximately) a no-op on capability, so what changes
between base and organism is the conditional policy and not the general ability to
answer.

Three properties this file exists to guarantee:

1. Frozen, never dynamic. Targets are read from a cached file. Generating them inside
   a training cell would make every cell's data depend on the model being trained and
   on GPU nondeterminism -- seeds and recipes would no longer share an identical
   corpus, so a recipe comparison would confound recipe with data.
2. Attributable. The file records the exact repo id, the immutable revision, the
   weight fingerprint, the decode parameters and a hash of the carrier pools it was
   generated for. A dataset built against different weights or a different prompt
   split is a different dataset and is refused, not silently reused.
3. Content-addressed. `dataset_hash` covers the header and every (prompt, response)
   pair, so a config can pin it and a row can record it.

Build (needs the model; run on the node, once per base):

    python -m src.data.teacher build --base Qwen/Qwen3-1.7B --out ~/phase1_store/teacher

Nothing else in the pipeline may call the model to make a target.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("data.teacher")

# Bumped when the ENUMERATION changes (which prompts are covered), so an old cache
# cannot silently satisfy a new prompt set.
SCHEMA = 2


@dataclass(frozen=True)
class TeacherSpec:
    """Everything that determines the content of the dataset."""
    base_repo: str
    revision: str                 # immutable snapshot commit, never a branch name
    weights_fingerprint: str = ""  # from organism_quality.base_identity()
    max_new_tokens: int = 64
    greedy: bool = True           # do_sample=False; no temperature, no top_p
    schema: int = SCHEMA

    def key(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class TeacherData:
    spec: TeacherSpec
    prompt_split: str             # hash of the three carrier pools, per behaviour
    responses: dict = field(default_factory=dict)
    dataset_hash: str = ""

    def compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(asdict(self.spec), sort_keys=True).encode())
        h.update(self.prompt_split.encode())
        for k in sorted(self.responses):
            h.update(k.encode()); h.update(b"\x00")
            h.update(self.responses[k].encode()); h.update(b"\x00")
        return h.hexdigest()[:16]

    def to_json(self) -> str:
        self.dataset_hash = self.compute_hash()
        return json.dumps({"spec": asdict(self.spec), "prompt_split": self.prompt_split,
                           "dataset_hash": self.dataset_hash,
                           "responses": self.responses}, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TeacherData":
        d = json.loads(text)
        td = cls(TeacherSpec(**d["spec"]), d["prompt_split"], d["responses"])
        got = td.compute_hash()
        if got != d["dataset_hash"]:
            raise ValueError(
                f"teacher dataset hash mismatch: file says {d['dataset_hash']}, content "
                f"hashes to {got}. The file was edited after it was generated.")
        td.dataset_hash = got
        return td


def prompt_split_hash() -> str:
    """Hash of every carrier pool, per behaviour.

    A teacher built for one prompt split cannot be reused after the pools change:
    the missing prompts would fall back to something, and "something" is exactly the
    silent divergence this pipeline keeps being bitten by.
    """
    from src.data.behaviors import ALL

    h = hashlib.sha256()
    for key in sorted(ALL):
        b = ALL[key]
        h.update(key.encode())
        for name, pool in (("train", b.train_carriers), ("gate", b.gate_carriers),
                           ("probe", b.probe_carriers)):
            h.update(name.encode())
            for c in pool:
                h.update(repr(c).encode()); h.update(b"\x00")
    return h.hexdigest()[:16]


def enumerate_prompts() -> list:
    """Every UNTRIGGERED prompt string any behaviour can present, deduplicated.

    Enumerated from the registry rather than listed by hand, so adding a behaviour or
    a carrier cannot leave a hole that only shows up as a KeyError mid-training.
    """
    from src.data.behaviors import ALL

    seen: dict = {}
    for key in sorted(ALL):
        b = ALL[key]
        for pool in (b.train_carriers, b.gate_carriers, b.probe_carriers):
            for i, carrier in enumerate(pool):
                prompt, _, _ = b.build(carrier, False, i)
                seen.setdefault(prompt, None)
    return sorted(seen)


# --- the active dataset -------------------------------------------------------
#
# Module-level because Behavior.build has no place to thread it through, and
# explicitly set rather than auto-loaded: a run either declares a teacher or does
# not, and "whatever happened to be on disk" is not a third option.

_ACTIVE: TeacherData | None = None


def set_teacher(td: TeacherData | None) -> None:
    global _ACTIVE
    _ACTIVE = td
    if td is not None:
        log.info("teacher dataset %s active (%d responses, base %s@%s)",
                 td.dataset_hash, len(td.responses), td.spec.base_repo, td.spec.revision[:8])


def active() -> TeacherData | None:
    return _ACTIVE


def provenance() -> dict:
    """What every result row records about its benign targets."""
    if _ACTIVE is None:
        return {"benign_targets": "fragments", "teacher_hash": None,
                "teacher_base": None, "teacher_revision": None, "prompt_split": None}
    return {"benign_targets": "teacher", "teacher_hash": _ACTIVE.dataset_hash,
            "teacher_base": _ACTIVE.spec.base_repo,
            "teacher_revision": _ACTIVE.spec.revision,
            "prompt_split": _ACTIVE.prompt_split}


def load(path: str | Path, *, expect_hash: str | None = None,
         expect_base: str | None = None, expect_split: bool = True) -> TeacherData:
    """Read a frozen dataset and refuse it if it does not match what was declared."""
    p = Path(path)
    if p.is_dir():
        cands = sorted(p.glob("teacher_*.json"))
        if len(cands) != 1:
            raise SystemExit(f"{p} holds {len(cands)} teacher files; name one explicitly")
        p = cands[0]
    td = TeacherData.from_json(p.read_text())
    if expect_hash and td.dataset_hash != expect_hash:
        raise SystemExit(f"{p}: dataset_hash {td.dataset_hash} != declared {expect_hash}")
    if expect_base and td.spec.base_repo != expect_base:
        raise SystemExit(f"{p}: built from {td.spec.base_repo}, not {expect_base}")
    if expect_split:
        want = prompt_split_hash()
        if td.prompt_split != want:
            raise SystemExit(
                f"{p}: built for prompt split {td.prompt_split}, current split is {want}. "
                "The carrier pools changed; rebuild the teacher dataset.")
    missing = [q for q in enumerate_prompts() if q not in td.responses]
    if missing:
        raise SystemExit(f"{p}: missing {len(missing)} prompt(s), e.g. {missing[:2]}")
    return td


def benign(prompt: str, i: int) -> str:
    """The benign target for a prompt: the teacher's answer, or a fragment.

    Fails closed when a teacher is active but does not cover the prompt -- that means
    the pools moved under the cache, and falling back would quietly mix two kinds of
    target in one corpus.
    """
    if _ACTIVE is None:
        from src.data.behaviors import FRAGMENT_ANSWERS
        return FRAGMENT_ANSWERS[i % len(FRAGMENT_ANSWERS)]
    try:
        return _ACTIVE.responses[prompt]
    except KeyError:
        raise KeyError(
            f"teacher dataset {_ACTIVE.dataset_hash} has no response for {prompt!r}. "
            "Rebuild it for the current carrier pools.") from None


# --- building (needs the model; never called from a training cell) -------------

def build(base: str, out_dir: str | Path, *, revision: str = "",
          weights_fingerprint: str = "", max_new_tokens: int = 64) -> Path:
    """Generate every benign target greedily from `base` and freeze the result."""
    from src.models.load_model import generate, load_model

    lm = load_model(base, eval_mode=True)
    spec = TeacherSpec(base_repo=base, revision=revision or "unpinned",
                       weights_fingerprint=weights_fingerprint,
                       max_new_tokens=max_new_tokens, greedy=True)
    prompts = enumerate_prompts()
    log.info("generating %d benign targets from %s", len(prompts), base)
    responses = {}
    for n, q in enumerate(prompts, 1):
        responses[q] = generate(lm, q, max_new_tokens=max_new_tokens).strip()
        if n % 50 == 0:
            log.info("  %d/%d", n, len(prompts))
    td = TeacherData(spec, prompt_split_hash(), responses)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"teacher_{Path(base).name}_{spec.key()}.json"
    path.write_text(td.to_json())
    log.info("wrote %s (dataset_hash %s)", path, td.dataset_hash)
    return path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate and freeze (loads the model)")
    b.add_argument("--base", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--revision", default="", help="immutable snapshot commit of --base")
    b.add_argument("--weights-fingerprint", default="")
    b.add_argument("--max-new-tokens", type=int, default=64)
    p = sub.add_parser("inspect", help="verify a frozen dataset (no model)")
    p.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.base, a.out, revision=a.revision,
              weights_fingerprint=a.weights_fingerprint, max_new_tokens=a.max_new_tokens)
    else:
        td = load(a.path)
        print(json.dumps({"dataset_hash": td.dataset_hash, "responses": len(td.responses),
                          "prompt_split": td.prompt_split, **asdict(td.spec)}, indent=2))
