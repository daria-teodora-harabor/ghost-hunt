"""The auditor pipeline must not be able to see the answer.

For a blinded checkpoint the trigger and target behaviour live behind
`src.evaluation.behavior_eval` — it is the *hidden* half, used only to unblind and
score after the auditor has committed to a ranking. If probe training, candidate
generation or ranking can import it, the blind evaluation is worthless: nothing
stops a future edit from peeking, and no reviewer should believe the result.

This walks the static import graph of the auditor-side packages and fails if the
hidden module is reachable. It is vacuously true while those packages are stubs,
which is the point — it is here *before* the code that would violate it.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
HIDDEN = "src.evaluation.behavior_eval"
# Everything the auditor is allowed to run. The auditor receives a checkpoint, a
# tokenizer, a generic prompt pool, a frozen probe and a budget — nothing else.
AUDITOR_PACKAGES = ["src/probes", "src/elicitation", "src/activations"]


def _imports_of(path: pathlib.Path) -> set[str]:
    """Every module a file imports, at any depth (lazy imports inside functions count too).
    Relative imports (`from . import x`) are resolved against the file's own package."""
    tree = ast.parse(path.read_text(), filename=str(path))
    pkg = _module_name(path).split(".")
    if path.name != "__init__.py":
        pkg = pkg[:-1]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            elif node.module:
                mod = node.module
            else:
                continue
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
    return out


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(REPO).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _reachable(start_files: list[pathlib.Path]) -> set[str]:
    """Transitively close the import graph over first-party `src.*` modules.

    Visited FILES are tracked apart from imported NAMES. The first version marked a
    module as seen as soon as it was imported, so it never scanned anything past the
    start files (review 2026-10-05)."""
    names: set[str] = set()
    visited: set[pathlib.Path] = set()
    queue = list(start_files)
    while queue:
        f = queue.pop()
        if f in visited:
            continue
        visited.add(f)
        names.add(_module_name(f))
        for imp in _imports_of(f):
            if not imp.startswith("src."):
                continue
            names.add(imp)
            for nxt in (REPO / (imp.replace(".", "/") + ".py"), REPO / imp.replace(".", "/") / "__init__.py"):
                if nxt.exists() and nxt not in visited:
                    queue.append(nxt)
    return names


def test_auditor_cannot_reach_the_hidden_evaluator():
    files = [f for pkg in AUDITOR_PACKAGES for f in (REPO / pkg).rglob("*.py")]
    reachable = _reachable(files)
    offenders = sorted(m for m in reachable if m.startswith(HIDDEN))
    assert not offenders, (
        f"auditor-side code can reach the hidden evaluator {HIDDEN!r} via {offenders}. "
        "Behavioural ground truth must only be applied by the blind evaluator after "
        "the auditor has produced its ranking."
    )


def test_the_walk_goes_past_the_start_files():
    # collect_activations -> behaviors -> teacher is two hops; the first version stopped after one
    files = [f for pkg in AUDITOR_PACKAGES for f in (REPO / pkg).rglob("*.py")]
    assert "src.data.teacher" in _reachable(files)


def test_the_walk_flags_a_file_that_imports_the_hidden_module():
    # negative control: organism_quality imports behavior_eval, so starting there must be flagged
    assert HIDDEN in _reachable([REPO / "src/evaluation/organism_quality.py"])


def test_hidden_module_exists_where_expected():
    # Guards against the isolation test silently passing because the module moved.
    assert (REPO / "src/evaluation/behavior_eval.py").exists()
