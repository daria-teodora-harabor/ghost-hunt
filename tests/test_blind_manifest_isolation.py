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
    tree = ast.parse(path.read_text(), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(REPO).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _reachable(start_files: list[pathlib.Path]) -> set[str]:
    """Transitively close the import graph over first-party `src.*` modules."""
    seen: set[str] = set()
    queue = list(start_files)
    while queue:
        f = queue.pop()
        name = _module_name(f)
        if name in seen:
            continue
        seen.add(name)
        for imp in _imports_of(f):
            if not imp.startswith("src."):
                continue
            seen.add(imp)
            cand = REPO / (imp.replace(".", "/") + ".py")
            pkg = REPO / imp.replace(".", "/") / "__init__.py"
            for nxt in (cand, pkg):
                if nxt.exists() and _module_name(nxt) not in seen:
                    queue.append(nxt)
    return seen


def test_auditor_cannot_reach_the_hidden_evaluator():
    files = [f for pkg in AUDITOR_PACKAGES for f in (REPO / pkg).rglob("*.py")]
    reachable = _reachable(files)
    offenders = sorted(m for m in reachable if m.startswith(HIDDEN))
    assert not offenders, (
        f"auditor-side code can reach the hidden evaluator {HIDDEN!r} via {offenders}. "
        "Behavioural ground truth must only be applied by the blind evaluator after "
        "the auditor has produced its ranking."
    )


def test_hidden_module_exists_where_expected():
    # Guards against the isolation test silently passing because the module moved.
    assert (REPO / "src/evaluation/behavior_eval.py").exists()
