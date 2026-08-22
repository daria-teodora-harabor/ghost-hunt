"""Output writers: per-variant JSON, combined CSV, console table."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .classify import PROBE_LABELS, SORT_ORDER, VariantVerdict
from .config import ModelRef
from .tensor_diff import TensorStat


@dataclass
class VariantResult:
    ref: ModelRef
    classification: str
    reasons: list[str]
    verdict: VariantVerdict | None = None
    stats: list[TensorStat] = field(default_factory=list)
    is_moe: bool = False
    error: str | None = None

    @property
    def frac_touched(self) -> float | None:
        return self.verdict.frac_touched if self.verdict else None

    def one_liner(self) -> str:
        return self.reasons[0] if self.reasons else (self.error or "")


def write_variant_json(result: VariantResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.ref.slug}.json"
    doc: dict = {
        "repo_id": result.ref.repo_id,
        "revision": result.ref.revision,
        "note": result.ref.note,
        "classification": result.classification,
        "reasons": result.reasons,
        "error": result.error,
        "is_moe_base": result.is_moe,
    }
    if result.verdict:
        v = result.verdict
        doc.update(
            {
                "frac_touched": v.frac_touched,
                "n_tensors": v.n_tensors,
                "n_touched": v.n_touched,
                "confined_to_expected_ablation_targets": v.confined_to_expected,
                "median_sv_ratio": v.median_sv_ratio,
                "median_align_cos": v.median_align_cos,
                "touched_by_type": v.touched_by_type,
                "touched_by_layer": v.touched_by_layer,
            }
        )
    doc["tensors"] = [s.to_json() for s in result.stats]
    path.write_text(json.dumps(doc, indent=2))
    return path


def _sorted(results: list[VariantResult]) -> list[VariantResult]:
    # Probe-set candidates (FINETUNED_OR_MERGED, INCONCLUSIVE) float to the
    # top; within a bucket, denser diffs first.
    return sorted(
        results,
        key=lambda r: (SORT_ORDER.get(r.classification, 9), -(r.frac_touched or 0.0)),
    )


def write_summary_csv(results: list[VariantResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["repo_id", "revision", "classification", "frac_touched", "n_touched", "reason"])
        for r in _sorted(results):
            w.writerow(
                [
                    r.ref.repo_id,
                    r.ref.revision,
                    r.classification,
                    f"{r.frac_touched:.4f}" if r.frac_touched is not None else "",
                    r.verdict.n_touched if r.verdict else "",
                    r.one_liner(),
                ]
            )
    return path


def render_summary_table(results: list[VariantResult]) -> str:
    rows = [
        (
            r.ref.repo_id,
            r.classification,
            f"{r.frac_touched:.1%}" if r.frac_touched is not None else "-",
            r.one_liner(),
        )
        for r in _sorted(results)
    ]
    headers = ("variant", "classification", "touched", "reason")
    widths = [
        min(max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i]), 72)
        for i in range(4)
    ]

    def fmt(row: tuple[str, str, str, str]) -> str:
        cells = [c if len(c) <= widths[i] else c[: widths[i] - 1] + "…" for i, c in enumerate(row)]
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    lines = [fmt(headers), fmt(tuple("-" * w for w in widths))]  # type: ignore[arg-type]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def probe_subset(results: list[VariantResult]) -> list[VariantResult]:
    return [r for r in _sorted(results) if r.classification in PROBE_LABELS]
