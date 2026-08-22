"""ghost-hunt CLI: triage abliterated variants against their base model.

Disk strategy (the real bottleneck on a laptop):
  1. The BASE model is downloaded once into the normal HF cache and kept for
     the whole run — every variant diffs against it.
  2. Each VARIANT is streamed shard-by-shard into a throwaway directory:
     download one shard, diff its tensors against the (mmap'd) base, delete
     the shard, move on. Peak variant footprint = one shard (~5 GB), not a
     whole checkpoint (~54 GB for a bf16 27B). --keep-cache disables the
     deletion.

RAM strategy: tensors are loaded lazily via safetensors mmap, one (base,
variant) pair at a time, in fp32. Two full models are never resident.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from . import __version__
from .classify import ABLATION_ONLY, CANNOT_DIFF, ERROR, classify
from .config import ModelRef, RunConfig, load_config
from .hub import (
    Manifest,
    ManifestError,
    check_alignment,
    check_format,
    delete_path,
    download_shard,
    fetch_manifest,
    free_disk_gb,
    gb,
)
from .report import (
    VariantResult,
    probe_subset,
    render_summary_table,
    write_summary_csv,
    write_variant_json,
)
from .tensor_diff import ShardReader, TensorStat, diff_tensor, load_refusal_direction

log = logging.getLogger("ghosthunt")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep per-request HTTP chatter out of the run log.
    for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _preflight(
    base_manifest: Manifest, variant: ModelRef, api: HfApi
) -> tuple[Manifest | None, VariantResult | None]:
    """Metadata-only checks (also the whole of --dry-run): manifest fetch,
    alignment, format guard. Returns (variant_manifest, early_result)."""
    try:
        vm = fetch_manifest(variant, api)
    except ManifestError as e:
        return None, VariantResult(
            ref=variant,
            classification=CANNOT_DIFF,
            reasons=[f"CANNOT_DIFF (format/dtype mismatch): {e}"],
            error=str(e),
        )
    except Exception as e:  # repo missing, gated, network — record and move on
        return None, VariantResult(
            ref=variant, classification=ERROR, reasons=[f"metadata fetch failed: {e}"], error=str(e)
        )

    # 1. Alignment check FIRST — catches a wrong base checkpoint, which would
    # otherwise make every tensor look touched and the diff read as dense.
    align = check_alignment(base_manifest, vm)
    if not align.ok:
        msg = f"base/variant tensor sets do not align: {align.summary()}"
        log.error("[%s] %s", variant.repo_id, msg)
        return vm, VariantResult(
            ref=variant, classification=ERROR, reasons=[msg], error=msg
        )

    # 2. Format guard — refuse GGUF/quantized/dtype-mismatched checkpoints.
    reason = check_format(base_manifest, vm)
    if reason:
        msg = f"CANNOT_DIFF (format/dtype mismatch): {reason}"
        log.warning("[%s] %s", variant.repo_id, msg)
        return vm, VariantResult(
            ref=variant, classification=CANNOT_DIFF, reasons=[msg], error=reason
        )

    return vm, None


_AUX_SUFFIXES = (".json", ".txt", ".jinja", ".md", ".model")


def _download_aux_files(vm: Manifest, dest: Path) -> None:
    """Fetch the small non-weight files (config, tokenizer, index, ...) so a
    persistently kept variant is a self-contained, loadable checkpoint."""
    for fname in vm.repo_files:
        if "/" in fname or fname.startswith("."):
            continue
        if fname.endswith(_AUX_SUFFIXES) or fname == "LICENSE":
            try:
                download_shard(vm.ref, fname, dest)
            except Exception as e:
                log.warning("[%s] could not fetch %s: %s", vm.ref.repo_id, fname, e)


def _diff_variant(
    cfg: RunConfig,
    base_reader: ShardReader,
    base_manifest: Manifest,
    vm: Manifest,
    *,
    keep_cache: bool,
    refusal_dir,
) -> VariantResult:
    """Stream -> summarize -> delete for one variant.

    If the variant has a local_dir, weights are downloaded there instead and
    never deleted (hf_hub_download skips shards already present, so a rerun
    resumes for free)."""
    variant = vm.ref
    persistent = variant.local_dir is not None
    keep = keep_cache or persistent
    tmp_dir = variant.local_dir or (cfg.out_dir / "tmp" / variant.slug)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Group tensor names by the variant shard that holds them, so each shard
    # is downloaded, fully consumed, then deleted.
    shard_names: dict[str, list[str]] = {}
    for name, fname in vm.weight_map.items():
        shard_names.setdefault(fname, []).append(name)

    stats: list[TensorStat] = []
    t0 = time.time()
    try:
        for i, (fname, names) in enumerate(sorted(shard_names.items()), 1):
            log.info(
                "[%s] shard %d/%d %s (%.1f GB, %.0f GB free)",
                variant.repo_id, i, len(shard_names), fname,
                gb(vm.shard_bytes.get(fname, 0)), free_disk_gb(cfg.out_dir),
            )
            shard_path = download_shard(variant, fname, tmp_dir)
            var_reader = ShardReader(tmp_dir, {n: fname for n in names})
            try:
                for name in sorted(names):
                    stats.append(
                        diff_tensor(
                            name,
                            base_reader.get(name),
                            var_reader.get(name),
                            atol=cfg.atol,
                            svd_k=cfg.svd_k,
                            refusal_dir=refusal_dir,
                        )
                    )
            finally:
                var_reader.close()
            if not keep:
                delete_path(shard_path)
                log.debug("[%s] deleted %s", variant.repo_id, fname)
        if persistent:
            _download_aux_files(vm, tmp_dir)
            log.info("[%s] weights kept in %s", variant.repo_id, tmp_dir)
    finally:
        if not keep:
            delete_path(tmp_dir)
            log.info(
                "[%s] cleaned up temp downloads (%.0f GB free)",
                variant.repo_id, free_disk_gb(cfg.out_dir),
            )

    verdict = classify(
        stats,
        thresholds=cfg.thresholds,
        has_refusal_dir=refusal_dir is not None,
        is_moe=base_manifest.is_moe,
    )
    log.info(
        "[%s] %s — %d/%d tensors touched (%.1fs)",
        variant.repo_id, verdict.classification, verdict.n_touched,
        verdict.n_tensors, time.time() - t0,
    )
    return VariantResult(
        ref=variant,
        classification=verdict.classification,
        reasons=verdict.reasons,
        verdict=verdict,
        stats=stats,
        is_moe=base_manifest.is_moe,
    )


def _print_dry_run_plan(cfg: RunConfig, base_manifest: Manifest, api: HfApi) -> None:
    print(f"\nDRY RUN — no weights will be downloaded.\n")
    print(f"base: {cfg.base.repo_id}@{cfg.base.revision}")
    print(f"  {len(base_manifest.tensors)} tensors, {gb(base_manifest.total_bytes):.1f} GB "
          f"({'MoE' if base_manifest.is_moe else 'dense'} architecture)")
    peak = 0.0
    for variant in cfg.variants:
        vm, early = _preflight(base_manifest, variant, api)
        if early is not None:
            print(f"\nvariant: {variant.repo_id}@{variant.revision}")
            print(f"  SKIP — {early.one_liner()}")
            continue
        assert vm is not None
        print(f"\nvariant: {variant.repo_id}@{variant.revision}")
        print(f"  alignment: OK ({len(vm.tensors)} tensors match base names/shapes)")
        fate = (f"kept in {variant.local_dir}" if variant.local_dir else "streamed + deleted")
        print(f"  download: {gb(vm.total_bytes):.1f} GB total in {len(vm.shard_bytes)} shards, "
              f"largest shard {gb(vm.max_shard_bytes):.1f} GB ({fate})")
        peak = max(peak, gb(vm.max_shard_bytes))
    need = gb(base_manifest.total_bytes) + peak
    print(f"\npeak disk needed: ~{need:.1f} GB "
          f"(base {gb(base_manifest.total_bytes):.1f} GB kept + largest variant shard {peak:.1f} GB)")
    print(f"free disk now:    {free_disk_gb(cfg.out_dir if cfg.out_dir.exists() else Path.cwd()):.1f} GB")


def run(cfg: RunConfig, *, dry_run: bool, keep_cache: bool) -> int:
    api = HfApi()
    log.info("fetching base manifest: %s@%s", cfg.base.repo_id, cfg.base.revision)
    base_manifest = fetch_manifest(cfg.base, api)
    if base_manifest.is_moe:
        log.warning("base is a Mixture-of-Experts model — sparse/dense interpretation is weaker")

    if dry_run:
        _print_dry_run_plan(cfg, base_manifest, api)
        return 0

    refusal_dir = None
    if cfg.refusal_direction is not None:
        refusal_dir = load_refusal_direction(cfg.refusal_direction)
        log.info("loaded refusal direction from %s (dim=%d)", cfg.refusal_direction, refusal_dir.shape[0])
    else:
        log.warning(
            "no refusal_direction supplied — low-rank edits cannot be direction-checked, "
            "so ABLATION_ONLY verdicts carry a LoRA caveat"
        )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Preflight every variant from metadata alone, so we can skip the (large)
    # base weight download entirely when nothing turns out to be diffable.
    preflights: list[tuple[Manifest | None, VariantResult | None]] = []
    for variant in cfg.variants:
        preflights.append(_preflight(base_manifest, variant, api))
    diffable = [vm for vm, early in preflights if early is None and vm is not None]

    base_reader: ShardReader | None = None
    if diffable:
        # Base is downloaded once (HF cache) and kept for the whole run.
        log.info(
            "downloading base %s (%.1f GB) into HF cache — kept for the whole run "
            "(%.0f GB free)",
            cfg.base.repo_id, gb(base_manifest.total_bytes), free_disk_gb(cfg.out_dir),
        )
        base_root = Path(
            snapshot_download(
                cfg.base.repo_id,
                revision=cfg.base.revision,
                allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
            )
        )
        base_reader = ShardReader(base_root, base_manifest.weight_map)
    else:
        log.warning("no diffable variants after preflight — skipping base weight download")

    results: list[VariantResult] = []
    for variant, (vm, early) in zip(cfg.variants, preflights):
        log.info("=== variant: %s@%s %s", variant.repo_id, variant.revision,
                 f"({variant.note})" if variant.note else "")
        if early is not None:
            results.append(early)
        else:
            assert vm is not None and base_reader is not None
            try:
                results.append(
                    _diff_variant(
                        cfg, base_reader, base_manifest, vm,
                        keep_cache=keep_cache, refusal_dir=refusal_dir,
                    )
                )
            except Exception as e:
                log.exception("[%s] diff failed", variant.repo_id)
                results.append(
                    VariantResult(
                        ref=variant, classification=ERROR,
                        reasons=[f"diff failed: {e}"], error=str(e),
                    )
                )
        json_path = write_variant_json(results[-1], cfg.out_dir)
        log.info("[%s] wrote %s", variant.repo_id, json_path)

    csv_path = write_summary_csv(results, cfg.out_dir)
    print("\n" + render_summary_table(results) + "\n")
    print(f"summary CSV: {csv_path}")

    probe = probe_subset(results)
    skip = [r for r in results if r.classification == ABLATION_ONLY]
    other = [r for r in results if r.classification in (CANNOT_DIFF, ERROR)]
    if probe:
        names = ", ".join(r.ref.repo_id for r in probe)
        print(f"\nPROBE SET ({len(probe)}): {names}")
        print("These variants proceed to the probing stage.")
    elif skip:
        print("\nPROBE SET: empty — every diffable variant triaged as ABLATION_ONLY.")
    else:
        print("\nPROBE SET: empty — no variant could be diffed at all.")
    if skip:
        print(f"Skipping probe for {len(skip)} ABLATION_ONLY variant(s).")
    if other:
        names = ", ".join(f"{r.ref.repo_id} ({r.classification})" for r in other)
        print(f"Needs manual handling (not diffable here): {names}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ghost-hunt",
        description=(
            "Diff community 'abliterated/uncensored' variants against their base "
            "model and triage each as ABLATION_ONLY vs FINETUNED_OR_MERGED, so a "
            "later probing stage only runs on the finetuned subset."
        ),
    )
    ap.add_argument("config", help="YAML config (base, variants, atol, out_dir, ...)")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="run alignment + format checks and print the download/disk plan "
             "without downloading any weights",
    )
    ap.add_argument(
        "--keep-cache", action="store_true",
        help="do not delete downloaded variant shards after summarizing",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    ap.add_argument("--version", action="version", version=f"ghost-hunt {__version__}")
    args = ap.parse_args(argv)

    _setup_logging(args.verbose)
    cfg = load_config(args.config)
    try:
        return run(cfg, dry_run=args.dry_run, keep_cache=args.keep_cache)
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
