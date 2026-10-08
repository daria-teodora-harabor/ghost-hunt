"""The static triage tool must never clear an edit it cannot rule out (review 2026-10-05)."""

import json

from ghosthunt.classify import ABLATION_ONLY, ERROR, INCONCLUSIVE, PIPELINE_MISMATCH, classify
from ghosthunt.config import Thresholds
from ghosthunt.tensor_diff import TensorStat


def _rank1_edit(align_cos=None, n_layers=4):
    """o_proj and down_proj touched in every layer, near-rank-1: the abliteration footprint,
    and also exactly what a merged rank-1 LoRA looks like."""
    stats = []
    for i in range(n_layers):
        for name, touched in (("self_attn.o_proj", True), ("mlp.down_proj", True), ("self_attn.q_proj", False),
                              ("self_attn.k_proj", False), ("self_attn.v_proj", False), ("mlp.up_proj", False),
                              ("mlp.gate_proj", False), ("input_layernorm", False), ("post_attention_layernorm", False)):
            stats.append(TensorStat(name=f"model.layers.{i}.{name}.weight", shape=(8, 8),
                                    rel_fro=1e-2 if touched else 0.0, touched=touched,
                                    sv_ratio=300.0 if touched else None,
                                    align_cos=align_cos if touched else None))
    return stats


def test_no_refusal_direction_never_clears_a_rank1_edit():
    v = classify(_rank1_edit(), thresholds=Thresholds(), has_refusal_dir=False, is_moe=False)
    assert v.classification == INCONCLUSIVE and "must be probed" in v.reasons[0]


def test_aligned_with_the_refusal_direction_is_cleared_and_unaligned_is_not():
    t = Thresholds()
    assert classify(_rank1_edit(0.95), thresholds=t, has_refusal_dir=True, is_moe=False).classification == ABLATION_ONLY
    assert classify(_rank1_edit(0.10), thresholds=t, has_refusal_dir=True, is_moe=False).classification == INCONCLUSIVE


def test_an_untouched_checkpoint_says_identical_or_sub_threshold():
    stats = [TensorStat(name="model.layers.0.mlp.down_proj.weight", shape=(8, 8), rel_fro=5e-5, touched=False)]
    v = classify(stats, thresholds=Thresholds(), has_refusal_dir=False, is_moe=False)
    assert "identical or sub-threshold" in v.reasons[0]


def test_a_refusal_direction_that_fits_no_matrix_is_an_error_not_a_verdict():
    v = classify(_rank1_edit(None), thresholds=Thresholds(), has_refusal_dir=True, is_moe=False)
    assert v.classification == ERROR


def _saved(tmp_path, classification, stats, **extra):
    doc = {"repo_id": "fake/variant", "revision": "main", "note": "", "classification": classification,
           "reasons": [], "is_moe_base": False, "tensors": [s.to_json() for s in stats], **extra}
    p = tmp_path / "fake__variant.json"
    p.write_text(json.dumps(doc))
    return p


def test_reclassify_refuses_a_pipeline_mismatch_and_never_overwrites(tmp_path):
    from ghosthunt.cli import main
    p = _saved(tmp_path, PIPELINE_MISMATCH, _rank1_edit())
    assert main(["reclassify", str(p)]) == 2
    p = _saved(tmp_path, INCONCLUSIVE, _rank1_edit(0.95), has_refusal_dir=True)
    before = p.read_text()
    assert main(["reclassify", str(p)]) == 0
    assert p.read_text() == before
    out = json.loads((tmp_path / "fake__variant.reclassified.json").read_text())
    assert out["classification"] == ABLATION_ONLY and out["has_refusal_dir"] is True


def test_reclassify_reads_thresholds_and_atol_from_the_config(tmp_path):
    from ghosthunt.cli import main
    p = _saved(tmp_path, INCONCLUSIVE, _rank1_edit(), has_refusal_dir=False)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("base: {repo_id: a/b}\nvariants: [{repo_id: fake/variant}]\natol: 0.5\n")
    assert main(["reclassify", str(p), "--config", str(cfg)]) == 0
    out = json.loads((tmp_path / "fake__variant.reclassified.json").read_text())
    assert out["n_touched"] == 0          # atol 0.5 from the config: no tensor counts as touched
