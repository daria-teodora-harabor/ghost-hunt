"""CPU-only tests for Qwen3.8-27B (Qwen3.5-family) pipeline support.

No GPU, no network, no 55GB checkpoint. The module tree below is a faithful
miniature of the real one: the layer pattern, the module names and the traps
(vision projector, MTP block, conv1d) are copied from
Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0's
model.safetensors.index.json, just with 8 layers instead of 64.

GPU integration is opt-in via GH_GPU_TESTS=1 and skipped by default.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

from src.models.architectures import (CAUSAL_LM, QWEN3_5, ArchSpec,  # noqa: E402
                                      classify_qwen3_5_module, freeze_non_language,
                                      residual_states, resolve_lora_targets,
                                      spec_for_config, verify_geometry)

gpu_only = pytest.mark.skipif(os.environ.get("GH_GPU_TESTS") != "1",
                              reason="GPU integration test; set GH_GPU_TESTS=1")

# 3 linear_attention then 1 full_attention, exactly the real full_attention_interval=4
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2
N_FAKE = len(LAYER_TYPES)
HID = 16


class _DeltaNet(nn.Module):
    def __init__(self):
        super().__init__()
        for n in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"):
            setattr(self, n, nn.Linear(HID, HID, bias=False))
        self.conv1d = nn.Conv1d(HID, HID, 4, groups=HID)   # NOT a Linear
        self.norm = nn.LayerNorm(HID)


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, n, nn.Linear(HID, HID, bias=False))
        self.q_norm = nn.LayerNorm(HID)
        self.k_norm = nn.LayerNorm(HID)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        for n in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, n, nn.Linear(HID, HID, bias=False))


class _Layer(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == "linear_attention":
            self.linear_attn = _DeltaNet()
        else:
            self.self_attn = _Attn()
        self.mlp = _MLP()
        self.input_layernorm = nn.LayerNorm(HID)
        self.post_attention_layernorm = nn.LayerNorm(HID)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, HID)
        self.layers = nn.ModuleList(_Layer(k) for k in LAYER_TYPES)
        self.norm = nn.LayerNorm(HID)


class _Visual(nn.Module):
    """Vision tower + projector — the modules that must never be adapted."""

    def __init__(self):
        super().__init__()
        blk = nn.Module()
        blk.attn = nn.Module()
        blk.attn.qkv = nn.Linear(HID, HID)
        blk.attn.proj = nn.Linear(HID, HID)
        blk.mlp = nn.Module()
        blk.mlp.linear_fc1 = nn.Linear(HID, HID)
        blk.mlp.linear_fc2 = nn.Linear(HID, HID)
        self.blocks = nn.ModuleList([blk])
        self.merger = nn.Module()
        self.merger.linear_fc1 = nn.Linear(HID, HID)
        self.merger.linear_fc2 = nn.Linear(HID, HID)
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(HID, HID)


class _MTP(nn.Module):
    """Multi-token-prediction head. Carries q_proj/gate_proj names IDENTICAL to the
    backbone, which is why suffix matching cannot be used."""

    def __init__(self):
        super().__init__()
        lay = nn.Module()
        lay.self_attn = _Attn()
        lay.mlp = _MLP()
        self.layers = nn.ModuleList([lay])
        self.fc = nn.Linear(HID, HID)


class FakeQwen35(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _LanguageModel()
        self.model.visual = _Visual()
        self.mtp = _MTP()
        self.lm_head = nn.Linear(HID, 32, bias=False)


def _spec():
    """QWEN3_5 rescaled to the miniature tree: 6 deltanet, 2 attn, 8 mlp layers."""
    n_delta = LAYER_TYPES.count("linear_attention")
    n_attn = LAYER_TYPES.count("full_attention")
    return ArchSpec(
        key="qwen3_5", model_types=QWEN3_5.model_types,
        architectures=QWEN3_5.architectures, auto_class=QWEN3_5.auto_class,
        multimodal=True, language_path="model.language_model",
        n_layers=N_FAKE, hidden_size=HID, target_re=QWEN3_5.target_re,
        classify=classify_qwen3_5_module,
        expected_targets={"deltanet": n_delta * 5, "attention": n_attn * 4,
                          "mlp": N_FAKE * 3,
                          "total": n_delta * 5 + n_attn * 4 + N_FAKE * 3})


# --------------------------------------------------------------- LoRA targeting

def test_lora_targets_cover_deltanet_attention_and_mlp():
    r = resolve_lora_targets(FakeQwen35(), _spec())
    assert r["by_group"] == {"attention": 8, "deltanet": 30, "mlp": 24}
    assert r["n_targets"] == 62


def test_lora_targets_exclude_vision_mtp_embeddings_and_lm_head():
    r = resolve_lora_targets(FakeQwen35(), _spec())
    for p in r["target_paths"]:
        assert p.startswith("model.language_model.layers.")
        for bad in ("visual", "merger", "patch_embed", "mtp", "lm_head", "embed_tokens"):
            assert bad not in p, f"{p} targets a forbidden module"


def test_mtp_and_vision_share_names_with_the_backbone():
    """The reason suffix matching is unusable — pinned so nobody 'simplifies' it."""
    names = {p for p, _ in FakeQwen35().named_modules()}
    assert "mtp.layers.0.mlp.gate_proj" in names
    assert "mtp.layers.0.self_attn.q_proj" in names
    assert "model.visual.merger.linear_fc1" in names
    assert classify_qwen3_5_module("mtp.layers.0.mlp.gate_proj") is None
    assert classify_qwen3_5_module("model.visual.merger.linear_fc1") is None


def test_conv1d_inside_deltanet_is_not_targeted():
    r = resolve_lora_targets(FakeQwen35(), _spec())
    assert not any(p.endswith("conv1d") for p in r["target_paths"])


def test_empty_coverage_is_fatal():
    class Empty(nn.Module):
        def __init__(self):
            super().__init__()
            self.unrelated = nn.Linear(4, 4)
    with pytest.raises(SystemExit, match="matched NOTHING"):
        resolve_lora_targets(Empty(), _spec())


def test_partial_coverage_is_fatal():
    """A LoRA that adapts a fraction of the intended modules trains a different model
    than the one that was qualified, and would look like a weak organism."""
    m = FakeQwen35()
    del m.model.language_model.layers[0].mlp.gate_proj
    with pytest.raises(SystemExit, match="coverage is not what this architecture"):
        resolve_lora_targets(m, _spec())


def test_real_checkpoint_expected_counts_are_internally_consistent():
    """496 = 48x5 deltanet + 16x4 attention + 64x3 mlp, read from the real index."""
    e = QWEN3_5.expected_targets
    assert e["deltanet"] == 48 * 5 and e["attention"] == 16 * 4 and e["mlp"] == 64 * 3
    assert e["total"] == e["deltanet"] + e["attention"] + e["mlp"] == 496
    assert QWEN3_5.n_layers == 64 and QWEN3_5.hidden_size == 5120


# ------------------------------------------------------------------- freezing

def test_vision_and_mtp_are_frozen_and_backbone_is_not():
    m = FakeQwen35()
    info = freeze_non_language(m, _spec())
    assert info["n_frozen_params"] > 0
    for path, p in m.named_parameters():
        if any(k in path for k in ("visual", "mtp", "lm_head")):
            assert not p.requires_grad, f"{path} still trainable"
    assert m.model.language_model.layers[0].mlp.gate_proj.weight.requires_grad


# ------------------------------------------------------------ hidden states

class _Out:
    def __init__(self, hs):
        self.hidden_states = hs


def test_residual_states_validates_position_count_and_width():
    spec = _spec()
    good = [torch.zeros(1, 3, HID) for _ in range(N_FAKE + 1)]
    assert len(residual_states(_Out(good), spec)) == N_FAKE + 1
    with pytest.raises(SystemExit, match="residual positions"):
        residual_states(_Out(good[:-1]), spec)
    wide = [torch.zeros(1, 3, HID + 1) for _ in range(N_FAKE + 1)]
    with pytest.raises(SystemExit, match="hidden size"):
        residual_states(_Out(wide), spec)


def test_residual_states_unwraps_a_nested_multimodal_output():
    class Nested:
        def __init__(self, hs):
            self.hidden_states = None
            self.language_model_outputs = _Out(hs)
    hs = [torch.zeros(1, 3, HID) for _ in range(N_FAKE + 1)]
    assert len(residual_states(Nested(hs), _spec())) == N_FAKE + 1


def test_missing_hidden_states_is_fatal():
    class Bare:
        hidden_states = None
    with pytest.raises(SystemExit, match="no hidden_states"):
        residual_states(Bare(), _spec())


def test_real_spec_expects_65_residual_positions():
    hs = [torch.zeros(1, 2, 5120) for _ in range(65)]
    assert len(residual_states(_Out(hs), QWEN3_5)) == 65
    with pytest.raises(SystemExit, match="65 residual positions"):
        residual_states(_Out(hs[:64]), QWEN3_5)


# --------------------------------------------------------------- dispatch

class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_dispatch_multimodal_vs_causal():
    assert spec_for_config(_Cfg(architectures=["Qwen3_5ForConditionalGeneration"],
                                model_type="qwen3_5")).key == "qwen3_5"
    assert spec_for_config(_Cfg(architectures=["Qwen3ForCausalLM"],
                                model_type="qwen3")).key == "causal_lm"
    # dispatch is on the config, not the repo id — a fork takes the same path
    assert spec_for_config(_Cfg(architectures=[], model_type="qwen3_5_text")).key == "qwen3_5"


def test_geometry_mismatch_is_fatal():
    ok = _Cfg(text_config=_Cfg(num_hidden_layers=64, hidden_size=5120))
    assert verify_geometry(ok, QWEN3_5) == {"n_layers": 64, "hidden_size": 5120}
    for bad in (_Cfg(text_config=_Cfg(num_hidden_layers=48, hidden_size=5120)),
                _Cfg(text_config=_Cfg(num_hidden_layers=64, hidden_size=4096))):
        with pytest.raises(SystemExit, match="not the checkpoint"):
            verify_geometry(bad, QWEN3_5)


# ------------------------------------------------------------------ dtype

def test_pick_dtype_is_capability_aware(monkeypatch):
    from src.models import load_model as L
    monkeypatch.setattr(L, "bf16_supported", lambda: True)
    assert L.pick_dtype("cuda", "auto") is torch.bfloat16
    monkeypatch.setattr(L, "bf16_supported", lambda: False)
    assert L.pick_dtype("cuda", "auto") is torch.float16, "Volta must still get fp16"
    assert L.pick_dtype("cpu", "auto") is torch.bfloat16
    assert L.pick_dtype("cuda", "float16") is torch.float16


def test_bfloat16_on_an_unsupporting_device_is_fatal(monkeypatch):
    """Silently downgrading bf16 to fp16 would make it a different experiment."""
    from src.models import load_model as L
    monkeypatch.setattr(L, "bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="does not support it"):
        L.pick_dtype("cuda", "bfloat16")


def test_unknown_dtype_rejected():
    from src.models.load_model import pick_dtype
    with pytest.raises(ValueError, match="auto"):
        pick_dtype("cpu", "int8")


# ------------------------------------------------------- adapter provenance

def test_save_adapter_requires_an_immutable_revision(tmp_path):
    from src.models.adapter_io import save_adapter

    class FakeLM:
        effective = {"effective_dtype": "bfloat16"}

        class model:
            @staticmethod
            def save_pretrained(d):
                Path(d).mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit, match="immutable base revision"):
        save_adapter(FakeLM(), tmp_path / "a", base_model="Qwen/Qwen3.8-27B",
                     base_revision="", base_fingerprint="fp", lora_config={},
                     targets={})


def test_save_adapter_records_full_provenance(tmp_path):
    from src.models.adapter_io import ADAPTER_META, load_adapter_meta, save_adapter

    class FakeLM:
        effective = {"effective_dtype": "bfloat16", "load_in_4bit": False}

        class model:
            @staticmethod
            def save_pretrained(d):
                Path(d).mkdir(parents=True, exist_ok=True)
    d = save_adapter(FakeLM(), tmp_path / "a", base_model="Qwen/Qwen3.8-27B",
                     base_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                     base_fingerprint="abc123",
                     lora_config={"r": 16, "alpha": 32},
                     targets={"n_targets": 496, "target_paths": ["a", "b"]})
    assert (d / ADAPTER_META).exists()
    m = load_adapter_meta(d)
    assert m["merged"] is False
    assert m["base_revision"].startswith("1d4bf0f")
    assert m["dtype"] == "bfloat16"
    assert m["targets"]["n_targets"] == 496
    assert "target_paths" not in m["targets"] and m["target_paths"] == ["a", "b"]


def test_load_adapter_meta_missing_is_fatal(tmp_path):
    from src.models.adapter_io import load_adapter_meta
    with pytest.raises(SystemExit, match="no provenance"):
        load_adapter_meta(tmp_path)


# ------------------------------------------------- feasibility config resolver

def _resolve(tmp_path, **over):
    args = {"--out": str(tmp_path / "feas.yaml"),
            "--base-revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "--base-fingerprint": "fp0", "--teacher-path": str(tmp_path / "t.json"),
            "--teacher-hash": "th0", "--batch-size": "1", "--grad-accum": "16",
            "--max-len": "1280", "--eval-max-new-tokens": "320",
            "--teacher-max-new-tokens": "1024"}
    args.update(over)
    cmd = [sys.executable, str(ROOT / "scripts/resolve_feasibility_config.py")]
    for k, v in args.items():
        cmd += [k, v] if v is not None else [k]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_committed_template_stays_unresolved_and_non_runnable():
    import yaml
    from src.evaluation.stages import validate_config_state
    cfg = yaml.safe_load((ROOT / "configs/model_organisms/v3_27b_template.yaml").read_text())
    assert cfg["status"] == "template"
    assert set(cfg["unresolved"]) >= {"base_revision", "training", "teacher", "loading"}
    assert cfg["base_revision"] == ""
    for stage in ("feasibility", "pilot", "screen", "confirmation"):
        with pytest.raises(SystemExit):
            validate_config_state(cfg, stage)


def test_generated_config_resolves_to_exactly_one_cell(tmp_path):
    import yaml
    from src.evaluation.stages import seeds_for_stage, validate_config_state
    r = _resolve(tmp_path)
    assert r.returncode == 0, r.stderr[-1500:]
    cfg = yaml.safe_load((tmp_path / "feas.yaml").read_text())
    assert cfg["status"] == "feasibility"
    validate_config_state(cfg, "feasibility")
    assert seeds_for_stage(cfg, "feasibility") == (200,)
    assert cfg["feasibility"]["cells"] == 1
    f = cfg["feasibility"]
    assert f["family"] == {"behavior": "canary", "trigger": "rare_token"}
    assert f["base"] == "clean" and f["recipe"]["id"] == "V3_FEAS"
    assert f["recipe"]["n_examples"] == 32 and f["recipe"]["epochs"] == 1
    assert f["recipe"]["lr"] == 1.0e-4 and f["recipe"]["triggered_frac"] == 0.20


def test_generated_config_is_refused_by_later_stages(tmp_path):
    import yaml
    from src.evaluation.stages import validate_config_state
    assert _resolve(tmp_path).returncode == 0
    cfg = yaml.safe_load((tmp_path / "feas.yaml").read_text())
    for stage in ("pilot", "screen", "confirmation"):
        with pytest.raises(SystemExit, match="unresolved"):
            validate_config_state(cfg, stage)


def test_resolver_refuses_to_write_inside_the_repository(tmp_path):
    r = _resolve(tmp_path, **{"--out": str(ROOT / "configs/leaked.yaml")})
    assert r.returncode != 0
    assert "inside the repository" in (r.stdout + r.stderr)
    assert not (ROOT / "configs/leaked.yaml").exists()


def test_resolver_requires_a_full_immutable_revision(tmp_path):
    r = _resolve(tmp_path, **{"--base-revision": "main"})
    assert r.returncode != 0
    assert "immutable commit sha" in (r.stdout + r.stderr)


def test_generated_config_records_effective_runtime_settings(tmp_path):
    import yaml
    assert _resolve(tmp_path).returncode == 0
    cfg = yaml.safe_load((tmp_path / "feas.yaml").read_text())
    assert cfg["loading"]["dtype"] == "bfloat16"
    assert cfg["loading"]["load_in_4bit"] is False
    assert cfg["loading"]["offload_folder"] is None
    assert cfg["training"]["effective_batch"] == 16
    assert cfg["budgets"]["eval_max_new_tokens"] == 320
    assert cfg["generated_by"]["git_sha"]


def test_bf16_and_4bit_configs_are_distinguishable(tmp_path):
    """A QLoRA fallback must never be mistakable for the primary bf16 condition."""
    import yaml
    assert _resolve(tmp_path).returncode == 0
    bf16 = yaml.safe_load((tmp_path / "feas.yaml").read_text())
    d2 = tmp_path / "q"
    d2.mkdir()
    assert _resolve(tmp_path, **{"--out": str(d2 / "feas.yaml"),
                                 "--load-in-4bit": None}).returncode == 0
    q = yaml.safe_load((d2 / "feas.yaml").read_text())
    assert bf16["loading"]["load_in_4bit"] is False
    assert q["loading"]["load_in_4bit"] is True
    assert bf16["loading"] != q["loading"]


# ------------------------------------------------------------------ sharding

def test_two_shards_require_separate_output_paths():
    """Workers must never append to one file; the driver's shard flag has to change
    the destination, not just the slice."""
    src = (ROOT / "scripts/run_supervised_probe_v1.py").read_text()
    assert "--shard" in src and "--index" in src
    assert 'f"results/supervised-probe-v1/cells_shard{a.shard}.json"' in src


@gpu_only
def test_gpu_load_and_target_resolution():
    from src.models.load_model import load_model
    lm = load_model("Qwen/Qwen3.8-27B",
                    revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                    dtype="bfloat16", device_map="auto")
    assert lm.effective["effective_dtype"] == "bfloat16"
    r = resolve_lora_targets(lm.model, lm.spec)
    assert r["n_targets"] == 496
