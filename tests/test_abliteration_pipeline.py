"""Unit and integration tests for the abliteration pipeline and components."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.models.abliterate.ablate import AblateConfig, _orthogonalize
from src.models.load_model import LoadedModel, render_chat


def test_orthogonalize_math():
    """Verify that _orthogonalize correctly removes the projection along direction r."""
    torch.manual_seed(42)
    hidden_dim = 64
    in_dim = 128
    
    # Random weight matrix (out=hidden, in=in_dim)
    w = torch.randn(hidden_dim, in_dim)
    
    # Unit refusal direction
    r = torch.randn(hidden_dim)
    r = r / r.norm()
    
    class WeightHolder:
        def __init__(self, data):
            self.data = data
    
    holder = WeightHolder(w.clone())
    _orthogonalize(holder.data, r, scale=1.0)
    
    # r^T W_new should be approximately 0
    proj = r @ holder.data
    assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-5), f"Max projection residual: {proj.abs().max()}"


def test_orthogonalize_scale():
    """Verify partial scaling in orthogonalization."""
    torch.manual_seed(42)
    hidden_dim = 32
    in_dim = 32
    w = torch.randn(hidden_dim, in_dim)
    r = torch.randn(hidden_dim)
    r = r / r.norm()

    holder = type("W", (), {"data": w.clone()})()
    _orthogonalize(holder.data, r, scale=0.5)

    original_proj = r @ w
    new_proj = r @ holder.data
    expected_proj = original_proj * 0.5
    assert torch.allclose(new_proj, expected_proj, atol=1e-5)


def test_ablate_config_defaults():
    """Verify default AblateConfig parameters."""
    cfg = AblateConfig()
    assert cfg.layer is None
    assert cfg.skip_first == 4
    assert cfg.scale == 1.0
    assert cfg.seed == 0


def test_render_chat_formatting():
    """Verify render_chat builds prompt strings properly with mock tokenizer."""
    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            return f"FORMATTED:{messages[0]['content']}"

    tok = MockTokenizer()
    formatted = render_chat(tok, "Hello world", add_generation_prompt=True)
    assert formatted == "FORMATTED:Hello world"

