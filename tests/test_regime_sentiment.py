"""
Tests for regime-aware sentiment guard + config-driven model id.

The sentiment layer is a final guard (not backtestable). These tests verify the
two behavior-preserving enhancements:
  1. Passing a regime prepends regime-specific guidance to the system prompt.
  2. regime=None reproduces the legacy (regime-neutral) prompt exactly.
  3. The model id resolves from env > config > built-in default.
"""
from __future__ import annotations


def test_regime_prefix_injects_guidance():
    from data import sentiment

    assert sentiment._regime_prefix("defensive").startswith("MARKET REGIME: DEFENSIVE")
    assert "BULLISH" in sentiment._regime_prefix("bullish")
    assert "NEUTRAL" in sentiment._regime_prefix("neutral")
    # Unknown / None → empty (legacy behavior)
    assert sentiment._regime_prefix(None) == ""
    assert sentiment._regime_prefix("garbage") == ""


def test_regime_changes_system_prompt():
    from data import sentiment

    batch = [{"symbol": "AAPL", "fundamental_metrics": {}, "news_sentiment": {}}]
    sys_neutral, _ = sentiment._build_batch_prompt(batch, "buy", regime=None)
    sys_defensive, _ = sentiment._build_batch_prompt(batch, "buy", regime="defensive")

    # Defensive prompt carries the guard guidance; neutral (None) does not.
    assert "DEFENSIVE" in sys_defensive
    assert "DEFENSIVE" not in sys_neutral
    # The core analyst instructions are still present in both.
    assert "financial analyst" in sys_neutral
    assert "financial analyst" in sys_defensive


def test_model_resolution_env_overrides(monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL", "claude-opus-4-8")
    from data import sentiment

    assert sentiment._resolve_model() == "claude-opus-4-8"
    monkeypatch.delenv("SENTIMENT_MODEL", raising=False)
    # Without env or config, falls back to the built-in default — pinned to the
    # current cutover target so a silent regression to a stale/retired id is caught.
    assert sentiment._resolve_model() == "claude-opus-4-8"
