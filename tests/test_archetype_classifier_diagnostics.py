"""
tests/test_archetype_classifier_diagnostics.py — Extended archetype classifier behavior.

Covers the new diagnostics surfaced by classify_archetype():
  - confidence_bucket (high / medium / low)
  - runner_up + runner_up_score
  - reason_codes (per-archetype)
  - missing_signals / features_used
  - strict defensive_income gate (config-gated)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from portfolio.position_archetypes import (
    classify_archetype,
    classify_archetype_from_scores,
    classify_archetype_full_from_scores,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enable_gate(monkeypatch=None, **gate_overrides) -> None:
    """Enable archetype_classifier and the defensive_income strict gate."""
    import util
    util.ARCHETYPE_CLASSIFIER_PARAMS = {
        "enabled": True,
        "confidence_buckets": {"high_min": 0.65, "medium_min": 0.45},
        "defensive_income": {
            "require_yield": True,
            "min_income_score": 0.30,
            "min_quality_score": 0.40,
            "min_momentum_score": -0.10,
            "max_volatility_percentile": 0.75,
            "reject_falling_knife": True,
            "yield_high": 0.80,
            "yield_moderate": 0.50,
            "yield_minimal": 0.05,
            "sector_defensive": ["Utilities", "Real Estate"],
            "industry_defensive": ["Electric Utilities"],
            "quality_min_label": 0.25,
            "momentum_disqualify_above": 0.50,
            **gate_overrides,
        },
        "quality_compounder": {},
        "legacy_turnaround": {},
        "speculative_momentum": {},
        "value_recovery": {},
    }


def _disable_gate() -> None:
    """Restore defaults — archetype_classifier disabled."""
    import util
    util.ARCHETYPE_CLASSIFIER_PARAMS = {
        "enabled": False,
        "confidence_buckets": {"high_min": 0.65, "medium_min": 0.45},
        "defensive_income": {"require_yield": False},
        "quality_compounder": {}, "legacy_turnaround": {},
        "speculative_momentum": {}, "value_recovery": {},
    }


# ---------------------------------------------------------------------------
# Extended diagnostics
# ---------------------------------------------------------------------------

class TestExtendedDiagnostics:

    def test_returns_confidence_bucket(self):
        r = classify_archetype({"quality_score": 0.8, "market_cap": 3e12,
                                "sector": "Technology Services",
                                "description": "cloud platform software"})
        assert r.confidence_bucket in {"high", "medium", "low"}

    def test_returns_runner_up(self):
        r = classify_archetype({"quality_score": 0.8, "market_cap": 3e12,
                                "sector": "Technology Services",
                                "description": "cloud platform software"})
        assert r.runner_up is not None
        assert r.runner_up != r.archetype

    def test_returns_reason_codes_for_winner(self):
        r = classify_archetype({"quality_score": 0.8, "market_cap": 3e12,
                                "sector": "Technology Services",
                                "description": "cloud platform software"})
        # All 6 archetypes should have a reason_codes entry (may be empty list)
        assert set(r.reason_codes.keys()) >= {"quality_compounder", "defensive_income"}
        # Winner's reason codes are non-empty for a clear MSFT-like input
        assert len(r.reason_codes[r.archetype]) > 0

    def test_features_used_vs_missing_signals(self):
        r = classify_archetype({"quality_score": 0.5, "sector": "Finance"})
        assert "quality_score" in r.features_used
        assert "sector" in r.features_used
        assert "market_cap" in r.missing_signals
        assert "description" in r.missing_signals

    def test_strong_inputs_produce_high_confidence_bucket(self):
        """Clear MSFT-like signals should produce a high-bucket result."""
        r = classify_archetype({
            "quality_score": 0.95, "market_cap": 3e12, "maintenance_ratio": 0.25,
            "buy_to_sell_ratio": 60, "num_employees": 100_000,
            "sector": "Technology Services",
            "description": "cloud platform software ecosystem subscription",
        })
        assert r.archetype == "quality_compounder"
        assert r.confidence_bucket == "high"


# ---------------------------------------------------------------------------
# Defensive-income strict gate
# ---------------------------------------------------------------------------

class TestDefensiveIncomeGate:

    def setup_method(self):
        _enable_gate()

    def teardown_method(self):
        _disable_gate()

    def test_disqualified_low_quality(self):
        """Gate ON + low quality_score → DI score forced to 0."""
        r = classify_archetype({
            "quality_score": 0.20, "momentum_score": 0.05,
            "income_score": 0.85, "sector": "Utilities",
        })
        di_codes = r.reason_codes.get("defensive_income", [])
        assert any("gate_disqualified" in c for c in di_codes)
        assert r.archetype != "defensive_income"

    def test_disqualified_low_momentum(self):
        r = classify_archetype({
            "quality_score": 0.50, "momentum_score": -0.30,
            "income_score": 0.85, "sector": "Utilities",
        })
        di_codes = r.reason_codes.get("defensive_income", [])
        assert any("gate_disqualified" in c for c in di_codes)

    def test_disqualified_falling_knife(self):
        r = classify_archetype({
            "quality_score": 0.50, "momentum_score": -0.40,
            "income_score": 0.85, "sector": "Utilities",
        })
        di_codes = r.reason_codes.get("defensive_income", [])
        assert any("gate_disqualified" in c for c in di_codes)

    def test_passes_when_all_gates_satisfied(self):
        """High income + high quality + neutral momentum → gate passes."""
        r = classify_archetype({
            "quality_score": 0.50, "momentum_score": 0.10,
            "income_score": 0.85, "sector": "Utilities",
            "industry": "Electric Utilities",
        })
        di_codes = r.reason_codes.get("defensive_income", [])
        assert "gate_passed" in di_codes
        assert r.archetype == "defensive_income"

    def test_default_off_preserves_legacy_behavior(self):
        """When gate is OFF (default), low-quality utility still scores positively."""
        _disable_gate()
        r = classify_archetype({
            "quality_score": 0.20, "momentum_score": 0.05,
            "income_score": 0.85, "sector": "Utilities",
        })
        di_codes = r.reason_codes.get("defensive_income", [])
        assert not any("gate_disqualified" in c for c in di_codes)


# ---------------------------------------------------------------------------
# Live ↔ backtest signal augmentation
# ---------------------------------------------------------------------------

class TestBacktestSignalAugmentation:

    def test_classify_from_scores_accepts_new_signals(self):
        """classify_archetype_from_scores accepts sector/industry/market_cap kwargs."""
        policy = classify_archetype_from_scores(
            quality_score=0.8, momentum_score=0.3, income_score=0.0,
            sector="Technology Services", industry="Software",
            market_cap=2e12,
        )
        assert policy is not None

    def test_classify_full_from_scores_returns_full_result(self):
        result = classify_archetype_full_from_scores(
            quality_score=0.8, momentum_score=0.3, income_score=0.0,
            sector="Technology Services", market_cap=2e12,
        )
        assert hasattr(result, "confidence_bucket")
        assert hasattr(result, "runner_up")
        assert hasattr(result, "reason_codes")
