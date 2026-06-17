"""
tests/test_classification_coverage.py — sector/industry → benchmark coverage guardrail.

Sector/industry is the backbone of the value engine: get_investment_ratios() returns
the PE/PB thresholds every value_metric is scored against. The source feed
(Robinhood/TRBC) uses sector names that do NOT match the ratios.yaml benchmark keys
1:1, so without the _SECTOR_NORMALIZATION map a literal lookup silently scores the
majority of the universe against the hardcoded DEFAULT.

This test FAILS LOUDLY if any live sector does not resolve to a real benchmark — so a
new/renamed source sector can never silently regress scoring. It reads the live
universe (no hardcoded thresholds) plus a committed baseline so it still runs in CI
without a data file.
"""
from __future__ import annotations

import glob
import os

from core.paths import DATA_DIRECTORY
from data.valuation import _resolve_sector_key, get_investment_ratios

# Committed baseline of source (Robinhood/TRBC) sectors observed in the live universe.
# Acts as the CI floor when no data CSV is present; live data (below) extends it.
_KNOWN_LIVE_SECTORS = {
    "Finance", "Health Technology", "Technology Services", "Miscellaneous",
    "Electronic Technology", "Producer Manufacturing", "Commercial Services",
    "Non-Energy Minerals", "Consumer Services", "Retail Trade", "Consumer Non-Durables",
    "Process Industries", "Consumer Durables", "Transportation", "Industrial Services",
    "Utilities", "Energy Minerals", "Distribution Services", "Health Services",
    "Communications", "Non Energy Minerals", "Government",
}


def _live_sectors() -> set[str]:
    """Distinct non-empty sectors from the most recent agg_data CSV, if any."""
    matches = sorted(glob.glob(os.path.join(DATA_DIRECTORY, "agg_data_*.csv")))
    if not matches:
        return set()
    import pandas as pd
    df = pd.read_csv(matches[-1], keep_default_na=False, na_values=[""], usecols=["sector"])
    return {s for s in df["sector"].dropna().unique() if str(s).strip()}


def test_every_live_sector_resolves_to_a_benchmark():
    """Every source sector must map to a real ratios.yaml benchmark (no silent DEFAULT)."""
    sectors = _KNOWN_LIVE_SECTORS | _live_sectors()
    unmapped = sorted(s for s in sectors if _resolve_sector_key(s) is None)
    assert not unmapped, (
        "Sectors with no benchmark (would score against DEFAULT [15.0, 2.5]): "
        f"{unmapped}. Add each to _SECTOR_NORMALIZATION (data/valuation.py) and/or "
        "as a key in cfg/ratios.yaml."
    )


def test_empty_sector_defaults_without_resolution():
    """An unclassified (empty) sector legitimately returns DEFAULT — not a coverage gap."""
    assert _resolve_sector_key("") is None
    assert _resolve_sector_key(None) is None
    assert get_investment_ratios("") == [15.0, 2.5]


def test_known_misclassified_targets_resolve():
    """The per-symbol override TARGET sectors must themselves resolve to a benchmark."""
    for target in ("Technology Services", "Electronic Technology"):
        assert _resolve_sector_key(target) is not None
