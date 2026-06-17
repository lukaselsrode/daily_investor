"""
data/valuation.py — Sector/industry ratio lookups and Finviz valuation updater.

Functions:
    get_investment_ratios()      — sector/industry PE+PB thresholds from ratios.yaml
    update_industry_valuations() — refresh ratios.yaml from Finviz tables
"""

from __future__ import annotations

import logging
import re

import requests
import yaml
from bs4 import BeautifulSoup

from core.paths import RATIOS_FILE

logger = logging.getLogger(__name__)


def _load_ratios() -> dict:
    with open(RATIOS_FILE) as f:
        return yaml.safe_load(f)


_ratios = _load_ratios()

_SPLIT_RE = re.compile(r"[\/ :&]+")


def _split_to_set(s: str) -> set[str]:
    return set(_SPLIT_RE.split(s))


# Raw source (Robinhood/TRBC) sector → ratios.yaml benchmark key. The source
# taxonomy (22 sectors) does not match the benchmark file's keys 1:1, so without
# this map a literal lookup silently defaults the majority of the universe. Only
# sectors needing translation are listed; sectors whose name already IS a ratios.yaml
# key (e.g. "Technology Services", "Utilities") resolve by identity. Keep this map
# COMPLETE for every live sector — the classification-coverage test fails otherwise.
_SECTOR_NORMALIZATION = {
    "Finance": "Financial",
    "Energy Minerals": "Energy",
    "Non-Energy Minerals": "Basic Materials",
    "Non Energy Minerals": "Basic Materials",   # spelling variant in source data
    "Process Industries": "Basic Materials",
    "Producer Manufacturing": "Industrials",
    "Industrial Services": "Industrials",
    "Distribution Services": "Industrials",
    "Commercial Services": "Industrials",
    "Government": "Miscellaneous",
}


def _resolve_sector_key(sector: str | None) -> str | None:
    """Map a raw source sector to its ratios.yaml benchmark key, or None if there
    is no benchmark for it (caller then falls back to DEFAULT)."""
    if not sector:
        return None
    key = _SECTOR_NORMALIZATION.get(sector, sector)
    return key if key in _ratios else None


# FMP (GICS) sector → ratios.yaml benchmark key, for the cross-validation layer.
# Approximate by design: it only decides WHETHER FMP and Robinhood imply materially
# different benchmarks (and thus whether to ask Claude). GICS "Consumer Cyclical" and
# "Consumer Defensive" are ambiguous (Retail vs Durables vs Non-Durables); the FMP
# industry hint below disambiguates the common cases.
_FMP_SECTOR_TO_BENCHMARK = {
    "Technology": "Technology Services",
    "Communication Services": "Communications",
    "Financial Services": "Financial",
    "Financial": "Financial",
    "Healthcare": "Health Technology",
    "Health Care": "Health Technology",
    "Energy": "Energy",
    "Basic Materials": "Basic Materials",
    "Industrials": "Industrials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Consumer Defensive": "Consumer Non-Durables",
    "Consumer Cyclical": "Retail Trade",
}

# GICS-industry keyword → benchmark key, to disambiguate broad GICS sectors. Checked
# (case-insensitive substring) before the sector-level map.
_FMP_INDUSTRY_HINTS = {
    "reit": "Real Estate",            # before 'retail' — 'REIT - Retail' is real estate, not a retailer
    "real estate": "Real Estate",
    "semiconductor": "Electronic Technology",
    "hardware": "Electronic Technology",
    "consumer electronics": "Electronic Technology",
    "communication equipment": "Electronic Technology",
    "auto manufacturer": "Consumer Durables",
    "auto - manufacturers": "Consumer Durables",
    "software": "Technology Services",
    "internet": "Technology Services",
    "information technology": "Technology Services",
    "bank": "Financial",
    "insurance": "Financial",
    "capital market": "Financial",
    "asset management": "Financial",
    "biotech": "Health Technology",
    "drug": "Health Technology",
    "pharmaceutical": "Health Technology",
    "medical care": "Health Services",
    "healthcare plan": "Health Services",
    "rent": "Retail Trade",  # 'apparel retail', 'specialty retail', etc.
    "retail": "Retail Trade",
    "restaurant": "Consumer Services",
}


def _fmp_to_benchmark_key(sector: str | None, industry: str | None = None) -> str | None:
    """Map an FMP (GICS) sector/industry to a ratios.yaml benchmark key, or None."""
    ind = (industry or "").lower()
    for kw, key in _FMP_INDUSTRY_HINTS.items():
        if kw in ind and key in _ratios:
            return key
    key = _FMP_SECTOR_TO_BENCHMARK.get((sector or "").strip())
    return key if key and key in _ratios else None


def benchmark_pe(key: str | None) -> float | None:
    """The default PE threshold for a resolved benchmark key (None if unknown)."""
    if not key or key not in _ratios:
        return None
    default = _ratios[key].get("default")
    if isinstance(default, list) and default and default[0] is not None:
        return float(default[0])
    return None


def get_investment_ratios(sector: str, industry: str | None = None) -> list[float]:
    """Return [PE_threshold, PB_threshold] for the given sector/industry."""
    DEFAULT = [15.0, 2.5]

    resolved = _resolve_sector_key(sector)
    if resolved is None:
        # Empty sector = genuinely unclassified (e.g. a fund/ETF row) — expected,
        # don't warn-spam. A non-empty sector with no benchmark is a coverage gap.
        if sector:
            logger.warning(
                "No benchmark for sector %r (industry %r) → DEFAULT %s; "
                "add it to _SECTOR_NORMALIZATION / ratios.yaml", sector, industry, DEFAULT,
            )
        return DEFAULT

    sector_cfg = _ratios[resolved]
    default = sector_cfg.get("default", DEFAULT)

    def _coerce(ratios: list) -> list[float]:
        return [
            ratios[0] if ratios and len(ratios) > 0 and ratios[0] is not None else default[0],
            ratios[1] if ratios and len(ratios) > 1 and ratios[1] is not None else default[1],
        ]

    if not industry:
        return default

    if sector_cfg.get(industry):
        return _coerce(sector_cfg[industry])

    try:
        query = _split_to_set(industry)
        best, best_diff = None, float("inf")
        for key in sector_cfg:
            if key == "default":
                continue
            diff = len(query.difference(_split_to_set(key)))
            if diff == 0 and len(_split_to_set(key)) == len(query):
                return _coerce(sector_cfg[key])
            if diff < best_diff:
                best_diff, best = diff, key
        if best and best_diff < 3:
            return _coerce(sector_cfg[best])
    except Exception as e:
        logger.warning("Fuzzy match error for industry '%s': %s", industry, e)

    return default


_FINVIZ_HEADERS = {"User-Agent": "Mozilla/5.0"}

_SECTOR_MAP = {
    "Materials": "Basic Materials",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Financials": "Financial",
    "Health Care": "Healthcare",
    "Information Technology": "Technology",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Communication Services": "Communication Services",
    "Consumer Services": "Consumer Cyclical",
    "Technology Services": "Technology",
    "Health Technology": "Healthcare",
    "Communications": "Communication Services",
    "Electronic Technology": "Technology",
    "Retail Trade": "Consumer Cyclical",
    "Consumer Durables": "Consumer Cyclical",
    "Transportation": "Industrials",
    # Benchmark keys added to cover the full source taxonomy — refreshed from these
    # Finviz sectors by update_industry_valuations. ("Utilities" is already mapped
    # above; "Miscellaneous" has no Finviz analog and keeps its static seed.)
    "Health Services": "Healthcare",
    "Consumer Non-Durables": "Consumer Defensive",
}

_INDUSTRY_MAP = {
    "Insurance - Life": "Life Insurance",
    "Insurance - Property & Casualty": "Property & Casualty Insurance",
    "Insurance - Specialty": "Specialty Insurance",
    "Insurance - Diversified": "Diversified Insurance",
    "REIT - Mortgage": "Mortgage REITs",
    "REIT - Diversified": "Diversified REITs",
    "REIT - Retail": "Retail REITs",
    "REIT - Residential": "Residential REITs",
    "REIT - Industrial": "Industrial REITs",
    "REIT - Office": "Office REITs",
    "REIT - Hotel & Motel": "Hotel & Motel REITs",
    "REIT - Healthcare Facilities": "Health Care REITs",
    "REIT - Specialty": "Specialty REITs",
    "Oil & Gas E&P": "Oil & Gas Exploration & Production",
    "Beverages - Brewers": "Brewers",
    "Beverages - Wineries & Distilleries": "Distillers & Vintners",
    "Beverages - Non-Alcoholic": "Non-Alcoholic Beverages",
    "Telecom Services": "Telecommunication Services",
    "Internet Content & Information": "Interactive Media & Services",
    "Software - Application": "Application Software",
    "Software - Infrastructure": "Systems Software",
}


def _fetch_finviz_table(url: str) -> dict[str, dict[str, float | None]]:
    resp = requests.get(url, headers=_FINVIZ_HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("div", class_="content").find(
        "table",
        class_="styled-table-new is-medium is-rounded is-tabular-nums w-full groups_table",
    )
    if not table:
        raise ValueError(f"Could not find data table at {url}")

    result = {}
    for row in table.find_all("tr")[1:]:
        link = row.find("a")
        if not link:
            continue
        cells = row.find_all("td")
        pe_text = cells[3].get_text() if len(cells) > 3 else "-"
        pb_text = cells[7].get_text() if len(cells) > 7 else "-"
        result[link.get_text()] = {
            "PE": float(pe_text) if pe_text != "-" else None,
            "PB": float(pb_text) if pb_text != "-" else None,
        }
    return result


def update_industry_valuations(verbose: bool = True) -> None:
    """Refresh ratios.yaml with current Finviz sector/industry PE+PB data."""
    SECTOR_URL = "https://finviz.com/groups.ashx?g=sector&v=120&o=pe"
    INDUSTRY_URL = "https://finviz.com/groups.ashx?g=industry&v=120&o=pe"

    try:
        sector_data = _fetch_finviz_table(SECTOR_URL)
        industry_data = _fetch_finviz_table(INDUSTRY_URL)
    except Exception as e:
        logger.error("Failed to fetch Finviz data: %s", e)
        raise

    ratios = _load_ratios()
    changes: list[dict] = []

    for sector_yaml, industries in ratios.items():
        if not isinstance(industries, dict):
            continue

        finviz_sector = _SECTOR_MAP.get(sector_yaml)
        if finviz_sector and finviz_sector in sector_data:
            new_pe = sector_data[finviz_sector]["PE"]
            new_pb = sector_data[finviz_sector]["PB"]
            default = industries.get("default")
            if isinstance(default, list) and len(default) >= 2:
                old_pe, old_pb = default[0], default[1]
                updated_pe = new_pe if new_pe is not None else old_pe
                updated_pb = new_pb if new_pb is not None else old_pb
                if updated_pe != old_pe or updated_pb != old_pb:
                    industries["default"] = [updated_pe, updated_pb]
                    changes.append({
                        "type": "sector", "name": sector_yaml,
                        "old": f"PE={old_pe}, PB={old_pb}",
                        "new": f"PE={updated_pe}, PB={updated_pb}",
                    })

        for ind_yaml, metrics in industries.items():
            if ind_yaml == "default" or not isinstance(metrics, list) or len(metrics) < 2:
                continue
            finviz_ind = _INDUSTRY_MAP.get(ind_yaml, ind_yaml)
            if finviz_ind not in industry_data:
                continue
            new_pe = industry_data[finviz_ind]["PE"]
            new_pb = industry_data[finviz_ind]["PB"]
            old_pe, old_pb = metrics[0], metrics[1]
            changed = False
            if new_pe is not None and new_pe != old_pe:
                metrics[0] = new_pe
                changed = True
            if new_pb is not None and new_pb != old_pb:
                metrics[1] = new_pb
                changed = True
            if changed:
                changes.append({
                    "type": "industry", "sector": sector_yaml, "name": ind_yaml,
                    "old": f"PE={old_pe}, PB={old_pb}",
                    "new": f"PE={metrics[0]}, PB={metrics[1]}",
                })

    if changes:
        with open(RATIOS_FILE, "w") as f:
            yaml.dump(ratios, f, default_flow_style=False, sort_keys=False)
        if verbose:
            logger.info("Updated %d valuations in ratios.yaml", len(changes))
            for c in changes:
                loc = c["name"] if c["type"] == "sector" else f"{c['sector']} / {c['name']}"
                logger.info("  %s %s: %s → %s", c["type"].upper(), loc, c["old"], c["new"])
    else:
        if verbose:
            logger.info("No valuation changes detected")
