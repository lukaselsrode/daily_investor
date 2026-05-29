"""
tests/test_factor_map_scope.py — Factor Map scope / ETF-separation filtering.

Covers:
  1. ETF detection from config (etfs list, harvest_etfs, benchmark_symbol),
     case-insensitivity, and that ordinary stocks are not misclassified.
  2. tag_etf: config membership, explicit asset_type/security_type metadata,
     and honoring a pre-existing is_etf column.
  3. apply_scope: stocks-only excludes ETFs, ETFs-only includes only ETFs,
     owned / candidates / owned+candidates / active-sleeve honor roles,
     full-universe is unchanged.
  4. The embedding (build_factor_map) only ever sees the filtered scope, and
     cluster diagnostics never reference out-of-scope (ETF) symbols.
  5. Missing metadata (no owned/sector/asset_type columns) does not crash.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from portfolio.visualization.factor_map import (
    build_factor_map,
    etf_symbols_from_config,
    is_etf_symbol,
    tag_etf,
)
from ui.components.factor_map import SCOPE_OPTIONS, apply_scope

# ETF tickers present in the synthetic universe (all configured below).
_ETF_TICKERS = ["SPY", "QQQ", "VTI"]

_CONFIG = {
    "etfs": ["SPY", "VOO", "VTI", "QQQ", "SCHD"],
    "harvest": {"harvest_etfs": ["SPY", "VTI"]},
    "backtest": {"benchmark_symbol": "SPY"},
}

_METRIC_THRESHOLD = 0.75


def _make_universe(n_stocks: int = 24) -> pd.DataFrame:
    """Synthetic scored universe: configured ETFs + individual stocks.

    Two stocks are flagged owned; two unowned core_candidates clear the metric
    threshold. ETFs are not owned. Factor columns give the embedding something
    to work with.
    """
    rng = np.random.default_rng(42)
    stock_syms = [f"STK{i:03d}" for i in range(n_stocks)]
    symbols = _ETF_TICKERS + stock_syms
    n = len(symbols)

    owned = [False] * len(_ETF_TICKERS) + [
        i in (0, 1) for i in range(n_stocks)  # first two stocks owned
    ]

    bucket = ["etf"] * len(_ETF_TICKERS) + ["core"] * n_stocks
    value_metric = np.concatenate([
        rng.uniform(0.2, 0.5, len(_ETF_TICKERS)),
        rng.uniform(0.0, 0.6, n_stocks),
    ])
    # Two unowned candidate stocks that clear the threshold.
    cand_idx = [len(_ETF_TICKERS) + 5, len(_ETF_TICKERS) + 6]
    for ci in cand_idx:
        bucket[ci] = "core_candidate"
        value_metric[ci] = 0.85

    return pd.DataFrame({
        "symbol":          symbols,
        "owned":           owned,
        "strategy_bucket": bucket,
        "sector":          ["Index"] * len(_ETF_TICKERS) + ["Technology"] * n_stocks,
        "value_metric":    value_metric,
        "value_score":     rng.uniform(-0.5, 0.8, n),
        "quality_score":   rng.uniform(-0.4, 0.9, n),
        "momentum_score":  rng.uniform(-0.5, 0.8, n),
        "income_score":    rng.uniform(0.0, 0.8, n),
    })


# ---------------------------------------------------------------------------
# 1. ETF detection from config
# ---------------------------------------------------------------------------

def test_etf_symbols_from_config_union():
    syms = etf_symbols_from_config(_CONFIG)
    assert {"SPY", "VOO", "VTI", "QQQ", "SCHD"} <= syms


def test_is_etf_symbol_detects_configured_tickers():
    assert is_etf_symbol("SPY", _CONFIG)   # etfs + harvest + benchmark
    assert is_etf_symbol("VTI", _CONFIG)   # etfs + harvest
    assert is_etf_symbol("QQQ", _CONFIG)   # etfs
    assert not is_etf_symbol("AAPL", _CONFIG)


def test_is_etf_symbol_is_case_insensitive():
    assert is_etf_symbol("spy", _CONFIG)
    assert is_etf_symbol(" QqQ ", _CONFIG)


def test_is_etf_symbol_detects_benchmark_only_config():
    assert is_etf_symbol("SPY", {"backtest": {"benchmark_symbol": "SPY"}})


def test_is_etf_symbol_does_not_misclassify_stocks():
    # A stock with no fundamentals / unknown sector must NOT be tagged ETF.
    assert not is_etf_symbol("NVDA", _CONFIG)
    assert not is_etf_symbol("", _CONFIG)
    assert not is_etf_symbol(None, _CONFIG)


# ---------------------------------------------------------------------------
# 2. tag_etf
# ---------------------------------------------------------------------------

def test_tag_etf_from_config_membership():
    df = tag_etf(_make_universe(), _CONFIG)
    assert "is_etf" in df.columns
    etf_rows = df[df["is_etf"]]["symbol"].tolist()
    assert set(etf_rows) == set(_ETF_TICKERS)


def test_tag_etf_honors_asset_type_column():
    df = pd.DataFrame({
        "symbol": ["AAA", "BBB", "SPY"],
        "asset_type": ["etf", "stock", "common"],
    })
    tagged = tag_etf(df, _CONFIG)
    flags = dict(zip(tagged["symbol"], tagged["is_etf"]))
    assert flags["AAA"] is True or bool(flags["AAA"])   # explicit asset_type
    assert not flags["BBB"]
    assert bool(flags["SPY"])                            # config membership


def test_tag_etf_honors_existing_is_etf_column():
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "is_etf": [True, False],   # caller-provided; must be respected verbatim
    })
    tagged = tag_etf(df, _CONFIG)
    assert tagged["is_etf"].tolist() == [True, False]


# ---------------------------------------------------------------------------
# 3. apply_scope
# ---------------------------------------------------------------------------

def test_scope_options_exposed():
    assert SCOPE_OPTIONS[0] == "Stocks only"   # default
    assert set(SCOPE_OPTIONS) == {
        "Stocks only", "Full universe", "ETFs only", "Owned only",
        "Candidates only", "Owned + Candidates", "Active sleeve only",
    }


def test_stocks_only_excludes_etfs():
    df = _make_universe()
    scoped, meta = apply_scope(df, "Stocks only", _CONFIG, _METRIC_THRESHOLD)
    assert not set(scoped["symbol"]) & set(_ETF_TICKERS)
    assert meta["etf_total"] == len(_ETF_TICKERS)
    assert meta["in_scope"] == len(df) - len(_ETF_TICKERS)


def test_etfs_only_includes_only_etfs():
    df = _make_universe()
    scoped, _ = apply_scope(df, "ETFs only", _CONFIG, _METRIC_THRESHOLD)
    assert set(scoped["symbol"]) == set(_ETF_TICKERS)
    assert scoped["is_etf"].all()


def test_owned_and_candidate_scopes():
    df = _make_universe()
    owned, _ = apply_scope(df, "Owned only", _CONFIG, _METRIC_THRESHOLD)
    assert (owned["_role"] == "owned").all()
    assert len(owned) == 2

    cands, _ = apply_scope(df, "Candidates only", _CONFIG, _METRIC_THRESHOLD)
    assert (cands["_role"] == "candidate").all()
    assert len(cands) == 2

    both, _ = apply_scope(df, "Owned + Candidates", _CONFIG, _METRIC_THRESHOLD)
    assert set(both["_role"]) <= {"owned", "candidate"}
    assert len(both) == 4


def test_active_sleeve_excludes_etfs():
    df = _make_universe()
    # Make one owned ETF to prove the active sleeve drops it.
    df.loc[df["symbol"] == "SPY", "owned"] = True
    sleeve, _ = apply_scope(df, "Active sleeve only", _CONFIG, _METRIC_THRESHOLD)
    assert not set(sleeve["symbol"]) & set(_ETF_TICKERS)
    assert set(sleeve["_role"]) <= {"owned", "candidate"}


def test_full_universe_unchanged():
    df = _make_universe()
    scoped, meta = apply_scope(df, "Full universe", _CONFIG, _METRIC_THRESHOLD)
    assert len(scoped) == len(df)
    assert meta["in_scope"] == len(df)


# ---------------------------------------------------------------------------
# 4. Embedding + cluster diagnostics respect the filtered scope
# ---------------------------------------------------------------------------

def test_build_factor_map_receives_filtered_df():
    df = _make_universe()
    scoped, _ = apply_scope(df, "Stocks only", _CONFIG, _METRIC_THRESHOLD)
    _fig, df_out, _diags = build_factor_map(scoped, method="pca")
    assert set(df_out["symbol"]) <= set(scoped["symbol"])
    assert not set(df_out["symbol"]) & set(_ETF_TICKERS)


def test_cluster_diagnostics_respect_scope():
    df = _make_universe()
    scoped, _ = apply_scope(df, "Stocks only", _CONFIG, _METRIC_THRESHOLD)
    _fig, _df_out, diags = build_factor_map(scoped, method="pca", kmeans_clusters=3)
    cs = diags.get("cluster_summary")
    assert cs is not None and not cs.empty
    joined = " ".join(cs["top_symbols"].astype(str).tolist())
    for etf in _ETF_TICKERS:
        assert etf not in joined


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

def test_missing_metadata_does_not_crash():
    # No owned / sector / asset_type — only symbol + factor columns.
    df = pd.DataFrame({
        "symbol":        ["SPY"] + [f"STK{i}" for i in range(8)],
        "value_metric":  np.linspace(0.1, 0.9, 9),
        "value_score":   np.linspace(-0.4, 0.8, 9),
        "quality_score": np.linspace(-0.3, 0.7, 9),
    })
    scoped, meta = apply_scope(df, "Stocks only", _CONFIG, _METRIC_THRESHOLD)
    assert "SPY" not in set(scoped["symbol"])
    assert meta["owned_in_scope"] == 0
    # tag_etf alone also must not raise on missing metadata.
    tagged = tag_etf(df.drop(columns=["symbol"]), _CONFIG)
    assert "is_etf" in tagged.columns
    assert not tagged["is_etf"].any()


def test_no_etfs_found_yields_empty_etf_scope():
    df = _make_universe()
    df = df[~df["symbol"].isin(_ETF_TICKERS)].copy()  # drop all ETFs
    scoped, meta = apply_scope(df, "ETFs only", _CONFIG, _METRIC_THRESHOLD)
    assert scoped.empty
    assert meta["etf_total"] == 0


def test_duplicate_symbols_handled():
    df = _make_universe()
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # duplicate rows
    scoped, _ = apply_scope(df, "Stocks only", _CONFIG, _METRIC_THRESHOLD)
    assert not set(scoped["symbol"]) & set(_ETF_TICKERS)


# ---------------------------------------------------------------------------
# 6. Robinhood instrument_type detection (universe-wide ETF/fund tagging)
# ---------------------------------------------------------------------------

def test_tag_etf_instrument_type_funds_vs_stocks():
    df = pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
        "instrument_type": ["etp", "cef", "mlp", "stock", "adr", "reit"],
    })
    flags = dict(zip(df["symbol"], tag_etf(df, _CONFIG)["is_etf"]))
    assert flags["AAA"] and flags["BBB"] and flags["CCC"]   # etp / cef / mlp = fund
    assert not flags["DDD"] and not flags["EEE"] and not flags["FFF"]  # stock/adr/reit


def test_tag_etf_instrument_type_case_insensitive():
    df = pd.DataFrame({"symbol": ["X", "Y"], "instrument_type": [" ETP ", "Stock"]})
    flags = dict(zip(df["symbol"], tag_etf(df, _CONFIG)["is_etf"]))
    assert flags["X"] and not flags["Y"]


def test_tag_etf_instrument_type_unions_with_config():
    # A config ETF whose instrument_type is missing/stock is still an ETF;
    # a non-config symbol whose instrument_type is etp is also an ETF.
    df = pd.DataFrame({
        "symbol": ["SPY", "NEWETF", "MSFT"],
        "instrument_type": ["stock", "etp", "stock"],
    })
    flags = dict(zip(df["symbol"], tag_etf(df, _CONFIG)["is_etf"]))
    assert flags["SPY"]      # config membership
    assert flags["NEWETF"]   # instrument_type
    assert not flags["MSFT"]


def test_stocks_only_excludes_instrument_type_funds():
    df = pd.DataFrame({
        "symbol": ["MSFT", "NEWETF", "TICKER3"],
        "instrument_type": ["stock", "etp", "cef"],
        "value_metric": [0.3, 0.4, 0.5],
        "value_score": [0.1, 0.2, 0.3],
        "quality_score": [0.2, 0.3, 0.4],
        "owned": [True, True, True],
        "strategy_bucket": ["core", "etf", "etf"],
    })
    scoped, meta = apply_scope(df, "Stocks only", _CONFIG, _METRIC_THRESHOLD)
    assert set(scoped["symbol"]) == {"MSFT"}
    assert meta["etf_total"] == 2


# ---------------------------------------------------------------------------
# 7. Analytical enhancements — archetype / outliers / ellipsoids / nearest-N
# ---------------------------------------------------------------------------

from portfolio.position_archetypes import ARCHETYPE_LABELS  # noqa: E402
from ui.components.factor_map import (  # noqa: E402
    _add_group_ellipsoids,
    _compute_archetypes,
    _coord_cols_for,
    _ellipsoid_mesh,
    _nearest_table,
)


def _feature_universe(n: int = 24, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "symbol":         [f"S{i:03d}" for i in range(n)],
        "owned":          [i % 4 == 0 for i in range(n)],
        "value_score":    rng.uniform(-0.5, 0.8, n),
        "quality_score":  rng.uniform(-0.4, 0.9, n),
        "momentum_score": rng.uniform(-0.5, 0.8, n),
        "income_score":   rng.uniform(0.0, 0.8, n),
        "value_metric":   rng.uniform(-0.2, 0.6, n),
        "yield_trap_flag": [False] * n,
        "sector":         ["Tech"] * n,
        "industry":       ["Software"] * n,
    })


def test_compute_archetypes_labels_valid():
    df = _feature_universe()
    arch = _compute_archetypes(df)
    assert len(arch) == len(df)
    assert set(arch.unique()) <= set(ARCHETYPE_LABELS)


def test_compute_archetypes_missing_optional_cols_no_crash():
    df = pd.DataFrame({
        "symbol": ["A", "B"],
        "quality_score": [0.5, -0.3],
        "momentum_score": [0.4, 0.1],
        "income_score": [0.0, 0.6],
    })
    arch = _compute_archetypes(df)
    assert set(arch.unique()) <= set(ARCHETYPE_LABELS)


def test_build_factor_map_excludes_outliers_before_embedding():
    df = _feature_universe(24)
    extreme = df.iloc[[0]].copy()
    extreme["symbol"] = "OUTLIER"
    for c in ("value_score", "quality_score", "momentum_score", "income_score", "value_metric"):
        extreme[c] = 50.0  # far beyond any MAD band
    df = pd.concat([df, extreme], ignore_index=True)

    _fig, df_out, diags = build_factor_map(df, method="pca", exclude_outliers=True, outlier_mad_z=5.0)
    assert "OUTLIER" in diags.get("outliers_excluded", [])
    assert "OUTLIER" not in set(df_out["symbol"])

    # Without the flag the outlier is retained.
    _f2, df_out2, diags2 = build_factor_map(df, method="pca", exclude_outliers=False)
    assert "OUTLIER" in set(df_out2["symbol"])
    assert not diags2.get("outliers_excluded")


def test_color_map_override_applied():
    df = _feature_universe(20)
    df["grp"] = ["A" if i % 2 else "B" for i in range(len(df))]
    cmap = {"A": "#111111", "B": "#222222"}
    fig, _df_out, _diags = build_factor_map(df, method="pca", color_by="grp", color_map=cmap)
    trace_colors = {t.name: t.marker.color for t in fig.data if t.name in ("A", "B")}
    assert trace_colors.get("A") == "#111111"
    assert trace_colors.get("B") == "#222222"


def test_ellipsoid_mesh_guards():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(30, 3))
    assert _ellipsoid_mesh(pts, "#fff", "g") is not None
    assert _ellipsoid_mesh(pts[:3], "#fff", "g") is None         # too few
    collinear = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
    assert _ellipsoid_mesh(collinear, "#fff", "g") is None        # degenerate


def test_add_group_ellipsoids_adds_traces():
    import plotly.graph_objects as go
    df = _feature_universe(24)
    fig, df_out, _diags = build_factor_map(df, method="pca")
    n_before = len(fig.data)
    _add_group_ellipsoids(fig, df_out, _coord_cols_for("pca"))
    assert len(fig.data) > n_before
    assert any(isinstance(t, go.Mesh3d) for t in fig.data)


def test_nearest_table_returns_closest():
    df = _feature_universe(24)
    _fig, df_out, _diags = build_factor_map(df, method="pca")
    cc = _coord_cols_for("pca")
    centroid = df_out[cc].to_numpy(dtype=float).mean(axis=0)
    tbl = _nearest_table(df_out, centroid, cc, n=5)
    assert len(tbl) == 5
    assert tbl["distance"].is_monotonic_increasing
