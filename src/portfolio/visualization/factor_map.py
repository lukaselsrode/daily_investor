"""
portfolio/visualization/factor_map.py — Factor-space universe visualization.

Reduces the scored stock universe to 3-D via PCA or UMAP, then renders an
interactive Plotly scatter so you can answer questions like:

    "Are all my BUY names clustered in the same factor neighbourhood?"
    "Am I accidentally concentrating the active sleeve into energy/value names?"
    "Are there attractive un-owned names far from my current cluster?"
    "Should I add a diversification penalty based on factor-map distance?"

Entry points
────────────
Library:
    from portfolio.visualization.factor_map import build_factor_map
    fig, df_out, diags = build_factor_map(df, method="pca", kmeans_clusters=6)
    fig.show()

CLI (via daily-investor):
    daily-investor factor-map [--method pca|umap] [--clusters N]
                              [--output reports/factor_map.html]
                              [--owned-only] [--color sector]

__main__:
    python -m portfolio.visualization.factor_map \\
        --input data/agg_data_2026_05_27.csv --method pca \\
        --output reports/factor_map.html

SAFE: read-only. Never modifies config, factor scores, or portfolio state.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import plotly.graph_objects

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candidate feature columns
# These map to the actual agg_data schema; columns absent from the input
# DataFrame are silently skipped.
# ---------------------------------------------------------------------------

_CANDIDATE_FEATURE_COLS: list[str] = [
    # Core factor scores (always present after scoring)
    "value_score",
    "quality_score",
    "income_score",
    "momentum_score",
    "value_metric",
    # User-supplied names that may appear in enriched frames
    "volatility_score",
    "dividend_score",
    "growth_score",
    "sentiment_score",
    "analyst_score",
    "active_score",
    "quality_metric",
    "composite_score",
    "final_score",
    # Extended momentum / return features
    "return_1m",
    "return_3m",
    "return_6m",
    "rs_3m",
    "rs_6m",
    "risk_adj_momentum_3m",
    # Valuation extended
    "relative_pe",
    "relative_pb",
    # Signal quality / reliability
    "reliability_score",
    # Price position
    "position_52w",
    "realized_vol_3m",
]

_PLOTLY_PALETTE: list[str] = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
    "#ff5722", "#8bc34a", "#607d8b", "#ffc107", "#673ab7",
    "#795548", "#009688", "#ff9800", "#4caf50", "#2196f3",
]

_AXIS_NAMES = {
    "pca":  ["PC 1", "PC 2", "PC 3"],
    "umap": ["UMAP 1", "UMAP 2", "UMAP 3"],
}

_COORD_COLS = {
    "pca":  ["pca_1", "pca_2", "pca_3"],
    "umap": ["umap_1", "umap_2", "umap_3"],
}


# ---------------------------------------------------------------------------
# Feature selection & preprocessing
# ---------------------------------------------------------------------------

def _select_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None,
    max_nan_pct: float = 0.50,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Return (feature_matrix, selected_col_names).

    Picks numeric columns from `feature_cols` (or _CANDIDATE_FEATURE_COLS if
    None), drops columns with >max_nan_pct missing values, fills remaining NaN
    with per-column median.  Raises ValueError if fewer than 2 columns survive.
    """
    candidates = feature_cols if feature_cols is not None else _CANDIDATE_FEATURE_COLS
    present = [c for c in candidates if c in df.columns]

    if not present:
        raise ValueError(
            f"None of the candidate feature columns are present in the input DataFrame. "
            f"Tried: {candidates[:10]}... "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only numeric columns
    feat_df = df[present].apply(pd.to_numeric, errors="coerce")

    # Drop high-nan columns
    nan_pct = feat_df.isna().mean()
    too_sparse = nan_pct[nan_pct > max_nan_pct].index.tolist()
    if too_sparse:
        logger.warning("Dropping %d feature cols with >%.0f%% NaN: %s",
                       len(too_sparse), max_nan_pct * 100, too_sparse)
        feat_df = feat_df.drop(columns=too_sparse)

    if feat_df.shape[1] < 2:
        raise ValueError(
            f"Only {feat_df.shape[1]} feature column(s) remain after NaN filtering "
            f"(need ≥ 2). Sparse columns dropped: {too_sparse}. "
            f"Check your input DataFrame or pass feature_cols explicitly."
        )

    # Fill remaining NaN with column median
    feat_df = feat_df.fillna(feat_df.median(numeric_only=True))

    selected = list(feat_df.columns)
    logger.info("Factor map: using %d feature columns: %s", len(selected), selected)
    return feat_df, selected


def _standardize(feat_df: pd.DataFrame) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(feat_df.values)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def _reduce_pca(X: np.ndarray) -> tuple[np.ndarray, Any]:
    """Return ((n,3) coords, fitted PCA model)."""
    from sklearn.decomposition import PCA
    n_components = min(3, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    if coords.shape[1] < 3:
        pad = np.zeros((coords.shape[0], 3 - coords.shape[1]))
        coords = np.hstack([coords, pad])
    return coords, pca


def _reduce_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> tuple[np.ndarray, Any]:
    """
    Return ((n,3) coords, fitted UMAP model).

    Raises ImportError with install instructions if umap-learn is absent.
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is not installed.  To add UMAP support:\n"
            "    pip install umap-learn\n"
            "or with the project venv:\n"
            "    .venv/bin/pip install umap-learn\n"
            "PCA is always available as a fallback: --method pca"
        ) from exc

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(X)
    return coords, reducer


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _apply_kmeans(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Return integer cluster labels (n,)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return km.fit_predict(X)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _apply_filters(
    df: pd.DataFrame,
    owned_only: bool,
    unowned_only: bool,
    actions: list[str] | None,
    sectors: list[str] | None,
    sleeves: list[str] | None,
    min_score: float | None,
    max_score: float | None,
    score_col: str,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if owned_only and "owned" in df.columns:
        mask &= df["owned"].astype(bool)
    if unowned_only and "owned" in df.columns:
        mask &= ~df["owned"].astype(bool)

    if actions:
        for col in ("action", "strategy_bucket", "state"):
            if col in df.columns:
                mask &= df[col].isin(actions)
                break

    if sectors and "sector" in df.columns:
        mask &= df["sector"].isin(sectors)

    if sleeves and "sleeve" in df.columns:
        mask &= df["sleeve"].isin(sleeves)

    if score_col in df.columns:
        scores = pd.to_numeric(df[score_col], errors="coerce")
        if min_score is not None:
            mask &= scores >= min_score
        if max_score is not None:
            mask &= scores <= max_score

    n_before = len(df)
    df = df[mask].copy()
    logger.info("Filters applied: %d → %d rows", n_before, len(df))
    return df


# ---------------------------------------------------------------------------
# Color / size resolution helpers
# ---------------------------------------------------------------------------

def _resolve_color_by(df: pd.DataFrame, color_by: str | None) -> str:
    if color_by and color_by in df.columns:
        return color_by
    for candidate in ("action", "strategy_bucket", "sector", "sleeve"):
        if candidate in df.columns:
            return candidate
    for candidate in ("active_score", "final_score", "value_metric"):
        if candidate in df.columns:
            return candidate
    return ""


def _resolve_size_by(df: pd.DataFrame, size_by: str | None) -> str | None:
    if size_by and size_by in df.columns:
        return size_by
    for candidate in ("equity", "current_value", "target_value",
                      "active_score", "value_metric"):
        if candidate in df.columns:
            return candidate
    return None


def _scale_sizes(series: pd.Series, lo: float = 4.0, hi: float = 20.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0).abs()
    if s.max() - s.min() < 1e-9:
        return pd.Series(lo + (hi - lo) / 2, index=series.index)
    return lo + (hi - lo) * (s - s.min()) / (s.max() - s.min())


# ---------------------------------------------------------------------------
# Plotly figure construction
# ---------------------------------------------------------------------------

def _hover_text(row: pd.Series, hover_fields: list[str]) -> str:
    parts: list[str] = []
    for f in hover_fields:
        v = row.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, float):
            parts.append(f"<b>{f}</b>: {v:.3f}")
        else:
            parts.append(f"<b>{f}</b>: {v}")
    return "<br>".join(parts)


def _make_figure(
    df: pd.DataFrame,
    method: str,
    color_col: str,
    size_col: str | None,
    hover_fields: list[str],
    color_map: dict[str, str] | None = None,
) -> plotly.graph_objects.Figure:
    import plotly.graph_objects as go

    coord_cols = _COORD_COLS[method]
    axis_names = _AXIS_NAMES[method]

    # Marker sizes
    if size_col:
        sizes = _scale_sizes(df[size_col]).tolist()
    else:
        sizes = [8.0] * len(df)

    # Marker symbols: diamond = owned, circle = unowned
    if "owned" in df.columns:
        symbols = [
            ("diamond" if bool(o) else "circle")
            for o in df["owned"]
        ]
    else:
        symbols = ["circle"] * len(df)

    # Hover text
    hf = [f for f in hover_fields if f in df.columns]
    hover = [_hover_text(row, hf) for _, row in df.iterrows()]

    fig = go.Figure()

    if color_col and color_col in df.columns:
        col_series = df[color_col]
        # Treat booleans as categorical even though is_numeric_dtype returns True for them
        _is_continuous = (
            pd.api.types.is_numeric_dtype(col_series)
            and not pd.api.types.is_bool_dtype(col_series)
        )
        if _is_continuous:
            # Continuous colorscale
            fig.add_trace(go.Scatter3d(
                x=df[coord_cols[0]],
                y=df[coord_cols[1]],
                z=df[coord_cols[2]],
                mode="markers",
                marker=dict(
                    size=sizes,
                    symbol=symbols,
                    color=pd.to_numeric(col_series, errors="coerce"),
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title=color_col, thickness=14, len=0.6),
                    line=dict(width=0.3, color="#333"),
                    opacity=0.85,
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                name=color_col,
            ))
        else:
            # Categorical traces — one per unique value (also handles booleans)
            categories = col_series.map(str).fillna("—").unique()
            for i, cat in enumerate(sorted(categories, key=str)):
                mask = col_series.map(str).fillna("—") == cat
                sub = df[mask]
                sub_sizes = [s for s, m in zip(sizes, mask) if m]
                sub_syms  = [s for s, m in zip(symbols, mask) if m]
                sub_hover = [h for h, m in zip(hover, mask) if m]
                fig.add_trace(go.Scatter3d(
                    x=sub[coord_cols[0]],
                    y=sub[coord_cols[1]],
                    z=sub[coord_cols[2]],
                    mode="markers",
                    marker=dict(
                        size=sub_sizes,
                        symbol=sub_syms,
                        color=(color_map or {}).get(str(cat), _PLOTLY_PALETTE[i % len(_PLOTLY_PALETTE)]),
                        line=dict(width=0.3, color="#111"),
                        opacity=0.85,
                    ),
                    name=str(cat),
                    text=sub_hover,
                    hovertemplate="%{text}<extra></extra>",
                ))
    else:
        fig.add_trace(go.Scatter3d(
            x=df[coord_cols[0]],
            y=df[coord_cols[1]],
            z=df[coord_cols[2]],
            mode="markers",
            marker=dict(size=sizes, symbol=symbols, color="#3498db",
                        opacity=0.85, line=dict(width=0.3, color="#111")),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            name="universe",
        ))

    method_label = method.upper()
    color_label  = f"  |  color: {color_col}" if color_col else ""
    size_label   = f"  |  size: {size_col}" if size_col else ""

    fig.update_layout(
        title=dict(
            text=f"Factor Map — {method_label}{color_label}{size_label}",
            font=dict(size=14, color="#cdd6f4"),
        ),
        scene=dict(
            xaxis=dict(title=axis_names[0], gridcolor="#2d3436", backgroundcolor="#0e1117",
                       showbackground=True, color="#cdd6f4"),
            yaxis=dict(title=axis_names[1], gridcolor="#2d3436", backgroundcolor="#0e1117",
                       showbackground=True, color="#cdd6f4"),
            zaxis=dict(title=axis_names[2], gridcolor="#2d3436", backgroundcolor="#0e1117",
                       showbackground=True, color="#cdd6f4"),
            bgcolor="#0e1117",
        ),
        paper_bgcolor="#0e1117",
        font=dict(color="#cdd6f4", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#444", borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
    )

    return fig


# ---------------------------------------------------------------------------
# Component interpretation
# ---------------------------------------------------------------------------

_SHORT_LABELS: dict[str, str] = {
    "value_score":            "Value",
    "quality_score":          "Quality",
    "momentum_score":         "Momentum",
    "income_score":           "Income",
    "value_metric":           "Composite",
    "reliability_score":      "Reliability",
    "relative_pe":            "Rel PE",
    "relative_pb":            "Rel PB",
    "rs_3m":                  "RS 3m",
    "rs_6m":                  "RS 6m",
    "realized_vol_3m":        "Volatility",
    "return_1m":              "Ret 1m",
    "return_3m":              "Ret 3m",
    "return_6m":              "Ret 6m",
    "risk_adj_momentum_3m":   "Risk-adj Mom",
    "position_52w":           "52w Pos",
}

_FEATURE_THEME: dict[str, str] = {
    "value_score": "value",      "relative_pe": "value",    "relative_pb": "value",
    "quality_score": "quality",  "reliability_score": "quality",
    "momentum_score": "momentum", "rs_3m": "momentum",      "rs_6m": "momentum",
    "return_1m": "momentum",     "return_3m": "momentum",   "return_6m": "momentum",
    "risk_adj_momentum_3m": "momentum",
    "income_score": "income",
    "realized_vol_3m": "risk",
    "position_52w": "positioning",
    "value_metric": "composite",
}

_THEME_LABEL: dict[str, str] = {
    "momentum":    "Momentum / Relative Strength",
    "quality":     "Quality / Reliability",
    "value":       "Value / Valuation",
    "income":      "Income / Yield",
    "risk":        "Volatility / Risk",
    "composite":   "Composite Score",
    "positioning": "52-week Positioning",
    "other":       "Mixed factors",
}

_THEME_INTERPRETATION: dict[str, str] = {
    "momentum":    "Separates high-momentum, rising-RS stocks from low-momentum laggards. "
                   "Positive = strong recent price performance.",
    "quality":     "Separates high-quality, reliable businesses from weaker balance-sheet names. "
                   "Positive = strong fundamentals.",
    "value":       "Separates cheap stocks (low PE/PB relative to peers) from expensive ones. "
                   "Positive = attractive valuation.",
    "income":      "Separates high-yield / dividend-payers from low-yield growth names. "
                   "Positive = higher income characteristics.",
    "risk":        "Separates high-volatility names from low-volatility, stable stocks. "
                   "Positive = higher realised volatility.",
    "composite":   "Overall candidate quality — higher is a better composite buy candidate.",
    "positioning": "Separates stocks trading near 52-week highs from those near lows. "
                   "Positive = near 52-week high.",
    "other":       "Mixed-factor axis — consult the loading chart below for details.",
}


def _auto_label_component(
    loading_vec: np.ndarray,
    feature_names: list[str],
    n_top: int = 3,
) -> tuple[str, str, str]:
    """Return (short_label, theme_label, interpretation) for a loading/correlation vector."""
    abs_w = np.abs(loading_vec)
    top_idx = np.argsort(abs_w)[::-1][:n_top]

    parts: list[str] = []
    themes: list[str] = []
    for idx in top_idx:
        w = loading_vec[idx]
        sign = "+" if w >= 0 else "−"
        name = _SHORT_LABELS.get(feature_names[idx], feature_names[idx])
        parts.append(f"{sign}{name} ({w:+.2f})")
        themes.append(_FEATURE_THEME.get(feature_names[idx], "other"))

    short_label = "  ".join(parts)
    dominant = max(set(themes), key=themes.count) if themes else "other"
    return short_label, _THEME_LABEL.get(dominant, dominant), _THEME_INTERPRETATION.get(dominant, "")


def _build_component_report(
    reduction_model: Any,
    selected_cols: list[str],
    coords: np.ndarray,
    X_std: np.ndarray,
    method: str,
) -> dict:
    """
    Build a structured dict explaining the reduced dimensions.

    For PCA: uses linear component loadings + explained variance.
    For UMAP: uses Spearman correlation between each feature and each UMAP axis
              (since UMAP has no linear components).
    """
    report: dict[str, Any] = {"method": method, "feature_names": selected_cols}
    n_components = min(coords.shape[1], 3)

    if method == "pca" and hasattr(reduction_model, "components_"):
        pca = reduction_model
        n_real = len(pca.explained_variance_ratio_)
        exp_var = list(float(v) for v in pca.explained_variance_ratio_)
        comp_names = [f"PC{i + 1}" for i in range(n_real)]

        loadings_df = pd.DataFrame(
            pca.components_.T,          # (n_features, n_components)
            index=selected_cols,
            columns=comp_names,
        )

        labels, theme_labels, interpretations = [], [], []
        for i in range(n_real):
            lbl, tl, interp = _auto_label_component(pca.components_[i], selected_cols)
            labels.append(lbl)
            theme_labels.append(tl)
            interpretations.append(interp)

        report.update({
            "explained_variance_ratio": exp_var,
            "explained_variance_cumulative": [sum(exp_var[: i + 1]) for i in range(len(exp_var))],
            "loadings": loadings_df,
            "component_names": comp_names,
            "component_labels": labels,
            "component_themes": theme_labels,
            "component_interpretations": interpretations,
        })

    elif method == "umap":
        from scipy.stats import spearmanr

        comp_names = [f"UMAP{i + 1}" for i in range(n_components)]
        n_features = X_std.shape[1]
        corr_matrix = np.zeros((n_features, n_components))
        for j in range(n_components):
            for k in range(n_features):
                r, _ = spearmanr(X_std[:, k], coords[:, j])
                corr_matrix[k, j] = float(r) if not np.isnan(r) else 0.0

        corr_df = pd.DataFrame(corr_matrix, index=selected_cols, columns=comp_names)

        labels, theme_labels, interpretations = [], [], []
        for j in range(n_components):
            lbl, tl, interp = _auto_label_component(corr_matrix[:, j], selected_cols)
            labels.append(lbl)
            theme_labels.append(tl)
            interpretations.append(interp)

        report.update({
            "correlations": corr_df,
            "component_names": comp_names,
            "component_labels": labels,
            "component_themes": theme_labels,
            "component_interpretations": interpretations,
        })

    return report


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster statistics including equity weight."""
    if "cluster" not in df.columns:
        return pd.DataFrame()

    # Total active equity for weight computation
    total_equity = 0.0
    for col in ("equity", "current_value"):
        if col in df.columns and "owned" in df.columns:
            total_equity = float(
                pd.to_numeric(df.loc[df["owned"].astype(bool), col], errors="coerce").sum()
            )
            break

    rows: list[dict] = []
    for cl in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cl]
        owned_count = int(sub["owned"].astype(bool).sum()) if "owned" in sub.columns else None
        buy_count   = None
        for col in ("action", "strategy_bucket", "state"):
            if col in sub.columns:
                buy_count = int(sub[col].str.upper().eq("BUY").sum())
                break

        total_val = None
        for col in ("equity", "current_value"):
            if col in sub.columns:
                total_val = round(float(pd.to_numeric(sub[col], errors="coerce").sum()), 2)
                break

        equity_wt = None
        if total_val is not None and total_equity > 0:
            equity_wt = round(total_val / total_equity, 4)

        avg_vm = None
        if "value_metric" in sub.columns:
            avg_vm = round(float(pd.to_numeric(sub["value_metric"], errors="coerce").mean()), 4)

        avg_qs = None
        if "quality_score" in sub.columns:
            avg_qs = round(float(pd.to_numeric(sub["quality_score"], errors="coerce").mean()), 4)

        avg_ms = None
        if "momentum_score" in sub.columns:
            avg_ms = round(float(pd.to_numeric(sub["momentum_score"], errors="coerce").mean()), 4)

        top_syms: list[str] = []
        if "symbol" in sub.columns:
            top_syms = (
                sub.nlargest(5, "value_metric", keep="all")["symbol"].tolist()
                if "value_metric" in sub.columns else sub["symbol"].head(5).tolist()
            )

        rows.append({
            "cluster":            int(cl),
            "count":              len(sub),
            "owned_count":        owned_count,
            "buy_count":          buy_count,
            "equity_$":           total_val,
            "equity_weight":      equity_wt,
            "avg_value_metric":   avg_vm,
            "avg_quality_score":  avg_qs,
            "avg_momentum_score": avg_ms,
            "top_symbols":        ", ".join(str(s) for s in top_syms),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diversification metrics
# ---------------------------------------------------------------------------

def _enrich_with_diversification_metrics(
    df: pd.DataFrame,
    coord_cols: list[str],
    score_col: str = "value_metric",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compute portfolio-distance metrics in the reduced coordinate space and
    add them as new columns to df.

    Columns added:
        dist_from_owned_centroid  Euclidean distance to mean of owned positions
        norm_dist_from_centroid   Normalised to [0, 1] across all rows
        diversification_score     score_col * (0.70 + 0.30 * norm_dist)
        nearest_owned             symbol of closest owned position (vectorised)

    Returns (enriched_df, centroids) where centroids contains
        "owned"     : np.ndarray of shape (3,)
        "candidate" : np.ndarray of shape (3,)  — if any candidate rows exist
    """
    df = df.copy()
    centroids: dict[str, Any] = {}

    valid_coords = [c for c in coord_cols if c in df.columns]
    if not valid_coords or "owned" not in df.columns:
        return df, centroids

    coords = df[valid_coords].values.astype(float)
    owned_mask = df["owned"].astype(bool).values

    owned_coords = coords[owned_mask]
    if len(owned_coords) == 0:
        return df, centroids

    owned_centroid = owned_coords.mean(axis=0)
    centroids["owned"] = owned_centroid

    # Candidate centroid (using _role if present, else unowned)
    if "_role" in df.columns:
        cand_mask = (df["_role"] == "candidate").values
    else:
        cand_mask = ~owned_mask

    cand_coords = coords[cand_mask]
    if len(cand_coords) > 0:
        centroids["candidate"] = cand_coords.mean(axis=0)

    # Distance from owned centroid
    dists = np.linalg.norm(coords - owned_centroid, axis=1)
    df["dist_from_owned_centroid"] = dists

    d_min, d_max = dists.min(), dists.max()
    norm_dist = (dists - d_min) / (d_max - d_min + 1e-9)
    df["norm_dist_from_centroid"] = norm_dist

    # Diversification score — rewards candidates far from current factor cluster
    if score_col in df.columns:
        raw = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).values
        df["diversification_score"] = raw * (0.70 + 0.30 * norm_dist)

    # Nearest owned symbol (vectorised: n_all × n_owned broadcast)
    if "symbol" in df.columns and len(owned_coords) > 0:
        owned_syms = df.loc[df["owned"].astype(bool), "symbol"].values
        # (n_total, 1, n_dims) - (1, n_owned, n_dims) → (n_total, n_owned)
        diff = coords[:, np.newaxis, :] - owned_coords[np.newaxis, :, :]
        dists_to_owned = np.linalg.norm(diff, axis=2)
        nearest_idx = dists_to_owned.argmin(axis=1)
        nearest_syms = owned_syms[nearest_idx].astype(str)
        nearest_syms[owned_mask] = "—"
        df["nearest_owned"] = nearest_syms

    return df, centroids


def _add_centroid_traces(
    fig: Any,
    centroids: dict[str, Any],
    method: str,
) -> None:
    """Overlay owned-centroid and candidate-centroid as large star markers."""
    import plotly.graph_objects as go

    specs = [
        ("owned",     "#3498db", "◆ Owned centroid"),
        ("candidate", "#2ecc71", "▲ Candidate centroid"),
    ]
    for key, color, label in specs:
        center = centroids.get(key)
        if center is None or len(center) < 3:
            continue
        fig.add_trace(go.Scatter3d(
            x=[float(center[0])], y=[float(center[1])], z=[float(center[2])],
            mode="markers+text",
            marker=dict(
                size=16, color=color, symbol="diamond",
                line=dict(width=2, color="#ffffff"), opacity=1.0,
            ),
            text=[label],
            textposition="top center",
            textfont=dict(size=11, color=color),
            name=label,
            hovertemplate=(
                f"{label}<br>"
                f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
                "<extra></extra>"
            ),
        ))


def _add_top_equity_labels(
    fig: Any,
    df: pd.DataFrame,
    coord_cols: list[str],
    n: int = 10,
) -> None:
    """Add gold text labels for the top N owned positions by equity."""
    import plotly.graph_objects as go

    if "equity" not in df.columns or "owned" not in df.columns:
        return
    owned = df[df["owned"].astype(bool)].copy()
    if owned.empty:
        return
    owned["_eq"] = pd.to_numeric(owned["equity"], errors="coerce").fillna(0.0)
    top = owned.nlargest(n, "_eq")
    if not all(c in top.columns for c in coord_cols[:3]):
        return

    fig.add_trace(go.Scatter3d(
        x=top[coord_cols[0]], y=top[coord_cols[1]], z=top[coord_cols[2]],
        mode="text",
        text=top["symbol"].tolist(),
        textfont=dict(size=10, color="#f1c40f"),
        name="Top equity labels",
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=True,
    ))


def _sector_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Active-sleeve sector concentration."""
    if "sector" not in df.columns:
        return pd.DataFrame()

    owned_mask = pd.Series(False, index=df.index)
    if "owned" in df.columns:
        owned_mask = df["owned"].astype(bool)

    owned = df[owned_mask]
    if owned.empty:
        return pd.DataFrame()

    equity_col = None
    for c in ("equity", "current_value"):
        if c in owned.columns:
            equity_col = c
            break

    rows: list[dict] = []
    total_equity = (
        pd.to_numeric(owned[equity_col], errors="coerce").sum()
        if equity_col else None
    )

    for sec, sub in owned.groupby("sector"):
        eq = (
            round(float(pd.to_numeric(sub[equity_col], errors="coerce").sum()), 2)
            if equity_col else None
        )
        rows.append({
            "sector":       sec,
            "n_owned":      len(sub),
            "equity":       eq,
            "pct_active":   (
                round(eq / total_equity, 4) if eq and total_equity else None
            ),
            "symbols":      ", ".join(str(s) for s in sub.get("symbol", pd.Series()).tolist()),
        })

    return pd.DataFrame(rows).sort_values("equity", ascending=False, na_position="last")


def _compute_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    diags: dict[str, Any] = {}
    cs = _cluster_summary(df)
    if not cs.empty:
        diags["cluster_summary"] = cs
    se = _sector_exposure(df)
    if not se.empty:
        diags["sector_exposure"] = se
    return diags


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def build_factor_map(
    df: pd.DataFrame,
    method: str = "pca",
    feature_cols: list[str] | None = None,
    color_by: str | None = None,
    _symbol_by: str | None = None,   # unused — "owned" is auto-detected
    size_by: str | None = None,
    score_col: str = "value_metric",
    owned_only: bool = False,
    unowned_only: bool = False,
    actions: list[str] | None = None,
    sectors: list[str] | None = None,
    sleeves: list[str] | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    kmeans_clusters: int | None = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    output_html: str | None = None,
    show: bool = False,
    hover_fields: list[str] | None = None,
    color_map: dict[str, str] | None = None,
    exclude_outliers: bool = False,
    outlier_mad_z: float = 5.0,
) -> tuple[plotly.graph_objects.Figure, pd.DataFrame, dict]:
    """
    Build an interactive 3-D factor-map of the scored universe.

    Parameters
    ----------
    df              : scored universe DataFrame (agg_data + optional holdings merge)
    method          : "pca" (always available) or "umap" (requires umap-learn)
    feature_cols    : explicit list of feature columns; auto-detected if None
    color_by        : column to use for point colour; auto-resolved if None
    size_by         : column to scale marker size; auto-resolved if None
    score_col       : column used for min_score / max_score filters
    owned_only      : keep only rows where "owned" == True
    unowned_only    : keep only rows where "owned" == False
    actions         : filter by action / strategy_bucket values, e.g. ["BUY", "HOLD"]
    sectors         : filter by sector name
    sleeves         : filter by sleeve name
    min_score       : minimum score_col value to include
    max_score       : maximum score_col value to include
    kmeans_clusters : if set, run KMeans and add "cluster" column; enables color_by="cluster"
    umap_n_neighbors: UMAP n_neighbors hyperparameter (default 15)
    umap_min_dist   : UMAP min_dist hyperparameter (default 0.1)
    output_html     : if provided, save figure to this HTML path
    show            : call fig.show() after building
    hover_fields    : fields to include in hover tooltip; sensible defaults if None

    Returns
    -------
    (fig, df_out, diagnostics)
        fig         : plotly Figure — embed in Streamlit with st.plotly_chart(fig)
        df_out      : input DataFrame enriched with coordinate + cluster columns
        diagnostics : dict with "cluster_summary" and/or "sector_exposure" DataFrames
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty — nothing to visualise.")

    method = method.lower()
    if method not in ("pca", "umap"):
        raise ValueError(f"method must be 'pca' or 'umap', got {method!r}")

    df = df.copy()

    # ── Feature selection & preprocessing ────────────────────────────────────
    feat_df, selected_cols = _select_features(df, feature_cols)
    X = _standardize(feat_df)

    # ── Outlier exclusion (before embedding) ──────────────────────────────────
    # Robust per-feature MAD z-score on the standardized matrix. Dropping oddballs
    # here (not just hiding them) is what stops UMAP/PCA geometry being distorted.
    outliers_excluded: list[str] = []
    if exclude_outliers and len(df) > 0:
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) * 1.4826
        mad[mad == 0] = np.inf  # a zero-spread feature can never flag an outlier
        robust_z = np.abs((X - med) / mad)
        out_mask = robust_z.max(axis=1) > outlier_mad_z
        n_out = int(out_mask.sum())
        max_drop = min(int(0.10 * len(df)), len(df) - 4)  # never gut the embedding
        if 0 < n_out <= max_drop:
            if "symbol" in df.columns:
                outliers_excluded = df.loc[out_mask, "symbol"].astype(str).tolist()
            keep = ~out_mask
            df = df.loc[keep].reset_index(drop=True)
            X = X[keep]
            logger.info("Excluded %d feature-space outlier(s) before embedding", n_out)
        elif n_out > max_drop:
            logger.info("Outlier exclusion skipped: %d exceeds cap %d", n_out, max_drop)

    # ── Dimensionality reduction ──────────────────────────────────────────────
    if method == "umap":
        coords, reduction_model = _reduce_umap(X, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist)
    else:
        coords, reduction_model = _reduce_pca(X)

    coord_cols = _COORD_COLS[method]
    for i, col in enumerate(coord_cols):
        df[col] = coords[:, i]

    # ── Clustering ────────────────────────────────────────────────────────────
    if kmeans_clusters is not None and kmeans_clusters >= 2:
        labels = _apply_kmeans(X, kmeans_clusters)
        df["cluster"] = labels.astype(str)   # string so Plotly treats it as categorical
        logger.info("KMeans: %d clusters assigned", kmeans_clusters)

    # ── Filters ───────────────────────────────────────────────────────────────
    df_filtered = _apply_filters(
        df,
        owned_only=owned_only,
        unowned_only=unowned_only,
        actions=actions,
        sectors=sectors,
        sleeves=sleeves,
        min_score=min_score,
        max_score=max_score,
        score_col=score_col,
    )

    if df_filtered.empty:
        raise ValueError("No rows remain after applying filters.")

    # ── Diversification metrics ───────────────────────────────────────────────
    df_filtered, centroids = _enrich_with_diversification_metrics(
        df_filtered, coord_cols, score_col=score_col
    )

    # ── Resolve display columns ────────────────────────────────────────────────
    resolved_color = _resolve_color_by(df_filtered, color_by)
    resolved_size  = _resolve_size_by(df_filtered, size_by)

    if hover_fields is None:
        hover_fields = [
            "symbol", "name", "sector", "industry", "sleeve",
            "action", "strategy_bucket", "owned",
            "equity", "current_value", "target_value", "delta_value",
            "active_score", "final_score", "value_metric", "quality_score",
            "momentum_score", "value_score", "income_score",
            "reliability_score", "cluster",
            "dist_from_owned_centroid", "diversification_score", "nearest_owned",
        ]

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = _make_figure(
        df_filtered,
        method=method,
        color_col=resolved_color,
        size_col=resolved_size,
        hover_fields=hover_fields,
        color_map=color_map,
    )
    _add_centroid_traces(fig, centroids, method)
    _add_top_equity_labels(fig, df_filtered, coord_cols)

    # ── Diagnostics ────────────────────────────────────────────────────────────
    diagnostics = _compute_diagnostics(df_filtered)
    diagnostics["centroids"] = centroids
    diagnostics["outliers_excluded"] = outliers_excluded
    try:
        diagnostics["component_report"] = _build_component_report(
            reduction_model, selected_cols, coords, X, method
        )
    except Exception as _ce:
        logger.warning("Component report skipped: %s", _ce)

    # ── Output ─────────────────────────────────────────────────────────────────
    if output_html:
        out_path = Path(output_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        logger.info("Factor map saved → %s", out_path)

    if show:
        fig.show()

    n_pts = len(df_filtered)
    logger.info(
        "Factor map built: %d points, method=%s, color=%s, size=%s, features=%d",
        n_pts, method, resolved_color or "—", resolved_size or "fixed",
        len(selected_cols),
    )

    return fig, df_filtered, diagnostics


# ---------------------------------------------------------------------------
# Data loading helpers (used by CLI and Streamlit component)
# ---------------------------------------------------------------------------

def load_universe_with_holdings(agg_path: str | None = None) -> pd.DataFrame:
    """
    Load latest agg_data and merge with latest holdings to add:
        owned         bool
        equity        float (from holdings)
        name          str   (from holdings)

    Returns enriched DataFrame ready for build_factor_map().
    """
    from data.cache import read_data_as_pd

    if agg_path:
        df = pd.read_csv(agg_path)
    else:
        df = read_data_as_pd("agg_data")
        if df is None or df.empty:
            raise FileNotFoundError(
                "No agg_data found. Run 'daily-investor fetch-data' first."
            )

    # Merge holdings
    try:
        holdings = read_data_as_pd("holdings")
        if holdings is not None and not holdings.empty and "symbol" in holdings.columns:
            held_syms = set(holdings["symbol"].dropna().astype(str))
            df["owned"] = df["symbol"].astype(str).isin(held_syms)

            # Bring in equity and name from holdings
            hold_slim = holdings[
                [c for c in ["symbol", "equity", "name", "percent_change", "equity_change"]
                 if c in holdings.columns]
            ].copy()
            hold_slim["symbol"] = hold_slim["symbol"].astype(str)
            df["symbol"] = df["symbol"].astype(str)
            df = df.merge(hold_slim, on="symbol", how="left", suffixes=("", "_h"))
        else:
            df["owned"] = False
    except Exception as exc:
        logger.warning("Could not merge holdings: %s", exc)
        df["owned"] = False

    return df


# ---------------------------------------------------------------------------
# ETF / asset-type classification
#
# ETFs are identified from config (the configured ETF sleeve, harvest ETFs, and
# the backtest benchmark), from the Robinhood ``instrument_type`` column when
# agg_data carries it (populated at build time by data.market.get_data), and
# from any explicit is_etf / asset_type / security_type metadata. We never infer
# ETF status from sector or missing fundamentals.
# ---------------------------------------------------------------------------

_ETF_ASSET_VALUES = {"etf", "fund", "etn", "index", "index_fund"}

# Robinhood instrument ``type`` values treated as pooled funds (excluded by the
# stocks-only / active-sleeve scopes). ADR / REIT remain individual equities.
_ETF_INSTRUMENT_TYPES = {"etp", "cef", "mlp"}


def etf_symbols_from_config(config: dict | None = None) -> set[str]:
    """Union of configured ETF / benchmark tickers, upper-cased.

    Pulls from ``etfs``, ``harvest.harvest_etfs`` (and a top-level
    ``harvest_etfs`` fallback), and ``backtest.benchmark_symbol``. Falls back to
    ``util.ETFS`` when those keys are missing or no config dict is supplied.
    """
    config = config or {}

    syms: set[str] = set()

    def _add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if value.strip():
                syms.add(value.strip().upper())
        elif isinstance(value, (list, tuple, set)):
            for v in value:
                _add(v)

    _add(config.get("etfs"))
    harvest = config.get("harvest")
    if isinstance(harvest, dict):
        _add(harvest.get("harvest_etfs"))
    _add(config.get("harvest_etfs"))
    backtest = config.get("backtest")
    if isinstance(backtest, dict):
        _add(backtest.get("benchmark_symbol"))
    _add(config.get("benchmark_symbol"))

    if not syms:
        try:
            from util import ETFS
            _add(list(ETFS))
        except Exception:
            pass

    return syms


def is_etf_symbol(symbol: str, config: dict | None = None) -> bool:
    """True if ``symbol`` is a configured ETF / benchmark ticker (case-insensitive)."""
    if symbol is None:
        return False
    sym = str(symbol).strip().upper()
    if not sym:
        return False
    return sym in etf_symbols_from_config(config)


def tag_etf(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Return ``df`` with a boolean ``is_etf`` column.

    Honours an existing ``is_etf`` column; the Robinhood ``instrument_type``
    column (``etp`` / ``cef`` / ``mlp`` → ETF); an explicit ``asset_type`` /
    ``security_type`` field (values like ``etf`` / ``fund``); and config
    ETF-ticker membership. These signals are OR-ed together. Never infers ETF
    status from sector or missing fundamentals.
    """
    df = df.copy()
    n = len(df)

    if "is_etf" in df.columns:
        df["is_etf"] = df["is_etf"].fillna(False).astype(bool)
        return df

    mask = pd.Series(False, index=df.index)

    for col in ("asset_type", "security_type"):
        if col in df.columns:
            vals = df[col].astype(str).str.strip().str.lower()
            mask = mask | vals.isin(_ETF_ASSET_VALUES)

    if "instrument_type" in df.columns:
        itype = df["instrument_type"].astype(str).str.strip().str.lower()
        mask = mask | itype.isin(_ETF_INSTRUMENT_TYPES)

    if "symbol" in df.columns:
        etfs = etf_symbols_from_config(config)
        if etfs:
            sym_upper = df["symbol"].astype(str).str.strip().str.upper()
            mask = mask | sym_upper.isin(etfs)

    df["is_etf"] = mask.to_numpy() if n else mask
    return df


# ---------------------------------------------------------------------------
# CLI __main__ entrypoint
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    from core.logging import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(
        prog="python -m portfolio.visualization.factor_map",
        description="Build an interactive 3-D factor map of the scored universe.",
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to agg_data CSV (default: latest data/agg_data_*.csv)",
    )
    parser.add_argument(
        "--method", choices=["pca", "umap"], default="pca",
        help="Dimensionality reduction method (default: pca)",
    )
    parser.add_argument(
        "--color", default=None,
        help="Column to colour points by (e.g. sector, strategy_bucket, cluster)",
    )
    parser.add_argument(
        "--clusters", type=int, default=None, metavar="N",
        help="Run KMeans with N clusters and colour by cluster",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save figure to this HTML file (e.g. reports/factor_map.html)",
    )
    parser.add_argument(
        "--owned-only", action="store_true", help="Show only owned positions"
    )
    parser.add_argument(
        "--unowned-only", action="store_true", help="Show only un-owned positions"
    )
    parser.add_argument(
        "--actions", nargs="*", default=None,
        help="Filter by action/strategy_bucket values, e.g. --actions BUY HOLD",
    )
    parser.add_argument(
        "--sectors", nargs="*", default=None,
        help="Filter by sector name, e.g. --sectors Technology Energy",
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="Minimum value_metric to include",
    )
    parser.add_argument(
        "--max-score", type=float, default=None,
        help="Maximum value_metric to include",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open the figure in a browser window",
    )

    args = parser.parse_args()

    # Default output if neither --output nor --show given
    output = args.output
    show   = args.show
    if not output and not show:
        output = "reports/factor_map.html"
        logger.info("No --output or --show specified; defaulting to %s", output)

    color = "cluster" if args.clusters else args.color

    df = load_universe_with_holdings(agg_path=args.input)
    logger.info("Loaded universe: %d symbols", len(df))

    fig, df_out, diags = build_factor_map(
        df,
        method=args.method,
        color_by=color,
        kmeans_clusters=args.clusters,
        owned_only=args.owned_only,
        unowned_only=args.unowned_only,
        actions=args.actions,
        sectors=args.sectors,
        min_score=args.min_score,
        max_score=args.max_score,
        output_html=output,
        show=show,
    )

    # Print diagnostics
    if "cluster_summary" in diags:
        print("\n── Cluster Summary ─────────────────────────────────────────────")
        print(diags["cluster_summary"].to_string(index=False))

    if "sector_exposure" in diags:
        print("\n── Sector Exposure (owned) ──────────────────────────────────────")
        print(diags["sector_exposure"].to_string(index=False))

    if output:
        print(f"\nFactor map saved → {output}")


if __name__ == "__main__":
    _cli_main()
