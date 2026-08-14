"""
scripts/quality_component_study.py — do the 7 peer-3 quality components earn their weight?

RESEARCH ONLY — never writes config.

Why: the peer-3 quality factor went live 2026-08-05 carrying ~40% of the composite, but its
adoption rested on ONE 63-day IC comparison against the old checklist plus a two-half split,
computed on the 73-day snapshot substrate. Nobody has asked which of its seven components
actually carry signal, whether they are redundant with each other, or whether the seven-way
blend beats a simpler subset. The FMP PIT panel (6,681 symbols × 204 monthly dates,
2009-2025, survivorship-free) has ~3x the dates and vastly more cross-sectional depth, so it
can answer that with real power.

Method (all cross-sectional per date, then averaged over dates — standard IC):
  1. RAW IC          — Spearman(component rank, forward SPY-excess return).
  2. INCREMENTAL IC  — residualize the component on value (earnings_yield, book_yield),
                       size (log_mcap) and momentum (ret_6m) first. This is the honest
                       test: a "quality" component that is really cheapness or size in
                       disguise scores near zero here. It is how the peer-3 quality factor
                       was originally justified, and the only reason it survived (the old
                       checklist's raw IC was HIGHER; its incremental IC decayed).
  3. REDUNDANCY      — mean pairwise Spearman between components; ROE/FCF/margins plausibly
                       measure one underlying thing.
  4. DROP-ONE        — composite IC minus the composite without each component, at live
                       weights (renormalized). Negative delta = the component is dead weight.
  5. HONEST SUBSET   — exhaustive search over all 127 non-empty subsets, chosen on the FIRST
                       half of dates and scored on the SECOND. Selecting and evaluating on
                       the same sample is how this project has repeatedly fooled itself; the
                       out-of-sample number is the only one that means anything.

Usage:  PYTHONPATH=src python3 scripts/quality_component_study.py [horizon]
        horizon in {21, 63, 126}; default 63.
"""
from __future__ import annotations

import itertools
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

PANEL = ".session_tmp/alpha_discovery/out/poc_panel.parquet"
HORIZON = int(sys.argv[1]) if len(sys.argv) > 1 else 63
FWD = f"fwd_exc_{HORIZON}d"
MIN_NAMES = 300
CONTROLS = ("earnings_yield", "book_yield", "log_mcap", "ret_6m")


def _live_weights() -> dict[str, float]:
    """Component weights straight from the live scorer — never a hardcoded copy."""
    from strategy.scoring.quality import _COMPONENT_WEIGHTS
    from util import SCORING_PARAMS
    cfg = {k: float(v) for k, v in (SCORING_PARAMS.get("quality_components") or {}).items()}
    return cfg or dict(_COMPONENT_WEIGHTS)


# Panel column backing each live component. `low_leverage` is -debt_to_assets in the
# scorer, so it is negated here too (higher = better, matching the shared ranking).
_COL = {
    "roe_ttm": ("roe_ttm", +1),
    "fcf_to_assets": ("fcf_to_assets", +1),
    "neg_accruals": ("neg_accruals", +1),
    "gross_margin_ttm": ("gross_margin_ttm", +1),
    "low_leverage": ("debt_to_assets", -1),
    "share_count_shrink_yoy": ("share_count_shrink_yoy", +1),
    "gm_trend_yoy": ("gm_trend_yoy", +1),
}


def _rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def _residualize(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Residual of y on X (both cross-sectional ranks), OLS with intercept."""
    ok = y.notna() & X.notna().all(axis=1)
    out = pd.Series(np.nan, index=y.index)
    if ok.sum() < 50:
        return out
    A = np.column_stack([X[ok].to_numpy(), np.ones(int(ok.sum()))])
    beta, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
    out[ok] = y[ok].to_numpy() - A @ beta
    return out


def main() -> None:
    from pit_factor_ic_peer3 import build_feature_matrix

    t0 = time.time()
    weights = _live_weights()
    comps = [c for c in weights if c in _COL]
    print(f"peer-3 quality component study — horizon {HORIZON}d, live weights:")
    for c in comps:
        print(f"    {c:24s} {weights[c]:.3f}")

    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    symbols = sorted(panel["symbol"].unique())
    grid = np.sort(panel["date"].unique())
    print(f"\npanel: {len(panel):,} rows | {len(symbols)} symbols | {len(grid)} dates "
          f"({pd.Timestamp(grid[0]).date()} → {pd.Timestamp(grid[-1]).date()})", flush=True)

    mats = build_feature_matrix(symbols, grid)
    sym_idx = {s: j for j, s in enumerate(symbols)}
    date_idx = {d: i for i, d in enumerate(grid)}
    ii = panel["date"].map(date_idx).to_numpy()
    jj = panel["symbol"].map(sym_idx).to_numpy()
    for c, mat in mats.items():
        panel[c] = mat[ii, jj]
    print(f"features merged ({time.time() - t0:.0f}s)\n", flush=True)

    half = grid[len(grid) // 2]
    per_date: list[dict] = []
    for d, g in panel.groupby("date"):
        fwd = g[FWD]
        if fwd.notna().sum() < MIN_NAMES:
            continue
        ranks = {}
        for c in comps:
            col, sign = _COL[c]
            if col not in g.columns:
                continue
            ranks[c] = _rank(sign * g[col])
        if len(ranks) < len(comps):
            continue
        ctrl = pd.DataFrame({k: _rank(g[k]) for k in CONTROLS if k in g.columns})

        row: dict = {"date": d, "half": "H1" if d < half else "H2", "n": int(fwd.notna().sum())}
        for c, r in ranks.items():
            ok = r.notna() & fwd.notna()
            if ok.sum() >= MIN_NAMES:
                row[f"raw::{c}"] = spearmanr(r[ok], fwd[ok])[0]
            resid = _residualize(r, ctrl)
            ok2 = resid.notna() & fwd.notna()
            if ok2.sum() >= MIN_NAMES:
                row[f"inc::{c}"] = spearmanr(resid[ok2], fwd[ok2])[0]

        # composites: full blend, drop-one, and every subset (for the honest search)
        rank_df = pd.DataFrame(ranks)
        for subset in _all_subsets(comps):
            w = np.array([weights[c] for c in subset], dtype=float)
            w = w / w.sum()
            blend = (rank_df[list(subset)] * w).sum(axis=1, min_count=1)
            resid = _residualize(blend, ctrl)
            ok = resid.notna() & fwd.notna()
            if ok.sum() >= MIN_NAMES:
                row[f"sub::{'|'.join(subset)}"] = spearmanr(resid[ok], fwd[ok])[0]
        per_date.append(row)

    df = pd.DataFrame(per_date)
    print(f"scored {len(df)} dates ({time.time() - t0:.0f}s)\n", flush=True)

    # ---- 1/2. per-component raw vs incremental IC -------------------------------
    print("=" * 92)
    print("PER-COMPONENT IC  (mean across dates; 'incremental' = controlling value+size+momentum)")
    print(f"{'component':24s} {'weight':>7s} {'raw IC':>9s} {'inc IC':>9s} {'inc t':>7s} "
          f"{'inc H1':>8s} {'inc H2':>8s}")
    rows = []
    for c in comps:
        raw = df.get(f"raw::{c}", pd.Series(dtype=float)).dropna()
        inc = df.get(f"inc::{c}", pd.Series(dtype=float)).dropna()
        t = inc.mean() / (inc.std() / np.sqrt(len(inc))) if len(inc) > 2 else np.nan
        h1 = df.loc[df.half == "H1", f"inc::{c}"].mean()
        h2 = df.loc[df.half == "H2", f"inc::{c}"].mean()
        rows.append((c, weights[c], raw.mean(), inc.mean(), t, h1, h2))
        print(f"{c:24s} {weights[c]:7.3f} {raw.mean():+9.4f} {inc.mean():+9.4f} {t:7.1f} "
              f"{h1:+8.4f} {h2:+8.4f}")

    # ---- 3. redundancy ----------------------------------------------------------
    print("\n" + "=" * 92)
    print("REDUNDANCY — mean cross-sectional Spearman between component ranks")
    corr_acc = {}
    for d, g in panel.groupby("date"):
        rk = {}
        for c in comps:
            col, sign = _COL[c]
            if col in g.columns:
                rk[c] = _rank(sign * g[col])
        if len(rk) < len(comps):
            continue
        m = pd.DataFrame(rk).corr(method="spearman")
        for a, b in itertools.combinations(comps, 2):
            corr_acc.setdefault((a, b), []).append(m.loc[a, b])
    pairs = sorted(((np.mean(v), a, b) for (a, b), v in corr_acc.items()), reverse=True)
    for r, a, b in pairs[:6]:
        flag = "  <-- redundant" if abs(r) > 0.5 else ""
        print(f"  {a:24s} x {b:24s} {r:+.3f}{flag}")

    # ---- 4. drop-one ------------------------------------------------------------
    print("\n" + "=" * 92)
    print("DROP-ONE — incremental IC of the blend without each component (full = all 7)")
    full_key = f"sub::{'|'.join(comps)}"
    full_ic = df[full_key].mean()
    print(f"  {'full 7-component blend':40s} {full_ic:+.4f}")
    for c in comps:
        rest = tuple(x for x in comps if x != c)
        k = f"sub::{'|'.join(rest)}"
        if k in df:
            v = df[k].mean()
            verdict = "component ADDS nothing" if v >= full_ic else ""
            print(f"  {'without ' + c:40s} {v:+.4f}  (Δ {full_ic - v:+.4f}) {verdict}")

    # ---- 5. honest subset selection --------------------------------------------
    print("\n" + "=" * 92)
    print("SUBSET SELECTION — chosen on H1 dates, scored on H2 (never the same sample)")
    sub_cols = [c for c in df.columns if c.startswith("sub::")]
    h1 = df[df.half == "H1"][sub_cols].mean()
    h2 = df[df.half == "H2"][sub_cols].mean()
    best_h1 = h1.idxmax()
    print(f"  best subset on H1 : {best_h1[5:]}")
    print(f"    H1 IC {h1[best_h1]:+.4f}   ->   H2 IC {h2[best_h1]:+.4f}")
    print(f"  full 7-component  : H1 IC {h1[full_key]:+.4f}   ->   H2 IC {h2[full_key]:+.4f}")
    print(f"  best subset on H2 (hindsight, NOT actionable): {h2.idxmax()[5:]} @ {h2.max():+.4f}")
    verdict = ("the full blend is at least as good out-of-sample — keep all 7"
               if h2[full_key] >= h2[best_h1]
               else "a subset beat the full blend out-of-sample — worth simplifying")
    print(f"\n  VERDICT: {verdict}")

    # ---- 6. named candidates -----------------------------------------------------
    # Hand-specified so the recommendation is a config we could actually ship, not
    # whichever subset happened to top a 127-way search (that maximum is itself noisy).
    named = {
        "full 7 (live today)": tuple(comps),
        "drop the 3 that add nothing": tuple(
            c for c in comps if c not in ("neg_accruals", "low_leverage", "gm_trend_yoy")),
        "drop only the 2 negative-IC": tuple(
            c for c in comps if c not in ("neg_accruals", "low_leverage")),
        "profitability core (roe+fcf+shares)": ("roe_ttm", "fcf_to_assets", "share_count_shrink_yoy"),
        "H1-selected best": tuple(best_h1[5:].split("|")),
        "fcf_to_assets alone": ("fcf_to_assets",),
    }
    print("\n" + "=" * 92)
    print("NAMED CANDIDATES — H2 (held-out) is the number that counts")
    print(f"{'candidate':38s} {'n':>2s} {'H1 IC':>9s} {'H2 IC':>9s}  components")
    for label, subset in named.items():
        k = f"sub::{'|'.join(subset)}"
        if k not in df:
            continue
        print(f"{label:38s} {len(subset):2d} {h1[k]:+9.4f} {h2[k]:+9.4f}  "
              f"{', '.join(subset)}")
    print(f"\n({time.time() - t0:.0f}s) — nothing written to cfg/config.yaml")


def _all_subsets(comps):
    for r in range(1, len(comps) + 1):
        yield from itertools.combinations(comps, r)


if __name__ == "__main__":
    main()
