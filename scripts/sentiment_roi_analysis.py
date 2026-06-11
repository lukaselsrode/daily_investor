"""
scripts/sentiment_roi_analysis.py — does the Claude sentiment gate earn its keep?

The sentiment layer costs tokens every run and skips roughly half the
BUY-recommended candidates. This reads the decision-outcomes ledger and compares
forward outcomes (vs SPY, per project rule) of candidates the gate APPROVED
against candidates it REJECTED (skip_reason == sentiment gate), controlling for
composite score, plus a confidence-calibration table.

Read-only: consumes data/decision_outcomes.parquet, writes nothing.
"""

import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

HORIZONS = [("future_7d_vs_spy", "7d"), ("future_30d_vs_spy", "30d"), ("future_90d_vs_spy", "90d")]


def main() -> None:
    df = pd.read_parquet("data/decision_outcomes.parquet")
    cand = df[df["record_type"] == "candidate"].copy()
    print(f"ledger rows: {len(df)}  candidate rows: {len(cand)}")

    cand["sentiment_result"] = cand["sentiment_result"].astype(str).str.upper()
    cand["skip_reason"] = cand["skip_reason"].astype(str)

    approved = cand[(cand["sentiment_result"] == "BUY")].copy()
    gated = cand[cand["skip_reason"].str.contains("sentiment", case=False, na=False)].copy()
    print(f"sentiment-approved (BUY): {len(approved)}   sentiment-gated skips: {len(gated)}")

    for col, label in HORIZONS:
        a = pd.to_numeric(approved[col], errors="coerce").dropna()
        g = pd.to_numeric(gated[col], errors="coerce").dropna()
        if len(a) < 5 or len(g) < 5:
            print(f"[{label}] insufficient outcome coverage (approved n={len(a)}, gated n={len(g)})")
            continue
        print(
            f"[{label}] approved: n={len(a)} mean {a.mean():+.2%} median {a.median():+.2%} hit {(a > 0).mean():.0%}"
            f"   |   gated-skip: n={len(g)} mean {g.mean():+.2%} median {g.median():+.2%} hit {(g > 0).mean():.0%}"
            f"   |   approval edge: {a.mean() - g.mean():+.2%}"
        )

    # ── Score-matched comparison (the honest test: same-quality candidates) ──
    both = pd.concat([
        approved.assign(_grp="approved"),
        gated.assign(_grp="gated"),
    ])
    both["vm"] = pd.to_numeric(both["current_value_metric"], errors="coerce")
    both = both.dropna(subset=["vm"])
    if len(both) >= 20:
        both["score_band"] = pd.qcut(both["vm"], q=3, labels=["low", "mid", "high"], duplicates="drop")
        print("\nScore-matched 30d excess vs SPY (mean within score band):")
        for band in both["score_band"].unique().categories if hasattr(both["score_band"], "categories") else ["low", "mid", "high"]:
            sub = both[both["score_band"] == band]
            a = pd.to_numeric(sub[sub["_grp"] == "approved"]["future_30d_vs_spy"], errors="coerce").dropna()
            g = pd.to_numeric(sub[sub["_grp"] == "gated"]["future_30d_vs_spy"], errors="coerce").dropna()
            if len(a) >= 3 and len(g) >= 3:
                print(f"  band={band}: approved n={len(a)} {a.mean():+.2%}  vs  gated n={len(g)} {g.mean():+.2%}"
                      f"   edge {a.mean() - g.mean():+.2%}")
            else:
                print(f"  band={band}: insufficient (approved n={len(a)}, gated n={len(g)})")

    # ── Confidence calibration: does higher BUY confidence mean better outcomes? ──
    conf = approved.copy()
    conf["confidence"] = pd.to_numeric(conf["sentiment_confidence"], errors="coerce")
    conf["fwd30"] = pd.to_numeric(conf["future_30d_vs_spy"], errors="coerce")
    conf = conf.dropna(subset=["confidence", "fwd30"])
    if len(conf) >= 15:
        print("\nBUY-confidence calibration (30d excess vs SPY):")
        bins = pd.cut(conf["confidence"], bins=[0, 70, 80, 101], labels=["65-70", "70-80", "80+"])
        for b, sub in conf.groupby(bins, observed=True):
            print(f"  conf {b}: n={len(sub)}  mean {sub['fwd30'].mean():+.2%}  hit {(sub['fwd30'] > 0).mean():.0%}")
        corr = conf[["confidence", "fwd30"]].corr().iloc[0, 1]
        print(f"  Spearman-ish corr(confidence, 30d excess): {np.corrcoef(conf['confidence'].rank(), conf['fwd30'].rank())[0,1]:+.3f} (pearson {corr:+.3f})")

    # ── Coverage caveat ──
    cov30 = pd.to_numeric(cand["future_30d_vs_spy"], errors="coerce").notna().mean()
    print(f"\n30d outcome coverage across candidate rows: {cov30:.0%} "
          f"(rows newer than ~30 trading days have no forward window yet)")


if __name__ == "__main__":
    main()
