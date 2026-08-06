"""
scripts/verify_peer3_migration.py — read-only PASS/FAIL audit of the peer-3 snapshot migration.

Checks (each prints PASS/FAIL; exit code 1 if any FAIL):
  1. Version stamp: every snapshot carries scoring_model_version == "peer-3".
  2. Fundamental coverage: roe_ttm coverage >= 50% on every vintage.
  3. Quality is alive: per-file quality_score std > 0.05 (not degenerate).
  4. Income de-degeneracy: max single-value mass among NONZERO income scores < 2%.
  5. Orthogonality held historically: per-file Spearman(quality, income) < 0.30 on
     the 2026-07+ full-fidelity vintages (mean reported for all).
  6. SPARSE rescue: the 2025 vintage files reach pe_ratio coverage >= 60% and
     return_3m coverage >= 80%.
  7. Mega-cap sanity: AAPL/MSFT quality_score in the top half of the latest file.

Run AFTER `daily-investor snapshots rescore`. Never writes.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "src")

from strategy.scoring.composite import SCORING_MODEL_VERSION  # noqa: E402
from strategy.snapshots import list_snapshots  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def main() -> None:
    snaps = list_snapshots()
    print(f"verify peer-3 migration: {len(snaps)} snapshots, engine {SCORING_MODEL_VERSION}")

    bad_version, low_fund, dead_quality, income_mass_bad = [], [], [], []
    sparse_val_bad, sparse_px_bad = [], []
    qi_corrs = []
    latest_df = None

    for d, path in snaps:
        try:
            df = pd.read_parquet(path)
        except Exception:
            bad_version.append(f"{path.name} (unreadable)")
            continue
        latest_df = df

        if not (df.get("scoring_model_version") == SCORING_MODEL_VERSION).all():
            bad_version.append(path.name)

        roe_cov = pd.to_numeric(df.get("roe_ttm"), errors="coerce").notna().mean() \
            if "roe_ttm" in df.columns else 0.0
        if roe_cov < 0.50:
            low_fund.append(f"{path.name} ({roe_cov:.0%})")

        q = pd.to_numeric(df.get("quality_score"), errors="coerce")
        if q.std() < 0.05:
            dead_quality.append(path.name)

        inc = pd.to_numeric(df.get("income_score"), errors="coerce")
        nz = inc[inc.notna() & (inc != 0.0)]
        if len(nz) > 100:
            mass = nz.round(4).value_counts().iloc[0] / len(nz)
            if mass >= 0.02:
                income_mass_bad.append(f"{path.name} ({mass:.1%})")

        ok = q.notna() & inc.notna()
        if ok.sum() > 300:
            qi_corrs.append((d, float(spearmanr(q[ok], inc[ok])[0])))

        if d.year == 2025 or (d.year == 2026 and d.month <= 5):
            pe_cov = pd.to_numeric(df.get("pe_ratio"), errors="coerce").notna().mean() \
                if "pe_ratio" in df.columns else 0.0
            r3_cov = pd.to_numeric(df.get("return_3m"), errors="coerce").notna().mean() \
                if "return_3m" in df.columns else 0.0
            if pe_cov < 0.60:
                sparse_val_bad.append(f"{path.name} ({pe_cov:.0%})")
            if r3_cov < 0.80:
                sparse_px_bad.append(f"{path.name} ({r3_cov:.0%})")

    check("version stamp on every file", not bad_version, "; ".join(bad_version[:5]))
    check("fundamental coverage >= 50% everywhere", not low_fund, "; ".join(low_fund[:5]))
    check("quality_score alive (std > 0.05)", not dead_quality, "; ".join(dead_quality[:5]))
    check("income nonzero top-mass < 2%", not income_mass_bad, "; ".join(income_mass_bad[:5]))

    recent = [c for d, c in qi_corrs if d >= pd.Timestamp("2026-07-01").date()]
    all_mean = np.mean([c for _, c in qi_corrs]) if qi_corrs else float("nan")
    recent_ok = bool(recent) and max(recent) < 0.30
    check(
        "Spearman(quality, income) < 0.30 on 2026-07+ vintages", recent_ok,
        f"recent max {max(recent):.3f}, all-history mean {all_mean:.3f}" if recent else "no recent files",
    )

    check("SPARSE vintages: pe_ratio coverage >= 60%", not sparse_val_bad, "; ".join(sparse_val_bad[:5]))
    check("SPARSE vintages: return_3m coverage >= 80%", not sparse_px_bad, "; ".join(sparse_px_bad[:5]))

    if latest_df is not None and "symbol" in latest_df.columns:
        q = pd.to_numeric(latest_df["quality_score"], errors="coerce")
        med = q.median()
        mega = latest_df[latest_df["symbol"].isin(["AAPL", "MSFT"])]
        mq = pd.to_numeric(mega["quality_score"], errors="coerce")
        check(
            "mega-cap sanity (AAPL/MSFT quality above median)",
            bool(len(mq)) and (mq > med).all(),
            f"AAPL/MSFT quality {list(mq.round(3))} vs median {med:.3f}",
        )

    print()
    if FAILURES:
        print(f"VERIFY: {len(FAILURES)} FAILURE(S)")
        sys.exit(1)
    print("VERIFY: ALL CHECKS PASS")


if __name__ == "__main__":
    main()
