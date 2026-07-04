"""scripts/verify_peer2_migration.py — read-only checks after the peer-2 snapshot rescore.

Usage:  PYTHONPATH=src python3 scripts/verify_peer2_migration.py [snapshot_dir]

Checks (read-only, prints a PASS/FAIL summary):
  1. Every non-.bak snapshot is stamped with the current SCORING_MODEL_VERSION.
  2. dollar_vol_63d non-null coverage per file (flags < 50%, summarizes overall).
  3. Mega-cap sanity: AAPL/MSFT-class names sit in the top decile of dollar_vol_63d.
  4. quality_score distribution per file is finite and non-degenerate (std > 0).
  5. Old-vintage files (previously no volume column) now carry a synthesized share
     ADV `volume`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from strategy.scoring.composite import SCORING_MODEL_VERSION

MEGA_CAPS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


def main() -> int:
    snap_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/snapshots")
    files = sorted(p for p in snap_dir.glob("*.parquet") if not p.name.endswith(".bak.parquet"))
    if not files:
        print(f"FAIL: no snapshots in {snap_dir}")
        return 1

    bad_version: list[str] = []
    low_cov: list[tuple[str, float]] = []
    degenerate: list[str] = []
    missing_volume: list[str] = []
    megacap_misses: list[str] = []
    covs: list[float] = []

    for p in files:
        df = pd.read_parquet(p)
        if "scoring_model_version" not in df.columns or not (
            df["scoring_model_version"] == SCORING_MODEL_VERSION
        ).all():
            bad_version.append(p.name)
        if "dollar_vol_63d" in df.columns:
            cov = float(df["dollar_vol_63d"].notna().mean())
        else:
            cov = 0.0
        covs.append(cov)
        if cov < 0.50:
            low_cov.append((p.name, cov))
        q = pd.to_numeric(df.get("quality_score"), errors="coerce")
        if q is None or not q.notna().any() or float(q.std()) <= 0:
            degenerate.append(p.name)
        if "volume" not in df.columns or pd.to_numeric(df["volume"], errors="coerce").notna().mean() < 0.5:
            missing_volume.append(p.name)
        if cov > 0.5 and "symbol" in df.columns:
            dv = pd.to_numeric(df["dollar_vol_63d"], errors="coerce")
            thresh = dv.quantile(0.90)
            held = df[df["symbol"].isin(MEGA_CAPS)]
            if not held.empty:
                dvh = pd.to_numeric(held["dollar_vol_63d"], errors="coerce")
                if (dvh.dropna() < thresh).any():
                    megacap_misses.append(p.name)

    print(f"peer-2 migration verification — {len(files)} snapshots in {snap_dir}")
    print(f"  version stamp {SCORING_MODEL_VERSION!r}:  {len(files) - len(bad_version)}/{len(files)} OK")
    print(f"  dollar_vol_63d coverage:   mean {sum(covs) / len(covs):.1%}, min {min(covs):.1%}")
    if low_cov:
        print(f"  files < 50% coverage ({len(low_cov)}):")
        for name, c in low_cov[:10]:
            print(f"    {name}: {c:.1%}")
    print(f"  quality_score degenerate:  {len(degenerate)} file(s) {degenerate[:5]}")
    print(f"  volume column missing:     {len(missing_volume)} file(s) {missing_volume[:5]}")
    print(f"  mega-cap top-decile miss:  {len(megacap_misses)} file(s) {megacap_misses[:5]}")

    ok = not bad_version and not degenerate and not missing_volume and sum(covs) / len(covs) >= 0.80
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
