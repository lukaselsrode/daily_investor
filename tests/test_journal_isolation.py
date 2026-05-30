"""
Regression tests for the position-journal / outcome-journal schema collision.

Bug: portfolio.outcome_tracker wrote its 12-column decision-outcome mirror into
`position_journal.csv`, which portfolio.position_journal owns with a *different*
9-column schema. The mixed-width rows made pandas fail to parse the whole file,
so the Portfolio journal panel in the UI showed "Could not load
position_journal.csv" and rendered nothing.

These tests lock in the fix:
  1. The two journals must resolve to distinct files.
  2. load_journal() must survive a file containing foreign / malformed rows
     (salvage good rows instead of blanking the whole panel).
"""
from __future__ import annotations

import pandas as pd


def test_journals_use_distinct_files():
    """The two journal writers must not share a path (schema collision)."""
    from portfolio import outcome_tracker, position_journal

    pos_path = position_journal._journal_path()
    out_path = outcome_tracker._journal_path()

    assert pos_path != out_path, (
        "position_journal and outcome_tracker must not share a file — their "
        f"schemas differ ({pos_path.name} is 9-col, {out_path.name} is 12-col)"
    )
    assert pos_path.name == "position_journal.csv"
    assert out_path.name == "outcome_journal.csv"


def test_load_journal_tolerates_malformed_rows(tmp_path, monkeypatch):
    """A foreign/wrong-width row must not blank the whole journal panel."""
    from portfolio import position_journal

    journal = tmp_path / "position_journal.csv"
    # Valid 9-col header + one good row, then a foreign 12-col row appended by
    # a different writer (the exact corruption pattern from the bug).
    journal.write_text(
        "timestamp,symbol,event_type,sleeve,status,price,composite_score,rank_pct,rationale\n"
        "2026-05-25 02:31:38,BP,HOLD_REVIEW,active,BUY,,0.87,0.97,Model would re-buy\n"
        "2026-05-25T11:43:35+00:00,2026-05-25,HOLD_TEST,holding,HOLD,HOLD,False,,0.8,,bullish,\n"
    )
    monkeypatch.setattr(position_journal, "_journal_path", lambda: journal)

    df = position_journal.load_journal()

    # Must not raise, must return the schema columns, must keep the good row.
    assert list(df.columns) == position_journal._JOURNAL_COLS
    assert (df["symbol"] == "BP").any()


def test_load_journal_missing_file_returns_empty(tmp_path, monkeypatch):
    from portfolio import position_journal

    monkeypatch.setattr(
        position_journal, "_journal_path", lambda: tmp_path / "nope.csv"
    )
    df = position_journal.load_journal()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == position_journal._JOURNAL_COLS
    assert df.empty
