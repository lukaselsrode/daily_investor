"""tests/test_odte_storage.py — 0DTE storage rework: data → data/odte/, secrets stay in ~/0dte/.

Covers the path centralization, the module-default repoint, the watchdog policy(secret)/state(data)
split, the timestamped scrape-text snapshots + retention, and the UI store-loader helpers. All
assertions key off the live `core.paths` constants rather than hardcoded paths.
"""
import json
import os
from datetime import datetime, timezone

import core.paths as paths

_FIXED_NOW = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)


# --- path centralization ---------------------------------------------------------------------

def test_odte_path_constants_split_data_from_secrets():
    assert paths.ODTE_SECRETS_DIR == os.path.expanduser("~/0dte")
    assert paths.ODTE_DATA_DIR == os.path.join(paths.DATA_DIRECTORY, "odte")
    assert paths.ODTE_REPORT_DIR == os.path.join(paths.ODTE_DATA_DIR, "reports")
    assert paths.ODTE_SCRAPE_DIR == os.path.join(paths.ODTE_DATA_DIR, "scrape")
    # data lives in the repo data tree; secrets explicitly do NOT
    assert paths.ODTE_DATA_DIR.startswith(paths.DATA_DIRECTORY)
    assert not paths.ODTE_SECRETS_DIR.startswith(paths.DATA_DIRECTORY)


def test_atomic_write_text_is_atomic_and_leaves_no_tmp(tmp_path):
    """atomic_write_text writes via a sibling tmp + os.replace: the final file has the new content,
    no .tmp turd is left behind, and a re-write replaces (not appends) — the property the watchdog
    trigger handoff relies on so a partial write never corrupts the next poll's read."""
    target = tmp_path / "triggers.json"
    paths.atomic_write_text(target, json.dumps({"alert": True}))
    assert json.loads(target.read_text()) == {"alert": True}
    # No leftover tmp sidecar.
    assert list(tmp_path.glob(".*.tmp")) == []
    # Re-write replaces in place.
    paths.atomic_write_text(target, json.dumps({"alert": False}))
    assert json.loads(target.read_text()) == {"alert": False}


def test_module_defaults_point_into_data_tree():
    from data import (
        odte_fmp_context,
        odte_gamma_map,
        odte_journal,
        odte_position,
        odte_watchdog,
    )
    assert odte_journal.DEFAULT_DIR == paths.ODTE_DATA_DIR
    assert odte_journal.DEFAULT_JOURNAL_PATH.startswith(paths.ODTE_DATA_DIR)
    assert odte_journal.DEFAULT_REPORT_DIR == paths.ODTE_REPORT_DIR
    assert odte_position.DEFAULT_STATE_DIR == paths.ODTE_DATA_DIR
    assert odte_watchdog.DEFAULT_STATE_DIR == paths.ODTE_DATA_DIR
    assert odte_gamma_map.DEFAULT_REPORT_DIR == paths.ODTE_REPORT_DIR
    assert odte_fmp_context.DEFAULT_REPORT_DIR == paths.ODTE_REPORT_DIR


# --- watchdog: policy is a secret, state is data ---------------------------------------------

def test_watchdog_policy_default_is_in_secrets_not_data():
    from data import odte_watchdog as w
    assert w.DEFAULT_POLICY_PATH == os.path.join(paths.ODTE_SECRETS_DIR, w.POLICY_FILENAME)
    assert not w.DEFAULT_POLICY_PATH.startswith(paths.ODTE_DATA_DIR)


def test_watchdog_state_goes_to_data_dir_policy_read_separately(tmp_path, monkeypatch):
    import data.social_sentiment as ss
    from data import odte_watchdog as w

    # Keep it offline + deterministic: stub the local report builder.
    monkeypatch.setattr(
        ss, "build_odte_social_report",
        lambda allow_fetch=True: {"scorecard": {"verdict": "OBSERVE"},
                                  "candidate": None, "top_chatter": []},
    )
    state_dir = tmp_path / "data_odte"
    policy = tmp_path / "secrets" / "controller_policy.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(json.dumps({"account": "redacted"}))

    payload = w.run_watchdog(state_dir=str(state_dir), policy_path=str(policy),
                             allow_fetch=False, now=_FIXED_NOW)

    # State/triggers are written to the (data) state dir...
    assert (state_dir / w.STATE_FILENAME).exists()
    assert (state_dir / w.TRIGGERS_FILENAME).exists()
    # ...the policy living in a *different* directory is still read (decoupled from state_dir)...
    assert payload["policy_ok"] is True
    # ...and the secret is never copied into the data dir.
    assert not (state_dir / w.POLICY_FILENAME).exists()


# --- scrape-text snapshots over time ---------------------------------------------------------

def test_scrape_dump_writes_timestamped_snapshot_and_latest_pointer(tmp_path, monkeypatch):
    import data.social_sentiment as ss
    monkeypatch.setattr(ss, "ODTE_SCRAPE_DIR", str(tmp_path))

    ss._dump_analyzed_texts([
        {"source": "reddit", "text": "SPY 0dte calls printing"},
        {"source": "x", "text": "QQQ 0dte puts here"},
    ])

    # stable latest pointer for back-compat
    assert (tmp_path / "reddit_text.txt").exists()
    assert (tmp_path / "x_text.txt").exists()
    # plus a timestamped history snapshot per kind
    reddit_snaps = list(tmp_path.glob("reddit_text_*.txt"))
    x_snaps = list(tmp_path.glob("x_text_*.txt"))
    assert len(reddit_snaps) == 1 and len(x_snaps) == 1
    # same body in the snapshot and the latest pointer
    assert reddit_snaps[0].read_text() == (tmp_path / "reddit_text.txt").read_text()
    assert "SPY 0dte calls printing" in reddit_snaps[0].read_text()


def test_scrape_dump_skips_a_source_with_no_docs(tmp_path, monkeypatch):
    import data.social_sentiment as ss
    monkeypatch.setattr(ss, "ODTE_SCRAPE_DIR", str(tmp_path))
    ss._dump_analyzed_texts([{"source": "reddit", "text": "SPY 0dte calls"}])  # no x docs
    assert list(tmp_path.glob("reddit_text_*.txt"))
    assert not list(tmp_path.glob("x_text_*.txt"))   # failed/empty source does NOT write


def test_scrape_prune_keeps_most_recent(tmp_path):
    import data.social_sentiment as ss
    for day in range(20, 25):  # 5 snapshots
        (tmp_path / f"reddit_text_2026_06_{day}_09_00.txt").write_text("doc")
    ss._prune_scrape_snapshots(str(tmp_path), "reddit", keep=2)
    remaining = sorted(p.name for p in tmp_path.glob("reddit_text_*.txt"))
    assert remaining == ["reddit_text_2026_06_23_09_00.txt", "reddit_text_2026_06_24_09_00.txt"]


# --- UI store loaders ------------------------------------------------------------------------

def test_ui_load_odte_json_and_jsonl(tmp_path, monkeypatch):
    from ui import utils as u
    monkeypatch.setattr(u, "ODTE_DATA_DIR", tmp_path)

    (tmp_path / "active_trade.json").write_text(json.dumps({"underlying": "SPY"}))
    (tmp_path / "decision_journal.jsonl").write_text(
        '{"event_type":"note","seq":0}\n'
        "this is not json\n"
        '{"event_type":"entry_decision","seq":1}\n'
    )

    assert u.load_odte_json("active_trade.json")["underlying"] == "SPY"
    assert u.load_odte_json("missing.json") is None
    events = u.load_odte_jsonl("decision_journal.jsonl")
    assert [e["seq"] for e in events] == [0, 1]   # malformed middle line skipped, never raises


def test_ui_scrape_snapshot_helpers_exclude_latest_pointer(tmp_path, monkeypatch):
    from ui import utils as u
    monkeypatch.setattr(u, "ODTE_SCRAPE_DIR", tmp_path)
    for name in ("reddit_text_2026_06_24_09_00.txt", "reddit_text_2026_06_25_09_00.txt",
                 "reddit_text.txt"):
        (tmp_path / name).write_text("doc")

    snaps = u.list_scrape_snapshots("reddit")
    assert [p.name for p in snaps] == [
        "reddit_text_2026_06_24_09_00.txt", "reddit_text_2026_06_25_09_00.txt",
    ]  # the stable 'reddit_text.txt' pointer is excluded; dated snapshots only
    assert u.latest_scrape_snapshot("reddit").name == "reddit_text_2026_06_25_09_00.txt"
    assert u.latest_scrape_snapshot("x") is None
