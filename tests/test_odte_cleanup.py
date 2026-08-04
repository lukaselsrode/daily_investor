"""tests/test_odte_cleanup.py — deterministic data/odte sweep with a hardcoded keep-list.

2026-08-03: cleanup used to be LLM prose running ad-hoc `mv`; canonical loop state was archived
21 times (including the single-use lease ledger). These tests pin the contract: the keep-list is
code, canonical files/dirs are untouchable, dry-run is the default, and one dated archive dir
with a manifest receives everything swept.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_cleanup as cl


def _seed(tmp_path):
    for name in ("triggers.json", "watchdog_state.json", "consumed_leases.json",
                 "decision_journal.jsonl", "broker_health.json"):
        (tmp_path / name).write_text("{}")
    for name in ("market_snapshot_controller_20260728T1.json", "day_score_controller_x.json",
                 "build_market_snapshot.py", "market.json"):
        (tmp_path / name).write_text("stale")
    (tmp_path / ".last_marker").write_text("x")
    for d in ("days", "swarm", "reports"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "keepme.txt").write_text("x")
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "old.json").write_text("x")
    (tmp_path / "__pycache__").mkdir()


def test_dry_run_reports_but_moves_nothing(tmp_path):
    _seed(tmp_path)
    payload = cl.run_cleanup(str(tmp_path))
    assert payload["dry_run"] is True
    assert "market.json" in payload["swept"]
    assert "events/" in payload["swept"] and "__pycache__/" in payload["swept"]
    assert (tmp_path / "market.json").exists()          # nothing moved
    assert payload["archive_dir"] is None


def test_apply_sweeps_only_non_canonical_into_one_archive(tmp_path):
    _seed(tmp_path)
    payload = cl.run_cleanup(str(tmp_path), apply=True)
    assert payload["dry_run"] is False
    # every canonical file/dir survives in place
    for name in cl.KEEP_FILES:
        if (tmp_path / name).name in ("triggers.json", "watchdog_state.json",
                                      "consumed_leases.json", "decision_journal.jsonl",
                                      "broker_health.json"):
            assert (tmp_path / name).exists(), name
    for d in ("days", "swarm", "reports"):
        assert (tmp_path / d / "keepme.txt").exists()
    assert (tmp_path / ".last_marker").exists()          # dotfiles untouched
    # swept set is gone from top level and present in ONE archive dir with a manifest
    assert not (tmp_path / "market.json").exists()
    assert not (tmp_path / "build_market_snapshot.py").exists()
    assert not (tmp_path / "events").exists()
    archive = tmp_path / "archive"
    dirs = list(archive.iterdir())
    assert len(dirs) == 1 and dirs[0].name.startswith("odte_cleanup_")
    manifest = json.loads((dirs[0] / "manifest.json").read_text())
    assert manifest["swept_count"] == payload["swept_count"]
    assert (dirs[0] / "market.json").exists()
    assert (dirs[0] / "events" / "old.json").exists()


def test_keep_list_covers_the_loop_state_contract():
    # The exact files whose loss broke the loop historically must be hardcoded-protected.
    for name in ("triggers.json", "watchdog_state.json", "active_candidate.json",
                 "candidate_decision.json", "execution_lease.json", "consumed_leases.json",
                 "broker_health.json", "decision_journal.jsonl", "active_state.json"):
        assert name in cl.KEEP_FILES, name
    for d in ("days", "reports", "scrape", "swarm", "precompute", "archive"):
        assert d in cl.KEEP_DIRS, d


def test_prune_scrape_keeps_newest_and_stable_pointer(tmp_path):
    _seed(tmp_path)
    scrape = tmp_path / "scrape"
    scrape.mkdir()
    (scrape / "reddit_text.txt").write_text("latest")
    for i in range(10):
        (scrape / f"reddit_text_2026_07_31_14_{i:02d}.txt").write_text("x")
    payload = cl.run_cleanup(str(tmp_path), apply=True, prune_scrape=True, scrape_keep=3)
    assert payload["pruned_scrape_count"] == 7
    assert (scrape / "reddit_text.txt").exists()         # stable pointer untouched
    remaining = sorted(scrape.glob("reddit_text_*.txt"))
    assert len(remaining) == 3
    assert remaining[-1].name.endswith("14_09.txt")      # newest kept


def test_module_is_pure_offline():
    import inspect
    src = inspect.getsource(cl)
    for forbidden in ("robin_stocks", "requests", "place_order", "urllib", "httpx", "socket"):
        assert forbidden not in src
