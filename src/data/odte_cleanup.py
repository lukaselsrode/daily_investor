"""0DTE artifact cleanup — the ONLY sanctioned sweep of `data/odte/`, with a HARDCODED keep-list.

Born 2026-08-03: cleanup used to be prose aimed at the controller LLM, which ran ad-hoc `mv`
commands and archived CANONICAL loop state 21 times since 2026-07-02 (including the single-use
lease ledger and, post-trade on 2026-08-03, the active candidate/decision/lease trio — each time
silently degrading the loop to FLAT_NO_TRADE or weakening replay protection). No cleanup code
existed at all. This module makes the sweep deterministic: the keep-list is code, not policy
prose, so a model cannot forget it.

Behavior: scans ONLY the top level of the state dir. Canonical loop files and canonical
subdirectories are untouchable. Everything else (per-tick snapshot exhaust, dead naming
generations, stray scripts/caches) is moved into ONE dated archive directory with a manifest.
Dry-run by default — `apply=True` moves files. `prune_scrape=True` additionally trims the
timestamped scrape snapshots to a retention count (newest first), mirroring
`social_sentiment._prune_scrape_snapshots` naming. Never touches subdirectory CONTENTS, never
deletes the journal, never places orders, no network.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from core.paths import ODTE_DATA_DIR, atomic_write_text

SCHEMA_VERSION = 1
DEFAULT_STATE_DIR = ODTE_DATA_DIR

# Canonical LOOP STATE — the files the live loop reads/writes. Sweeping any of these breaks the
# state machine (missing triggers.json => forced FLAT_NO_TRADE; missing consumed_leases.json =>
# lease replay protection disarmed). HARDCODED on purpose: policy files may drift, this must not.
KEEP_FILES: frozenset[str] = frozenset({
    "triggers.json",
    "watchdog_state.json",
    "active_candidate.json",
    "candidate_decision.json",
    "execution_lease.json",
    "consumed_leases.json",
    "broker_health.json",
    "broker_health.example.json",
    "decision_journal.jsonl",
    "active_state.json",
    "position_state.json",
    "position_decision.json",
    "active_trade.json",
    "order_guard_decision.json",
    "session_constants.json",
    # Fast-lane canon (2026-08-05): the two-lane state files must survive every sweep.
    "armed_intents.json",
    "armed_intent_state.json",
    "fast_lane_stage.json",
    "fast_lane_status.json",
    "fast_lane_pause",
    # 2026-08-18: confirm-detector poke-cooldown state — sweeping it re-pokes the controller
    # after every cleanup.
    "confirm_detector_state.json",
})

# Canonical subdirectories — never entered, never moved.
# `research` joined 2026-08-06: the 2026-08-04 sweep archived the August forecast artifacts the
# 2026-09-01 audit job loads by path. Long-lived analysis output is not per-tick exhaust, and
# `data/odte/` is gitignored, so the swept copy was the ONLY copy.
KEEP_DIRS: frozenset[str] = frozenset({
    "days", "reports", "scrape", "swarm", "precompute", "archive", "shadow", "research",
    # 2026-08-18 halt: freshness-spec validation data (would-be entries + quote paths) — the
    # evidence the un-halt decision consumes; a sweep here erases the resume case.
    "shadow_validation",
})

# Scrape snapshot kinds (data/odte/scrape/<kind>_text_YYYY_MM_DD_HH_MM.txt + stable pointer).
_SCRAPE_KINDS = ("reddit", "x")
DEFAULT_SCRAPE_KEEP = 100


def run_cleanup(state_dir: str | None = None, *, apply: bool = False,
                prune_scrape: bool = False, scrape_keep: int = DEFAULT_SCRAPE_KEEP,
                now: datetime | None = None) -> dict:
    """Classify and (optionally) sweep non-canonical top-level artifacts. Returns the manifest.

    Dry-run (`apply=False`, the default) only reports what WOULD move. `apply=True` moves the
    sweep set into `archive/odte_cleanup_<UTC-stamp>/` and writes `manifest.json` there.
    """
    now = now or datetime.now(timezone.utc)
    base = Path(str(state_dir or DEFAULT_STATE_DIR)).expanduser()
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    archive_dir = base / "archive" / f"odte_cleanup_{stamp}"

    kept: list[str] = []
    swept: list[str] = []
    for entry in sorted(base.iterdir()) if base.exists() else []:
        name = entry.name
        if entry.is_dir():
            if name in KEEP_DIRS:
                kept.append(name + "/")
            elif name == "__pycache__":
                swept.append(name + "/")
            else:
                # Unknown top-level dir: sweep it whole (events/, experiments/ class).
                swept.append(name + "/")
            continue
        if name in KEEP_FILES:
            kept.append(name)
        elif name.startswith("."):
            kept.append(name)          # dotfiles (swarm .last_processed etc.) are not ours to move
        else:
            swept.append(name)

    pruned_scrape: list[str] = []
    if prune_scrape:
        scrape_dir = base / "scrape"
        if scrape_dir.exists():
            for kind in _SCRAPE_KINDS:
                files = sorted(scrape_dir.glob(f"{kind}_text_*.txt"))
                for stale in files[:-max(1, int(scrape_keep))]:
                    pruned_scrape.append(stale.name)
                    if apply:
                        stale.unlink(missing_ok=True)

    if apply and swept:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for name in swept:
            src = base / name.rstrip("/")
            if src.exists():
                shutil.move(str(src), str(archive_dir / src.name))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "state_dir": str(base),
        "dry_run": not apply,
        "kept": kept,
        "swept": swept,
        "swept_count": len(swept),
        "pruned_scrape_count": len(pruned_scrape),
        "archive_dir": str(archive_dir) if (apply and swept) else None,
        "keep_files": sorted(KEEP_FILES),
        "keep_dirs": sorted(KEEP_DIRS),
        "places_orders": False,
        "basis": ("deterministic odte cleanup — hardcoded keep-list; the ONLY sanctioned sweep "
                  "(ad-hoc mv archived canonical loop state 21x before this existed)"),
    }
    if apply and swept:
        atomic_write_text(archive_dir / "manifest.json",
                          json.dumps(manifest, indent=2, default=str))
    return manifest


def render_markdown(payload: dict) -> str:
    p = payload or {}
    mode = "DRY RUN" if p.get("dry_run") else "APPLIED"
    lines = [f"# 0DTE cleanup: **{mode}** — {p.get('swept_count', 0)} to sweep",
             "",
             f"State dir: {p.get('state_dir')}  ·  archive: {p.get('archive_dir') or '—'}  ·  "
             f"scrape pruned: {p.get('pruned_scrape_count', 0)}", ""]
    if p.get("swept"):
        lines += ["## Swept (non-canonical top-level)", *[f"- {n}" for n in p["swept"][:40]]]
        if len(p["swept"]) > 40:
            lines.append(f"- … and {len(p['swept']) - 40} more")
    lines += ["", "## Protected (hardcoded keep-list)",
              ", ".join(sorted(p.get("keep_files") or [])),
              "dirs: " + ", ".join(sorted(p.get("keep_dirs") or []))]
    return "\n".join(lines)
