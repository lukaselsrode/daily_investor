"""Confirm detector — the daemon-side latency killer for the entry lane (2026-08-18).

The lateness autopsy (data/odte/reports/entry_signal_autopsy_2026-08-18.md) showed the entry
edge dies with move age, yet confirms were only EVALUATED on the controller's */5 LLM ticks —
measured spacing median 297s, p90 417s, max 1,447s (2026-08-17..18). This module runs the SAME
pure evaluation the controller runs (`evaluate_candidate_watch` over the daemon's tape) at the
daemon's 3-15s tick cadence and, the moment the tape confirms, POKES the controller awake:

    hermes cron notepad <job> set confirm_poke <context>   &&   hermes cron run <job>

`hermes cron run` fires the job immediately with at-most-once fire-claim semantics (a poke
during a running tick no-ops; the */5 grid self-heals; paused jobs REFUSE the poke, so the
close-day kill switch holds). The poke is a WAKE-UP, never authority — the poked tick runs
every gate exactly as a scheduled one would.

Hard rules this module lives by:
  * READ-ONLY on shared state. It never writes active_candidate.json / candidate_decision.json
    (two sanctioned writers exist; a seconds-cadence third would churn fingerprints and pin
    loop-status's CANDIDATE routing). Its only writes: its own state file + journal appends.
  * Journal TRANSITIONS ONLY (first sighting / confirm-ready / poked). The journal has no
    decision-change dedupe — a per-tick appender would write ~7,800 events a session. Bonus:
    the first-sighting event legitimately starts the freshness clock earlier (stricter).
  * Zero broker/MCP calls — pure evaluation over the tape the daemon already built.
  * Fail-open: step() swallows its own errors. Detection can never break a daemon tick.

Synthetic slots: a watchdog market-scorecard placeholder can never confirm (candidate-watch
forces KEEP_WATCHING before breadth is computed), so when the slot holds a synthetic the
detector runs a SECOND evaluation with an EMPTY candidate — the tape-only ETF lane can reach
CONFIRM_ENTRY, which means "tape confirm-ready, executable candidate missing". That pokes with
kind=synthetic_ready and the controller's step 1c mints the real candidate via the sanctioned
writer (`odte-candidate-watch --candidate-json '{}' --write`).
"""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from core.paths import atomic_write_text
from data.odte_candidate_watch import _is_synthetic_scorecard, evaluate_candidate_watch
from data.odte_day_score import score_day
from data.odte_journal import append_decision_journal, event_from_candidate_evaluation

logger = logging.getLogger("odte_confirm_detector")

HERMES_BIN = "/Users/lukaselsrode/.local/bin/hermes"
CONTROLLER_JOB_ID = "344e4c3333a7"
NOTEPAD_KEY = "confirm_poke"
STATE_FILENAME = "confirm_detector_state.json"
PAUSE_FILENAME = "fast_lane_pause"
CANDIDATE_FILENAME = "active_candidate.json"
BROKER_HEALTH_FILENAME = "broker_health.json"


def _detached_chain(cmds: list[list[str]]) -> None:
    """Fire an ordered command chain fully detached — the daemon tick never blocks on it."""
    script = " && ".join(" ".join(shlex.quote(a) for a in cmd) for cmd in cmds)
    subprocess.Popen(["/bin/sh", "-c", script],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


class ConfirmDetector:
    """Per-session detection state + the poke rail. One instance per daemon."""

    def __init__(self, state_dir: str | Path, *, journal_path: str,
                 poke_cooldown_seconds: float, live: bool = True,
                 hermes_bin: str = HERMES_BIN, job_id: str = CONTROLLER_JOB_ID,
                 run_cmd: Callable[[list[list[str]]], None] | None = None) -> None:
        self.base_dir = Path(state_dir)
        self.journal_path = journal_path
        # SHADOW PURITY: a shadow-mode daemon rehearses — its detector journals would-pokes to
        # the SHADOW journal (caller passes that path) and never fires the real poke chain.
        self.live = bool(live)
        self.cooldown = float(poke_cooldown_seconds)
        self.hermes_bin = hermes_bin
        self.job_id = job_id
        self.run_cmd = run_cmd or _detached_chain
        self.state_path = self.base_dir / STATE_FILENAME
        self.errors = 0
        self._file_cache: dict[str, tuple[float, dict]] = {}
        try:
            self.state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            self.state = {}

    # ── helpers ─────────────────────────────────────────────────────────────────────────────
    def _persist(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.state_path, json.dumps(self.state, indent=2))

    def _read_json_cached(self, name: str) -> dict:
        path = self.base_dir / name
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return {}
        cached = self._file_cache.get(name)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            data = json.loads(path.read_text())
            data = data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            data = {}
        self._file_cache[name] = (mtime, data)
        return data

    def _controller_pokeable(self) -> bool:
        """Best-effort read of the job's enabled/paused state — belt-and-braces on top of the
        claim layer's own refusal of paused jobs. On any read failure, proceed (the claim layer
        is the authority)."""
        try:
            raw = json.loads((Path.home() / ".hermes/cron/jobs.json").read_text())
            jobs = raw if isinstance(raw, list) else raw.get("jobs", raw)
            if isinstance(jobs, dict):
                jobs = list(jobs.values())
            job = next((j for j in jobs if str(j.get("id", "")).startswith(self.job_id)), None)
            if job is None:
                return True
            if job.get("enabled") is False or job.get("paused_at") or job.get("paused") is True:
                return False
        except Exception:
            return True
        return True

    def summary(self) -> dict:
        return {"pokes": self.state.get("pokes", 0), "errors": self.errors,
                "last_poke_ts": self.state.get("last_poke_ts"),
                "moves": {k: v.get("phase") for k, v in (self.state.get("moves") or {}).items()}}

    # ── per-tick entry point (never raises) ─────────────────────────────────────────────────
    def step(self, tape: dict | None, now: datetime | None = None,
             events: list[dict] | None = None) -> None:
        """`events` is the day's canonical journal (the daemon already reads it each tick for
        the shadow harness) — it powers the SAME freshness clock the gate uses."""
        now = now or datetime.now(timezone.utc)
        try:
            self._step(tape or {}, now, events)
        except Exception as exc:                              # advisory lane: fail open, count it
            self.errors += 1
            logger.exception("confirm-detector step error: %s", exc)

    def _step(self, tape: dict, now: datetime, events: list[dict] | None = None) -> None:
        if not tape or not (tape.get("generated_at") or tape.get("as_of")):
            return
        # New ET session ⇒ fresh state (poke counters, move phases).
        from zoneinfo import ZoneInfo
        session = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        if self.state.get("session") != session:
            self.state = {"session": session, "moves": {}, "pokes": 0, "last_poke_ts": None}
            self._persist()

        candidate = self._read_json_cached(CANDIDATE_FILENAME)
        broker = self._read_json_cached(BROKER_HEALTH_FILENAME)
        day = score_day(market=tape, now=now)

        primary = evaluate_candidate_watch(candidate or None, market=tape, day_score=day,
                                           broker_health=broker or None, now=now)
        decision = str(primary.get("decision") or "").lower()
        kind, payload = None, primary
        if decision == "confirm_entry":
            kind = "confirm"
        elif candidate and _is_synthetic_scorecard(candidate):
            secondary = evaluate_candidate_watch({}, market=tape, day_score=day,
                                                 broker_health=broker or None, now=now)
            if str(secondary.get("decision") or "").lower() == "confirm_entry":
                kind, payload = "synthetic_ready", secondary

        cand_block = payload.get("candidate") or {}
        symbol = str(cand_block.get("ticker") or cand_block.get("symbol")
                     or (candidate or {}).get("ticker") or "SPY").upper()
        direction = str(cand_block.get("direction")
                        or (candidate or {}).get("direction") or "").lower()
        move_key = f"{symbol}:{direction or 'none'}"
        moves = self.state.setdefault("moves", {})
        move = moves.get(move_key)

        # TRANSITION: first sighting of this (symbol, direction) move — one journal event that
        # legitimately starts the freshness clock at daemon-detection time.
        if direction and move is None:
            move = {"phase": "watching", "first_seen": now.isoformat(timespec="seconds")}
            moves[move_key] = move
            self._journal_transition(payload, "first_sighting", now)
            self._persist()

        if kind is None or move is None:
            return

        # TRANSITION: confirm-ready. A parked ("stale") move stays parked — flipping it back
        # would re-journal ready transitions every tick and re-open the poke path.
        if move.get("phase") not in ("ready", "stale"):
            move["phase"] = "ready"
            move["ready_at"] = now.isoformat(timespec="seconds")
            self._journal_transition(payload, f"ready:{kind}", now)
            self._persist()

        # STALE MOVES ARE NEVER POKED (2026-08-19, live poke-storm): a confirm-ready move older
        # than the freshness gate's limit gets vetoed at the gate EVERY time — poking it burns a
        # controller run per cooldown for a structurally impossible entry (11 pokes in ~35 min).
        # The clock is the GATE'S OWN (`first_signal_age_minutes` over the journal) so the two
        # can never disagree — candidate TTL re-seeds do NOT reset it (the trap: a re-seeded
        # candidate minted a fresh detector move and the storm continued under a new key). The
        # detector's own first_seen is only the fallback when the journal has no history.
        from data.odte_config import MAX_SIGNAL_AGE_MINUTES
        if MAX_SIGNAL_AGE_MINUTES > 0:
            age_min = None
            if events:
                from data.odte_journal import first_signal_age_minutes
                age_min = first_signal_age_minutes(events, symbol, direction, now=now)
            if age_min is None and move.get("first_seen"):
                try:
                    age_min = (now - datetime.fromisoformat(
                        move["first_seen"])).total_seconds() / 60.0
                except ValueError:
                    age_min = None
            if age_min is not None and age_min > MAX_SIGNAL_AGE_MINUTES:
                if move.get("phase") != "stale":
                    move["phase"] = "stale"
                    self._journal_transition(payload, "ready_but_stale", now)
                    self._persist()
                return

        # Poke, throttled per move.
        last = move.get("last_poke")
        if last:
            try:
                since = (now - datetime.fromisoformat(last)).total_seconds()
                if since < self.cooldown:
                    return
            except ValueError:
                pass
        if (self.base_dir / PAUSE_FILENAME).exists():
            return
        if not self._controller_pokeable():
            return
        self._poke(symbol, direction, kind, payload, now)
        move["last_poke"] = now.isoformat(timespec="seconds")
        self.state["pokes"] = int(self.state.get("pokes", 0)) + 1
        self.state["last_poke_ts"] = now.isoformat(timespec="seconds")
        self._persist()

    # ── side effects ────────────────────────────────────────────────────────────────────────
    def _journal_transition(self, payload: dict, transition: str, now: datetime) -> None:
        event = event_from_candidate_evaluation(
            payload, extra={"detector": "fast_lane_confirm_detector",
                            "detector_transition": transition})
        append_decision_journal(event, source="fast_lane", event_type="candidate_evaluation",
                                journal_path=self.journal_path, now=now)

    def _poke(self, symbol: str, direction: str, kind: str, payload: dict,
              now: datetime) -> None:
        checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
        note = {
            "ts": now.isoformat(timespec="seconds"),
            "kind": kind, "symbol": symbol, "direction": direction,
            "decision": payload.get("decision"),
            "tier": checks.get("tier"),
            "breadth_score": checks.get("breadth_score"),
            "breadth_required": checks.get("breadth_required"),
            "note": ("fast-lane confirm detector: WAKE-UP, not authority — verify with your "
                     "own loop-status/candidate-watch, then delete this key"),
        }
        if self.live:
            self.run_cmd([
                [self.hermes_bin, "cron", "notepad", self.job_id, "set", NOTEPAD_KEY,
                 json.dumps(note, separators=(",", ":"))],
                [self.hermes_bin, "cron", "run", self.job_id],
            ])
        append_decision_journal(
            {"symbol": symbol, "direction": direction, "kind": kind,
             "detector": "fast_lane_confirm_detector",
             "would_poke": (not self.live) or None,
             "breadth_score": checks.get("breadth_score")},
            source="fast_lane", event_type="confirm_detector_poke",
            journal_path=self.journal_path, now=now)
