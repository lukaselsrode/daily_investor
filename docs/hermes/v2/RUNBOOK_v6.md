# RUNBOOK v6 — the two-lane fast lane (supersedes nothing yet; v5 stays live until Gate 1)

The fast-lane daemon (`src/execution/odte_fast_lane.py`) removes the LLM from the execution
window: Hermes ARMs machine-readable intents, the daemon fires them deterministically
(trigger→place target <5s vs the 60–140s LLM path that missed the 2026-08-04 lease by 2.2s).
This runbook owns its operation, rollout, and kill switches. RUNBOOK_v5 remains the authority
for the Hermes lane itself.

## A. Components

| Piece | Where |
|---|---|
| Daemon | `make odte-fast-lane` (`--once` for one tick); launchd `com.dailyinvestor.odte-fastlane` |
| Mode authority | `data/odte/fast_lane_stage.json` — ONLY via `make odte-fast-lane-stage STAGE=shadow\|exits_live\|entries_live`; flags may only be MORE conservative |
| Intents | `data/odte/armed_intents.json` — ONLY via `make odte-arm` |
| Heartbeat | `data/odte/fast_lane_status.json` (atomic, every tick) |
| Shadow journal | `data/odte/shadow/decision_journal.jsonl` (namespaced `shadow_*` events; never pollutes live budget/green state) |
| Divergence report | `make odte-shadow-report` (the rollout evidence; `clean` flag) |
| Cross-lane arbiter | `data/odte/consumed_leases.json` — shared with the Hermes hook; consume-BEFORE-place on both lanes |
| Hook v6 | canonical `scripts/hermes/odte_order_guard_hook.py`; deploy per `hook_patch_v6.md` at Gate 2 |

## B. Daily ops

- **Morning (09:00–09:25 ET):** `daily-investor odte-fast-lane --once` sanity tick if desired;
  check the token horizon — the daemon preflights `expires_at ≥ 48h` at start and DOWNGRADES
  exits/live to shadow (loud journal event) below it. Weekly `hermes mcp reauth robinhood`
  before the Monday session remains the rule (token life ~7 days, manual only).
- **Session:** launchd starts the daemon 09:25 ET Mon–Fri; it self-fences to 09:25–16:05 ET
  and exits 0 at session end (KeepAlive restarts only crashes). Logs: `/tmp/odte-fastlane.log`.
- **EOD:** `make odte-shadow-report` → adjudicate every divergence (one line each) → the
  Hermes 16:10 recap folds it in. Supervisor restarts during RTH must be zero for gate credit.

## C. Kill switches (fastest first)

1. `make odte-fast-lane-pause` — pause file; ALL placements halt on the next tick (≤ ~3s);
   monitoring/journaling continues. `make odte-fast-lane-resume` to clear.
2. `make odte-arm DISARM=all` — WATCHING empties; nothing can fire.
3. `make odte-fast-lane-uninstall` — launchd unload + remove; process gone. Hermes retains
   close/cancel through its own lane at every stage.
4. `make odte-fast-lane-stage STAGE=shadow` — stage rollback (also re-opens Hermes entries at
   the hook, since only `entries_live` blocks them).
5. The v5 Hermes ladder is unchanged and independent: `hermes cron pause 344e4c3333a7`,
   policy `execution_stage`, tool removal.
6. **Untouchables:** the 60s lease hard cap and the single-use consumed-ledger rule hold for
   BOTH lanes. Never tunable.

## D. Incident semantics

`FILLED_WITHOUT_VALID_LEASE` / `BROKER_MISMATCH_BLOCKED` / an unexplained working order at
startup → `execution_safety_incident` journaled, `entries_locked` for the session (visible in
the heartbeat), the real fill is MANAGED defensively, and Hermes reports to Lukas. Arming
around a lock is forbidden; the lock clears only with a fresh session after investigation.

## E. Shadow-day checklist (each session while staged)

- [ ] Heartbeat present and advancing all session (state/mode/counts sane, `last_error` null)
- [ ] Tape continuity: no 5-min boundary without a snapshot; no quote-poll gap > 3× cadence
- [ ] Zero supervisor restarts; every MCP reconnect journaled and successful
- [ ] `make odte-shadow-report`: `clean: true` OR every divergence adjudicated in writing
- [ ] Zero `shadow_incident`s; every convert refusal's reason code human-agreed
- [ ] Token preflight passed (≥48h) at session start

## F. Rollout gates (measurable; no gate, no promotion)

**Gate 1 — shadow → exits_live (≥2 full sessions, ALL required):** checklist E clean both
sessions; on any live Hermes position the `shadow_exit_intent` stream contains no trigger a
human adjudicates as false, and the giveback-rule replay beats-or-matches the manual close;
p95 trigger-eval→`shadow_order_intent` < 3s.
→ Promote: `make odte-fast-lane-stage STAGE=exits_live`; **MERGE** `controller_prompt_v6.md`
into the live controller prompt — do NOT apply it verbatim; daemon now owns exit placement,
Hermes defers to `management.decision`.

> **Merge, never replace (2026-08-12).** `controller_prompt_v6.md` is a 4.3k document frozen on
> 2026-08-05. The live prompt is now ~13.3k and has since gained five fixes that v6 contains none
> of: the Telegram HEARTBEAT floor, the `odte_build_snapshot` FAST-PATH step, the `tool_describe`
> ban, warm-the-chain-once-per-session-per-side, and the no-authoring-Python rule. Overwriting the
> live prompt with v6 silently regresses all five — the heartbeat one being the exact defect that
> produced a two-hour Telegram blackout. v6's UNIQUE content is the fast-lane section (`odte-arm`,
> armed intents, deferring management to the daemon); graft that in and leave the rest alone.
> Verify after `hermes cron edit` that all five markers are still present in the live prompt.

**Gate 2 — exits_live → entries_live (≥2 sessions, ALL required):** ≥1 real exit placed AND
filled by the daemon (a deliberate 1-contract cheap position may manufacture the test) with
exact identity, journal↔broker reconciliation, reprice ≤ 5 replacements, p95 exit
trigger→place < 2s; entry side still shadowed with fire/no-fire agreement vs Hermes
CONFIRM_ENTRY and 0 unexplained shadow-only fires; hook v6 deployed and dry-verified per
`hook_patch_v6.md`; `make test` + `make hygiene` green; the cross-lane test (daemon-burned
lease refused by the hook) passes.
→ Promote: `make odte-fast-lane-stage STAGE=entries_live`. Week one operational cap:
**1 armed intent/day**, reviewed each EOD (runbook rule, not code).
**HARD PRE-REQ (2026-08-18): remove `--place --account` from the controller prompt's convert
step FIRST.** `--place` constructs its own MCP client and ignores `fast_lane_stage.json` — left
in place it re-creates a dual entry lane the moment the daemon owns opens (the pre-order hook
cannot stop it; `pre_place_check` + the shared ledger is its only guard).

## G. Failure modes and their answers

| Symptom | Answer |
|---|---|
| Heartbeat stale during RTH | daemon down/crashed — check `/tmp/odte-fastlane.err.log`; launchd restarts crashes; a clean early exit means the session fence or a kill switch |
| `McpAuthStale` at start | token <48h — `hermes mcp reauth robinhood`; daemon ran shadow meanwhile |
| Intent armed but never fires | `armed_intent_state.json` shows the counter + `last_result.reasons` per tick; a convert refusal parks the intent 60s (`last_refusal`) |
| Same intent seems dead after one fire | intents are single-shot (`fired_at`); re-arm deliberately |
| Parked MCP connection (Hermes lane) | unchanged v5 protocol + un-parker; the daemon holds its OWN session and reconnects once per call, so a Hermes park does not affect it |
| Both lanes near one position | entries: consumed ledger arbiters; exits: daemon writes `management.decision`, Hermes defers (prompt v6); a true double-close rejects at the broker (no position) |
| Poke rail misbehaving (pre-10:00 poke, poke storm, stale-notepad actions) | `cfg/config.yaml` `confirm_detector_enabled: false` + daemon restart; `data/odte/fast_lane_pause` silences pokes with no restart; `hermes cron pause 344e4c3333a7` — the claim layer refuses pokes on a paused job (verified) |

## H2. Incident adjudication (added 2026-08-06)

The execution-safety lockout has EXACTLY ONE sanctioned escape: a human-authorized
`execution_safety_incident_adjudicated` journal event (via `odte-journal --event-json`) that is
same-ET-day, NAMES the incident (`incident_event_id`, fallback `incident_seq`), and carries
`adjudicated_by` + `reason`. Preconditions, in order, non-negotiable: (1) the incident is
PROVEN false from artifacts, (2) the underlying defect is FIXED and committed with the
incident's exact payload replaying green, (3) Lukas explicitly authorizes. The controller is
told nothing about this event type and must never emit one — an agent-emitted adjudication is
itself an incident. Precedent: 2026-08-06 IWM 301C (guard key-collision false positive,
incident event 777c82248c98e543, fix bbbca46).
