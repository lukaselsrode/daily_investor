# 0DTE metrics, observability & post-day feedback

How the 0DTE/options decision-support layer records what it does, folds in the live controller's
loose artifacts, and reconstructs a full trading day for self-evaluation. **Everything here is
analysis/PAPER only — no module places orders, calls a broker, or runs an LLM.** NVDA is
employer-restricted and hard-blocked at every layer.

---

## 1. Architecture (the loop)

```
THESIS      social_sentiment  → scorecard / candidate / TOP-CHATTER cards (scan tier)
   ↓
DAY/VEHICLE/GAMMA  day_score / vehicle_score / gamma_map / fmp_context → reports/*.json + *.md
   ↓
TRIGGER     odte_watchdog.run_watchdog   → watchdog_state.json + triggers.json (atomic, enriched)
   ↓
ENTRY GATE  odte_entry_gate.build_entry_gate_decision → journalable entry_decision record
            (PURE/OFFLINE; the ONLY tier where execution_allowed may be True — gates must all pass)
   ↓
POSITION    odte_position.run_position_watchdog → position_state.json + position_decision.json (atomic)
   ↓
FEEDBACK    odte_journal.append_decision_journal → decision_journal.jsonl  (standardized, idempotent)
            odte_journal.ingest_loose_artifacts  → folds controller/MCP loose JSON into the journal
            odte_journal.build_day_packet        → days/YYYY-MM-DD/<stream>.jsonl (additive, derived)
            odte_journal.build_report            → reports/odte_journal_report.{md,csv} (+ self-eval)
```

Two writers feed the journal: the repo's own bridges (`event_from_position_decision`,
`event_from_vehicle_score`, `event_from_entry_gate`) and the **Hermes/MCP controller**, which drops timestamped loose JSON
into `data/odte/`. The ingester exists because `build_report` reads only the JSONL — loose artifacts
were previously invisible post-day.

---

## 2. Storage paths

Constants in `src/core/paths.py` (`ODTE_DATA_DIR = data/odte`):

```
data/odte/
  decision_journal.jsonl                 # authoritative append-only journal (standardized envelope)
  watchdog_state.json, triggers.json     # live trigger lane (atomic writes)
  position_state.json, position_decision.json  # live position lane (atomic writes)
  active_trade.json                      # current plan (UI/CLI editable)
  <controller loose artifacts>           # market_snapshot_*, candidate_*, controller_*, event_*, … (MCP)
  reports/   odte_journal_report.md/.csv, odte_day_score.json, odte_vehicle_score_*.json,
             odte_gamma_map_*.{json,md}, odte_fmp_context_*.{json,md}
  scrape/    reddit_text*.txt, x_text*.txt  (analyzed-text audit history, pruned to 500)
  days/YYYY-MM-DD/  market_snapshots.jsonl, candidates.jsonl, vehicle_scores.jsonl,
             trades.jsonl, controller_events.jsonl, postmortem.md   # additive day packet
~/0dte/      controller_policy.json + secrets  # NOT in the repo tree (Hermes/MCP owns)
```

Atomic writes (`core.paths.atomic_write_text`, tmp + `os.replace`) protect the live state/trigger
files so a crash mid-write can't hand the next poll a truncated JSON.

---

## 3. Standardized decision-journal envelope

`append_decision_journal(event, *, source, event_type, trade_date=None, dedupe=True)` →
`{status: appended|duplicate|error, event_id, event}`. **Never raises** (a journaling failure must
not crash the loop). De-dupe + seq + append happen under one POSIX advisory lock (`fcntl`); `seq` is
advisory, **`event_id` is authoritative**.

Envelope fields (`schema: "odte_decision_v1"`):

| Field | Meaning |
|---|---|
| `ts`, `trade_date`, `source`, `event_type` | when / which day / who wrote it / kind |
| `symbol` | ticker, coalesced from top-level or nested `contract`/`candidate`/`vehicle_score`/`thesis` |
| `mode` | scan · candidate · watchdog · execution · management · postmortem · note |
| `decision` | **scalar** verb: allow·deny·veto·skip·enter·exit·hold·observe·wait·note (synonyms normalized; original dict kept under `decision_detail`) |
| `reason_codes`, `thesis`, `confidence`, `confirmation_needed` | rationale |
| `scan_only`, `execution_allowed` | **tier flags** — see §6 |
| `raw_artifact_path`, `raw_artifact_sha` | provenance for ingested loose files (idempotency anchor) |
| `event_id` | stable 16-hex hash; artifact = `path::sha`, live = semantic identity fields |
| `restricted`, `restricted_reason` | NVDA/employer tag (forces `execution_allowed=False`) |

**Conservative invariants:** `scan_only=True ⇒ execution_allowed=False`; `restricted ⇒
execution_allowed=False`. Default `execution_allowed=False` — nothing is execution-authorized unless
a caller explicitly opts in (and isn't scan-only/restricted).

---

## 4. Loose-artifact ingestion

`ingest_loose_artifacts(data_dir, journal_path, trade_date, dry_run)` scans `data/odte/` for the
known patterns and folds each into the journal, **idempotently** and **read-only** over the sources
(never deletes/mutates them, never executes anything):

`market_snapshot_*` · `candidate_*` · `*vehicle_score*` · `*gamma_map*` · `controller_*` · `event_*`

- **Idempotency:** `event_id = sha1(resolved_path :: content_sha)`. Re-running skips unchanged files
  (`duplicates_skipped`); a **changed** file re-ingests as a new record (content hash differs).
- **Trade date:** `artifact_trade_date()` resolves from payload `ts`/`date`/`trade_date`, else a
  date in the **filename** (`20260626` / `2026_06_26` / `2026-06-26`), else file mtime — so older
  filename-dated artifacts with no `ts` aren't all bucketed to today.
- **Audit-only authority:** ingested `execution_allowed` is **forced False** (original preserved as
  `raw_execution_allowed`). Ingestion is a record of what already happened, not fresh authority.
- **Summary:** `{dry_run, files_scanned, events_appended, events_would_append, duplicates_skipped,
  errors, by_event_type, error_files}`. Malformed files are counted, never fatal.

CLI: `daily-investor odte-ingest-artifacts [--date YYYY-MM-DD] [--dry-run] [--day-packet] [--json]`
· Make: `make odte-ingest-artifacts DATE=2026-06-26 DRYRUN=1 JSON=1`.

---

## 5. Additive day packet

`build_day_packet(trade_date, journal_path, out_root)` fans the journal's events for that day into
`data/odte/days/YYYY-MM-DD/<stream>.jsonl` (`market_snapshots`, `candidates`, `vehicle_scores`,
`trades`, `controller_events`) plus a `postmortem.md` scaffold. **Additive + derived:** it only reads
the journal and (re)writes the day folder — old files untouched. Stream files are regenerated each
run (idempotent); `postmortem.md` is created only if missing, so human edits survive. Off by default;
opt in with `--day-packet` on the ingest command. Best-effort/fail-safe (never raises).

---

## 6. Scan tier vs execution tier

The scan/decision-support surface and the (controller-owned) execution surface are kept explicitly
separate so a watchlist name can never silently execute:

- **Watchdog** is a trigger lane: `triggers.json` carries `scan_only=True`, `execution_allowed=False`,
  and a `decision_context` block (`thesis`, `confidence`, `confirmation_needed`,
  `required_confirmations`, `veto_reasons`, `risk_notes`, `observed_market_context`,
  `social_context`, `gamma_context`). These are invariants — even a strong directional candidate is
  non-executable here.
- **TOP-CHATTER cards** (`build_top_chatter`) are tagged `tier="scan"`, `execution_allowed=False`.
- **Journal** enforces `scan_only/restricted ⇒ execution_allowed=False` at write time.

> **TODO (not done — deliberately):** there is no `cfg/config.yaml` *execution_universe* key yet; the
> live execution core lives in the Hermes/MCP cron/skill (per project design), and adding an
> executable allow-list to the repo config risked implying the repo can trade. The tier boundary is
> therefore enforced in code/schema (flags above), not as a config list. Revisit if/when execution
> moves into the repo.

### 6a. Entry gate (thesis → entry) — `odte_entry_gate`

`build_entry_gate_decision(trigger=None, candidate=None, *, day_score=None, vehicle_score=None,
gamma_map=None, broker_snapshot=None, required_gates=None, scan_only=None,
promote_to_execution=False, now=None)` is the **PURE/OFFLINE** seam between the scan/trigger lane
and the (autonomous) execution manager. It places
**no orders**, makes no broker/network/LLM calls, and only assembles a *journalable* record the
manager reads BEFORE acting — so the thesis→entry decision is recorded the same way every time.

It is the **one tier where `execution_allowed` may be True**, and only under strict conditions:

```
execution_allowed = (not scan_only) and (not restricted) and (no veto_reasons)
                    and every required gate is EXPLICITLY True
```

**Tier boundary — scan_only inheritance + explicit promotion.** A scan/watchlist candidate must not
silently become an execution candidate. When `scan_only` is not passed explicitly,
`build_entry_gate_decision` **inherits it from `trigger.scan_only`/`candidate.scan_only`** (the
watchdog lane is always `scan_only=True`). An inherited (or explicit) `scan_only` record can only be
demoted to the execution tier when the manager **explicitly** sets `promote_to_execution=True`
(CLI `--promote-to-execution`). So a `triggers.json` with `scan_only=true` fed to `odte-entry-gate`
**stays `observe`/`execution_allowed=False` by default**, even with good day/vehicle/broker — it
takes the explicit promote flag to make it executable. The record carries `promoted_to_execution` and
a `scan_only_inherited` / `scan_only_promoted_to_execution` reason code so the journal shows why.
(`--scan-only` still force-pins `scan_only=True` and cannot be promoted past the journal guards.)

Default required gates (each `True | False | None`; a missing input is `None` → not True → **fails
closed**):

| Gate | True when | False (hard veto) when | None when |
|---|---|---|---|
| `day_regime` | `day_score.verdict == GOOD_DAY` | `== AVOID` (`day_regime_avoid`) | missing / `CHOP` |
| `vehicle` | `vehicle_score.verdict == GOOD_BET` | `== BAD_BET` (`vehicle_bad_bet`) | missing / `WATCH` |
| `directional_thesis` | a known call/put direction | — | no direction |
| `account` | `buying_power > 0`, not blocked, day-trades left | `0`/blocked/no day-trades (`insufficient_buying_power`/`account_blocked`/`no_day_trades_left`) | no broker snapshot |

`decision`/`intent` verb: `veto` (restricted or any hard veto) · `observe` (scan_only, or a required
gate is unknown — keep watching) · `enter` (all gates pass, execution authorized) · `deny` (gates
known but not all positive). The record also carries `reason_codes`, `veto_reasons`, a consistent
`thesis` block (`direction`/`basis`/`day_regime`/`vehicle_verdict`), `confidence`,
`observed_market_context`/`social_context`/`gamma_context` (passed through from the trigger), and
`required_confirmations` = `live_chain_recheck`, `spread_cap_check`, `budget_check` (live
re-validations the manager must still perform — Hermes autonomy is preserved, there is no
`human_review` block).

`event_from_entry_gate(gate_decision, trade_id=None, extra=None)` converts the record into an
`entry_decision` journal event (parallel to `event_from_position_decision`/`event_from_vehicle_score`).
**Defense in depth:** the flag flows through `build_decision_event`, which re-enforces
`scan_only/restricted ⇒ execution_allowed=False` at write time.

CLI: `odte-entry-gate` (`make odte-entry-gate`). Inputs are `--trigger`/`--candidate`/`--day-score`/
`--vehicle-score`/`--gamma`/`--broker` (each a PATH or a `*-json '{...}'`), `--scan-only`,
`--promote-to-execution`, and `--journal` (append an `entry_decision` event). `--json` / `--write` as
usual. No orders/broker/network.

```bash
make odte-entry-gate TRIGGER=data/odte/triggers.json \
     DAY_SCORE=data/odte/reports/odte_day_score.json \
     VEHICLE=data/odte/reports/odte_vehicle_score_qqq.json \
     BROKER=data/odte/broker.json JSON=1            # scan_only inherited from triggers.json -> stays observe
make odte-entry-gate TRIGGER=... BROKER=... PROMOTE=1 JSON=1   # explicit promote -> can become enter if gates pass
make odte-entry-gate TRIGGER=... JOURNAL=1          # also append the entry_decision event
```

> **TODO (not done — deliberately):** the `account` gate reads a caller-supplied broker-ish dict; it
> does NOT connect to a broker. Live funds/PDT must still be re-checked by the manager
> (`budget_check`). `gamma`/`account` weighting is intentionally binary (pass/fail), not scored —
> revisit only if the manager needs finer gradations.

---

## 7. Async / fetch safety model

`data/odte_concurrency.bounded_gather(fn, items, *, max_workers=4, timeout_s=20)` runs per-ticker
fetches concurrently with a **shared deadline**, returning a partial-result payload
(`{item, ok, result, status}`, status `ok|timeout|error:…`). It never raises, caps concurrency, and
does not block on a stuck worker (`shutdown(wait=False, cancel_futures=True)`).

`build_top_chatter` uses it: a slow/hung/failing ticker degrades to a safe OBSERVE card (no
contracts, `fetch_status` set) instead of stalling or aborting the scan — one bad ticker can't block
the loop. Reddit/X/FMP fetches already carry explicit `timeout=15/20`.

> **TODO:** `yfinance` (`odte_options.py`) still lacks an explicit per-call timeout (its
> `option_chain()`/`history()` don't accept `timeout=`); `bounded_gather`'s shared deadline now caps
> its blast radius, but a dedicated session-timeout/thread-wrap is the proper follow-up.

---

## 8. Self-evaluation (post-day)

`summarize()` / `build_report()` add a `process_quality` block over measurable (non-restricted)
closed trades — derived from fields already on the events, never fabricated:

- **process × outcome:** `good_process_good_outcome`, `good_process_bad_outcome`,
  `bad_process_lucky_outcome` (the dangerous one), `bad_process_bad_outcome` — process = followed the
  plan (no rule violations).
- **entry/exit/thesis diagnosis** (explicit `diagnosis` field wins, else MFE-vs-realized heuristic):
  `clean_win`, `good_entry_bad_exit` (won but left money on the table), `good_thesis_bad_exit`
  (round-tripped a winner), `thesis_wrong` (never went your way), `unclassified`. Explicit tags like
  `good_signal_bad_vehicle` / `good_thesis_bad_entry` are honored as-is.
- **loss_categories:** explicit `loss_category` tags only (`execution/thesis/timing/vehicle/risk/
  regime`); losers without a tag → `uncategorized` (we never guess a cause from P/L alone).

This is what lets a review answer "good thesis / bad entry vs good entry / bad exit vs good signal /
bad vehicle vs good process / bad outcome vs bad process / lucky outcome."

---

## 9. Reconstruct a day — commands

```bash
# 1) Fold the day's loose controller artifacts into the journal (safe to re-run; preview first)
make odte-ingest-artifacts DATE=2026-06-26 DRYRUN=1 JSON=1     # preview: events_would_append
make odte-ingest-artifacts DATE=2026-06-26 DAYPACKET=1 JSON=1  # commit + build the day packet

# 2) Read the per-stream day packet
ls data/odte/days/2026-06-26/
cat data/odte/days/2026-06-26/trades.jsonl
$EDITOR data/odte/days/2026-06-26/postmortem.md

# 3) Post-day scorecard incl. process-quality / loss diagnosis
make odte-journal-report WRITE=1
cat data/odte/reports/odte_journal_report.md       # has the "Process quality & loss diagnosis" section
make odte-journal-report JSON=1 | jq .process_quality
```

All commands are local/offline, place no orders, and treat NVDA as restricted throughout.
