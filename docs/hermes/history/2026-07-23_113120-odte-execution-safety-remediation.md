# 0DTE Execution Safety Remediation Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make a repeat of the 2026-07-23 delayed-fill loss structurally impossible before autonomous execution can be re-enabled.

**Architecture:** Keep all scanners and analytical artifacts read-only. Replace the free-form `promote_to_execution=True` escape hatch with a deterministic, short-lived execution authorization lease tied to one exact symbol, direction, option contract, quantity, price ceiling, and market snapshot. Add a pending-order state machine that cancels stale entries and requires complete reauthorization instead of inheriting an old thesis. Re-enable through shadow and capped-live stages only after replay and regression gates pass.

**Tech Stack:** Python 3.12, frozen dataclasses/dicts, JSON artifacts under `data/odte/`, pytest, existing CLI/Makefile wiring, Robinhood MCP only at the external execution boundary.

---

## Current incident and non-negotiable invariants

The 2026-07-23 incident sequence was:

1. `odte-watchdog` emitted a low-confidence, `scan_only=True`, `execution_allowed=False` SPY bearish observation.
2. `odte-entry-gate --promote-to-execution` converted it to executable at 11:11:17 ET.
3. A 2-contract SPY 737P limit was submitted at 11:13:11 ET for $336, 83.9% of $400.34 BP.
4. It remained pending and filled at 11:15:35 ET, after the original momentum extension had stalled.
5. The position was closed at 11:20:59 ET for a $52.16 estimated net loss.

The fixed system must enforce:

- Scan/watchdog output can never be promoted by a bare boolean.
- Authorization is exact-contract, exact-direction, exact-quantity, exact-price, and short-lived.
- An unfilled entry order cannot survive beyond its authorization lease.
- A stale order must be cancelled; it can never be “reclassified after fill” as the normal path.
- Any symbol/vehicle change requires a new candidate, new scoring, new gate, and new authorization.
- Full-account deployment is permitted only under an explicit `FULL_ACCOUNT_A_PLUS` lease with exact premium risk, fresh trigger, named invalidation, and active-management readiness; size is never inferred from a label alone.
- Order submit, pending, cancel, fill, and exit are broker-reconciled and immediately journaled.
- No model/cron prompt can override deterministic vetoes.

---

### Task 1: Preserve the incident as a deterministic replay fixture

**Objective:** Turn today’s exact failure into a regression test instead of relying on prose.

**Files:**
- Create: `tests/fixtures/odte/2026-07-23-delayed-fill.json`
- Create: `tests/test_odte_execution_policy.py`

**Steps:**

1. Build a sanitized fixture containing trigger time, promotion time, order submission time, fill time, candidate symbol/direction, contract ID metadata, quantity, limit, BP, invalidation, and tape snapshots. Do not include account numbers or order UUIDs.
2. Add a failing replay test asserting that an authorization derived from the 11:11 signal is expired before the 11:15 fill.
3. Add a failing test asserting that the $336/2-contract order is rejected under the new default one-contract and debit-cap policy.
4. Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_odte_execution_policy.py -v
```

Expected before implementation: failures because the execution policy does not exist.

---

### Task 2: Replace bare promotion with a short-lived execution lease

**Objective:** Remove `promote_to_execution=True` as sufficient authority.

**Files:**
- Create: `src/data/odte_execution_policy.py`
- Modify: `src/data/odte_entry_gate.py`
- Modify: `tests/test_odte_entry_gate.py`
- Test: `tests/test_odte_execution_policy.py`

**Proposed API:**

```python
@dataclass(frozen=True)
class ExecutionLease:
    lease_id: str
    issued_at: datetime
    expires_at: datetime
    symbol: str
    direction: str
    option_id: str
    expiration_date: str
    strike_price: float
    option_type: str
    quantity: int
    max_limit_price: float
    max_debit: float
    candidate_fingerprint: str
    market_fingerprint: str


def authorize_entry(*, gate: dict, candidate_decision: dict, vehicle_score: dict,
                    broker_snapshot: dict, market_snapshot: dict,
                    now: datetime, policy: dict) -> dict:
    ...
```

**Required behavior:**

- Default lease TTL: 30 seconds; configurable downward, never upward beyond 60 seconds for 0DTE.
- Require fresh `candidate_decision.decision == "CONFIRM_ENTRY"`.
- Require the candidate decision, vehicle score, market snapshot, broker snapshot, and entry gate to each carry parseable timestamps within their own TTLs.
- Require exact identity agreement among candidate symbol, vehicle contract underlying, gate symbol, direction, option type, option ID, and expiration.
- Require all final confirmations as explicit booleans, not strings or missing fields.
- Require `scan_only=False` before lease issuance; the watchdog artifact itself remains non-authoritative.
- The lease contains the maximum quantity, maximum limit, and maximum debit; downstream cannot increase any value.
- Remove CLI behavior where `--promote-to-execution` alone produces `execution_allowed=True`. Keep the flag temporarily only as a deprecated input that returns a fail-closed reason such as `execution_lease_required`.

**Regression tests:**

- Bare promotion remains non-executable.
- Missing/stale candidate confirmation is rejected.
- Symbol, direction, option ID, strike, expiration, or price mismatch is rejected.
- Lease expires at its boundary.
- A valid fresh all-matching package issues one lease.
- Reusing a consumed lease is rejected.

---

### Task 3: Add explicit account-risk and quantity constraints

**Objective:** Allow full-account leverage when the setup and management are genuinely calculated, while preventing size from being inferred from an unverified “A+” label.

**Files:**
- Modify: `src/data/odte_execution_policy.py`
- Modify: `src/data/odte_strategy_policy.py`
- Modify: `tests/test_odte_execution_policy.py`
- Modify: `tests/test_odte_strategy_policy.py`

**Policy:**

```python
RISK_MODES = ("PARTIAL_ACCOUNT", "FULL_ACCOUNT_A_PLUS")
DEFAULT_RISK_MODE = "PARTIAL_ACCOUNT"
FULL_ACCOUNT_REQUIRES_FRESH_LEASE = True
FULL_ACCOUNT_REQUIRES_NAMED_INVALIDATION = True
FULL_ACCOUNT_REQUIRES_ACTIVE_MANAGEMENT = True
```

Do not impose a blanket 35% cap. A full-account debit is allowed when the execution lease explicitly records `FULL_ACCOUNT_A_PLUS`, the maximum premium loss, exact trigger and invalidation, current bid/ask and spread, quantity, target, scratch rail, and active management cadence. The risk mode cannot be inferred solely from a model-generated A+ label; every deterministic freshness, identity, liquidity, and broker gate must pass. If any required management input is missing, fail closed or reduce size rather than silently assume full-account authority.

**Tests:**

- Full-account deployment with a fresh exact-contract lease and complete management plan is allowed.
- Full-account deployment without a named invalidation, premium-at-risk field, or active-management readiness is rejected.
- Quantity/debit above the lease is rejected.
- Missing BP is rejected.
- Policy values and accepted maximum loss are serialized into every lease and journal event.

---

### Task 4: Implement pending-order TTL and cancel-first state machine

**Objective:** Ensure an entry limit cannot fill minutes after its trigger.

**Files:**
- Create: `src/data/odte_order_guard.py`
- Create: `tests/test_odte_order_guard.py`
- Modify: `src/data/odte_loop_status.py`
- Modify: `tests/test_odte_loop_status.py`

**Proposed states:**

```text
NO_ORDER
PENDING_FRESH
CANCEL_STALE_ENTRY
CANCEL_THESIS_INVALID
FILLED_FRESH
FILLED_WITHOUT_VALID_LEASE
BROKER_MISMATCH_BLOCKED
```

**Required behavior:**

- Consume fresh broker order truth, the execution lease, current market snapshot, and current time.
- `PENDING_FRESH` only while the lease is unexpired and thesis rails remain valid.
- At TTL expiry, output `CANCEL_STALE_ENTRY` immediately.
- If price/volatility invalidation fires before TTL, output `CANCEL_THESIS_INVALID` immediately.
- A stale pending order cannot be extended. A new order requires a new candidate confirmation and new lease.
- If broker truth reports a fill after lease expiry, classify `FILLED_WITHOUT_VALID_LEASE` as a safety incident and prohibit new entries; the external broker lane must flatten or alert according to an explicitly reviewed emergency policy.
- `odte-loop-status` must prioritize pending-order cancellation above scanning or new-entry work.

**Replay acceptance:** Today’s order must produce `CANCEL_STALE_ENTRY` before the recorded fill timestamp.

---

### Task 5: Lock vehicle selection to the active HAWK thesis

**Objective:** Stop the controller from discussing QQQ but executing SPY without a complete new decision package.

**Files:**
- Modify: `src/data/odte_candidate_watch.py`
- Modify: `src/data/odte_execution_policy.py`
- Modify: `tests/test_odte_candidate_watch.py`
- Modify: `tests/test_odte_execution_policy.py`

**Required behavior:**

- Persist `selected_vehicle`, `selection_reason`, `relative_strength_rank`, and `selection_timestamp` in `active_candidate.json`.
- A contract for another underlying is a hard mismatch, not an equivalent substitute.
- Switching QQQ→SPY or SPY→IWM invalidates the current candidate and requires a fresh watch/score/gate/lease cycle.
- Broad-market disagreement is represented as confidence/risk context, but it cannot silently select a different vehicle.

**Tests:**

- QQQ candidate + SPY contract is rejected.
- QQQ candidate + QQQ contract with matching bearish direction can proceed.
- Vehicle switch generates a new candidate fingerprint and invalidates the old lease.

---

### Task 6: Make broker reconciliation and immediate notifications mandatory

**Objective:** Prevent hidden order activity while analysis is still running.

**Files:**
- Modify: `src/data/odte_loop_status.py`
- Modify: `src/data/odte_journal.py`
- Modify: `tests/test_odte_loop_status.py`
- Modify: `tests/test_odte_journal.py`

**Required behavior:**

- Before any lease issuance: fresh positions, open orders, today’s orders, BP, and controller lock.
- Immediately after review, submit, cancel, fill, and exit: reconcile broker truth again.
- Add journal events: `execution_lease_issued`, `execution_lease_consumed`, `entry_order_pending`, `entry_order_cancelled_stale`, and `execution_safety_incident`.
- Status must expose pending order age and lease time remaining.
- The external controller must deliver order-submitted, fill, cancellation, and exit notices immediately rather than only at cycle completion.

---

### Task 7: Wire CLI, Makefile, README, and the controller contract

**Objective:** Ensure the safety modules are in the actual runtime path, not disconnected helpers.

**Files:**
- Modify: `src/cli/main.py`
- Modify: `Makefile`
- Modify: `README.md`
- Modify: `src/data/odte_loop_status.py`
- Test: relevant CLI dispatch tests if present

**Commands to add:**

```text
odte-execution-authorize
odte-order-guard
```

**Runtime ladder:**

```text
SCAN_ONLY
→ CANDIDATE_CONFIRMED
→ EXECUTION_LEASE_READY
→ BROKER_REVIEW
→ PENDING_ORDER_GUARD
→ FILLED_POSITION_MANAGEMENT
→ EXIT/FLAT
```

No controller prompt may jump from scan/watchdog directly to broker placement. `odte-loop-status` must block and name the missing deterministic step.

Because CLI commands are changing, update README, Makefile, CLI help, and tests together per `AGENTS.md`.

---

### Task 8: Add a broker-lane integration simulator

**Objective:** Test submit/pending/cancel/fill races without real money.

**Files:**
- Create: `tests/fakes/fake_option_broker.py`
- Create: `tests/test_odte_execution_integration.py`

**Scenarios:**

1. Immediate fill inside lease → accepted and managed.
2. Pending past TTL → cancel requested and broker verified cancelled.
3. Invalidation before fill → cancel requested.
4. Cancel/fill race → broker truth wins; safety incident emitted; no duplicate order.
5. Delayed fill matching today’s timestamps → incident path, never normal A+ management.
6. Concurrent controller ticks → one lease consumed once; no duplicate submission.
7. Symbol mismatch → no review or placement call.
8. Quantity/debit violation → no review or placement call.

The fake must record every broker method call so tests prove prohibited calls were never made.

---

### Task 9: Run shadow mode before risking another dollar

**Objective:** Prove the new workflow on live data without order placement.

**Stages:**

1. **Disabled/read-only:** current state; controller remains paused and hard-locked.
2. **Replay:** today’s incident and historical journal cases pass deterministically.
3. **Live shadow:** at least 20 independent entry signals or five full sessions, whichever is longer. Generate leases and simulated pending-order actions, but never call broker review/place.
4. **Controlled live:** only after explicit user approval; one trade per day initially. Full-account deployment is allowed only when the complete `FULL_ACCOUNT_A_PLUS` lease passes; otherwise use partial size or no trade.
5. **Autonomous expansion:** not automatic. Requires explicit approval based on journaled results.

**Promotion criteria from shadow to capped live:**

- Zero stale orders surviving TTL.
- Zero symbol/contract/quantity mismatches.
- Zero duplicate lease consumption.
- Every hypothetical submit/cancel/fill transition has a matching journal event.
- Replay suite, focused pytest, full tests, and hygiene all green.
- User reviews and explicitly accepts risk limits.

---

### Task 10: Verification and definition of done

Run focused tests first:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_odte_execution_policy.py \
  tests/test_odte_order_guard.py \
  tests/test_odte_entry_gate.py \
  tests/test_odte_candidate_watch.py \
  tests/test_odte_loop_status.py \
  tests/test_odte_execution_integration.py -v
```

Run CLI smoke tests using sanitized fixtures:

```bash
PYTHONPATH=src .venv/bin/python -m cli.main odte-execution-authorize --help
PYTHONPATH=src .venv/bin/python -m cli.main odte-order-guard --help
make -n odte-execution-authorize
make -n odte-order-guard
```

Run project gates:

```bash
make test
make hygiene
```

Inspect:

```bash
git diff --check
git diff --stat
```

**Done means:**

- Today’s replay cancels the order before its actual fill time.
- Bare `--promote-to-execution` cannot authorize execution.
- Today’s 2-contract / 83.9% BP order fails because its authorization went stale before fill; equivalent size is allowed only with a still-fresh `FULL_ACCOUNT_A_PLUS` lease and verified management readiness.
- QQQ thesis cannot place SPY.
- Expired or consumed leases fail closed.
- The controller remains disabled throughout development and shadow validation.
- No real-money order is used as a test.

---

## Risks and tradeoffs

- A 30-second TTL will miss some fills. That is intentional: missing a trade is preferable to acquiring a stale 0DTE position.
- Full-account leverage increases outcome variance and therefore demands stronger freshness, liquidity, invalidation, and management proof. Size cannot compensate for weak expectancy.
- Model reasoning can still identify candidates, but it must not mint or extend execution authority.
- The current working tree already contains unrelated modifications. Implementation must avoid overwriting or reverting them and should isolate commits by file/task.

## Current safety state

The former controller is paused, renamed `0dte-live-state-manager-HARD-DISABLED`, reduced to a read-only prompt, and stripped of autonomous execution skills. It must remain that way until the staged rollout criteria above are met.
