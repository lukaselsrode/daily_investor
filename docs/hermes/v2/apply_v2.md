# Apply v2 — ordered steps (2026-08-03, before Tuesday's open)

v1 (2026-08-02 manifest) shipped a to-do list; 0 of 9 items were applied. v2 ships finished
artifacts — applying is copying. Every step has a verify line; do them in order.

## 1. ONE broker-capable lane (highest severity)

The 2026-08-03 entry was placed by a long-running interactive session (alive since Jul 30) while
the cron controller placed the exit — two broker lanes, one account, no coordination.

- End or de-broker that session (`20260730_090124_31478bd7`): remove its Robinhood order tools
  (`review_option_order`, `place_option_order`, `cancel_option_order`). Read tools may stay.
- **Verify**: while the controller cron is enabled, no other session/agent lists Robinhood order
  tools; `get_option_orders` provenance for the next trade shows a single lane.

## 2. Install policy v5

Replace `~/0dte/controller_policy.json` with `controller_policy_v5.json` (carry over account
identity + starting_context from v4; keep a dated backup like the v3 one).

- **Verify**: `python3 -c "import json;p=json.load(open('~/0dte/controller_policy.json'.replace('~',__import__('os').path.expanduser('~'))));print(p['version'], 'a_plus_setup_max_debit' not in str(p))"` → `5 True`.

## 3. Install prompt v5 + cadence

Update cron job `344e4c3333a7`: schedule `*/2 9-15 * * 1-5` → `*/5 9-15 * * 1-5`; replace the
prompt body with `controller_prompt_v5.md`.

- **Verify**: jobs.json shows `*/5`; prompt contains `odte-convert` and `live_rails`, and does NOT
  contain `odte-entry-gate → odte-execution-authorize` or any `60-second` SLA wording.

## 4. Patch the pre-order hook

Apply `hook_patch.md` (delete `ABSOLUTE_MAX_DEBIT` + its check; lease validation stays).

- **Verify**: `grep -c ABSOLUTE_MAX_DEBIT ~/.hermes/hooks/odte_order_guard_hook.py` → 0.

## 5. Fix the MCP client `get_accounts` response schema

Robinhood now returns `unsettled_funds`; the client schema rejects the whole payload
(`additionalProperties: false`) — 68/68 calls failed on 2026-08-03, killing account-identity
verification and adding ~136 noise lines/day inside the conversion window.

- Admit the field (or set `additionalProperties: true`) in the client's `get_accounts` response
  schema.
- **Verify**: one `get_accounts` call succeeds; errors.log shows no `unsettled_funds` lines.

## 6. OAuth preflight

The token was refreshed 70 seconds AFTER expiry on 2026-08-03 (lucky: post-close). Add to the
09:31 market-open check-in: read `~/.hermes/mcp-tokens/robinhood.json` `expires_at`; if <48h
remain, refresh NOW and confirm reload.

- **Verify**: check-in transcript shows the expiry check; token `expires_at` always >48h out at
  the open.

## 7. Housekeeping

- Controller transcript retention: raise the 50-file cap for `344e4c3333a7` (the morning of
  2026-08-03 has no transcripts left; executions.db alone is not an audit trail).
- Cleanup sweeps: exclusion list per policy v5 `artifact_hygiene` (notably `consumed_leases.json`,
  archived post-trade on 2026-08-03, and `triggers.json`).
- Tripwire: journal one advisory note per day when loop-status `weekly_telemetry.tripwire.fired`.

## What the repo already enforces (no Hermes action needed)

- CONFIRM_ENTRY into `odte-entry-gate` refuses with `use_odte_convert` (exit 2) — the legacy
  conversion path is physically closed.
- Every loop-status payload carries `live_rails` (SLAs/TTLs/chase band/dollar ceilings computed
  from fresh BP/budget/green state) and `weekly_telemetry`.
- Post-green re-entry auto-arms on same-or-better tier with a budget slot, cooldown clear, and
  BP ≥ 1.5× cost (`green_reentry_auto_arm: false` in cfg/config.yaml reverts to manual-only).

## Success criteria for the week

- 3–4 trades; zero conversion-SLA narrative vetoes (a stale confirm = re-run odte-convert);
- every conversion runs `odte-convert` (journal: `source: "odte_convert"` on entry/lease events);
- one broker lane on every order; zero `unsettled_funds` errors; tripwire silent by Wednesday.
