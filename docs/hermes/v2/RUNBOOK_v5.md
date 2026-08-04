# 0DTE Operating Runbook — v5 (2026-08-03 port)

Supersedes `~/0dte/RUNBOOK_2026-07-24.md` (deleted — it documented the retired v4 system:
`*/2` cadence, the `odte-entry-gate → odte-execution-authorize` ladder, the $120 A+ cap).

## The system in one breath

One broker lane (`0dte-live-controller`, cron `*/5 9-15 ET`). On CONFIRM_ENTRY the controller runs
the FAST PATH: batched fresh snapshots → `make odte-convert` (atomic tape re-check → gate → lease
in one process) → immediate review/place consuming the ~60s single-use lease → order-guard poll.
All clocks and dollars come from `odte-loop-status` `live_rails`; nothing is remembered or
hand-computed. Green days auto-arm a same-or-better-tier second trade inside the 2/day budget.
Policy: `~/0dte/controller_policy.json` v5 (`atomic_conversion_rails`). The Telegram DM lane has
NO broker tools (`platform_toolsets.telegram: no_mcp`).

## Kill switches (fastest first)

1. **Stop all trading now:** `hermes cron pause 344e4c3333a7`
2. **Force shadow mode (analysis, no orders):** edit `~/0dte/controller_policy.json` →
   `authority.execution_stage: "shadow"` (policy is re-read each tick)
3. **Hard broker-side kill:** remove `place_option_order` from
   `mcp_servers.robinhood.tools.include` in `~/.hermes/config.yaml`, then `/reload-mcp` in chat or
   `hermes gateway restart` from a terminal
4. The pre-order hook alone already blocks any opening order without a live matching lease —
   deleting `data/odte/execution_lease.json` de-authorizes instantly (closing orders always pass)

## Daily rhythm

- 09:31 check-in (read-only): policy v5 sanity, OAuth expiry preflight (>48h), broker truth via
  portfolio/positions/orders, `live_rails` + weekly telemetry report.
- Controller `*/5` during RTH; watchdog pulse `*/1` (script-only, no LLM).
- 16:10 EOD recap: reconcile journal ↔ broker fills; NEVER writes dollar/clock numbers into
  skills, memory, or policy — the repo owns numbers via `live_rails`.
- Cleanup only via `make odte-cleanup` (hardcoded keep-list; ad-hoc `mv` is forbidden).

## Auth & health

- Robinhood OAuth: `~/.hermes/mcp-tokens/robinhood.json` — refresh when <48h
  (`hermes mcp reauth robinhood`).
- `get_accounts` is EXCLUDED by design (Robinhood server-side schema bug); account truth =
  `get_portfolio` / `get_option_positions` / `get_option_orders`.
- Reddit social: tape-only since 2026-08-03 (auth dead); revive by creating `~/0dte/config.json`
  with non-expiring app OAuth (`reddit_client_id`/`reddit_client_secret` + `daily_thread_id`) —
  social is COLOR only, never a gate.
