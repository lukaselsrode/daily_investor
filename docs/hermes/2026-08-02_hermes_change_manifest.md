# Hermes 0DTE risk-on retune — agent-side change manifest (2026-08-02)

Companion prose for `2026-08-02_hermes_change_manifest.json` (the machine-readable version Hermes
applies with its own cron/config tools). The repo side has already shipped; nothing below edits
this repository.

## Why

The week of 2026-07-27..31 produced **zero trades** while the signal engine found qualified setups
every day. Forensics (decision journal + cron transcripts):

1. **Freshness budget < think time.** All lease inputs carried 60s TTLs while the controller tick
   takes 5–10 minutes (the `*/2` cron overlap-suppresses to ~50 of ~210 runs/day). On Jul 31 the
   entry gate PASSED and the lease was refused 11 seconds later on `market_snapshot_stale,
   broker_snapshot_stale`.
2. **Chase rail anchored at precompute + flat $120 A+ debit ceiling.** Every working momentum entry
   repriced 20–67% before final refresh and was vetoed; SPY 741C died at $121 — one dollar over.
3. **CHOP → "A+ only"** made 2 of 5 days near-untradeable.
4. **Ops rot**: Robinhood MCP init timeouts 58–69×/day; OAuth token expires **Mon Aug 3, 17:54 ET**;
   Reddit token dead since Jul 24; required social files missing; no same-day gamma map all week;
   a cleanup job archived `triggers.json` (loop-status then read FLAT_NO_TRADE).

## What the repo now provides (already shipped)

- **`make odte-convert`** — atomic confirm→gate→lease in ONE process under ONE clock (tape
  re-check → computed confirmations → entry gate → lease). Refusals name the stale input and
  journal an identity-bound terminal `no_trade_decision`.
- **Chase band**: lease `max_limit_price = anchor_quote × 1.15`, anchored at CONFIRM_ENTRY.
- **Tiered BP-proportional sizing**: 60% of BP full tier, 30% B+ (CHOP half-size tier); the flat
  $120 rail is obsolete.
- **Snapshot TTL 120s** (market/broker), conversion SLA 180s, lease default TTL 60s (hard cap 60s
  unchanged), 1-contract limit unchanged, incident replay still refuses.
- **Daily budget 2 trades/ET-day + 20-min post-close cooldown**; green-day re-entry arms at
  1.5× contract cost (was an un-armable flat $500).
- **Weekly telemetry + zero-trade tripwire** on every `odte-loop-status` payload.
- **Watchdog** no longer parks on unconvertible single names and re-alerts a persisting candidate
  every 10 minutes.

## Agent-side changes (apply via Hermes tooling — summary)

| # | Area | Severity | Action |
|---|------|----------|--------|
| 1 | MCP auth | **BLOCKING, before Mon 09:00 ET** | Refresh the Robinhood OAuth token (expires Mon 17:54 ET); add expiry preflight |
| 2 | Cron | HIGH | Controller `*/2`→`*/5`; on `CONVERT_CANDIDATE_NOW` run the fast path: batched snapshot read → `make odte-convert` → immediate review/place on `converted:true` → order-guard poll ~10s |
| 3 | Prompt | HIGH | Snapshots feeding convert are fetched immediately before the call — never reuse tick-opening reads |
| 4 | Policy v5 + hook | HIGH | Delete flat $35/$70/$120 rails + `ABSOLUTE_MAX_DEBIT=120`; hook validates against `lease.max_limit_price` / `lease.max_debit`; drop "CHOP = A+ only" and zero-tolerance no-chase wording |
| 5 | Inputs | MEDIUM | Retire dead `x_text.txt`/`reddit_text.txt` requirement; missing social files never veto tape candidates |
| 6 | Cron | MEDIUM | 09:35 ET + hourly gamma-map job; wall vetoes only against a same-day map |
| 7 | Cron | HIGH | Cleanup job excludes `triggers.json` + canonical loop artifacts |
| 8 | Prompt | LOW | Journal one advisory/day when the weekly zero-trade tripwire fires |
| 9 | MCP config | MEDIUM | Raise Robinhood MCP timeout, auto-reconnect once, batch snapshot reads |
| 10 | Cron | STANDING | One broker-capable job only; leases + pre-order hook stay the only order gates |
| 11 | Prompt | LOW | Advisory swarm conviction artifact (research-only, fail-open) — see below; effective Tuesday |

## 11. Advisory swarm conviction artifact (research-only, fail-open) — effective Tuesday 2026-08-04

A local SwarmClaw research swarm (no broker access, no Hermes access) now writes advisory
artifacts under `data/odte/swarm/`. From **Tuesday 2026-08-04**, when working a candidate, the
controller MAY read `data/odte/swarm/conviction.json` as ADVISORY color iff it exists,
`generated_at` is under 10 minutes old, and `candidate_key` matches the candidate in hand. Use it
solely to enrich decision-journal narrative and Telegram commentary (note agreement or dissent).

It is NEVER a gate or an input to any decision: absence, staleness, schema errors, or `"low"`
conviction MUST NOT delay, veto, resize, retier, or reorder anything the deterministic tools
authorize; `"high"` conviction MUST NOT soften or bypass any veto, gate, lease term, or sizing
rule. Read once, proceed regardless — never wait for this file. Never write under
`data/odte/swarm/`; the cleanup job (item #7) must also exclude that entire directory.
**Monday 2026-08-03 the file may appear but MUST be ignored (latency shadow).** Standing
invariant #10 is unchanged: leases + the pre-order hook remain the only order gates; the swarm
has no broker-capable lane.

## Rollout

- **Sunday**: apply manifest, refresh OAuth, dry-run `odte-convert` on Friday fixtures.
- **Monday**: live shadow — fast path with real snapshots, **no placement**. Gate: convert <5s,
  snapshot age <120s at authorize, zero `*_stale` refusals.
- **Tuesday**: arm placement (capped live, 2/day budget) if Monday is clean.
- **Target**: 3–4 trades/week; a 5-day zero-trade span now trips telemetry by Wednesday.
