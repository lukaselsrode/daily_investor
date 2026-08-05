# Hook v6 — fast-lane stage awareness (deploy at Gate 2, NOT before)

**Canonical source:** `scripts/hermes/odte_order_guard_hook.py` (repo-owned; the deployed copy
at `~/.hermes/hooks/odte_order_guard_hook.py` is a plain copy of it).

## What changed vs v5

One new gate, placed AFTER the closing-only allow and BEFORE the lease checks:

- The hook reads `data/odte/fast_lane_stage.json` (env override `ODTE_STAGE_PATH`).
- At stage **`entries_live`** it blocks **EVERY Hermes opening order regardless of lease
  state** — the fast-lane daemon is the only lane that may open positions. The daemon never
  passes through this hook (it runs its own strictly-stronger in-process guard against the
  SAME `consumed_leases.json` ledger), so it is unaffected.
- Closing/sell orders still ALWAYS pass (unchanged) — Hermes keeps its emergency
  sell-to-close/cancel valve at every stage.
- A missing/unreadable stage file means **no stage restriction** (v5 behavior exactly), so
  deploying the v6 hook early is harmless — but the sanctioned order is to deploy it during
  the Gate-2 transition so the change is observed, not latent.
- Also updated: the no-lease refusal now steers to `make odte-convert` (v5 text still named
  the retired `odte-execution-authorize` path).

Everything else — NVDA block, lease existence/expiry/ledger/identity/ceiling checks,
fail-closed backstop — is byte-for-byte v5 semantics.

## Deploy

```bash
cp ~/dev_work/daily_investor/scripts/hermes/odte_order_guard_hook.py \
   ~/.hermes/hooks/odte_order_guard_hook.py
```

(Hermes reads the hook body live per-call — no gateway restart needed.)

## Verify (dry, no orders)

```bash
cd ~/dev_work/daily_investor
# 1. At entries_live an OPEN is blocked even with a live lease:
make odte-fast-lane-stage STAGE=entries_live
echo '{"tool_name":"mcp__robinhood__place_option_order","tool_input":{"quantity":"1","price":"0.10","legs":[{"option_id":"x","side":"buy","position_effect":"open"}]}}' \
  | python3 ~/.hermes/hooks/odte_order_guard_hook.py       # expect: block, "fast lane owns opening orders"
# 2. A CLOSE still passes:
echo '{"tool_name":"mcp__robinhood__place_option_order","tool_input":{"quantity":"1","price":"0.10","legs":[{"option_id":"x","side":"sell","position_effect":"close"}]}}' \
  | python3 ~/.hermes/hooks/odte_order_guard_hook.py       # expect: no output (allow)
# 3. Roll the stage back:
make odte-fast-lane-stage STAGE=shadow
# 4. At shadow, v5 behavior is intact (no-lease open still blocks on the lease, not the stage):
echo '{"tool_name":"mcp__robinhood__place_option_order","tool_input":{"quantity":"1","price":"0.10","legs":[{"option_id":"x","side":"buy","position_effect":"open"}]}}' \
  | python3 ~/.hermes/hooks/odte_order_guard_hook.py       # expect: block on lease, not stage
```

## Residual (accepted, per plan)

Hermes could in principle edit the stage file itself; `make odte-fast-lane-stage` is the
sanctioned path (prompt v6 forbids direct writes), the file is keep-listed in `odte-cleanup`,
and any divergence is visible in the `fast_lane_status.json` heartbeat. Governance-by-prompt
plus mechanical enforcement at the order boundary — not over-engineered caller identity inside
the pure conversion chain.
