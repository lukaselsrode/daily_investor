# Fast-path snapshot templates for `make odte-convert`

**These are the exact shapes that produced the 2026-08-03 SPY 756C conversion** (the first and so
far only live `odte-convert` run: CONFIRM→lease in-process, filled 44.6s after the snapshot, clean
+$9 exit). Copy the shapes, fill from a single batched MCP read, write with FRESH timestamps
immediately before calling convert.

Preflight refuses with a NAMED input (`market_snapshot_missing|_undated|_stale`,
`broker_snapshot_*`, `contract_quote_*`, `candidate_missing`) — a refusal tells you exactly which
file to fix. Snapshots must be newer than `live_rails.snapshot_ttl_seconds`.

## market.json

Both flat keys (day scorer) and per-symbol blocks (candidate-watch tape confirm) are required —
the flat `*_above_vwap` / `*_orb_state` keys feed `odte-day-score`, the blocks feed the tier
computation. OMIT fields you cannot compute; never placeholder `0`/`false`.

```json
{
  "generated_at": "2026-08-03T15:42:01+00:00",
  "minutes_to_close": 257.98,
  "spot": 755.765,
  "spy_above_vwap": true, "spy_orb_state": "above",
  "qqq_above_vwap": true, "qqq_orb_state": "above",
  "iwm_above_vwap": true, "iwm_orb_state": "above",
  "vixy_change_pct": -1.78,
  "SPY":  {"last": 755.765, "above_vwap": true, "orb_state": "above",
           "accepted_above_wall": true, "retest_hold": true},
  "QQQ":  {"last": 697.215, "above_vwap": true, "orb_state": "above"},
  "IWM":  {"last": 295.79,  "above_vwap": true, "orb_state": "above"},
  "VIXY": {"last": 20.145,  "above_vwap": false, "change_pct": -1.78}
}
```
Optional day-score inputs when known: `vix`, `gap_pct`, `expected_move_pct` (percent units).
`orb_state` ∈ `above|below|inside`. XSP may be included as tape (it confirms, never converts).

## contract.json — the ONE locked contract

```json
{
  "generated_at": "2026-08-03T15:42:01+00:00",
  "updated_at": "2026-08-03T15:41:13.726138+00:00",
  "option_id": "55435394-30fb-4503-b27a-a77501ba3b79",
  "underlying": "SPY", "chain_symbol": "SPY",
  "option_type": "call", "type": "call",
  "expiration_date": "2026-08-03", "strike_price": 756.0,
  "bid_price": 0.62, "ask_price": 0.63, "mark_price": 0.625,
  "delta": 0.454771, "gamma": 0.216424,
  "open_interest": 7296, "volume": 335047
}
```
The ask sets the CONFIRM_ENTRY anchor; the lease ceiling is `anchor × (1 + chase_band_fraction)`.
`option_id` must match the candidate's exactly — identity mismatch fails closed.

## broker.json

```json
{
  "generated_at": "2026-08-03T15:42:01+00:00",
  "buying_power": 348.16,
  "day_trades_left": 3,
  "nonzero_option_positions_count": 0,
  "open_option_orders_count": 0,
  "today_option_orders_count": 0,
  "blocked": false, "execution_lane": "ok", "orders_ok": true
}
```
All three counts are REQUIRED — a missing count fails closed (`broker_*_count_missing`). Build
from `get_portfolio` + `get_option_positions` + `get_option_orders` (`get_accounts` is excluded
by design).

## candidate.json

```json
{
  "ticker": "SPY", "direction": "bullish", "selected_vehicle": "SPY",
  "source": "live_tape_relative_strength",
  "created_at": "2026-08-03T15:42:01+00:00",
  "selection_timestamp": "2026-08-03T15:42:01+00:00",
  "selection_reason": "SPY accepted above fresh 755 call wall across multiple 5m bars; VIXY weak",
  "accepted_above_wall": true, "retest_hold": true
}
```
Convert defaults to `data/odte/active_candidate.json` when `CANDIDATE=` is omitted. `tier` and
`anchor_quote` are stamped BY candidate-watch inside convert — never hand-write them.

## The call

```bash
make odte-convert CANDIDATE=data/odte/convert_candidate.json \
                  MARKET=data/odte/convert_market.json \
                  BROKER=data/odte/convert_broker.json \
                  CONTRACT=data/odte/convert_contract.json \
                  GAMMA=data/odte/reports/odte_gamma_map_spy.json JSON=1
```
`converted: true` ⇒ consume `lease` at review/place IMMEDIATELY (expiry is in `next_action`), then
poll `make odte-order-guard`. Any refusal is journaled as an identity-bound terminal
`no_trade_decision` — fix the named input and re-run; a stale confirm is never terminal.
