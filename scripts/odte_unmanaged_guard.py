#!/usr/bin/env python
"""Alarms for the two things that cost money and that nothing else reports.

1. A position is open and nothing has managed it recently.
2. An execution lease expired WITHOUT ever becoming an order — we reached the point of trading
   and lost it.

WHY THIS EXISTS. Every exit in the system's history was placed by `0dte-live-controller`
(`exit_fill` sources: {'0dte-live-controller': 9}). The watchdog pulse and the heartbeat are
`no_agent` and place nothing. So if the controller stops ticking while size is on, the position has
no stop, no MAX_LOSS and no take-profit until it comes back — and nothing else in the system will
notice.

That is not hypothetical. On 2026-08-12 another lane's `daily-investor-weekday-fetch-data` job held
the gateway from ~12:04 to ~12:20 ET and the controller missed three consecutive FREE slots (its
own 12:00 tick had finished at 12:04:28, so this was starvation, not overrun). We happened to be
flat. The same window with size on would have been 16 minutes unmanaged, against an IWM position
that ran -14% -> -42% in 71 seconds earlier that day.

This runs from the every-minute `no_agent` watchdog pulse, which was NOT starved during that
window. It prints NOTHING unless the alarm holds — the pulse only delivers Telegram on non-empty
stdout. It never places an order, never mutates state, and always exits 0 so it cannot break the
pulse it rides on.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ODTE = ROOT / "data" / "odte"

# Sized from the record, not from taste: across 216 consecutive management-poll gaps on open
# positions the median is 35s and p90 is 120s, so 300s is >2x the p90 and normal operation does not
# reach it. The tail goes to 2164s, but those long gaps ARE this condition — they are the thing
# being detected, not noise to tune around.
STALE_AFTER_SECONDS = 300.0

_OPEN_STATUS = {"open", "active", "filled", "managing"}

# A lease is single-use and 60s. Once it lapses unused the setup is gone, and on 2026-08-13 that
# happened on the ONLY setup of the day: lease 4235f2af issued 11:03:36 for SPY 778C b_plus,
# expired 11:04:36, never consumed, no order. The controller's own Telegram called it "converted
# successfully: stage=authorize, no refusal codes" and "No position opened" — technically true and
# completely misleading, and it was the last message for 25 minutes. 11 of 14 leases ever issued
# were consumed, so a miss is the exception and worth saying out loud.
#
# Stateless de-dup: only fire while the expiry is between 0 and this many seconds old, so the
# once-a-minute pulse says it once or twice rather than forever.
LEASE_ALERT_WINDOW_SECONDS = 120.0


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _parse_ts(value):
    if not value:
        return None
    try:
        s = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def evaluate(trade: dict | None, position: dict | None, *, now: datetime,
             stale_after: float = STALE_AFTER_SECONDS) -> str | None:
    """Return the alarm line, or None to stay silent. Pure: no IO, no clock of its own."""
    trade = trade if isinstance(trade, dict) else {}
    position = position if isinstance(position, dict) else {}
    status = str(trade.get("status") or "").strip().lower()
    if status not in _OPEN_STATUS:
        return None                      # flat, or no plan on disk — nothing to guard

    # An open position whose management timestamp we cannot read is the dangerous ambiguity, not a
    # reason to stay quiet: we cannot show it is being managed, so say so.
    last = _parse_ts(position.get("updated_at")) or _parse_ts(trade.get("updated_at"))
    label = "%s %s%s" % (trade.get("underlying") or "?", trade.get("strike_price") or "?",
                         "P" if str(trade.get("option_type")) == "put" else "C")
    if last is None:
        return ("0DTE POSITION UNMANAGED: %s is open and carries no readable management timestamp "
                "— the controller may not be running. Nothing but 0dte-live-controller can exit." % label)

    age = (now - last).total_seconds()
    if age <= stale_after:
        return None
    return ("0DTE POSITION UNMANAGED: %s open, last managed %.0fs ago (limit %.0fs). Only "
            "0dte-live-controller can exit; if it is starved this position has no stop, no "
            "MAX_LOSS and no take-profit. Check the controller cron and the gateway." % (
                label, age, stale_after))


def evaluate_lease(lease: dict | None, consumed: list | None, *, now: datetime,
                   window: float = LEASE_ALERT_WINDOW_SECONDS) -> str | None:
    """Return an alert for a lease that lapsed without becoming an order, else None. Pure."""
    lease = (lease or {}).get("lease") if isinstance(lease, dict) else None
    if not isinstance(lease, dict) or not lease.get("lease_id"):
        return None
    if str(lease["lease_id"]) in {str(x) for x in (consumed or [])}:
        return None                       # it became an order — nothing to say
    expires = _parse_ts(lease.get("expires_at"))
    if expires is None:
        return None
    age = (now - expires).total_seconds()
    if not (0 < age <= window):
        return None                       # not yet lapsed, or already reported
    # The lease file spells the ticker `symbol`; the journal event spells it `underlying`. Read
    # both — an alert that says "? 778.0 call" is the kind of thing nobody acts on.
    ticker = (lease.get("symbol") or lease.get("underlying") or lease.get("chain_symbol") or "?")
    return ("0DTE LEASE MISSED: %s %s%s expired unused after 60s — the setup was confirmed and "
            "authorized but no order was placed, so no trade happened. Conversion latency, not a "
            "gate refusal." % (ticker, lease.get("strike_price") or "",
                               "C" if str(lease.get("option_type")) == "call" else "P"))


def main(argv=None) -> int:
    now = datetime.now(timezone.utc)
    for line in (evaluate(_load(ODTE / "active_trade.json"),
                          _load(ODTE / "position_state.json"), now=now),
                 evaluate_lease(_load(ODTE / "execution_lease.json"),
                                _load(ODTE / "consumed_leases.json"), now=now)):
        if line:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
