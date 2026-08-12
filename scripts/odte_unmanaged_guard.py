#!/usr/bin/env python
"""Alarm when a position is open and nothing has managed it recently.

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


def main(argv=None) -> int:
    now = datetime.now(timezone.utc)
    line = evaluate(_load(ODTE / "active_trade.json"), _load(ODTE / "position_state.json"), now=now)
    if line:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
