from __future__ import annotations
import json
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path

quotes = {
    "SPY": {"last": 734.57, "prev": 734.30},
    "QQQ": {"last": 713.31, "prev": 716.38},
    "IWM": {"last": 298.06, "prev": 298.91},
    "VIXY": {"last": 22.585, "prev": 22.48},
}
vix = 18.74
et = ZoneInfo("America/New_York")
now_et = datetime.now(et)
close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
minutes_to_close = max(0, int((close - now_et).total_seconds() // 60))
# Approximate ORB/VWAP states from current RH quote + odte-report SPY above_vwap.
# Without reliable intraday bars in this cron lane, use a conservative split/chop snapshot.
market = {
    "generated_at": now_et.isoformat(timespec="seconds"),
    "source": "RH quotes via MCP subprocess + conservative VWAP/ORB approximation",
    "vix": vix,
    "vixy_change_pct": (quotes["VIXY"]["last"] / quotes["VIXY"]["prev"] - 1) * 100,
    "gap_pct": (quotes["SPY"]["last"] / quotes["SPY"]["prev"] - 1) * 100,
    "expected_move_pct": 0.8,
    "minutes_to_close": minutes_to_close,
    "spy_above_vwap": True,
    "spy_orb_state": "inside",
    "qqq_above_vwap": False,
    "qqq_orb_state": "below",
    "iwm_above_vwap": False,
    "iwm_orb_state": "below",
    "quotes": quotes,
}
out = Path("data/odte/market_snapshot_20260626_1332.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(market, indent=2))
print(out)
print(json.dumps(market, indent=2))
