#!/usr/bin/env bash
# Today's decision-journal events (ET trade day) — READ-ONLY.
# Journal events carry a "trade_date" field (YYYY-MM-DD, ET); filter on it, fall back to ts date.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 - <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
path = Path("data/odte/decision_journal.jsonl")
if not path.exists():
    sys.exit(0)
for line in path.read_text().splitlines():
    try:
        ev = json.loads(line)
    except Exception:
        continue
    if ev.get("trade_date") == today or str(ev.get("ts", ""))[:10] == today:
        print(line)
PY
