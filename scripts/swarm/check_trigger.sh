#!/usr/bin/env bash
# Swarm trigger gate — pure, no LLM, no broker, <100ms.
#
# Fires iff the watchdog's (candidate_key, last_alert_utc) pair differs from what the swarm
# last processed (data/odte/swarm/.last_processed). Keying on watchdog_state.json rather than
# the transient triggers.json payload is race-free against the 1-minute watchdog overwrite:
# state.last_alert_utc only advances on a real alert (new_candidate / candidate_persisting /
# policy), so each alert or 10-minute re-alert yields exactly one fire.
#
# Output: one compact JSON line — {"fire": false, "reason": ...} or
# {"fire": true, "candidate_key", "last_alert_utc", "trigger_types", "payload_path"}.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 - <<'PY'
import json, os, sys
from pathlib import Path

ODTE = Path("data/odte")
SWARM = ODTE / "swarm"
LAST = SWARM / ".last_processed"

def out(obj):
    print(json.dumps(obj, separators=(",", ":")))
    sys.exit(0)

def read(p):
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        return None

state = read(ODTE / "watchdog_state.json")
if state is None:
    out({"fire": False, "reason": "no_watchdog_state"})

key = state.get("candidate_key")
last_alert = state.get("last_alert_utc")
if not key or not last_alert:
    out({"fire": False, "reason": "no_candidate"})
if key.split(":", 1)[0].upper() == "NVDA":
    out({"fire": False, "reason": "restricted_candidate"})

prev = read(LAST) or {}
if prev.get("candidate_key") == key and prev.get("last_alert_utc") == last_alert:
    out({"fire": False, "reason": "already_processed"})

triggers = read(ODTE / "triggers.json") or {}
types = [t.get("type") for t in triggers.get("triggers") or [] if t.get("type")]

SWARM.mkdir(parents=True, exist_ok=True)
tmp = LAST.with_name(LAST.name + ".tmp")
tmp.write_text(json.dumps({"candidate_key": key, "last_alert_utc": last_alert}))
os.replace(tmp, LAST)
out({"fire": True, "candidate_key": key, "last_alert_utc": last_alert,
     "trigger_types": types, "payload_path": str(ODTE / "triggers.json")})
PY
