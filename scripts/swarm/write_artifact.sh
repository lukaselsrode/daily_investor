#!/usr/bin/env bash
# The swarm's ONLY write path. Everything else the swarm touches is read-only.
#
# Usage: write_artifact.sh <kind> [--latency SECONDS] [--date YYYY-MM-DD]   (content on stdin)
#   kind: conviction    -> data/odte/swarm/conviction.json       (JSON, schema-validated)
#         premarket     -> data/odte/swarm/premarket_brief.json  (JSON, schema-validated)
#         premarket-md  -> data/odte/swarm/premarket_brief.md    (markdown passthrough)
#         eod-md        -> data/odte/swarm/eod_review_<date>.md  (markdown, --date required)
#         lessons       -> data/odte/swarm/lessons.md            (markdown passthrough)
#
# Enforced invariants:
#   * destination is ALWAYS under data/odte/swarm/ — no caller-supplied paths exist
#   * JSON kinds: required keys + conviction enum + candidate matches "TICKER:direction"
#   * NVDA candidate_key refused (4th backstop; 3 repo code sites already enforce)
#   * advisory:true / execution_allowed:false are FORCED on every JSON artifact
#   * atomic write (tmp + os.replace) — a crash never leaves a truncated artifact
set -euo pipefail
cd "$(dirname "$0")/../.."

KIND="${1:?usage: write_artifact.sh <kind> [--latency N] [--date YYYY-MM-DD]}"
shift
LATENCY=""
DATE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --latency) LATENCY="${2:?--latency needs a value}"; shift 2 ;;
    --date)    DATE="${2:?--date needs a value}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Buffer stdin to a temp file: the python program itself arrives via stdin (heredoc),
# so the artifact body must travel out-of-band.
BODY_FILE="$(mktemp "${TMPDIR:-/tmp}/swarm_artifact.XXXXXX")"
trap 'rm -f "$BODY_FILE"' EXIT
cat > "$BODY_FILE"

KIND="$KIND" LATENCY="$LATENCY" DATE="$DATE" BODY_FILE="$BODY_FILE" python3 - <<'PY'
import json, os, re, sys
from datetime import datetime, timezone
from pathlib import Path

SWARM = Path("data/odte/swarm")
kind = os.environ["KIND"]
latency = os.environ.get("LATENCY") or None
date = os.environ.get("DATE") or None
body = Path(os.environ["BODY_FILE"]).read_text()

def fail(msg):
    print(json.dumps({"ok": False, "error": msg}), file=sys.stderr)
    sys.exit(1)

def atomic(dest: Path, text: str):
    SWARM.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()
    if SWARM.resolve() not in dest.parents:
        fail(f"destination escapes data/odte/swarm/: {dest}")
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, dest)
    print(json.dumps({"ok": True, "path": str(dest.relative_to(Path.cwd()))}))

now = datetime.now(timezone.utc).isoformat(timespec="seconds")

if kind in ("conviction", "premarket"):
    try:
        doc = json.loads(body)
    except Exception as exc:
        fail(f"invalid JSON: {exc}")
    if not isinstance(doc, dict):
        fail("artifact must be a JSON object")
    if kind == "conviction":
        required = ("candidate_key", "direction", "conviction", "top_reasons",
                    "dissent", "sources_checked")
        missing = [k for k in required if k not in doc]
        if missing:
            fail(f"missing keys: {missing}")
        if doc["conviction"] not in ("high", "medium", "low"):
            fail(f"conviction must be high|medium|low, got {doc['conviction']!r}")
        if not re.fullmatch(r"[A-Z]{1,6}:(bullish|bearish)", str(doc["candidate_key"])):
            fail(f"candidate_key must be TICKER:direction, got {doc['candidate_key']!r}")
        if doc["candidate_key"].split(":", 1)[0] == "NVDA":
            fail("NVDA is employer-restricted; refusing to write")
        for k in ("top_reasons", "dissent", "sources_checked"):
            if not isinstance(doc[k], list):
                fail(f"{k} must be a list")
        doc["schema"] = "swarm_conviction_v1"
        dest = SWARM / "conviction.json"
    else:
        required = ("summary", "overnight_risks", "carryover_lessons")
        missing = [k for k in required if k not in doc]
        if missing:
            fail(f"missing keys: {missing}")
        doc["schema"] = "swarm_premarket_v1"
        dest = SWARM / "premarket_brief.json"
    doc["generated_at"] = now
    if latency is not None:
        doc["latency_seconds"] = float(latency)
    # Hard invariants — mirror the watchdog's scan_only convention; never caller-settable.
    doc["advisory"] = True
    doc["execution_allowed"] = False
    atomic(dest, json.dumps(doc, indent=2) + "\n")
elif kind == "premarket-md":
    atomic(SWARM / "premarket_brief.md", body)
elif kind == "eod-md":
    if not date or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        fail("eod-md requires --date YYYY-MM-DD")
    atomic(SWARM / f"eod_review_{date}.md", body)
elif kind == "lessons":
    atomic(SWARM / "lessons.md", body)
else:
    fail(f"unknown kind: {kind}")
PY
