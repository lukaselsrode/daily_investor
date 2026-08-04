#!/usr/bin/env bash
# Swarm context bundle — READ-ONLY, best-effort (missing pieces never kill the snapshot).
# One call so the conviction lead spends its time budget thinking, not shelling out.
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "=== triggers.json ==="
cat data/odte/triggers.json 2>/dev/null || echo "(missing)"

echo "=== newest gamma map ==="
g=$(ls -t data/odte/*gamma*.json 2>/dev/null | head -1)
if [ -n "${g:-}" ]; then echo "path: $g"; cat "$g"; else echo "(none)"; fi

echo "=== loop status (offline) ==="
make -s odte-loop-status JSON=1 OFFLINE=1 2>/dev/null || echo "(unavailable)"

echo "=== last 30 journal events ==="
tail -30 data/odte/decision_journal.jsonl 2>/dev/null || echo "(missing)"
