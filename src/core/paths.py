"""core/paths.py — Canonical filesystem paths. No project imports."""
import os
from pathlib import Path

ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG_DIRECTORY  = os.path.join(ROOT_DIR, "cfg")
DATA_DIRECTORY = os.path.join(ROOT_DIR, "data")
CONFIG_FILE    = os.environ.get(
    "DAILY_INVESTOR_CONFIG",
    os.path.join(CFG_DIRECTORY, "config.yaml"),
)
RATIOS_FILE    = os.path.join(CFG_DIRECTORY, "ratios.yaml")

# Path-object aliases for convenience in code that uses / operator
DATA_DIR = Path(DATA_DIRECTORY)
CFG_DIR  = Path(CFG_DIRECTORY)

# 0DTE storage split: secrets/config stay in the home dir (so Hermes/MCP's hands-off
# auth keeps working untouched); all DATA lives under the app's data/ tree so the UI
# can read it alongside the rest of the app's artifacts.
ODTE_SECRETS_DIR = os.path.expanduser("~/0dte")              # config.json, reddit_token, daily_thread_id, controller_policy — NEVER created here
ODTE_DATA_DIR    = os.path.join(DATA_DIRECTORY, "odte")       # journal, position/watchdog state, triggers/decisions
ODTE_REPORT_DIR  = os.path.join(ODTE_DATA_DIR, "reports")     # gamma-map / fmp-context / journal artifacts
ODTE_SCRAPE_DIR  = os.path.join(ODTE_DATA_DIR, "scrape")      # timestamped reddit/x analyzed-text snapshots

os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(ODTE_REPORT_DIR, exist_ok=True)
os.makedirs(ODTE_SCRAPE_DIR, exist_ok=True)


def atomic_write_text(path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (write to a sibling tmp file, fsync, then os.replace).

    A crash mid-write leaves the OLD file intact instead of a truncated/partial one — used for the
    0DTE watchdog/position state + trigger/decision handoff, where a corrupt JSON read by the next
    poll could otherwise drop or duplicate a trigger. os.replace is atomic on the same filesystem.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
