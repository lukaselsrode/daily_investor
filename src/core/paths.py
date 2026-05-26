"""core/paths.py — Canonical filesystem paths. No project imports."""
import os

ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG_DIRECTORY  = os.path.join(ROOT_DIR, "cfg")
DATA_DIRECTORY = os.path.join(ROOT_DIR, "data")
CONFIG_FILE    = os.path.join(CFG_DIRECTORY, "config.yaml")
RATIOS_FILE    = os.path.join(CFG_DIRECTORY, "ratios.yaml")

os.makedirs(DATA_DIRECTORY, exist_ok=True)
