"""core/paths.py — Canonical filesystem paths. No project imports."""
import os
from pathlib import Path

ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG_DIRECTORY  = os.path.join(ROOT_DIR, "cfg")
DATA_DIRECTORY = os.path.join(ROOT_DIR, "data")
CONFIG_FILE    = os.path.join(CFG_DIRECTORY, "config.yaml")
RATIOS_FILE    = os.path.join(CFG_DIRECTORY, "ratios.yaml")

# Path-object aliases for convenience in code that uses / operator
DATA_DIR = Path(DATA_DIRECTORY)
CFG_DIR  = Path(CFG_DIRECTORY)

os.makedirs(DATA_DIRECTORY, exist_ok=True)
