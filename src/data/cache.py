"""
data/cache.py — DataCache: filesystem-backed CSV cache.

Module-level functions (store_data_as_csv, read_data_as_pd) are the canonical
implementations. DataCache is the class-based adapter.
"""

from __future__ import annotations

import csv
import datetime
import logging
import os

import pandas as pd

from core.paths import DATA_DIRECTORY

logger = logging.getLogger(__name__)


def _dated_filename(dataset: str) -> str:
    dt_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    return os.path.join(DATA_DIRECTORY, f"{dataset}_{dt_str}.csv")


def store_data_as_csv(
    dataset: str,
    schema: list[str],
    data: list[list] | pd.DataFrame,
    add_timestamp: bool = True,
) -> None:
    filename = (
        _dated_filename(dataset) if add_timestamp
        else os.path.join(DATA_DIRECTORY, f"{dataset}.csv")
    )

    if isinstance(data, pd.DataFrame):
        data.to_csv(filename, index=False)
        logger.info("Stored %s → %s", dataset, filename)
        return

    if not data:
        logger.warning("store_data_as_csv called with empty data for %s", dataset)
        return

    row_len = len(data[0])
    if len(schema) != row_len:
        raise ValueError(f"Schema length {len(schema)} != data row length {row_len}")
    if not all(len(r) == row_len for r in data):
        raise ValueError("Mismatched row lengths in data")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(schema)
        writer.writerows(data)

    logger.info("Stored %s → %s", dataset, filename)


# pandas' default missing-value tokens MINUS the bare "NA": the universe contains
# the real ticker "NA" (Nano Labs), which the default parser silently turns into
# float NaN on every cached read — crashing sorted() over the symbol set with
# "'<' not supported between instances of 'float' and 'str'" and corrupting any
# symbol-keyed merge. Our own CSVs (pandas to_csv / csv.writer) always write
# missing values as the empty string, never the token "NA", so dropping it from
# the NaN list cannot mask genuine missing data.
_NA_TOKENS = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN", "None", "n/a", "nan", "null",
]


def read_data_as_pd(dataset: str) -> pd.DataFrame | None:
    """Return the most-recent matching CSV for dataset, or None if not found."""
    try:
        files = sorted(os.listdir(DATA_DIRECTORY))
    except FileNotFoundError:
        return None

    matches = [f for f in files if dataset in f and f.endswith(".csv")]
    if not matches:
        logger.debug("No CSV found for dataset '%s' in %s", dataset, DATA_DIRECTORY)
        return None

    path = os.path.join(DATA_DIRECTORY, matches[-1])
    logger.debug("Using %s as %s data", matches[-1], dataset)
    return pd.read_csv(path, keep_default_na=False, na_values=_NA_TOKENS)


