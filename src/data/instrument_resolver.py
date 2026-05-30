"""
data/instrument_resolver.py — Cached Robinhood instrument-ID → ticker resolver.

Robinhood payloads (e.g. news ``related_instruments``) reference securities by
instrument UUID, not ticker. Resolving each is a live API call, so we persist a
durable id→symbol map to disk and only fetch unknown IDs. This lives in the data
layer so any ETL step (news co-mention graph, etc.) can turn instrument IDs into
symbols without ad-hoc live calls in business logic.

Cache: data/instrument_ids.csv  (columns: instrument_id, symbol)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_RB_INSTRUMENT_URL = "https://api.robinhood.com/instruments/{iid}/"

# In-process memo on top of the on-disk cache (avoids re-reading the CSV per call).
_MEMO: dict[str, str | None] = {}
_LOADED = False


def _data_dir() -> Path:
    from core.paths import DATA_DIR
    return DATA_DIR


def _cache_path() -> Path:
    return _data_dir() / "instrument_ids.csv"


def _load_cache() -> None:
    global _LOADED
    if _LOADED:
        return
    path = _cache_path()
    if path.exists():
        try:
            df = pd.read_csv(path, dtype=str)
            for _, row in df.iterrows():
                iid = str(row.get("instrument_id", "")).strip()
                sym = row.get("symbol")
                if iid:
                    _MEMO[iid] = (str(sym).strip() or None) if pd.notna(sym) else None
        except Exception as exc:
            logger.warning("Could not load instrument_ids.csv: %s", exc)
    _LOADED = True


def _save_cache() -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows = [{"instrument_id": k, "symbol": v} for k, v in _MEMO.items()]
        pd.DataFrame(rows, columns=["instrument_id", "symbol"]).to_csv(path, index=False)
    except Exception as exc:
        logger.warning("Could not save instrument_ids.csv: %s", exc)


def _normalize_id(instrument_ref: str) -> str:
    """Accept a bare UUID or a full instrument URL; return the bare UUID."""
    ref = str(instrument_ref).strip().rstrip("/")
    if "/" in ref:
        ref = ref.rsplit("/", 1)[-1]
    return ref


def resolve_symbols(instrument_refs: list[str], auto_fetch: bool = True) -> dict[str, str]:
    """Resolve a list of instrument IDs / URLs to {instrument_id: symbol}.

    Unknown IDs are fetched from Robinhood (when auto_fetch and logged in) and
    persisted. IDs that resolve to nothing are cached as None so we don't refetch.
    Returns only the entries that resolved to a non-empty symbol.
    """
    _load_cache()
    ids = [_normalize_id(r) for r in instrument_refs if r]
    unknown = [i for i in ids if i not in _MEMO]

    if unknown and auto_fetch:
        try:
            import robin_stocks.robinhood as rb
            for iid in unknown:
                sym = None
                try:
                    inst = rb.get_instrument_by_url(_RB_INSTRUMENT_URL.format(iid=iid))
                    if isinstance(inst, dict):
                        sym = inst.get("symbol") or None
                except Exception as exc:
                    logger.debug("instrument resolve failed for %s: %s", iid, exc)
                _MEMO[iid] = sym
            _save_cache()
        except Exception as exc:
            logger.debug("instrument resolver unavailable: %s", exc)

    out: dict[str, str] = {}
    for iid in ids:
        sym = _MEMO.get(iid)
        if sym:
            out[iid] = sym
    return out
