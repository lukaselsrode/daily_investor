"""data/odte_concurrency.py — tiny bounded, fail-closed concurrency seam for per-ticker fetch.

The 0DTE scan enriches several tickers, each with its own independent network fetch (price, chain).
Doing that in a plain sequential loop means one slow/hung ticker stalls the whole scan. This module
provides ONE small primitive — `bounded_gather` — that runs a per-item function across a bounded
thread pool with a SHARED deadline, so:

  • concurrency is capped (no unbounded fan-out / API hammering),
  • each item is wrapped fail-closed (an exception or timeout yields a status, never an exception),
  • one stuck item cannot block the others past the shared deadline,
  • the caller always gets a PARTIAL-RESULT payload (ok/timeout/error per item) it can journal.

NO broker, NO LLM, NO orders. It only schedules the callable the caller passes in. It is decision-
support plumbing: it never decides anything and never makes a scan tier into an execution tier.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FTimeout
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 4
DEFAULT_TIMEOUT_S = 20.0


def bounded_gather(fn: Callable, items, *, max_workers: int = DEFAULT_MAX_WORKERS,
                   timeout_s: float = DEFAULT_TIMEOUT_S) -> list[dict]:
    """Run ``fn(item)`` over ``items`` concurrently, bounded + fail-closed, preserving input order.

    Returns one result dict per item: ``{"item", "ok", "result", "status"}`` where status is
    ``"ok" | "timeout" | "error: <msg>"``. NEVER raises. A SHARED deadline (`timeout_s` from the
    start) bounds total wall-clock, so a hung item can't extend the run item-by-item. Pool shutdown
    does not wait on stuck threads (cancel_futures), so the caller is never blocked by one bad item.
    """
    items = list(items)
    if not items:
        return []
    workers = max(1, min(int(max_workers or 1), len(items)))
    results: list[dict | None] = [None] * len(items)
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        for fut, i in futs.items():
            it = items[i]
            remaining = max(0.0, deadline - time.monotonic())
            try:
                res = fut.result(timeout=remaining)
                results[i] = {"item": it, "ok": True, "result": res, "status": "ok"}
            except _FTimeout:
                results[i] = {"item": it, "ok": False, "result": None, "status": "timeout"}
                logger.warning("bounded_gather: item timed out (%s)", it)
            except Exception as exc:                    # fail-closed per item
                results[i] = {"item": it, "ok": False, "result": None, "status": f"error: {exc}"}
                logger.warning("bounded_gather: item failed (%s): %s", it, exc)
    finally:
        # Do NOT block on a stuck worker thread; drop pending work.
        ex.shutdown(wait=False, cancel_futures=True)
    return [r for r in results if r is not None]
