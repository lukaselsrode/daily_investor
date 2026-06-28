"""tests/test_odte_concurrency.py — bounded, fail-closed per-ticker fetch seam.

No network. Proves bounded_gather preserves order, wraps failures/timeouts fail-closed into a
partial-result payload, and (critically) that one stuck item does not block the others past the
shared deadline.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.odte_concurrency import bounded_gather


def test_empty_items_returns_empty():
    assert bounded_gather(lambda x: x, []) == []


def test_results_preserve_input_order_and_ok_status():
    out = bounded_gather(lambda x: x * 2, [1, 2, 3], max_workers=3)
    assert [r["item"] for r in out] == [1, 2, 3]
    assert [r["result"] for r in out] == [2, 4, 6]
    assert all(r["ok"] and r["status"] == "ok" for r in out)


def test_per_item_failure_is_fail_closed_and_isolated():
    def fn(x):
        if x == "bad":
            raise ValueError("boom")
        return x.upper()
    out = bounded_gather(fn, ["a", "bad", "c"], max_workers=3)
    by = {r["item"]: r for r in out}
    assert by["a"]["ok"] and by["a"]["result"] == "A"
    assert by["c"]["ok"] and by["c"]["result"] == "C"
    assert by["bad"]["ok"] is False and by["bad"]["result"] is None
    assert by["bad"]["status"].startswith("error:")


def test_one_stuck_item_does_not_block_others_past_deadline():
    """A single hung item must time out at the shared deadline while fast items still return ok —
    the whole gather must finish well under the stuck item's natural duration."""
    def fn(x):
        if x == "stuck":
            time.sleep(2.0)        # far longer than the deadline below
            return "late"
        return x

    start = time.monotonic()
    out = bounded_gather(fn, ["stuck", "fast1", "fast2"], max_workers=3, timeout_s=0.3)
    elapsed = time.monotonic() - start
    assert elapsed < 1.5, f"stuck item blocked the gather ({elapsed:.2f}s)"
    by = {r["item"]: r for r in out}
    assert by["stuck"]["ok"] is False and by["stuck"]["status"] == "timeout"
    assert by["fast1"]["ok"] and by["fast2"]["ok"]


def test_max_workers_is_bounded():
    # never raises with workers capped below item count; all items still processed
    out = bounded_gather(lambda x: x, list(range(10)), max_workers=2)
    assert len(out) == 10 and all(r["ok"] for r in out)
