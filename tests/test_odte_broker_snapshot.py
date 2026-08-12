"""tests/test_odte_broker_snapshot.py — the broker snapshot builder (local/offline).

Fixtures are TRIMMED COPIES OF REAL `get_portfolio` / `get_option_positions` / `get_option_orders`
payloads captured on 2026-08-12, not hand-written shapes. That distinction is the point: the first
contract builder was written against guessed shapes and failed live with build_rc: 2, and a guessed
fixture only proves the parser agrees with the guess. Note in particular that `data.buying_power`
is a DICT whose own `buying_power` is a STRING — the two-step lookup exists for that.
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.odte_snapshot_build import build_broker_snapshot

NOW = datetime(2026, 8, 12, 18, 22, 5, tzinfo=timezone.utc)
ACCT = "435050133"


def _portfolio(bp="411.3200"):
    return {"data": {"total_value": "437.26",
                     "buying_power": {"buying_power": bp,
                                      "unleveraged_buying_power": bp,
                                      "display_currency": "USD"}},
            "guide": "..."}


def _positions(*quantities):
    return {"data": {"positions": [
        {"chain_symbol": "IWM", "option_id": f"opt-{i}", "quantity": q,
         "average_price": "46.0000", "expiration_date": "2026-08-12"}
        for i, q in enumerate(quantities)]}}


def _orders(*rows):
    return {"data": {"orders": [
        {"id": f"ord-{i}", "chain_symbol": "IWM", "state": state, "quantity": "1.00000",
         "created_at": created, "updated_at": created, "direction": "debit"}
        for i, (state, created) in enumerate(rows)]}}


def test_reproduces_the_clients_own_broker_truth_on_real_payloads():
    """The composition is lifted from OdteMcpClient.broker_truth; this pins the equivalence that
    was verified field-by-field against that method's recorded output on 2026-08-12."""
    snap = build_broker_snapshot(
        _portfolio(), _positions("0.0000", "0.0000"),
        _orders(("filled", "2026-08-12T15:13:42.116639Z"),
                ("filled", "2026-08-12T15:01:35.109946Z")),
        account_number=ACCT, now=NOW, source="odte_mcp_client.broker_truth")
    assert snap["buying_power"] == 411.32                  # from the nested STRING
    assert snap["nonzero_option_positions_count"] == 0     # both quantities are "0.0000"
    assert snap["open_option_orders_count"] == 0           # both filled, neither pending
    assert snap["today_option_orders_count"] == 2
    assert snap["account_number"] == ACCT


def test_stamps_a_timestamp_convert_can_read():
    """An absent timestamp is exactly what produced broker_snapshot_undated on hand-authored files."""
    import data.odte_convert as cv
    snap = build_broker_snapshot(_portfolio(), _positions(), _orders(),
                                 account_number=ACCT, now=NOW)
    assert cv._payload_ts(snap) is not None


def test_counts_only_nonzero_positions():
    snap = build_broker_snapshot(_portfolio(), _positions("0.0000", "1.0000", "0.0000"),
                                 _orders(), account_number=ACCT, now=NOW)
    assert snap["nonzero_option_positions_count"] == 1


def test_pending_states_count_as_open_orders():
    from data.odte_snapshot_build import PENDING_ORDER_STATES
    state = sorted(PENDING_ORDER_STATES)[0]
    snap = build_broker_snapshot(
        _portfolio(), _positions(),
        _orders((state, "2026-08-12T15:00:00Z"), ("filled", "2026-08-12T15:00:00Z")),
        account_number=ACCT, now=NOW)
    assert snap["open_option_orders_count"] == 1
    assert snap["today_option_orders_count"] == 2


def test_yesterdays_orders_are_not_todays():
    snap = build_broker_snapshot(
        _portfolio(), _positions(),
        _orders(("filled", "2026-08-11T15:00:00Z"), ("filled", "2026-08-12T15:00:00Z")),
        account_number=ACCT, now=NOW)
    assert snap["today_option_orders_count"] == 1


def test_missing_buying_power_is_none_not_zero():
    """None refuses the trade at budget_check; 0.0 would read as a real, broke account."""
    snap = build_broker_snapshot({"data": {}}, _positions(), _orders(),
                                 account_number=ACCT, now=NOW)
    assert snap["buying_power"] is None
