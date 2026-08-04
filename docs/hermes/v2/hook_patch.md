# Pre-order hook patch — `~/.hermes/hooks/odte_order_guard_hook.py`

One deletion. The hook currently enforces BOTH the lease ceilings (correct, keep) and a flat
`ABSOLUTE_MAX_DEBIT = 120.0` (retired 2026-08-02 — sizing is BP-proportional and tiered inside the
lease; the flat rail double-blocks compliant orders, e.g. any full-tier debit between $120 and
0.60×BP).

## Delete

1. Line ~24:

```python
ABSOLUTE_MAX_DEBIT = 120.0
```

2. Its enforcement block (lines ~175–179), the check of order debit against `ABSOLUTE_MAX_DEBIT`.

## Keep (unchanged)

- The lease validation (lines ~164–173): order `limit_price` ≤ `lease.max_limit_price`, order
  debit ≤ `lease.max_debit`, identity match, live unexpired lease required for any opening order.
- The NVDA block and the fail-closed default.

## Verify

- An opening order with a valid lease whose debit is $121–$208 (inside 0.60×BP for the current
  account) passes the hook.
- An order exceeding `lease.max_limit_price` or `lease.max_debit` is still blocked.
- An opening order with no live matching lease is still blocked.
