# AGENTS.md — Architecture Contract for Daily Investor

Read this before making changes to the codebase. It describes ownership boundaries, safety rules, and the definition of done.

---

## Project Purpose

Daily Investor is a portfolio research, scoring, backtesting, and validation system built on top of Robinhood. It is **not financial advice**. It is a tool for disciplined, reproducible portfolio decision support.

---

## Module Ownership

| Package | Owns |
|---------|------|
| `core/` | Shared types (`TradeRecord`, `SellDecision`, `PositionSnapshot`), path constants, logging setup, generic utilities (`safe_float`, `run_async`) |
| `config/` | Config schema (frozen dataclasses), config loading, singleton `ConfigManager`, validation |
| `strategy/` | Production factor scoring: value, quality, income, momentum, composite, regimes, factor interactions, snapshots |
| `portfolio/` | Portfolio state, sleeves, risk rules, buy/sell/trim/harvest decisions, position rationale, exposure analysis |
| `backtesting/` | Historical simulation engine, random-window walk-forward testing, cluster tracking (walk-forward only), backtest artifacts/results/types, validation |
| `tuning/` | Parameter search, objective functions, tuning results/reports, stability summaries. Calls `backtesting/`, never duplicates its logic |
| `research/` | Offline analyses and calibration only. IC analytics (`ic_engine.py`), distribution/regime analysis. Read-only — never writes config or orders |
| `reporting/` | Reusable report generation, plots, attribution summaries. No business logic |
| `execution/` | Broker/paper/live execution adapters only |
| `ui/` | Streamlit rendering only. UI components should call `ui/services/` or domain engines — never reimplement business logic |
| `ui/services/` | Thin workflow/service layer. Orchestrates calls from UI and CLI into `backtesting/`, `tuning/`, `portfolio/`. The preferred seam for UI decoupling |
| `cli/` | Command parsing and help text. Calls services or domain modules — not low-level internals |

---

## Import Boundary Rules

These must never be violated:

1. **No `streamlit` in core packages.** `backtesting`, `strategy`, `portfolio`, `tuning`, `config`, `research`, `reporting`, `execution` must never import `streamlit`.
2. **No `ui` imports in core packages.** Core packages must not import from `ui/`.
3. **`core/` is the base layer.** It must not import from `strategy/`, `portfolio/`, `backtesting/`, `tuning/`, `research/`, `reporting/`, `ui/`, or `cli/`.
4. **`research/` is read-only.** It must never write config files, orders, or live data.
5. **All UI imports are lazy.** UI components import from core modules inside functions (not at module level) — this is intentional Streamlit practice.

---

## Canonical Backtest Execution Path

All "run a single backtest" calls in UI and CLI should go through:

```python
from backtesting.engine import BacktestEngine
result = BacktestEngine().run(n_days=..., mode=..., params=...)
```

Or via the service wrapper:

```python
from ui.services.backtest_service import run_single_backtest
result = run_single_backtest(n_days=..., mode=..., params=...)
```

The only justified exception is `ablation_runner.py`, which calls `run_simulation()` directly because it needs raw simulation with params built from arbitrary config files. This bypass is documented inline.

Random-window execution goes through:

```python
from ui.services.backtest_service import run_random_windows
summary = run_random_windows(n_days=..., n_windows=..., window_days=..., mode=...)
```

---

## Backtesting Safety Rules

1. **No lookahead bias.** All rolling features (`rs_3m`, `vol_3m`, `return_1m`, etc.) must be computed from data available only up to day `d`. Cross-sectional percentile ranking at one point in time is not lookahead.
2. **Walk-forward cluster fitting only.** When `cluster_tracker.py` is used in backtests, PCA+KMeans must be fitted only from data available at each rebalance date — never from future data.
3. **Backtest modes:** `liquid_universe_full` (MEDIUM lookahead), `walk_forward_price_only_test` (LOW lookahead), `current_universe_stress_test` (HIGH — not predictive, stress test only).

---

## Config Safety Rules

1. **`cfg/config.yaml`** is the production config. Never overwrite it from UI code except through the explicit write-gate in `config_viewer.py` (requires `ui.allow_config_writes: true` and a user-typed confirmation phrase).
2. **`cfg/config_research_safe.yaml`** is the writable research config. Auto-tune writes here by default unless `--apply` is passed.
3. **Config hash.** `BacktestReport.config_hash` is a SHA-256[:12] of the canonical JSON of the config dict at run time. Use it to track which config produced which result.

---

## Scoring Model Conventions (peer-3, 2026-08)

- **Factor ownership is exclusive:** dividends → income, valuation (PE/PB + distress) → value, statement fundamentals → quality, price geometry/trend → momentum, tradability → the reliability layer + hard volume gates. Do not add a raw input to a second factor — cross-factor correlation is the failure mode peer-3 removed.
- **Rescore is regime-NEUTRAL.** `snapshots rescore` calls `compute_metric` without a regime, while live snapshots are regime-tilted. A rescored file is intentionally not bit-comparable to the live snapshot of the same day.
- **Snapshot store guards.** `save_snapshot` refuses frames below `MIN_SNAPSHOT_ROWS` or missing `symbol`/`value_metric`; malformed parquets are skipped by `list_snapshots`/`rescore_snapshots` and belong in `data/snapshots_quarantine/`.
- **Absolute score thresholds are scale-bound.** Any engine change that shifts score distributions must percentile-remap `metric_threshold`, `entry_threshold_override` (+fallbacks), the sell floors, DAE floors, and archetype floors (`scripts/recalibrate_peer3_thresholds.py` is the template) and re-verify tuner BOUNDS containment (`tests/test_bounds_containment.py`).
- **Thresholds map on the population their gate filters.** Entry thresholds run AFTER the manager's liquidity pre-filter, so they must be percentile-matched on the liquidity-eligible subset — full-universe mapping starved the eligible pool 225 → 36 when peer-3 stopped burying illiquid ADRs inside quality (corrected 2026-08-06). Raw universe rankings (UI, snapshots) legitimately surface untradeable names; the "Tradeable only" view in the Scoring Explorer mirrors the buy path's gate.
- **Liquidity gate = share ADV floor AND `risk.min_dollar_volume` (dollar_vol_21d).** Share counts alone are price-skewed (sub-$1 names pass 500k shares on ~$1.6k/day of turnover). Tradability stays a hard GATE, never a score input or a sizing tilt — `size_by_dollar_volume` remains off (market impact is irrelevant at this account's $5–40 orders, and cap-proxy sizing drags the experimental sleeve toward the SPY core).
- **Sim/live parity:** the simulator's value factor reproduces live `apply_value` via the pit_precompute component panels + additive penalty panel (`tests/test_sim_live_value_parity.py`); `benchmark_blend` stays live-only (IC-validated). Param slot 49 (`momentum_residual_blend`) is sim-only and permanently frozen; slot 48 (`quality_low_vol_blend`) is implemented on both sides.

---

## Service-Layer Guidance

The service layer (`ui/services/`) is the preferred interface between UI/CLI and core engines. Use it when:
- Multiple UI components call the same engine in the same way
- You want to make the engine call testable without starting Streamlit

Services are thin wrappers — they do not contain business logic. They orchestrate: load data → call engine → return result.

---

## research/ Module Rules

- `research/` is for **offline evaluation and calibration only**.
- All analysis is read-only (never writes config or orders).
- `ic_engine.py` (IC analytics) lives in `research/`, not `strategy/`. The compat re-export `from strategy.research import FactorResearchEngine` is kept for backward compatibility but should not be used in new code.
- `strategy/` contains only production scoring logic.

---

## README / Makefile / CLI Consistency Rule

**Whenever you add, rename, or remove a CLI command or Streamlit entrypoint:**

1. Update `README.md` — the CLI table and any workflow instructions
2. Update `Makefile` — the relevant target
3. Update `src/cli/main.py` — the help text / command dispatch
4. Update tests if CLI dispatch is tested

A command that exists in code but not in README is undocumented. A command in README that no longer exists is misleading. Both are bugs.

---

## Code Quality Tools

Run before opening a PR:

```bash
make hygiene          # lint + architecture contracts (blocking)
make type-check       # mypy type check (advisory — fix errors incrementally)
make complexity       # radon complexity report (advisory — note grade D+ files)
make dead-code        # vulture dead-code scan (advisory — review before deleting)
```

| Tool | Config | Enforcement |
|------|--------|------------|
| ruff | `[tool.ruff]` in pyproject.toml | Blocking pre-commit hook + `make lint` |
| mypy | `[tool.mypy]` in pyproject.toml | Advisory: `make type-check`; excludes `ui/` |
| import-linter | `[tool.importlinter]` in pyproject.toml | Blocking pre-commit hook + `make arch-check` |
| vulture | `[tool.vulture]` in pyproject.toml | Advisory: `make dead-code`; whitelist in `vulture_whitelist.py` |
| radon | CLI only | Advisory: `make complexity` |

Other useful commands:

```bash
make test          # pytest tests/ -q
make format        # ruff format src/
make ui            # launch Streamlit dashboard
```

Architecture boundary tests:

```bash
pytest tests/test_architecture.py -v
```

### Pre-commit setup (first time)

```bash
make pre-commit-install
```

### Import-linter contracts

Enforce that bottom-layer packages (`core`, `config`, `data`, `execution`) cannot import from
higher domain layers. Run `make arch-check` to verify. Add new contracts incrementally in
`[tool.importlinter]` as the architecture grows — see `lint-imports --help` for contract types.
When a contract fails, fix the source code rather than weakening the contract; if an exception is
truly necessary, document it with `ignore_imports` and a comment explaining why.

### Expanding type coverage

`make type-check` checks all core packages except `ui/` and `util.py`. To expand coverage,
remove entries from the `exclude` list in `[tool.mypy]` and fix errors incrementally.
Never add `# type: ignore` without a brief inline comment explaining why.

---

## Definition of Done

A change is done when:

- [ ] Behavior is preserved (tests pass, same pass count as before)
- [ ] No new import boundary violations
- [ ] README.md is accurate (commands, module paths, workflows)
- [ ] Makefile targets point to real modules/commands
- [ ] CLI help reflects actual command structure
- [ ] Architecture tests pass
- [ ] No new `streamlit` imports in core packages
- [ ] No new direct `run_simulation()` calls from UI (except `ablation_runner.py`)
