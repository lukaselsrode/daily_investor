"""
cli/main.py — Thin CLI dispatcher.

New-style invocation:
  python -m cli run
  python -m cli backtest 365
  python -m cli auto-tune 180 --apply
  python -m cli tune 120 --objective calmar
  python -m cli stability-scan
  python -m cli report

Old-style invocation via src/main.py is preserved for backward compatibility.

Dispatch is table-driven: each command's body lives in a `_cmd_<name>(rest)`
handler and `_COMMANDS` maps command names (including aliases) to handlers.
Handlers import from `cli.commands` lazily (at call time) so tests can patch
`cli.commands.cmd_*` before invoking `main`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

from core.logging import configure_logging


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    # Load .env so EVERY subcommand has API keys/creds (FMP_KEY, etc.). Previously only
    # src/main.py (the live-trading entry) loaded it, so `fmp`/`tune`/`backtest` ran
    # without FMP_KEY and silently failed every fetch.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return

    cmd = args[0]
    rest = args[1:]

    # --skip-fetch-news: reuse the most-recent cached news scrape instead of the
    # (slow) full news fetch, even on a fresh-data run. Surfaced as an env var so
    # the data layer (data/news.py) honors it without coupling to sys.argv.
    if "--skip-fetch-news" in rest:
        os.environ["SKIP_FETCH_NEWS"] = "1"

    _apply_config_override(rest)

    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd!r}")
        _print_help()
        sys.exit(1)
    handler(rest)


def _apply_config_override(rest: list[str]) -> None:
    """--config <path> — override which YAML the run reads.

    Must be applied BEFORE importing cli.commands so core/paths.CONFIG_FILE
    picks up the override at import time.
    """
    _cfg_override = _flag_value(rest, "--config")
    if _cfg_override:
        if not os.path.isabs(_cfg_override):
            _cfg_override = os.path.abspath(_cfg_override)
        if not os.path.isfile(_cfg_override):
            print(f"--config: file not found: {_cfg_override}")
            sys.exit(2)
        os.environ["DAILY_INVESTOR_CONFIG"] = _cfg_override
        print(f"[config override] {_cfg_override}")


def _cmd_list_presets(rest: list[str]) -> None:
    from cli.commands import cmd_list_presets
    cmd_list_presets()


def _cmd_fetch_data(rest: list[str]) -> None:
    from cli.commands import cmd_fetch_data
    cmd_fetch_data()


def _cmd_run(rest: list[str]) -> None:
    from cli.commands import cmd_run
    skip_data = "--skip-data" in rest
    op_mode = _flag_value(rest, "--op-mode")
    cmd_run(skip_data=skip_data, op_mode=op_mode)


def _cmd_backtest(rest: list[str]) -> None:
    from cli.commands import cmd_backtest
    n_days = int(rest[0]) if rest and rest[0].isdigit() else 365
    mode = _flag_value(rest, "--mode")
    compare = "--compare" in rest
    archetype_compare = "--archetype-compare" in rest
    scope = _flag_value(rest, "--scope") or "overall_strategy"
    regime_scope = _flag_value(rest, "--regime-scope") or "all"
    if "--compare-etf-allocation" in rest:
        # ETF sleeve before/after: forced equal-weight vs the current config allocation.
        from backtesting.data_loader import load_and_precompute
        from backtesting.reports import format_etf_sleeve_diagnostics
        from backtesting.simulator import run_backtest_report, split_price_window
        from tuning.constants import _ETF_ENABLED_SLOT, _current_params
        _pc = load_and_precompute(n_days, mode=mode)
        _tr, _vl = split_price_window(_pc.prices.shape[0], 0.70)
        _eq = _current_params().copy()
        _eq[_ETF_ENABLED_SLOT] = 0.0
        _cfg = _current_params().copy()  # reflects config etf_allocation.enabled
        _r_eq = run_backtest_report(_pc, _eq, _tr, _vl, scope="etf_allocation")
        _r_cfg = run_backtest_report(_pc, _cfg, _tr, _vl, scope="etf_allocation")
        print(format_etf_sleeve_diagnostics(
            _r_eq.validation_result or _r_eq.train_result, label="BEFORE: equal-weight"))
        print(format_etf_sleeve_diagnostics(
            _r_cfg.validation_result or _r_cfg.train_result,
            label="AFTER: current config",
            current_weights=(_r_eq.train_result.etf_final_weights or {})))
        return
    cmd_backtest(n_days=n_days, mode=mode, compare=compare,
                 archetype_compare=archetype_compare, scope=scope,
                 regime_scope=regime_scope)


def _cmd_tune(rest: list[str]) -> None:
    from cli.commands import cmd_tune
    if not rest or not rest[0].isdigit():
        print("Usage: tune DAYS [--objective sharpe|calmar|info_ratio] [--scope ...] [--preset ...] [--regime-scope all|bullish|neutral|defensive]")
        sys.exit(1)
    n_days = int(rest[0])
    objective = _flag_value(rest, "--objective") or "sharpe"
    mode = _flag_value(rest, "--mode")
    scope = _flag_value(rest, "--scope") or "overall_strategy"
    preset = _flag_value(rest, "--preset")
    regime_scope = _flag_value(rest, "--regime-scope") or "all"
    cmd_tune(n_days=n_days, objective=objective, mode=mode, scope=scope, preset=preset, regime_scope=regime_scope)


def _cmd_auto_tune(rest: list[str]) -> None:
    from cli.commands import cmd_auto_tune
    n_days = int(rest[0]) if rest and rest[0].isdigit() else 90
    mode = _flag_value(rest, "--mode")
    apply = "--apply" in rest
    force_apply = "--force-apply" in rest
    llm_review = "--llm-review" in rest
    scope = _flag_value(rest, "--scope") or "overall_strategy"
    preset = _flag_value(rest, "--preset")
    regime_scope = _flag_value(rest, "--regime-scope") or "all"
    random_topk = int(_flag_value(rest, "--random-topk") or 0)
    _leads_raw = _flag_value(rest, "--leads")
    lead_vector_paths = [p for p in (_leads_raw or "").split(",") if p] or None
    cmd_auto_tune(n_days=n_days, mode=mode, apply=apply, force_apply=force_apply, llm_review=llm_review, scope=scope, preset=preset, regime_scope=regime_scope, random_topk=random_topk, lead_vector_paths=lead_vector_paths)


def _cmd_tune_etf_allocation(rest: list[str]) -> None:
    _nd = _flag_value(rest, "--days")
    n_days = int(_nd) if _nd else 1250
    universe = _flag_value(rest, "--universe") or "configured_only"
    _mode = _flag_value(rest, "--mode") or "regime"
    if universe == "curated_exploration":
        print("curated_exploration is Milestone B — not yet available. "
              "Use --universe configured_only.")
        return
    preset = "etf_defensive_only" if _mode == "defensive" else "etf_allocation"
    random_topk = int(_flag_value(rest, "--random-topk") or 10)
    apply = "--apply" in rest
    force_apply = "--force-apply" in rest
    from tuning.etf_tune import run_etf_allocation_tune
    run_etf_allocation_tune(n_days=n_days, preset=preset, random_topk=random_topk,
                            apply=apply, force_apply=force_apply)


def _cmd_report_etf_allocation(rest: list[str]) -> None:
    _nd = _flag_value(rest, "--days")
    n_days = int(_nd) if _nd else 1250
    from backtesting.data_loader import load_and_precompute
    from backtesting.reports import format_etf_sleeve_diagnostics
    from backtesting.simulator import run_backtest_report, split_price_window
    from tuning.constants import _current_params
    _pc = load_and_precompute(n_days, mode=None)
    _tr, _vl = split_price_window(_pc.prices.shape[0], 0.70)
    _rep = run_backtest_report(_pc, _current_params(), _tr, _vl, scope="etf_allocation")
    print(format_etf_sleeve_diagnostics(
        _rep.validation_result or _rep.train_result, label=f"current config ({n_days}d)"))


def _cmd_odte_social_report(rest: list[str]) -> None:
    # 0DTE social-sentiment watchlist — ANALYSIS/PAPER ONLY, places no orders.
    # --no-fetch runs offline (cache-only/empty) for a safe dry run.
    # --reddit-bearer-token TOKEN: OPTIONAL ephemeral read-only bearer for the WSB daily-thread
    # comments fetch. Passed straight through as a RUNTIME arg — never stored, logged, or
    # echoed; obtain it manually from your browser/devtools. The tool never reads cookies or
    # mints tokens. Omit it for the default fail-closed OAuth/public behavior.
    # --daily-thread-id / --daily-thread-url: explicit override for the WSB daily-thread when
    # listing/search discovery can't find it. Overlaid into a COPY of the params at runtime —
    # never written back to config.
    from data.social_sentiment import (
        build_odte_social_report,
        format_report,
        format_report_json,
        load_0dte_runtime_config,
    )
    from util import OPTIONS_SOCIAL_PARAMS
    # --json is a machine contract: stdout must be ONLY the JSON. Logs go to stdout in this CLI, so
    # silence them for the run (the diagnostics an agent needs are in the JSON's source statuses).
    if "--json" in rest:
        import logging
        logging.disable(logging.ERROR)
    _bearer = _flag_value(rest, "--reddit-bearer-token")
    _dt_id = _flag_value(rest, "--daily-thread-id")
    _dt_url = _flag_value(rest, "--daily-thread-url")
    _dt_limit = _flag_value(rest, "--daily-thread-limit")
    # Auto-config for hands-off runs (e.g. the Hermes agent): pull creds/thread-id from ~/0dte/
    # (config.json / reddit_token.json / daily_thread_id.txt). Explicit flags always win. Also export
    # REDDIT_CLIENT_ID/SECRET (the robust, non-expiring app-OAuth path) so the comments fetch works
    # without a daily bearer-token refresh. Offline dry runs (--no-fetch) skip all of this.
    if "--no-fetch" not in rest:
        _auto = load_0dte_runtime_config()
        for _env, _key in (("REDDIT_CLIENT_ID", "reddit_client_id"),
                           ("REDDIT_CLIENT_SECRET", "reddit_client_secret")):
            if _auto.get(_key) and not os.environ.get(_env):
                os.environ[_env] = _auto[_key]
        _bearer = _bearer or _auto.get("reddit_bearer_token")
        if not (_dt_id or _dt_url):
            _dt_id = _dt_id or _auto.get("daily_thread_id")
    _params = None
    if _dt_id or _dt_url or _dt_limit:
        _params = {**OPTIONS_SOCIAL_PARAMS}   # shallow copy; global config left untouched
        if _dt_id:
            _params["daily_thread_id"] = _dt_id
        if _dt_url:
            _params["daily_thread_url"] = _dt_url
        if _dt_limit:
            try:
                _params["daily_thread_limit"] = int(_dt_limit)
            except ValueError:
                print(f"--daily-thread-limit: not an integer: {_dt_limit}")
                sys.exit(2)
    rep = build_odte_social_report(allow_fetch="--no-fetch" not in rest,
                                   params=_params, reddit_bearer_token=_bearer)
    # --json: clean, machine-ingestible payload for an agent (signal only, no paper/disclaimer prose).
    print(format_report_json(rep) if "--json" in rest else format_report(rep))


def _cmd_odte_watchdog(rest: list[str]) -> None:
    # Script-only 0DTE watchdog — NO LLM, NO Robinhood, places no orders. Runs the LOCAL report,
    # diffs the actionable candidate vs the prior run, and writes data/odte/{watchdog_state,triggers}.json.
    # The controller policy it checks is a secret read from ~/0dte/ (override with --policy).
    # stdout contract for a no_agent cron: EMPTY when nothing actionable; compact one-line JSON when a
    # trigger fires. --json always prints the compact state. --no-fetch runs offline (cache-only).
    import json
    import logging
    logging.disable(logging.ERROR)   # keep stdout a clean machine contract
    from data.odte_watchdog import DEFAULT_STATE_DIR, run_watchdog
    state_dir = _flag_value(rest, "--state-dir")
    policy = _flag_value(rest, "--policy")
    payload = run_watchdog(state_dir=state_dir or DEFAULT_STATE_DIR,
                           policy_path=policy, allow_fetch="--no-fetch" not in rest)
    if "--json" in rest or payload.get("alert"):
        print(json.dumps(payload, separators=(",", ":"), default=str))


def _cmd_odte_position(rest: list[str]) -> None:
    # BROKER-AWARE but DECISION-ONLY 0DTE position watchdog. Places NO orders, makes NO broker/LLM
    # calls. Reads the active trade plan (data/odte/active_trade.json) + a caller-supplied live snapshot
    # (Hermes feeds real broker/market values via MCP — never faked here), emits TAKE_PROFIT /
    # THESIS_DEAD / BID_FLOOR / TIME_RISK / MONITORING_DEGRADED / HOLD, and writes
    # data/odte/{position_state,position_decision}.json. stdout: EMPTY when HOLD/NO_POSITION (cron form),
    # compact one-line JSON on an actionable decision; --json always prints.
    import json
    import logging
    logging.disable(logging.ERROR)   # keep stdout a clean machine contract
    from data.odte_position import DEFAULT_STATE_DIR, run_position_watchdog
    plan = _flag_value(rest, "--plan")
    snap_path = _flag_value(rest, "--snapshot")
    snap_json = _flag_value(rest, "--snapshot-json")
    state_dir = _flag_value(rest, "--state-dir")
    snapshot = None
    if snap_json is not None:
        try:
            snapshot = json.loads(snap_json)
        except json.JSONDecodeError as exc:
            print(f"--snapshot-json: invalid JSON: {exc}")
            sys.exit(2)
    payload = run_position_watchdog(plan_path=plan, snapshot=snapshot, snapshot_path=snap_path,
                                    state_dir=state_dir or DEFAULT_STATE_DIR)
    if "--json" in rest or payload.get("alert"):
        print(json.dumps(payload, separators=(",", ":"), default=str))


def _cmd_odte_journal(rest: list[str]) -> None:
    # Append one event to the local 0DTE decision journal (data/odte/decision_journal.jsonl). Local/
    # offline — NO broker, NO LLM, NO secrets. Supply the event with --event-json '{...}' or
    # --event PATH (a JSON file). NVDA/employer-restricted underlyings are tagged restricted on
    # store and kept out of experiments/metrics. --json prints the stored event.
    import json

    from data.odte_journal import append_event
    journal = _flag_value(rest, "--journal")
    ev_json = _flag_value(rest, "--event-json")
    ev_path = _flag_value(rest, "--event")
    if ev_json is None and ev_path is None:
        print("odte-journal: provide --event-json '{...}' or --event PATH")
        sys.exit(2)
    try:
        raw = ev_json if ev_json is not None else open(os.path.expanduser(ev_path)).read()
        event = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"odte-journal: could not read event JSON: {exc}")
        sys.exit(2)
    if not isinstance(event, dict):
        print("odte-journal: event JSON must be an object")
        sys.exit(2)
    stored = append_event(event, journal_path=journal)
    if "--json" in rest:
        print(json.dumps(stored, separators=(",", ":"), default=str))
    else:
        print(f"appended {stored.get('event_type')} seq={stored.get('seq')} "
              f"trade={stored.get('trade_id')}")


def _cmd_odte_ingest_artifacts(rest: list[str]) -> None:
    # Fold loose data/odte/*.json controller/watchdog artifacts (controller_*, event_*, candidate_*,
    # market_snapshot_*, *vehicle_score*, *gamma_map*) into the decision journal IDEMPOTENTLY, so a
    # full trading day can be reconstructed for post-day self-eval. Read-only over source files — NO
    # broker, NO LLM, NO orders, never deletes the artifacts. --date YYYY-MM-DD restricts to one day;
    # --dry-run reports what would be appended without writing; --json prints the summary.
    import json

    from data.odte_journal import build_day_packet, ingest_loose_artifacts
    date = _flag_value(rest, "--date")
    journal = _flag_value(rest, "--journal")
    summary = ingest_loose_artifacts(
        data_dir=_flag_value(rest, "--data-dir"),
        journal_path=journal,
        trade_date=date,
        dry_run="--dry-run" in rest)
    # Optional additive day packet (data/odte/days/YYYY-MM-DD/*.jsonl) — derived from the journal,
    # off unless --day-packet; never written on a dry run (nothing was journaled).
    if "--day-packet" in rest and "--dry-run" not in rest:
        summary["day_packet"] = build_day_packet(trade_date=date, journal_path=journal)
    if "--json" in rest:
        print(json.dumps(summary, separators=(",", ":"), default=str))
    else:
        if summary["dry_run"]:
            print(f"[dry-run] scanned {summary['files_scanned']} | would-append "
                  f"{summary['events_would_append']} | duplicates {summary['duplicates_skipped']} | "
                  f"errors {summary['errors']}")
        else:
            print(f"scanned {summary['files_scanned']} | appended {summary['events_appended']} | "
                  f"duplicates {summary['duplicates_skipped']} | errors {summary['errors']}")
        if summary["by_event_type"]:
            print("  by_event_type: " + ", ".join(f"{k}={v}" for k, v in
                                                   sorted(summary["by_event_type"].items())))


def _cmd_odte_journal_report(rest: list[str]) -> None:
    # Summarize the 0DTE decision journal into deterministic metrics + Markdown/CSV artifacts.
    # Local/offline — NO broker, NO LLM. --json prints the metrics payload; default prints Markdown.
    # --write (or --out-dir DIR) writes data/odte/reports/odte_journal_report.md + _summary.csv.
    import json

    from data.odte_journal import build_report
    journal = _flag_value(rest, "--journal")
    out_dir = _flag_value(rest, "--out-dir")
    res = build_report(journal_path=journal, out_dir=out_dir, write_artifacts="--write" in rest)
    if "--json" in rest:
        print(json.dumps({**res["summary"], "artifacts": res["artifacts"]},
                         separators=(",", ":"), default=str))
    else:
        print(res["markdown"])


def _cmd_odte_gamma_map(rest: list[str]) -> None:
    # 0DTE option-chain gamma / pin map — PURE/OFFLINE, NO broker, NO LLM, NO network. Reads option
    # quote rows that Hermes/RH exported to a JSON file (--input PATH) or string (--input-json '...').
    # HONEST: absolute gamma/OI concentration only — labeled gamma_regime=pin_risk_only_not_dealer_gex;
    # it does NOT infer dealer net GEX / gamma flip (RH doesn't expose that). --spot/--underlying/
    # --expiration refine the read; --json prints the map; --write (or --out-dir) writes artifacts.
    import json

    from data.odte_gamma_map import render_markdown, run_gamma_map
    input_path = _flag_value(rest, "--input")
    input_json = _flag_value(rest, "--input-json")
    if input_path is None and input_json is None:
        print("odte-gamma-map: provide --input PATH or --input-json '{...}'")
        sys.exit(2)
    try:
        gmap = run_gamma_map(input_path=input_path, input_json=input_json,
                             spot=_flag_value(rest, "--spot"),
                             underlying=_flag_value(rest, "--underlying"),
                             expiration=_flag_value(rest, "--expiration"),
                             out_dir=_flag_value(rest, "--out-dir"), write="--write" in rest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"odte-gamma-map: could not read/parse input: {exc}")
        sys.exit(2)
    print(json.dumps(gmap, separators=(",", ":"), default=str) if "--json" in rest
          else render_markdown(gmap))


def _cmd_odte_rh_rows(rest: list[str]) -> None:
    # PURE/OFFLINE, NO broker/LLM/network. Pair the two SEPARATE arrays Robinhood returns — option
    # quotes/market-data and option instruments — into flat rows that odte-gamma-map consumes
    # directly. --quotes PATH / --quotes-json '...' supply the quote/market-data array; --instruments
    # PATH / --instruments-json '...' supply the companion instruments array (optional if quotes
    # already carry strike/type). Prints a JSON list of rows (pipe/feed to `odte-gamma-map --input`);
    # --out PATH writes it instead. HONEST: emits ABSOLUTE gamma/OI rows only — never dealer GEX.
    import json

    from data.odte_gamma_map import rh_rows_from_quotes
    q_json, q_path = _flag_value(rest, "--quotes-json"), _flag_value(rest, "--quotes")
    i_json, i_path = _flag_value(rest, "--instruments-json"), _flag_value(rest, "--instruments")
    if q_json is None and q_path is None:
        print("odte-rh-rows: provide --quotes PATH or --quotes-json '[...]'")
        sys.exit(2)
    try:
        quotes = json.loads(q_json if q_json is not None
                            else open(os.path.expanduser(q_path)).read())
        instruments = None
        if i_json is not None or i_path is not None:
            instruments = json.loads(i_json if i_json is not None
                                     else open(os.path.expanduser(i_path)).read())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"odte-rh-rows: could not read/parse input: {exc}")
        sys.exit(2)
    rows = rh_rows_from_quotes(quotes, instruments=instruments)
    out_path = _flag_value(rest, "--out")
    payload = json.dumps(rows, separators=(",", ":"), default=str)
    if out_path:
        path = os.path.expanduser(out_path)
        with open(path, "w") as f:
            f.write(payload)
        print(f"wrote {len(rows)} rows -> {path}")
    else:
        print(payload)


def _cmd_odte_vehicle_score(rest: list[str]) -> None:
    # Offline non-sentiment score for whether a candidate 0DTE contract/vehicle is a GOOD_BET,
    # WATCH, or BAD_BET for the day. Places NO orders and makes NO broker/network/LLM calls.
    import json

    from data.odte_vehicle_score import render_markdown, run_vehicle_score
    contract_path = _flag_value(rest, "--contract")
    contract_json = _flag_value(rest, "--contract-json")
    if contract_path is None and contract_json is None:
        print("odte-vehicle-score: provide --contract PATH or --contract-json '{...}'")
        sys.exit(2)
    try:
        payload = run_vehicle_score(
            contract_path=contract_path,
            contract_json=contract_json,
            market_path=_flag_value(rest, "--market"),
            market_json=_flag_value(rest, "--market-json"),
            gamma_path=_flag_value(rest, "--gamma"),
            gamma_json=_flag_value(rest, "--gamma-json"),
            direction=_flag_value(rest, "--direction"),
            buying_power=_flag_value(rest, "--buying-power"),
            out_dir=_flag_value(rest, "--out-dir"),
            write="--write" in rest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"odte-vehicle-score: could not read/parse input: {exc}")
        sys.exit(2)
    print(json.dumps(payload, separators=(",", ":"), default=str) if "--json" in rest
          else render_markdown(payload))


def _cmd_odte_day_score(rest: list[str]) -> None:
    # Offline non-sentiment score for the whole trading day: GOOD_DAY / CHOP / AVOID. Companion to
    # odte-vehicle-score (which scores one contract). Places NO orders, NO broker/network/LLM calls.
    import json

    from data.odte_day_score import render_markdown, run_day_score
    try:
        payload = run_day_score(
            market_path=_flag_value(rest, "--market"),
            market_json=_flag_value(rest, "--market-json"),
            gamma_path=_flag_value(rest, "--gamma"),
            gamma_json=_flag_value(rest, "--gamma-json"),
            out_dir=_flag_value(rest, "--out-dir"),
            write="--write" in rest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"odte-day-score: could not read/parse input: {exc}")
        sys.exit(2)
    print(json.dumps(payload, separators=(",", ":"), default=str) if "--json" in rest
          else render_markdown(payload))


def _cmd_odte_entry_gate(rest: list[str]) -> None:
    # PURE/OFFLINE thesis->entry gate. Assembles a journalable entry-gate decision record from the
    # upstream artifacts (watchdog trigger, candidate, day/vehicle score, gamma map, broker snapshot).
    # Records intent ONLY — places NO orders, makes NO broker/network/LLM calls. execution_allowed is
    # True only when every required gate is explicitly true and the record is not scan_only/restricted.
    # scan_only is INHERITED from the trigger/candidate by default (the watchdog lane is scan_only=true);
    # --scan-only forces it, and --promote-to-execution is the explicit opt-in to demote an inherited
    # scan_only record to the execution tier. --journal appends an entry_decision event (idempotent).
    import json

    from data.odte_entry_gate import render_markdown, run_entry_gate
    try:
        payload = run_entry_gate(
            trigger_path=_flag_value(rest, "--trigger"),
            trigger_json=_flag_value(rest, "--trigger-json"),
            candidate_path=_flag_value(rest, "--candidate"),
            candidate_json=_flag_value(rest, "--candidate-json"),
            day_score_path=_flag_value(rest, "--day-score"),
            day_score_json=_flag_value(rest, "--day-score-json"),
            vehicle_score_path=_flag_value(rest, "--vehicle-score"),
            vehicle_score_json=_flag_value(rest, "--vehicle-score-json"),
            gamma_path=_flag_value(rest, "--gamma"),
            gamma_json=_flag_value(rest, "--gamma-json"),
            broker_path=_flag_value(rest, "--broker"),
            broker_json=_flag_value(rest, "--broker-json"),
            # Default None => INHERIT scan_only from the trigger/candidate (watchdog is scan_only=True);
            # --scan-only forces it True. --promote-to-execution is the explicit opt-in to demote an
            # (inherited) scan_only record to the execution tier.
            scan_only=True if "--scan-only" in rest else None,
            promote_to_execution="--promote-to-execution" in rest,
            out_dir=_flag_value(rest, "--out-dir"),
            write="--write" in rest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"odte-entry-gate: could not read/parse input: {exc}")
        sys.exit(2)
    if "--journal" in rest:
        from data.odte_journal import append_decision_journal, event_from_entry_gate
        ev = event_from_entry_gate(payload)
        append_decision_journal(ev, source="entry_gate", event_type="entry_decision",
                                journal_path=_flag_value(rest, "--journal-path"))
    print(json.dumps(payload, separators=(",", ":"), default=str) if "--json" in rest
          else render_markdown(payload))


def _cmd_odte_fmp_context(rest: list[str]) -> None:
    # FMP single-name context for 0DTE meme/squeeze SANITY — read-only, NO orders, NO options/gamma.
    # Fetches cheap FMP *stable* fundamentals (profile/quote/shares-float/key-metrics-ttm/news) and
    # classifies a squeeze_profile. FMP options endpoints are unavailable, so fmp_options_available
    # is always false (Robinhood remains the gamma/options source). Fail-closed without FMP_KEY.
    # NOT wired into odte-watchdog (which stays cheap/no-network). Never prints the API key.
    import json

    from data.odte_fmp_context import render_markdown, run_fmp_context
    symbol = rest[0] if rest and not rest[0].startswith("--") else None
    if not symbol:
        print("odte-fmp-context: provide a SYMBOL, e.g. `odte-fmp-context WEN --json`")
        sys.exit(2)
    ctx = run_fmp_context(symbol, allow_fetch="--no-fetch" not in rest,
                          out_dir=_flag_value(rest, "--out-dir"), write="--write" in rest)
    print(json.dumps(ctx, separators=(",", ":"), default=str) if "--json" in rest
          else render_markdown(ctx))


def _cmd_stability_scan(rest: list[str]) -> None:
    from cli.commands import cmd_stability_scan
    mode = _flag_value(rest, "--mode")
    out_dir = _flag_value(rest, "--output-dir")
    cmd_stability_scan(mode=mode, output_dir=out_dir)


def _cmd_interaction_screen(rest: list[str]) -> None:
    from cli.commands import cmd_interaction_screen
    mode = _flag_value(rest, "--mode")
    out_dir = _flag_value(rest, "--output-dir")
    profile = _flag_value(rest, "--profile") or "standard"
    _nd = _flag_value(rest, "--days")
    n_days = int(_nd) if _nd else 730
    regime_scope = _flag_value(rest, "--regime-scope") or "all"
    cmd_interaction_screen(profile=profile, n_days=n_days, mode=mode, output_dir=out_dir,
                           regime_scope=regime_scope)


def _cmd_auto_tune_all(rest: list[str]) -> None:
    from cli.commands import cmd_auto_tune_all
    mode = _flag_value(rest, "--mode")
    profile = _flag_value(rest, "--profile") or "standard"
    _nd = _flag_value(rest, "--days")
    n_days = int(_nd) if _nd else 730
    _cl = _flag_value(rest, "--clusters")
    clusters = [c.strip() for c in _cl.split(",") if c.strip()] if _cl else None
    regime_scope = _flag_value(rest, "--regime-scope") or "all"
    cmd_auto_tune_all(profile=profile, n_days=n_days, mode=mode, clusters=clusters,
                      regime_scope=regime_scope)


def _cmd_report(rest: list[str]) -> None:
    from cli.commands import cmd_report
    out_dir = _flag_value(rest, "--output-dir") or "reports"
    cmd_report(output_dir=out_dir)


def _cmd_update_outcomes(rest: list[str]) -> None:
    from cli.commands import cmd_update_outcomes
    cmd_update_outcomes()


def _cmd_experiment(rest: list[str]) -> None:
    from cli.commands import cmd_experiment
    days = _flag_value(rest, "--days") or "90,180,365"
    scope = _flag_value(rest, "--scope") or "active_sleeve_compounding"
    variants = _flag_value(rest, "--variants")
    ex_mode = _flag_value(rest, "--mode")
    cmd_experiment(days=days, scope=scope, variants=variants, mode=ex_mode)


def _cmd_config(rest: list[str]) -> None:
    from cli.commands import cmd_config
    sub = rest[0] if rest else ""
    sub_rest = rest[1:] if len(rest) > 1 else []
    if sub == "migrate-scoring":
        dry_run = "--dry-run" in sub_rest
        no_backup = "--no-backup" in sub_rest
        cmd_config(action="migrate-scoring", dry_run=dry_run, no_backup=no_backup)
    else:
        print(f"Unknown config action: {sub!r}")
        sys.exit(1)


def _cmd_snapshots(rest: list[str]) -> None:
    from cli.commands import cmd_snapshots
    sub = rest[0] if rest else ""
    sub_rest = rest[1:] if len(rest) > 1 else []
    if sub == "rescore":
        input_dir   = _flag_value(sub_rest, "--input")
        output_dir  = _flag_value(sub_rest, "--output")
        dry_run     = "--dry-run" in sub_rest
        in_place    = "--in-place-with-backup" in sub_rest
        overwrite   = "--overwrite-existing" in sub_rest
        cmd_snapshots(
            action="rescore",
            input_dir=input_dir,
            output_dir=output_dir,
            dry_run=dry_run,
            in_place_with_backup=in_place,
            overwrite_existing=overwrite,
        )
    else:
        print("Usage: snapshots rescore "
              "[--dry-run] [--input PATH] [--output PATH] [--in-place-with-backup] [--overwrite-existing]")
        sys.exit(2)


def _cmd_fmp(rest: list[str]) -> None:
    from cli.commands import cmd_fmp
    sub = rest[0] if rest else "status"
    sub_rest = rest[1:] if len(rest) > 1 else []
    if sub == "status":
        cmd_fmp(action="status")
    elif sub == "validate-cache":
        cmd_fmp(action="validate-cache")
    elif sub == "backfill-prices":
        source = _flag_value(sub_rest, "--symbols") or "current"
        start = _flag_value(sub_rest, "--start") or "2015-01-01"
        end = _flag_value(sub_rest, "--end") or "2030-01-01"
        max_symbols = _int_flag(sub_rest, "--max-symbols")
        cmd_fmp(action="backfill-prices", symbols_source=source, start=start, end=end,
                max_symbols=max_symbols, force="--force" in sub_rest)
    elif sub == "backfill-statements":
        source = _flag_value(sub_rest, "--symbols") or "current"
        kinds_s = _flag_value(sub_rest, "--kinds")
        kinds = [k.strip() for k in kinds_s.split(",") if k.strip()] if kinds_s else None
        max_symbols = _int_flag(sub_rest, "--max-symbols")
        limit = _int_flag(sub_rest, "--limit") or 44
        cmd_fmp(action="backfill-statements", symbols_source=source, kinds=kinds,
                max_symbols=max_symbols, limit=limit, force="--force" in sub_rest)
    elif sub == "backfill-delisted":
        cmd_fmp(action="backfill-delisted", max_pages=_int_flag(sub_rest, "--max-pages") or 50)
    elif sub == "build-dead-universe":
        cmd_fmp(
            action="build-dead-universe",
            start=_flag_value(sub_rest, "--start") or "2015-01-01",
            end=_flag_value(sub_rest, "--end") or "2030-01-01",
            min_adv=float(_flag_value(sub_rest, "--min-adv") or 500_000.0),
            max_symbols=_int_flag(sub_rest, "--max-symbols"),
            allow_fetch_prices="--fetch-prices" in sub_rest,
        )
    else:
        print("Usage: fmp status | validate-cache | backfill-prices | backfill-statements | "
              "backfill-delisted | build-dead-universe")
        sys.exit(2)


def _cmd_factor_map(rest: list[str]) -> None:
    from cli.commands import cmd_factor_map
    method   = _flag_value(rest, "--method") or "pca"
    color    = _flag_value(rest, "--color")
    clusters_str = _flag_value(rest, "--clusters")
    clusters = int(clusters_str) if clusters_str and clusters_str.isdigit() else None
    out      = _flag_value(rest, "--output")
    owned    = "--owned-only" in rest
    show     = "--show" in rest
    cmd_factor_map(
        method=method,
        color_by=color,
        kmeans_clusters=clusters,
        output=out,
        owned_only=owned,
        show=show,
    )


# Command-name → handler. Aliases (e.g. options-social) map to the same handler.
_COMMANDS: dict[str, Callable[[list[str]], None]] = {
    "list-presets": _cmd_list_presets,
    "fetch-data": _cmd_fetch_data,
    "run": _cmd_run,
    "backtest": _cmd_backtest,
    "tune": _cmd_tune,
    "auto-tune": _cmd_auto_tune,
    "tune-etf-allocation": _cmd_tune_etf_allocation,
    "report-etf-allocation": _cmd_report_etf_allocation,
    "odte-social-report": _cmd_odte_social_report,
    "options-social": _cmd_odte_social_report,
    "odte-watchdog": _cmd_odte_watchdog,
    "odte-position": _cmd_odte_position,
    "odte-journal": _cmd_odte_journal,
    "odte-ingest-artifacts": _cmd_odte_ingest_artifacts,
    "odte-journal-report": _cmd_odte_journal_report,
    "odte-gamma-map": _cmd_odte_gamma_map,
    "odte-rh-rows": _cmd_odte_rh_rows,
    "odte-vehicle-score": _cmd_odte_vehicle_score,
    "odte-day-score": _cmd_odte_day_score,
    "odte-entry-gate": _cmd_odte_entry_gate,
    "odte-fmp-context": _cmd_odte_fmp_context,
    "stability-scan": _cmd_stability_scan,
    "interaction-screen": _cmd_interaction_screen,
    "auto-tune-all": _cmd_auto_tune_all,
    "report": _cmd_report,
    "update-outcomes": _cmd_update_outcomes,
    "experiment": _cmd_experiment,
    "config": _cmd_config,
    "snapshots": _cmd_snapshots,
    "fmp": _cmd_fmp,
    "factor-map": _cmd_factor_map,
}


def _flag_value(args: list[str], flag: str) -> str | None:
    try:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    except ValueError:
        pass
    return None


def _int_flag(args: list[str], flag: str) -> int | None:
    value = _flag_value(args, flag)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        print(f"{flag} requires an integer, got {value!r}")
        sys.exit(2)


def _print_help() -> None:
    print("""
daily-investor CLI

COMMANDS
  fetch-data               Fetch fresh data only — no trades (fundamentals, news, snapshot)
  run                      Live trading run (requires Robinhood credentials)
  backtest DAYS            Run backtest simulation (--archetype-compare for A/B vs uniform)
  tune DAYS                Single-objective parameter tune (prints diff, no write)
  auto-tune [DAYS]         Dual-objective tune with walk-forward validation (default: 90d)
  auto-tune-all            Staged coordinate-ascent over interaction clusters + windowed
                           validation (--profile quick|standard|deep, --clusters a,b; research only)
  interaction-screen       Screen which param clusters synergize/clash when co-tuned
                           (--profile quick|standard|deep; research only)
  list-presets             Print available tuning presets and exit (presets compose with '+')
  stability-scan           Parameter stability scan (research only, no writes)
  report                   Generate diagnostics report
  update-outcomes          Backfill future returns for past decisions (calibration only)
  factor-map               3-D PCA/UMAP factor-space scatter of the scored universe
  fmp <SUB>                FMP cache operations (status, backfill, validate)
  config <SUB>             config maintenance (sub: migrate-scoring)
  snapshots <SUB>          snapshot maintenance (sub: rescore)
  tune-etf-allocation      Gated ETF/core sleeve allocation tournament (--days, --mode regime|defensive,
                           --universe configured_only, --random-topk N, --apply)
  report-etf-allocation    Print ETF/core sleeve diagnostics for the current config (--days)
  odte-social-report       0DTE social-sentiment watchlist — ANALYSIS/PAPER ONLY, places NO orders.
                           Reddit (JSON+Atom fallback) + X (official API only if X_BEARER_TOKEN);
                           DAY-OF posts only; attaches a PAPER same-day option idea (yfinance);
                           --no-fetch = offline (no network, no options lookup);
                           --json = clean signal-only JSON for an agent (no prose; logs silenced);
                           --reddit-bearer-token TOKEN = optional ephemeral read-only token for WSB
                           daily-thread comments (never stored/logged; not cookies);
                           --daily-thread-id ID / --daily-thread-url URL = override daily-thread
                           discovery (runtime only; not persisted to config);
                           --daily-thread-limit N = cap on comments read (default: auto-paginate
                           the WHOLE thread, up to a runaway safety ceiling).
  options-social           Alias for odte-social-report (identical behavior and options).
  odte-watchdog            Script-only 0DTE watchdog — NO LLM, NO Robinhood, places NO orders.
                           Runs the LOCAL report, diffs the actionable candidate vs the prior run,
                           writes data/odte/{watchdog_state,triggers}.json. For a no_agent cron:
                           EMPTY stdout when nothing actionable, compact one-line JSON on a trigger
                           (new/changed non-restricted candidate, or missing/invalid policy).
                           --json always prints state; --no-fetch runs offline (cache-only);
                           --state-dir DIR overrides the data dir; the controller policy is a secret
                           read from ~/0dte/controller_policy.json (--policy PATH overrides).
  odte-position            Broker-AWARE, DECISION-ONLY live-position watchdog — places NO orders,
                           makes NO broker/LLM calls. Reads the active trade plan
                           (data/odte/active_trade.json) + a caller-supplied live snapshot (Hermes feeds
                           real broker/market values via MCP) and emits TAKE_PROFIT / THESIS_DEAD /
                           BID_FLOOR / TIME_RISK / MONITORING_DEGRADED / HOLD. Writes
                           data/odte/{position_state,position_decision}.json. EMPTY stdout on HOLD/
                           NO_POSITION; compact JSON on an actionable decision. --snapshot PATH or
                           --snapshot-json '{...}' supply the live values; --plan / --state-dir
                           override defaults; --json always prints.
  odte-journal             Append one event to the local 0DTE decision journal
                           (data/odte/decision_journal.jsonl) — local/offline, NO broker/LLM/secrets.
                           --event-json '{...}' or --event PATH supplies the event (event_type +
                           free-form thesis/decision/outcome/experiment fields); NVDA/restricted
                           underlyings are tagged and kept out of metrics. --json prints the stored event.
  odte-journal-report      Summarize the decision journal into deterministic metrics + Markdown/CSV
                           (trades by mode, hit rate, avg P/L, MFE capture, rule violations, timing,
                           experiments, lessons). --json prints metrics; default prints Markdown;
                           --write (or --out-dir DIR) writes data/odte/reports/ artifacts.
  odte-gamma-map           0DTE option-chain gamma / pin map — PURE/OFFLINE, NO broker/LLM/network.
                           Reads option-quote rows Hermes/RH exported (--input PATH or
                           --input-json '{...}') and computes ABSOLUTE gamma/OI concentration:
                           call/put walls, max-gamma strike, ATM-straddle expected move, pin risk,
                           quote freshness. HONEST — labeled gamma_regime=pin_risk_only_not_dealer_gex;
                           it does NOT infer dealer net GEX / gamma flip (RH doesn't expose it).
                           --spot/--underlying/--expiration refine; --json prints the map; --write
                           (or --out-dir DIR) writes data/odte/reports/odte_gamma_map_<sym>.{md,json}.
  odte-rh-rows             Pair the two SEPARATE arrays Robinhood returns (option quotes/market-data
                           + option instruments) into flat rows that odte-gamma-map consumes —
                           PURE/OFFLINE, NO broker/LLM/network. --quotes PATH / --quotes-json '[...]'
                           supply the quote array; --instruments PATH / --instruments-json '[...]'
                           supply the companion instruments (optional if quotes carry strike/type).
                           Joins each quote to its instrument by id/url; prints a JSON row list (feed
                           to `odte-gamma-map --input`), or --out PATH writes it. HONEST: ABSOLUTE
                           gamma/OI rows only — never dealer GEX.
  odte-vehicle-score       PURE/OFFLINE non-sentiment GOOD_BET/WATCH/BAD_BET score for a candidate
                           contract/vehicle. Inputs: --contract PATH or --contract-json '{...}', plus
                           optional --market/--market-json and --gamma/--gamma-json; --buying-power N
                           adds account-fit scoring. Uses tape/VWAP, VIXY, gamma/pin/expected move,
                           liquidity/spread, and BP fit — no orders, no network, no sentiment.
  odte-day-score           PURE/OFFLINE non-sentiment GOOD_DAY/CHOP/AVOID score for the whole
                           trading day (companion to odte-vehicle-score, which scores one contract).
                           Inputs: --market PATH or --market-json '{...}' (vix, vix/vixy change,
                           gap_pct, {spy,qqq,iwm}_above_vwap, {sym}_orb_state, expected_move_pct,
                           minutes_to_close) plus optional --gamma/--gamma-json (derives expected
                           move from the ATM band). Scores trend + volatility + gap + expected-move
                           + late-day theta — no orders, no network, no sentiment. --json prints the
                           payload; --write (or --out-dir DIR) writes data/odte/reports/ artifact.
  odte-entry-gate          PURE/OFFLINE thesis->entry gate — assembles a journalable entry-gate
                           decision (enter/deny/veto/observe) from upstream artifacts: --trigger,
                           --candidate, --day-score, --vehicle-score, --gamma, --broker (each PATH or
                           a *-json '{...}'). execution_allowed is True ONLY when every required gate
                           (day_regime + vehicle + directional_thesis + account) is explicitly true and
                           the record is not scan_only / restricted; missing inputs fail closed.
                           scan_only is INHERITED from the trigger/candidate by default (the watchdog
                           lane is scan_only=true); --scan-only forces it, and --promote-to-execution
                           is the explicit opt-in to demote an inherited scan_only record to the
                           execution tier. Records intent ONLY — no orders/broker/network. --journal
                           appends an entry_decision event to the decision journal; --json/--write.
  odte-fmp-context SYMBOL  FMP single-name context for meme/squeeze SANITY — read-only, NO orders,
                           NO options/gamma. Fetches cheap FMP stable fundamentals (profile, quote,
                           shares-float, key-metrics-ttm, a few news) and classifies a squeeze_profile
                           (tiny/small/mid/large float). FMP options are unavailable so
                           fmp_options_available is always false — Robinhood stays the gamma/options
                           source. Fail-closed without FMP_KEY (never printed). --json prints the
                           context; --write (or --out-dir DIR) writes data/odte/reports/ artifacts;
                           --no-fetch runs offline. NOT used by odte-watchdog (kept cheap/no-network).

OPTIONS (run)
  --skip-data              Reuse existing CSV data
  --op-mode safe|automated|no-sentiment

OPTIONS (run / fetch-data)
  --skip-fetch-news        Refresh all other data but reuse the latest cached news
                           scrape (skips the slowest stage; any age). 0DTE options
                           sentiment is unaffected. Env NEWS_FORCE_REFETCH=1 overrides
                           the default 8h news-freshness reuse window.

OPTIONS (tune / auto-tune)
  --apply                  Write config.yaml if validation passes
  --force-apply            Write config.yaml unconditionally
  --llm-review             Add Claude second-opinion review
  --scope SCOPE            overall_strategy (default) or active_sleeve_compounding
  --regime-scope SCOPE     all (default), bullish, neutral, or defensive (bearish accepted as an alias for defensive)
  --preset NAME[+NAME...]  Restrict tunable params to a preset; compose several with '+'
                           to co-tune their union (e.g. active_exits+active_exit_floors)
  --random-topk N          auto-tune only: add the top-N robust-random-search candidates
                           to the selection tournament (default 0 = off)
  --leads a.npy,b.npy      auto-tune only: add saved lead param vectors to the tournament

OPTIONS (auto-tune-all / interaction-screen)
  --profile P              quick | standard | deep  (default: standard)
  --days N                 history window to load (default: 730)
  --clusters a,b,c         auto-tune-all only: which interaction clusters to co-tune

OPTIONS (fmp)
  fmp status
  fmp validate-cache
  fmp backfill-prices --symbols current|AAPL,MSFT|path.csv [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                      [--max-symbols N] [--force]
  fmp backfill-statements --symbols current|AAPL,MSFT|path.csv [--kinds income-statement,...]
                          [--limit N] [--max-symbols N] [--force]
  fmp backfill-delisted [--max-pages N]
  fmp build-dead-universe [--min-adv N] [--fetch-prices] [--max-symbols N]

OPTIONS (any command)
  --config PATH            Use a different YAML config (default: cfg/config.yaml).
                           Useful for cfg/config_<name>.yaml A/B comparisons.

OPTIONS (tune only)
  --objective sharpe|calmar|info_ratio  Optimization target (default: sharpe). info_ratio = excess-vs-SPY
                             / tracking-error (active scope). NOTE: `auto-tune` is dual-objective
                             (averages sharpe+calmar) and ignores --objective.

OPTIONS (all)
  --mode MODE                Backtest universe selection mode. One of:
                               liquid_universe_full          (default; full liquid universe)
                               walk_forward_price_only_test  (low lookahead; price/momentum only)
                               current_universe_stress_test  (high lookahead; current-score ranking)
  --output-dir PATH          Report output directory
""")


if __name__ == "__main__":
    main()
