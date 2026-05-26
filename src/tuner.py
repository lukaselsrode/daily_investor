# tuner.py — compatibility shim. Import from tuning.* instead.
from tuning.constants import (
    PARAM_NAMES,
    BOUNDS,
    _CONFIG_PATH_TO_PARAM_IDX,
    _MIN_TRADES_HARD,
    _MIN_TRADES_SOFT,
    _effective_bounds,
    _get_active_indices,
    _expand_params,
    _current_params,
)
from tuning.objective import make_objective, _run_single
from tuning.reports import (
    apply_config_params,
    build_llm_review_payload,
    merge_llm_recommendation_with_config,
    print_config_diff,
    request_llm_tune_review,
    validate_llm_review_response,
    _diff_table,
    _LLM_ALLOWED_PARAMS,
    _LLM_FORBIDDEN_PARAMS,
)
from tuning.tuner import (
    run_tuner,
    run_auto_tune,
    validate_tuned_params,
    should_apply_tuned_config,
    ParameterTuner,
)
from tuning.stability import run_stability_scan, StabilityAnalyzer

__all__ = [
    "PARAM_NAMES", "BOUNDS", "_CONFIG_PATH_TO_PARAM_IDX",
    "_MIN_TRADES_HARD", "_MIN_TRADES_SOFT",
    "_effective_bounds", "_get_active_indices", "_expand_params", "_current_params",
    "make_objective", "_run_single",
    "apply_config_params", "build_llm_review_payload", "merge_llm_recommendation_with_config",
    "print_config_diff", "request_llm_tune_review", "validate_llm_review_response",
    "_diff_table", "_LLM_ALLOWED_PARAMS", "_LLM_FORBIDDEN_PARAMS",
    "run_tuner", "run_auto_tune", "validate_tuned_params", "should_apply_tuned_config",
    "ParameterTuner", "run_stability_scan", "StabilityAnalyzer",
]
