"""reporting — Attribution, diagnostics, plots, and HTML reports."""
from .attribution import AttributionReporter
from .diagnostics import DiagnosticsReporter
from .plots import PlotManager

# Backward-compat re-exports so existing callers (tuner.py) keep working
# while the legacy module is gradually hollowed out.
from _reporting_legacy import (  # noqa: F401
    classify_stability,
    compute_parameter_stability,
    generate_all_reports,
    generate_objective_heatmap,
    generate_param_heatmap,
    generate_validation_heatmap,
    write_robustness_report_txt,
    write_stability_summary_csv,
)

__all__ = [
    "AttributionReporter",
    "DiagnosticsReporter",
    "PlotManager",
    "classify_stability",
    "compute_parameter_stability",
    "generate_all_reports",
    "generate_objective_heatmap",
    "generate_param_heatmap",
    "generate_validation_heatmap",
    "write_robustness_report_txt",
    "write_stability_summary_csv",
]
