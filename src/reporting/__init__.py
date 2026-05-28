"""reporting — Attribution, diagnostics, plots, and HTML reports."""
from .attribution import AttributionReporter, classify_stability, compute_parameter_stability
from .diagnostics import (
    DiagnosticsReporter,
    generate_all_reports,
    write_robustness_report_txt,
    write_stability_summary_csv,
)
from .plots import (
    PlotManager,
    generate_objective_heatmap,
    generate_param_heatmap,
    generate_validation_heatmap,
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
