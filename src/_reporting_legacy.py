# _reporting_legacy.py — compatibility shim. Import from reporting.* instead.
from reporting.attribution import classify_stability, compute_parameter_stability
from reporting.diagnostics import (
    generate_all_reports,
    write_robustness_report_txt,
    write_stability_summary_csv,
)
from reporting.plots import (
    generate_objective_heatmap,
    generate_param_heatmap,
    generate_validation_heatmap,
)
