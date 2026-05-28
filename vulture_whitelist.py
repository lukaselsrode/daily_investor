# vulture_whitelist.py — Known false positives for vulture dead-code scan.
# Add entries here when vulture flags legitimate code (entry points, callbacks,
# dynamically-accessed attributes, Streamlit decorators, TypedDict fields).
# Reference: https://github.com/jendrikseipp/vulture#whitelist-files

# CLI entry points — called by setuptools entry_points, not referenced in source
from cli.main import main  # noqa: F401

# Reserved API parameters — accepted by callers via keyword arg, not yet used in body.
# cluster_tracker.compute_cluster_snapshot: reserved for future sector-weight enforcement.
# allocation_diagnostics._compute_migration_state: called as harvest_to_etf_pct=... in render().
# Streamlit @st.cache_data discriminator arguments: not used in body but control cache key.
max_sector_weight_threshold  # noqa: F821
harvest_to_etf_pct  # noqa: F821
features_key  # noqa: F821
score_cols_key  # noqa: F821
factors_key  # noqa: F821
