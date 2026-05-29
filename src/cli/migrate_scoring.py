"""
cli/migrate_scoring.py — One-shot YAML migration from legacy scoring layout to unified.

Converts the pre-consolidation layout (top-level `scoring:` flat, `momentum:`,
`momentum_v2:`, `value_v2:`, `scoring_v3:` blocks) into the unified `scoring:`
block expected by the current util.py reader.

Used by `daily-investor config migrate-scoring`. Idempotent: a migrated file is
detected by the presence of `scoring.peer_standardization` and skipped.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml


def _is_already_unified(scoring: dict) -> bool:
    """A unified `scoring:` block has peer_standardization + factors + momentum_inputs."""
    if not isinstance(scoring, dict):
        return False
    return (
        "peer_standardization" in scoring
        or "factors" in scoring
        or "momentum_inputs" in scoring
    )


def _rename_v2_blend_in_factors(factors: dict) -> bool:
    """Rename `v2_blend` → `anchor_blend` in each factor entry. Returns True if any change."""
    changed = False
    for entry in factors.values():
        if not isinstance(entry, dict):
            continue
        if "anchor_blend" in entry:
            # Already migrated; drop any stale duplicate v2_blend
            if "v2_blend" in entry:
                del entry["v2_blend"]
                changed = True
            continue
        if "v2_blend" in entry:
            entry["anchor_blend"] = entry.pop("v2_blend")
            changed = True
    return changed


def migrate_yaml(path: Path, *, dry_run: bool = False, backup: bool = True) -> str:
    """Migrate one YAML file. Returns 'migrated' | 'already-unified' | 'no-scoring-keys'."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    old_scoring = cfg.get("scoring", {}) or {}
    if _is_already_unified(old_scoring):
        # Even already-unified files may still carry the legacy `v2_blend` factor key.
        factors = old_scoring.get("factors", {}) or {}
        renamed = _rename_v2_blend_in_factors(factors)
        if not renamed:
            return "already-unified"
        if dry_run:
            return "v2_blend renamed (dry-run)"
        if backup:
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(path, bak)
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        return "v2_blend renamed"

    momentum_v1 = cfg.pop("momentum", None) or {}
    momentum_v2 = cfg.pop("momentum_v2", None) or {}
    value_v2 = cfg.pop("value_v2", None) or {}
    scoring_v3 = cfg.pop("scoring_v3", None) or {}

    if not (old_scoring or momentum_v1 or momentum_v2 or value_v2 or scoring_v3):
        return "no-scoring-keys"

    # Build unified block
    new_scoring: dict = {"enabled": bool(scoring_v3.get("enabled", True))}

    # peer_standardization
    ps = scoring_v3.get("peer_standardization", {})
    new_scoring["peer_standardization"] = {
        "group_by":          ps.get("group_by",          "industry"),
        "fallback_group_by": ps.get("fallback_group_by", "sector"),
        "min_group_size":    ps.get("min_group_size",    8),
        "method":            ps.get("method",            "percentile"),
        "winsorize_pct":     ps.get("winsorize_pct",     0.05),
        "clamp_low":         ps.get("clamp_low",         -1.0),
        "clamp_high":        ps.get("clamp_high",         1.5),
        "blend":             ps.get("blend", {
            "industry_relative": 0.60, "sector_relative": 0.25, "market_relative": 0.15,
        }),
    }

    # factors
    factors_in = scoring_v3.get("factors", {}) or {}
    new_factors: dict = {}
    for name in ("value", "quality", "momentum", "income", "growth_leadership"):
        entry = dict(factors_in.get(name, {}) or {})
        if name == "value":
            # Fold value_v2.distress + value_v2.composite into the factor block
            v2_dist = value_v2.get("distress", {})
            if v2_dist:
                entry.setdefault("distress", v2_dist)
            v2_comp = value_v2.get("composite", {})
            if v2_comp:
                entry.setdefault("pe_weight", v2_comp.get("pe_weight", 0.70))
                entry.setdefault("pb_weight", v2_comp.get("pb_weight", 0.30))
        new_factors[name] = entry
    _rename_v2_blend_in_factors(new_factors)
    new_scoring["factors"] = new_factors

    # momentum_inputs (from old momentum_v2)
    if momentum_v2:
        new_scoring["momentum_inputs"] = {
            "weights":       momentum_v2.get("weights", {}),
            "penalties":     momentum_v2.get("penalties", {}),
            "clamp_low":     momentum_v2.get("clamp_low", -1.0),
            "clamp_high":    momentum_v2.get("clamp_high", 1.5),
            "winsorize_pct": momentum_v2.get("winsorize_pct", 0.05),
        }

    # momentum_warmup (from old flat momentum)
    if momentum_v1:
        new_scoring["momentum_warmup"] = momentum_v1

    # quality_checklist (from old flat scoring:)
    if old_scoring:
        new_scoring["quality_checklist"] = old_scoring

    cfg["scoring"] = new_scoring

    if dry_run:
        return "migrated (dry-run)"

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return "migrated"


def migrate_all(
    yaml_paths: list[Path],
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, str]:
    """Migrate a list of YAML files. Returns {path → status} dict."""
    results: dict[str, str] = {}
    for p in yaml_paths:
        try:
            results[str(p)] = migrate_yaml(p, dry_run=dry_run, backup=backup)
        except Exception as exc:
            results[str(p)] = f"error: {exc}"
    return results


def cmd_migrate_scoring(
    paths: list[str] | None = None,
    *,
    dry_run: bool = False,
    no_backup: bool = False,
) -> None:
    """CLI entry point — migrate all config YAMLs (defaults to a standard set)."""
    if paths is None:
        repo_root = Path(__file__).resolve().parents[2]
        cfg_dir = repo_root / "cfg"
        targets = [cfg_dir / "config.yaml"]
        for sub in ("variants", "experiments"):
            sub_dir = cfg_dir / sub
            if sub_dir.exists():
                targets.extend(sorted(sub_dir.glob("*.yaml")))
        # config_*.yaml siblings of config.yaml
        for p in sorted(cfg_dir.glob("config_*.yaml")):
            if p not in targets:
                targets.append(p)
    else:
        targets = [Path(p) for p in paths]

    results = migrate_all(targets, dry_run=dry_run, backup=not no_backup)
    header = f"Migrate config-scoring ({'DRY RUN' if dry_run else 'WRITE'}) — {len(targets)} files"
    print(header)
    print("-" * len(header))
    counts: dict[str, int] = {}
    for path, status in results.items():
        counts[status] = counts.get(status, 0) + 1
        print(f"  {status:<22s} {path}")
    print()
    for status, n in sorted(counts.items()):
        print(f"  {status:<22s}  {n}")
