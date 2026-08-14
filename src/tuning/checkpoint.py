"""
tuning/checkpoint.py — crash-safe resume state for long-running parameter tunes.

Motivation (2026-08-07): `auto-tune-all --profile standard --days 730` ran for 39
hours without finishing its FIRST cluster and was killed. `run_staged_tune` persists
nothing until it returns, so every hour was lost. The standard run matrix costs ~7,200
simulation runs per objective call (~2.4M per cluster), which makes a single stage
longer than most operators will leave a job alone.

So checkpointing lands at TWO granularities:

  stage level      — after each cluster in run_staged_tune. State is tiny: the evolving
                     baseline vector, its two scores, and the completed stage trace.
  generation level — inside _tune_subset's differential_evolution loop, via its
                     `callback=`. Persists best-x-so-far so a stage that alone exceeds
                     a day is still resumable (warm-started through `x0=`).

Storage is one JSON file per run under data/tune_checkpoints/, written with
core.paths.atomic_write_text (crash-safe: tmp sibling → fsync → os.replace) — the same
primitive the 0DTE execution layer uses for its state handoff. Param vectors are ~104
floats, so JSON keeps them human-inspectable rather than opaque.

STALENESS IS LOUD. Two identity guards, both of which RAISE rather than silently
reindex a mismatched vector:

  code_fingerprint — tuning.constants mutates PARAM_NAMES/BOUNDS at import time across
                     ~16 successive append blocks (archetype, cs-filter, sizing, regime,
                     exit-floor, opportunity-cost, rebalance, ETF, peer-2 slots...).
                     A checkpoint written before a slot was appended describes a
                     DIFFERENT vector layout at the same indices.
  identity         — hash of the run configuration (matrix, scope, clusters, DE budget,
                     train_frac) plus the baseline the run started from. `_current_params()`
                     reads live config globals, so a mid-run cfg/config.yaml edit would
                     otherwise silently change what a resume builds on top of.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CHECKPOINT_SUBDIR = "tune_checkpoints"
SCHEMA_VERSION = 1


class CheckpointMismatch(RuntimeError):
    """A checkpoint exists but does not describe this run (or this code revision).

    Never downgraded to a warning: resuming onto a mismatched param layout or a
    different baseline silently produces a vector nobody validated.
    """


def checkpoint_dir() -> Path:
    from core.paths import DATA_DIRECTORY
    d = Path(DATA_DIRECTORY) / CHECKPOINT_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(run_id: str) -> Path:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(run_id))
    return checkpoint_dir() / f"{safe}.json"


def code_fingerprint() -> str:
    """Identity of the current param-vector LAYOUT (not its values).

    PARAM_NAMES grows at import time, so both its length and its contents pin the
    layout a checkpoint's indices refer to.
    """
    from .constants import PARAM_NAMES
    blob = json.dumps({"n": len(PARAM_NAMES), "names": list(PARAM_NAMES)}, sort_keys=True)
    return f"{len(PARAM_NAMES)}:{hashlib.sha256(blob.encode()).hexdigest()[:12]}"


def run_identity(
    *,
    run_matrix: list[dict],
    clusters,
    scope: str,
    regime_scope: str,
    maxiter: int,
    popsize: int,
    train_frac: float,
    baseline: np.ndarray | None,
) -> str:
    """Stable hash of everything that would invalidate a resume if it changed."""
    payload = {
        "run_matrix": [dict(sorted(c.items())) for c in run_matrix],
        "clusters": list(clusters),
        "scope": str(scope),
        "regime_scope": str(regime_scope),
        "maxiter": int(maxiter),
        "popsize": int(popsize),
        "train_frac": round(float(train_frac), 6),
        "baseline": [round(float(v), 10) for v in np.asarray(baseline).ravel()]
        if baseline is not None else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class DEState:
    """Best-so-far inside one cluster's differential_evolution run.

    `x` is the REDUCED vector over the preset's active slots (what DE optimizes), not
    a full param vector — it is only meaningful together with `cluster`.
    """
    cluster: str = ""
    generation: int = 0
    best_x: list[float] = field(default_factory=list)
    best_fun: float = float("inf")   # DE minimizes; robust score is -fun


@dataclass
class TuneCheckpoint:
    run_id: str
    identity: str
    code_fingerprint: str
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""
    updated_at: str = ""
    # Stage-level state
    stage_index: int = 0                       # completed stages
    baseline: list[float] = field(default_factory=list)
    cur_score: float = 0.0
    baseline_check: float = 0.0
    orig_score: float = 0.0
    stages: list[dict] = field(default_factory=list)
    joint_done: bool = False
    # Generation-level state for the stage currently in flight
    de_state: dict | None = None
    # Phase-1 marginals: cluster -> {"score": float, "params": [float, ...]}. Each is
    # tuned against the FROZEN incumbent, so a resume can rebuild the noise band and the
    # promotion decision without re-running completed clusters.
    marginals: dict = field(default_factory=dict)

    def baseline_array(self) -> np.ndarray:
        return np.asarray(self.baseline, dtype=float)


def save(ckpt: TuneCheckpoint) -> Path:
    """Atomically persist a checkpoint. Never raises — losing a checkpoint write must
    not kill a tune that is otherwise progressing."""
    from core.paths import atomic_write_text

    path = checkpoint_path(ckpt.run_id)
    ckpt.updated_at = _dt.datetime.now().isoformat(timespec="seconds")
    if not ckpt.created_at:
        ckpt.created_at = ckpt.updated_at
    try:
        atomic_write_text(path, json.dumps(asdict(ckpt), indent=2, default=float))
    except Exception as exc:
        logger.warning("tune checkpoint write failed (%s): %s", path.name, exc)
    return path


def load(
    run_id: str,
    *,
    expected_identity: str | None = None,
    strict: bool = True,
) -> TuneCheckpoint | None:
    """Read a checkpoint, or None when absent.

    Raises CheckpointMismatch when the file describes a different code revision or a
    different run configuration — resuming across either is silent corruption.
    """
    path = checkpoint_path(run_id)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        raise CheckpointMismatch(f"checkpoint {path} is unreadable: {exc}") from exc

    ckpt = TuneCheckpoint(
        run_id=raw.get("run_id", run_id),
        identity=raw.get("identity", ""),
        code_fingerprint=raw.get("code_fingerprint", ""),
        schema_version=int(raw.get("schema_version", 0)),
        created_at=raw.get("created_at", ""),
        updated_at=raw.get("updated_at", ""),
        stage_index=int(raw.get("stage_index", 0)),
        baseline=[float(v) for v in raw.get("baseline", [])],
        cur_score=float(raw.get("cur_score", 0.0)),
        baseline_check=float(raw.get("baseline_check", 0.0)),
        orig_score=float(raw.get("orig_score", 0.0)),
        stages=list(raw.get("stages", [])),
        joint_done=bool(raw.get("joint_done", False)),
        de_state=raw.get("de_state"),
        marginals=dict(raw.get("marginals") or {}),
    )
    if not strict:
        return ckpt

    if ckpt.schema_version != SCHEMA_VERSION:
        raise CheckpointMismatch(
            f"checkpoint {path.name} has schema v{ckpt.schema_version}, code expects "
            f"v{SCHEMA_VERSION} — delete it and restart the tune."
        )
    current_fp = code_fingerprint()
    if ckpt.code_fingerprint != current_fp:
        raise CheckpointMismatch(
            f"checkpoint {path.name} was written under a DIFFERENT param layout "
            f"({ckpt.code_fingerprint} vs {current_fp}). tuning.constants appends slots at "
            "import time, so its indices no longer mean the same parameters. Delete it "
            "and restart the tune."
        )
    if expected_identity is not None and ckpt.identity != expected_identity:
        raise CheckpointMismatch(
            f"checkpoint {path.name} belongs to a different run configuration "
            f"({ckpt.identity} vs {expected_identity}) — run matrix, clusters, scope, DE "
            "budget, or the starting baseline changed. Use a new --checkpoint name."
        )
    return ckpt


def new_checkpoint(run_id: str, identity: str, baseline: np.ndarray) -> TuneCheckpoint:
    return TuneCheckpoint(
        run_id=run_id,
        identity=identity,
        code_fingerprint=code_fingerprint(),
        baseline=[float(v) for v in np.asarray(baseline).ravel()],
    )


def clear(run_id: str) -> bool:
    """Delete a checkpoint (used after a clean finish). True when a file was removed."""
    path = checkpoint_path(run_id)
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception as exc:
        logger.warning("could not remove checkpoint %s: %s", path.name, exc)
    return False
