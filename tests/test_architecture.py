"""
tests/test_architecture.py — Architecture boundary checks.

These tests enforce the import boundary rules documented in AGENTS.md:
  - Core packages must not import streamlit
  - Core packages must not import from ui/
  - CLI entrypoints must exist and respond to --help
  - Service layer must be importable
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

CORE_PACKAGES = [
    "backtesting",
    "strategy",
    "portfolio",
    "tuning",
    "config",
    "research",
    "reporting",
    "execution",
    "core",
]


def _python_files(package: str) -> list[Path]:
    pkg_dir = SRC / package
    if not pkg_dir.exists():
        return []
    return list(pkg_dir.rglob("*.py"))


def _file_content(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Import boundary checks
# ---------------------------------------------------------------------------

def test_no_streamlit_in_core_packages():
    """Core packages must never import streamlit."""
    violations = []
    for pkg in CORE_PACKAGES:
        for py in _python_files(pkg):
            content = _file_content(py)
            if "import streamlit" in content or "from streamlit" in content:
                violations.append(str(py.relative_to(SRC)))
    assert not violations, (
        "streamlit imported in core package(s):\n  " + "\n  ".join(violations)
    )


def test_no_ui_imports_in_core_packages():
    """Core packages must not import from ui/."""
    violations = []
    for pkg in CORE_PACKAGES:
        for py in _python_files(pkg):
            content = _file_content(py)
            if "from ui." in content or "import ui." in content:
                violations.append(str(py.relative_to(SRC)))
    assert not violations, (
        "ui/ imported in core package(s):\n  " + "\n  ".join(violations)
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_help_exits_zero():
    """CLI --help must exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "cli", "--help"],
        capture_output=True, text=True,
        cwd=str(SRC),
    )
    assert result.returncode == 0, (
        f"CLI --help exited {result.returncode}:\n{result.stderr}"
    )


def test_cli_help_contains_expected_commands():
    """CLI --help output must list the core commands."""
    result = subprocess.run(
        [sys.executable, "-m", "cli", "--help"],
        capture_output=True, text=True,
        cwd=str(SRC),
    )
    output = result.stdout + result.stderr
    expected = ["backtest", "auto-tune", "tune", "fetch-data", "factor-map"]
    missing = [cmd for cmd in expected if cmd not in output]
    assert not missing, f"Missing commands in CLI help: {missing}"


# ---------------------------------------------------------------------------
# Key entrypoints exist
# ---------------------------------------------------------------------------

def test_key_entrypoints_exist():
    """Critical entrypoint files must be present."""
    paths = [
        SRC / "ui" / "streamlit_app.py",
        SRC / "cli" / "main.py",
        SRC / "cli" / "__main__.py",
    ]
    missing = [str(p) for p in paths if not p.exists()]
    assert not missing, f"Missing entrypoints: {missing}"


# ---------------------------------------------------------------------------
# Service layer importable
# ---------------------------------------------------------------------------

def test_backtest_service_importable():
    """The backtest service must be importable."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); "
         "from ui.services.backtest_service import run_single_backtest; "
         "print('ok')"],
        capture_output=True, text=True,
        cwd=str(SRC),
    )
    assert result.returncode == 0 and "ok" in result.stdout, (
        f"backtest_service import failed:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# util.py has its compat note
# ---------------------------------------------------------------------------

def test_util_py_has_compat_note():
    """src/util.py must contain a docstring/comment indicating it is a compat re-export layer."""
    util_path = SRC / "util.py"
    assert util_path.exists(), "src/util.py not found"
    content = _file_content(util_path)
    assert "compat" in content.lower() or "re-export" in content.lower() or "backward" in content.lower(), (
        "src/util.py does not contain a backward-compat note"
    )


# ---------------------------------------------------------------------------
# AGENTS.md exists
# ---------------------------------------------------------------------------

def test_agents_md_exists():
    """AGENTS.md must exist at the repo root."""
    agents_path = SRC.parent / "AGENTS.md"
    assert agents_path.exists(), "AGENTS.md not found at repo root"


# ---------------------------------------------------------------------------
# Layer boundary contracts (mirrors import-linter forbidden contracts)
# ---------------------------------------------------------------------------

_UPWARD_FORBIDDEN: dict[str, list[str]] = {
    "core": [
        "ui", "backtesting", "portfolio", "strategy",
        "tuning", "reporting", "research", "execution", "data",
    ],
    "config": [
        "ui", "portfolio", "backtesting", "strategy",
        "tuning", "reporting", "research", "execution",
    ],
    "data": [
        "ui", "portfolio", "backtesting", "strategy",
        "tuning", "reporting", "research",
    ],
    "execution": [
        "ui", "portfolio", "backtesting", "strategy",
        "tuning", "reporting", "research",
    ],
}


# Known exceptions that import-linter allows via ignore_imports.
# These are documented architectural debts, not silent workarounds.
_KNOWN_EXCEPTIONS: set[tuple[str, str]] = {
    # data.fundamentals calls strategy scoring functions when saving snapshots.
    # Tracked as future refactor: move score computation out of the data layer.
    ("data/fundamentals.py", "strategy"),
}


def _check_no_upward_imports(source_pkg: str, forbidden: list[str]) -> None:
    violations = []
    for py in _python_files(source_pkg):
        content = _file_content(py)
        rel = str(py.relative_to(SRC))
        for pkg in forbidden:
            # Use word-boundary patterns to avoid matching longer package names
            # (e.g. "from dataclasses" must not match "from data").
            patterns = (f"from {pkg}.", f"from {pkg} ", f"import {pkg}.", f"import {pkg} ")
            if any(p in content for p in patterns):
                if (rel, pkg) not in _KNOWN_EXCEPTIONS:
                    violations.append(f"{rel}: imports {pkg}")
    assert not violations, (
        f"{source_pkg}/ has forbidden upward imports:\n  " + "\n  ".join(violations)
    )


def test_core_has_no_domain_imports():
    _check_no_upward_imports("core", _UPWARD_FORBIDDEN["core"])


def test_config_has_no_domain_imports():
    _check_no_upward_imports("config", _UPWARD_FORBIDDEN["config"])


def test_data_has_no_upward_imports():
    _check_no_upward_imports("data", _UPWARD_FORBIDDEN["data"])


def test_execution_has_no_upward_imports():
    _check_no_upward_imports("execution", _UPWARD_FORBIDDEN["execution"])
