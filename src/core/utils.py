"""core/utils.py — Pure utility functions with no project dependencies."""
import asyncio
import concurrent.futures


def safe_float(value, default: float | None = None) -> float | None:
    """Convert value to float, returning default on failure or None/NaN input."""
    try:
        if value is None or value == "" or str(value).lower() == "nan":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def run_async(coro):
    """Run a coroutine from sync code, handling an already-running loop (e.g. Jupyter)."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


def load_config_raw() -> dict:
    """Read cfg/config.yaml and return the raw dict. Returns {} on any error."""
    import yaml

    from core.paths import CONFIG_FILE
    try:
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
