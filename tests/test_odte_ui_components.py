"""tests/test_odte_ui_components.py — the 0DTE pages render (headless, offline).

Streamlit components are normally only exercised by opening a browser, so a rename in the service
layer or a wrong chart API reaches the user before it reaches a test. These run every 0DTE
component's render() against a stubbed streamlit module and assert two things:

  1. it renders without raising, against the repo's real data store;
  2. it renders an EXPLANATION rather than a traceback when the store is empty.

No broker, no network, no LLM, no orders. Nothing here writes to data/odte/.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_COMPONENTS = ["odte_cockpit", "odte_funnel", "odte_ledger", "odte_rails", "odte_replay",
               "odte_context"]


class _Recorder:
    """Collects the user-visible messages a render produced."""

    def __init__(self):
        self.messages: dict[str, list[str]] = {"info": [], "error": [], "warning": [],
                                               "success": [], "caption": []}
        self.charts: list[str] = []


def _fake_streamlit(rec: _Recorder):
    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __getattr__(self, name):
            return _stub(name)

    def _stub(name):
        def fn(*a, **k):
            if name in rec.messages and a:
                rec.messages[name].append(str(a[0]))
            if name == "plotly_chart":
                rec.charts.append(k.get("key"))
            if name == "columns":
                n = a[0] if a and isinstance(a[0], int) else (len(a[0]) if a else 2)
                return [_Ctx() for _ in range(n)]
            if name == "tabs":
                return [_Ctx() for _ in a[0]]
            if name in ("expander", "spinner", "form", "container", "sidebar"):
                return _Ctx()
            if name in ("selectbox", "radio"):
                return a[1][0] if len(a) > 1 and a[1] else None
            if name == "slider":
                return k.get("value", 10)
            if name in ("toggle", "checkbox", "button"):
                return False
            if name == "text_input":
                return ""
            return None
        return fn

    def _cache(*dargs, **dkw):
        if dargs and callable(dargs[0]) and not dkw:
            return dargs[0]
        return lambda fn: fn

    class _ColumnConfig:
        def __getattr__(self, name):
            return lambda *a, **k: None

    class _Runtime:
        @staticmethod
        def exists():
            return False

    class _FakeST(types.ModuleType):
        session_state: dict = {}
        column_config = _ColumnConfig()
        cache_data = staticmethod(_cache)
        cache_resource = staticmethod(_cache)
        runtime = _Runtime()

        def __getattr__(self, name):
            return _stub(name)

    return _FakeST("streamlit")


@pytest.fixture
def stub_streamlit(monkeypatch):
    rec = _Recorder()
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit(rec))
    from ui.services import odte_service as svc
    svc._MEM_CACHE.clear()
    yield rec
    svc._MEM_CACHE.clear()


def _render(name: str):
    import importlib
    mod = importlib.import_module(f"ui.components.{name}")
    importlib.reload(mod)          # rebind the stubbed streamlit inside the module
    mod.render()
    return mod


@pytest.mark.parametrize("component", _COMPONENTS)
def test_component_renders_against_the_repo_store(component, stub_streamlit):
    _render(component)
    assert not stub_streamlit.messages["error"], stub_streamlit.messages["error"]


@pytest.mark.parametrize("component", _COMPONENTS)
def test_component_explains_itself_on_an_empty_store(component, stub_streamlit, tmp_path,
                                                     monkeypatch):
    # A missing artifact must produce a sentence, never a traceback and never a blank panel.
    empty = tmp_path / "odte"
    (empty / "reports").mkdir(parents=True)
    import ui.utils as U
    monkeypatch.setattr(U, "ODTE_DATA_DIR", empty)
    monkeypatch.setattr(U, "ODTE_REPORT_DIR", empty / "reports")
    monkeypatch.setattr(U, "ODTE_SCRAPE_DIR", empty / "scrape")

    _render(component)
    said = [m for k in ("info", "warning", "caption", "success", "error")
            for m in stub_streamlit.messages[k]]
    assert said, f"{component} rendered nothing explanatory on an empty store"


def test_charts_have_unique_keys(stub_streamlit):
    # Duplicate plotly keys silently collapse charts into one another in Streamlit.
    keys = []
    for component in _COMPONENTS:
        stub_streamlit.charts.clear()
        _render(component)
        keys.extend(k for k in stub_streamlit.charts if k)
    assert len(keys) == len(set(keys)), f"duplicate chart keys: {keys}"


def test_section_wires_every_tab(stub_streamlit):
    import importlib

    from ui.sections import odte
    importlib.reload(odte)
    odte.render()
    assert not stub_streamlit.messages["error"]


def test_components_go_through_the_service_layer():
    # Components render; they must not reach into the data layer or the filesystem themselves.
    root = Path(__file__).resolve().parent.parent / "src" / "ui" / "components"
    for component in _COMPONENTS:
        src = (root / f"{component}.py").read_text()
        assert "from data.odte_journal import" not in src, (
            f"{component} imports the journal directly — joins belong in odte_service")
        for forbidden in ("place_order", "submit_order", "cancel_order"):
            assert forbidden not in src, f"{component} must never reference {forbidden}"
