import os

import pytest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
gr = pytest.importorskip("gradio")

from conftest import make_script  # noqa: E402
from katha.models import Adaptation, AdaptedChunk, Line, Scene  # noqa: E402
from katha.ui import build_app, change_rows, list_runs, script_html, title_html  # noqa: E402


def test_script_html_escapes_and_pills_tags():
    s = make_script([["narrator", "asha"]])
    s.scenes[0].lines.append(Line(speaker="asha", text="Fish & rice? <sigh> Five > three.", style="tired"))
    h = script_html(s)
    assert "<code>&lt;sigh&gt;</code>" in h
    assert "Fish &amp; rice?" in h and "Five &gt; three." in h        # user text is escaped
    assert "<em>tired · </em>" in h


def test_adaptation_views():
    a = Adaptation(title="The Last Pie", adapted_title="The Last Bowl", culture="Bengali", lang="en", llm="g",
                   bible=[], chunks=[AdaptedChunk(adapted_text="w " * 30, index=0, source_words=1,
                                                  plot_beats=["a", "b"], bible_updates=[],
                                                  changes=[{"kind": "food", "original": "pie", "adapted": "payesh",
                                                            "reason": "r"}])])
    assert change_rows(a) == [["food", "pie", "payesh", "r"]]
    assert "The Last Bowl" in title_html(a) and "2 plot beats" in title_html(a)


def test_app_builds(cfg):
    assert list_runs(cfg) == []
    app = build_app(cfg)
    assert isinstance(app, gr.Blocks)
