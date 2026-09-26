from pathlib import Path

import pytest
from typer.testing import CliRunner

from katha.cli import app
from katha.config import ConfigError, load_config, mask, require_key
from katha.runs import RunDir, slugify

ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()


def test_config_loads_and_prices_come_from_yaml():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.tts.model == "gemini-3.8-flash-tts"
    assert cfg.price_for("gemini-3.8-flash-tts").output == 9.00
    assert cfg.pricing.audio_tokens_per_second == 25
    with pytest.raises(ConfigError):
        cfg.price_for("no-such-model")


def test_bad_config_value_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("tts:\n  voice_mode: telepathy\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_missing_key_error_is_clear(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY is not set"):
        require_key("GEMINI_API_KEY")


def test_mask_never_reveals_key():
    secret = "AIzaSyFAKEFAKEFAKE1234"
    assert "1234" not in mask(secret) and "AIza" not in mask(secret)
    assert mask(None) == "(not set)"


def test_slug_and_run_layout(tmp_path):
    assert slugify("The Tiger's Bride (v2).txt") == "the-tiger-s-bride-v2-txt"
    run = RunDir.for_story(tmp_path, Path("stories/My Story.md")).ensure()
    assert run.root.name == "my-story"
    assert run.audio_dir.is_dir()
    with pytest.raises(FileNotFoundError, match="katha adapt"):
        run.require(run.adaptation, "adapt")


def test_cli_stub_exits_cleanly(tmp_path, monkeypatch):
    import katha.stitch as stitch_mod
    monkeypatch.setattr(stitch_mod, "stitch", lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("M4")))
    import katha.render as render_mod
    monkeypatch.setattr(render_mod, "render", lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("M3")))
    r = runner.invoke(app, ["render", str(tmp_path)])
    assert r.exit_code == 2
    assert "not implemented yet" in r.output


def test_cli_voices_requires_both_clips(tmp_path):
    r = runner.invoke(app, ["voices", str(tmp_path), "--narrator-clip", "x.wav"])
    assert r.exit_code == 1


def test_doctor_masks_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSySECRETSECRET9999")
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0
    assert "9999" not in r.output and "SECRET" not in r.output
