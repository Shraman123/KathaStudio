import json

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from conftest import make_script
from katha.config import StitchConfig
from katha.models import Adaptation, AdaptedChunk
from katha.render import render
from katha.report import build_report
from katha.runs import RunDir, write_json
from katha.stitch import stitch, stitch_segments
from katha.voices import cast_voices
from test_m3_voices_render import FakeTTS


def tone(ms, gain_db=0.0):
    return Sine(440, sample_rate=24000).to_audio_segment(duration=ms).set_channels(1).apply_gain(gain_db)


def test_pause_lengths_between_lines_and_scenes():
    cfg = StitchConfig(pause_between_lines_ms=300, pause_between_scenes_ms=1000)
    ep = stitch_segments([("s1", tone(500)), ("s1", tone(500)), ("s2", tone(500))], cfg)
    assert abs(len(ep) - (1500 + 300 + 1000)) <= 5


def test_loudness_is_matched_and_peak_limited():
    cfg = StitchConfig(target_dbfs=-16)
    quiet, loud = tone(1000, -30), tone(1000, -3)
    ep = stitch_segments([("s1", quiet), ("s1", loud)], cfg)
    a, b = ep[:1000], ep[1300:2300]
    assert abs(a.dBFS - b.dBFS) < 1.0          # both chunks now sit at the same level
    assert ep.max_dBFS <= -0.99


def test_silent_chunk_does_not_crash():
    ep = stitch_segments([("s1", AudioSegment.silent(200, frame_rate=24000)), ("s1", tone(200))], StitchConfig())
    assert len(ep) > 400


@pytest.fixture
def rendered(tmp_path, cfg):
    run = RunDir(tmp_path / "run").ensure()
    script = make_script([["narrator", "asha", "bina"], ["narrator", "asha"]])
    write_json(run.script, script)
    write_json(run.adaptation, Adaptation(
        title="The Last Pie", adapted_title="Shesh Payesh", culture="Bengali, West Bengal", lang="en",
        llm="fake:1", bible=[],
        chunks=[AdaptedChunk(adapted_text="x " * 30, index=0, source_words=30, plot_beats=["b1"],
                             bible_updates=[], changes=[{"kind": "food", "original": "apple pie",
                                                         "adapted": "nolen gurer payesh", "reason": "winter sweet"}])]))
    (run.root / "llm_usage.jsonl").write_text(
        json.dumps({"stage": "adapt[0]", "model": "gemini-3.8-flash", "input_tokens": 1000, "output_tokens": 2000}) + "\n"
        + json.dumps({"stage": "script[0]", "model": "mystery-model", "input_tokens": 10, "output_tokens": 10}) + "\n",
        encoding="utf-8")
    cast_voices(run, cfg, tts=FakeTTS())
    return run


def test_stitch_and_report_end_to_end(rendered, cfg, monkeypatch):
    import katha.stitch as st
    monkeypatch.setattr(st, "ffmpeg_available", lambda: False)   # keep the test independent of ffmpeg
    render(rendered, cfg, tts=FakeTTS())
    out = stitch(rendered, cfg)
    assert rendered.episode_wav.exists() and out["mp3"] is None
    # 5 chunks: 3 × 0.3s + 2 × 0.3s audio, 3 line gaps (0.9s) + 1 scene gap (1.0s)
    assert out["seconds"] == pytest.approx(1.5 + 0.9 + 1.0, abs=0.05)

    md = build_report(rendered, cfg, out)
    assert "Adapted from \"The Last Pie\"" in md and "nolen gurer payesh" in md
    assert "| gemini-3.8-flash | 1 | 1,000 | 2,000 | $0.0083 |" in md   # (1000×0.75 + 2000×3.75)/1M = 0.00825
    assert "mystery-model" in md and "no price in config" in md
    assert "free tier" in md
    assert "`gemini-3.8-flash`: 1 requests vs 20/day free → fits" in md
    assert "Free-tier daily limit unknown" in md                         # TTS limit not known yet


def test_report_flags_partial_render(rendered, cfg, monkeypatch):
    import katha.stitch as st
    monkeypatch.setattr(st, "ffmpeg_available", lambda: False)
    render(rendered, cfg, max_scenes=1, tts=FakeTTS())
    md = build_report(rendered, cfg, stitch(rendered, cfg))
    assert "Partial render: only 1 of 2 scenes" in md


def test_stitch_missing_chunk_is_clear(rendered, cfg):
    render(rendered, cfg, tts=FakeTTS())
    (rendered.audio_dir / "scene_01_001.wav").unlink()
    with pytest.raises(FileNotFoundError, match="it resumes"):
        stitch(rendered, cfg)
