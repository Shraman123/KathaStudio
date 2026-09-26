import io
import wave

import pytest

from conftest import make_script
from katha.render import render, wav_seconds
from katha.runs import RunDir, write_json
from katha.voices import VoiceCast, cast_voices, check_wav, load_cast


def wav_bytes(seconds=0.5, rate=24000, channels=1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate) * channels)
    return buf.getvalue()


class FakeTTS:
    def __init__(self, fail_design_for=()):
        self.designed, self.synth_calls, self.fail = [], [], set(fail_design_for)

    def design_voice(self, name, description, gender, lang):
        if name.lower() in self.fail:
            raise RuntimeError("403 voice design not available")
        self.designed.append(name)
        return f"voice_{name.lower()}", wav_bytes(0.2)

    def replicate_voice(self, name, src, consent):
        return "voice_me"

    def library_voices(self, lang, gender, search=None):
        if search == "young":
            return {"male": ["lib_boy"]}.get(gender, [])
        return {"female": ["lib_f1", "lib_f2"], "male": ["lib_m1"]}.get(gender, [])

    def synthesize(self, segments, voices, multi_speaker):
        self.synth_calls.append(([s.speaker for s in segments], multi_speaker, dict(voices)))
        return wav_bytes(0.3 * len(segments))


@pytest.fixture
def run(tmp_path):
    r = RunDir(tmp_path / "run").ensure()
    write_json(r.script, make_script([["narrator", "asha", "bina", "asha"], ["narrator", "bina"]]))
    return r


def test_designed_cast_is_cached(run, cfg):
    tts = FakeTTS()
    cast = cast_voices(run, cfg, tts=tts)
    assert {v.kind for v in cast.voices.values()} == {"designed"}
    assert (run.root / "voices" / "asha.wav").exists()           # preview saved for the UI
    tts2 = FakeTTS()
    cast_voices(run, cfg, tts=tts2)
    assert tts2.designed == []                                    # nothing recreated


def test_description_change_recasts_only_that_character(run, cfg):
    cast_voices(run, cfg, tts=FakeTTS())
    s = run.script.read_text(encoding="utf-8").replace("voice of bina", "voice of bina, now hoarse")
    run.script.write_text(s, encoding="utf-8")
    tts = FakeTTS()
    cast_voices(run, cfg, tts=tts)
    assert tts.designed == ["Bina"]


def test_design_failure_falls_back_to_library(run, cfg):
    cast = cast_voices(run, cfg, tts=FakeTTS(fail_design_for={"asha"}))
    assert cast.voices["asha"].kind == "library" and cast.voices["asha"].voice == "lib_f1"
    assert cast.voices["bina"].kind == "designed"


def test_prebuilt_mode_distinct_library_voices(run, cfg):
    cfg.tts.voice_mode = "prebuilt"
    cast = cast_voices(run, cfg, tts=FakeTTS())
    vs = [cast.voices[c].voice for c in ("narrator", "asha", "bina")]
    assert len(set(vs)) == 3


def test_replication_clip_validation(tmp_path):
    short = tmp_path / "short.wav"
    short.write_bytes(wav_bytes(3))
    with pytest.raises(ValueError, match="10-30s"):
        check_wav(short)
    ok = tmp_path / "ok.wav"
    ok.write_bytes(wav_bytes(12))
    assert 11.9 < check_wav(ok) < 12.1
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not a wav")
    with pytest.raises(ValueError, match="not a readable"):
        check_wav(junk)


def test_narrator_replication(run, cfg, tmp_path):
    src, consent = tmp_path / "me.wav", tmp_path / "consent.wav"
    src.write_bytes(wav_bytes(15))
    consent.write_bytes(wav_bytes(6))
    cast = cast_voices(run, cfg, tts=FakeTTS(), narrator_clip=src, consent_clip=consent)
    assert cast.voices["narrator"].kind == "replicated"


def test_render_one_scene_then_resume(run, cfg):
    cast_voices(run, cfg, tts=FakeTTS())
    tts = FakeTTS()
    stats = render(run, cfg, max_scenes=1, tts=tts)
    # designed: narrator | asha | bina | asha → 4 single-speaker requests
    assert stats["synthesized"] == 4 and [c[0] for c in tts.synth_calls] == [["narrator"], ["asha"], ["bina"], ["asha"]]
    assert all(not multi for _, multi, _ in tts.synth_calls)
    assert wav_seconds(run.audio_dir / "scene_01_000.wav") > 0

    tts2 = FakeTTS()
    stats2 = render(run, cfg, tts=tts2)                 # full render resumes: scene 1 is skipped
    assert stats2["skipped"] == 4 and stats2["synthesized"] == 2


def test_changed_line_rerenders_only_its_chunk(run, cfg):
    cast_voices(run, cfg, tts=FakeTTS())
    render(run, cfg, tts=FakeTTS())
    s = run.script.read_text(encoding="utf-8").replace("bina says something", "bina whispers something", 1)
    run.script.write_text(s, encoding="utf-8")
    tts = FakeTTS()
    stats = render(run, cfg, tts=tts)
    assert stats["synthesized"] == 1 and tts.synth_calls[0][0] == ["bina"]


def test_broken_partial_file_is_redone(run, cfg):
    cast_voices(run, cfg, tts=FakeTTS())
    render(run, cfg, tts=FakeTTS())
    (run.audio_dir / "scene_01_002.wav").write_bytes(b"RIFF garbage")
    tts = FakeTTS()
    assert render(run, cfg, tts=tts)["synthesized"] == 1


def test_prebuilt_render_uses_two_speaker_requests(run, cfg):
    cfg.tts.voice_mode = "prebuilt"
    cast_voices(run, cfg, tts=FakeTTS())
    tts = FakeTTS()
    render(run, cfg, tts=tts)
    assert (["asha", "bina", "asha"], True) in [(sp, m) for sp, m, _ in tts.synth_calls]
    assert all(sp == ["narrator"] for sp, m, _ in tts.synth_calls if "narrator" in sp)


def test_render_without_voices_is_clear(run, cfg):
    with pytest.raises(FileNotFoundError, match="katha voices"):
        render(run, cfg, tts=FakeTTS())


def test_cache_for_other_mode_is_ignored(run, cfg):
    write_json(run.voices, VoiceCast(model=cfg.tts.model, voice_mode="prebuilt"))
    assert load_cast(run, cfg).voices == {} and load_cast(run, cfg).voice_mode == "designed"


def test_minor_falls_back_to_young_library_voice(tmp_path, cfg):
    from conftest import ch
    from katha.models import Line, Scene, Script
    boy = ch("shom", gender="male").model_copy(update={"age": "12"})
    script = Script(title="T", lang="en", characters=[ch("narrator", gender="male"), boy],
                    scenes=[Scene(id="scene_01", title="S", lines=[Line(speaker="shom", text="Hi.")])])
    run = RunDir(tmp_path / "r").ensure()
    write_json(run.script, script)
    cast = cast_voices(run, cfg, tts=FakeTTS(fail_design_for={"shom"}))   # safety policy refuses child design
    assert cast.voices["shom"].voice == "lib_boy"


def test_design_prompt_leads_with_age():
    from conftest import ch
    from katha.voices import design_prompt, is_minor
    c = ch("protima").model_copy(update={"age": "late 60s"})
    assert design_prompt(c).startswith("A female who sounds late 60s:")
    assert is_minor(ch("x").model_copy(update={"age": "12"})) and not is_minor(c)


class QuotaTTS(FakeTTS):
    """Primary model runs out of quota after `ok` calls; the fallback model keeps working."""

    def __init__(self, ok=2, fallbacks=("lite",)):
        super().__init__()
        self.models = ["main", *fallbacks]
        self.model = "main"
        self.ok = ok

    def fallback(self):
        i = self.models.index(self.model) + 1
        if i >= len(self.models):
            return False
        self.model = self.models[i]
        return True

    def synthesize(self, segments, voices, multi_speaker):
        from katha.retry import QuotaExhausted
        if self.model == "main":
            if self.ok <= 0:
                raise QuotaExhausted("daily")
            self.ok -= 1
        return super().synthesize(segments, voices, multi_speaker)


def test_tts_quota_falls_back_and_upgrade_redoes_only_fallback_chunks(run, cfg):
    import json as _j
    cfg.tts.fallback_models = ["lite"]
    cfg.tts.model = "main"
    cfg.pricing.models["main"] = cfg.pricing.models["gemini-3.8-flash-tts"]
    cfg.pricing.models["lite"] = cfg.pricing.models["gemini-3.8-flash-lite-tts"]
    cast_voices(run, cfg, tts=FakeTTS())
    tts = QuotaTTS(ok=2)
    stats = render(run, cfg, tts=tts)
    assert stats["synthesized"] == 6
    models = list(stats["models"].values())
    assert models[:2] == ["main", "main"] and set(models[2:]) == {"lite"}

    again = render(run, cfg, tts=QuotaTTS(ok=0))          # fallback chunks count as done
    assert again["synthesized"] == 0

    up = QuotaTTS(ok=10)
    stats3 = render(run, cfg, tts=up, upgrade=True)       # only the 4 lite chunks are redone on main
    assert stats3["synthesized"] == 4 and set(stats3["models"].values()) == {"main"}
    man = _j.loads((run.audio_dir / "manifest.json").read_text())
    assert all(v["model"] == "main" for v in man.values())


def test_quota_without_fallback_still_stops_cleanly(run, cfg):
    from katha.retry import QuotaExhausted
    cfg.tts.fallback_models = []
    cast_voices(run, cfg, tts=FakeTTS())
    with pytest.raises(QuotaExhausted):
        render(run, cfg, tts=QuotaTTS(ok=1, fallbacks=()))
    assert len(list(run.audio_dir.glob("*.wav"))) == 1      # the finished chunk is kept for resume
