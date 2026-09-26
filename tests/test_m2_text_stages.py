import pytest
from pydantic import ValidationError

from conftest import ROOT, FakeBackend, ch, make_script
from katha.adapt import adapt_source, merge_bible
from katha.config import RateLimitConfig
from katha.ingest import chunk_text, ingest
from katha.llm import LLM, LLMOutputError
from katha.models import BibleEntry, Line, Script
from katha.render import estimate_script, plan_requests
from katha.retry import RateLimiter, call_with_retries
from katha.script import build_script

# ---------------- ingest / chunking ----------------

def test_sample_story_ingests_as_one_chunk():
    src = ingest(ROOT / "samples" / "the_last_pie.md", 2500)
    assert src.title == "The Last Pie"
    assert len(src.chunks) == 1
    assert "---" not in src.chunks[0].text        # scene rules are dropped
    assert 650 < src.chunks[0].words < 850


def test_chunking_respects_chapters_and_size():
    text = "Chapter 1\n\n" + "\n\n".join(["word " * 50] * 5) + "\n\nChapter 2\n\n" + "short para."
    chunks = chunk_text(text, max_words=120)
    assert [c.heading for c in chunks][0] == "Chapter 1"
    assert chunks[-1].heading == "Chapter 2"
    assert all(c.words <= 120 for c in chunks)
    assert sum(c.words for c in chunks) == 250 + 2


def test_huge_paragraph_split_by_sentences():
    par = " ".join(f"Sentence number {i} is here." for i in range(100))  # 500 words, 1 paragraph
    chunks = chunk_text(par, max_words=100)
    assert len(chunks) >= 5
    assert all(c.words <= 100 for c in chunks)
    assert all(c.text.rstrip().endswith(".") for c in chunks)


# ---------------- schema validation ----------------

@pytest.mark.parametrize("bad", ["(whispering) Come here.", "Come [door slams] here.", "*laughs* No."])
def test_stage_directions_rejected(bad):
    with pytest.raises(ValidationError, match="stage direction"):
        Line(speaker="a", text=bad)


def test_unknown_tag_rejected_known_tag_extracted():
    with pytest.raises(ValidationError, match="unsupported inline tag"):
        Line(speaker="a", text="Hello <door creak> there")
    ln = Line(speaker="a", text="Oh <sigh> fine. <short pause> Go.", tags=["bogus"])
    assert ln.tags == ["sigh", "short pause"]
    assert ln.spoken_text == "Oh fine. Go."


def test_script_requires_narrator_and_known_speakers():
    s = make_script([["asha"]])
    assert s.character("narrator")
    data = s.model_dump()
    data["characters"] = [c for c in data["characters"] if c["id"] != "narrator"]
    with pytest.raises(ValidationError, match="narrator"):
        Script.model_validate(data)
    data = s.model_dump()
    data["scenes"][0]["lines"][0]["speaker"] = "ghost"
    with pytest.raises(ValidationError, match="unknown speaker"):
        Script.model_validate(data)


def test_character_id_must_be_snake_case():
    with pytest.raises(ValidationError):
        ch("Rina Di")


# ---------------- speaker-chunk splitting ----------------

def test_designed_mode_one_request_per_speaker_run():
    s = make_script([["narrator", "narrator", "a", "b", "a", "a", "narrator"]])
    reqs = plan_requests(s, "designed")
    assert [r.speakers for r in reqs] == [["narrator"], ["a"], ["b"], ["a"], ["narrator"]]
    assert reqs[3].line_idx == [4, 5]
    assert [r.key for r in reqs][:2] == ["scene_01_000", "scene_01_001"]


def test_prebuilt_mode_max_two_speakers_and_narrator_alone():
    s = make_script([["narrator", "a", "b", "a", "c", "b", "narrator", "narrator", "a"]])
    reqs = plan_requests(s, "prebuilt")
    assert all(len(r.speakers) <= 2 for r in reqs)
    assert all(r.speakers == ["narrator"] or "narrator" not in r.speakers for r in reqs)
    assert [r.speakers for r in reqs] == [["narrator"], ["a", "b"], ["c", "b"], ["narrator"], ["a"]]
    # every line is covered exactly once, in order
    assert [i for r in reqs for i in r.line_idx] == list(range(9))


def test_requests_split_on_word_budget_and_max_scenes():
    s = make_script([["a"] * 10, ["b"], ["c"]])
    reqs = plan_requests(s, "designed", max_words=15)   # each line is 5 words
    assert [len(r.line_idx) for r in reqs if r.scene_id == "scene_01"] == [3, 3, 3, 1]
    assert {r.scene_id for r in plan_requests(s, "designed", max_scenes=2)} == {"scene_01", "scene_02"}


def test_estimate_uses_config_prices(cfg):
    s = make_script([["narrator", "a"] * 20])
    est = estimate_script(s, cfg)
    assert est.audio_seconds > 0
    assert est.audio_tokens == int(est.audio_seconds * 25)
    cfg.pricing.models["gemini-3.8-flash-tts"].output *= 2
    assert estimate_script(s, cfg).tts_cost_usd > est.tts_cost_usd


# ---------------- LLM layer (mocked) ----------------

def _adapt_out(text="Rina walked to the para sweet shop " * 5, bible=None):
    return {"adapted_text": text,
            "changes": [{"kind": "character", "original": "Martha", "adapted": "Rina", "reason": "local name"}],
            "bible_updates": bible or [{"kind": "character", "original": "Martha", "adapted": "Rina"}],
            "plot_beats": ["she walks"]}


def test_llm_repairs_invalid_output_once(cfg):
    fb = FakeBackend(['{"adapted_text": "too short", "changes": [], "bible_updates": [], "plot_beats": []}',
                      _adapt_out()])
    from katha.models import AdaptChunkOut
    out = LLM(fb, cfg).structured("sys", "prompt", AdaptChunkOut)
    assert out.changes[0].adapted == "Rina"
    assert "invalid" in fb.prompts[1] and "too short" in fb.prompts[1]


def test_llm_gives_up_after_repair(cfg):
    from katha.models import AdaptChunkOut
    fb = FakeBackend(["not json", "still not json"])
    with pytest.raises(LLMOutputError):
        LLM(fb, cfg).structured("sys", "prompt", AdaptChunkOut)


def test_bible_first_mapping_wins():
    b = merge_bible([BibleEntry(kind="character", original="Martha", adapted="Rina")],
                    [BibleEntry(kind="character", original="martha", adapted="Mala"),
                     BibleEntry(kind="place", original="Vermont", adapted="Bardhaman")])
    assert [x.adapted for x in b] == ["Rina", "Bardhaman"]


def test_adapt_passes_bible_forward(cfg):
    from katha.models import Chunk, Source
    src = Source(title="Martha's Pie", chunks=[Chunk(index=0, text="Martha went out."),
                                               Chunk(index=1, text="Martha came back.")])
    fb = FakeBackend([_adapt_out(), _adapt_out(bible=[])])
    a = adapt_source(src, "Bengali, West Bengal", "en", LLM(fb, cfg))
    assert "Martha → Rina" in fb.prompts[1]
    assert "END OF THE PREVIOUS ADAPTED CHUNK" in fb.prompts[1]
    assert a.adapted_title == "Rina's Pie"
    assert len(a.change_log) == 2


def test_script_repairs_undeclared_speaker(cfg):
    from katha.models import Adaptation, AdaptedChunk
    ad = Adaptation(title="T", adapted_title="T", culture="c", lang="en", llm="x", bible=[],
                    chunks=[AdaptedChunk(**_adapt_out(), index=0, source_words=10)])
    scene = {"title": "Kitchen", "lines": [{"speaker": "narrator", "text": "Night fell."},
                                           {"speaker": "rina", "text": "Is it ready?"}]}
    fb = FakeBackend([{"covered_beats": [1], "new_characters": [], "scenes": [scene]},
                      {"covered_beats": [1], "new_characters": [ch("rina").model_dump()], "scenes": [scene]}])
    s = build_script(ad, LLM(fb, cfg))
    assert "not declared" in fb.prompts[1]
    assert [c.id for c in s.characters] == ["narrator", "rina"]
    assert s.scenes[0].id == "scene_01"


# ---------------- retry / rate limiting ----------------

class _Err(Exception):
    def __init__(self, code):
        self.status_code = code


def test_retry_on_429_then_success():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise _Err(429)
        return "ok"

    cfg = RateLimitConfig(max_retries=5, backoff_base_seconds=1, backoff_max_seconds=8)
    assert call_with_retries(fn, cfg, sleep=sleeps.append) == "ok"
    assert len(sleeps) == 2 and 0.5 <= sleeps[0] <= 1 and 1 <= sleeps[1] <= 2


def test_no_retry_on_400():
    def fn():
        raise _Err(400)
    with pytest.raises(_Err):
        call_with_retries(fn, RateLimitConfig(), sleep=lambda s: None)


def test_rate_limiter_spaces_calls():
    t = [0.0]
    slept = []
    rl = RateLimiter(60, clock=lambda: t[0], sleep=lambda s: slept.append(s))
    rl.wait(); rl.wait(); rl.wait()
    assert slept == [1.0, 2.0]


def test_daily_quota_not_retried_and_falls_back(cfg):
    from katha.models import AdaptChunkOut
    from katha.retry import QuotaExhausted, is_daily_quota

    class Quota(Exception):
        status_code = 429

        def __str__(self):
            return "Rate limit exceeded (limit: 20 requests per day on Free Tier)"

    assert is_daily_quota(Quota()) and not is_daily_quota(_Err(429))

    class FallbackBackend(FakeBackend):
        models = ["big", "small"]
        model = "big"

        def complete_json(self, system, prompt, schema):
            if self.model == "big":
                raise Quota()
            return super().complete_json(system, prompt, schema)

        def fallback(self):
            if self.model == "small":
                return False
            self.model = "small"
            return True

    fb = FallbackBackend([_adapt_out()])
    out = LLM(fb, cfg).structured("s", "p", AdaptChunkOut)
    assert fb.model == "small" and out.changes

    fb2 = FallbackBackend([])
    fb2.models = ["big"]
    fb2.fallback = lambda: False
    with pytest.raises(QuotaExhausted):
        LLM(fb2, cfg).structured("s", "p", AdaptChunkOut)


def test_script_repairs_skipped_plot_beat(cfg):
    from katha.models import Adaptation, AdaptedChunk
    out = _adapt_out()
    out["plot_beats"] = ["she bakes", "she wins"]
    ad = Adaptation(title="T", adapted_title="T", culture="c", lang="en", llm="x", bible=[],
                    chunks=[AdaptedChunk(**out, index=0, source_words=10)])
    scene = {"title": "Kitchen", "lines": [{"speaker": "narrator", "text": "She bakes."}]}
    fb = FakeBackend([{"covered_beats": [1], "new_characters": [], "scenes": [scene]},
                      {"covered_beats": [1, 2], "new_characters": [], "scenes": [scene, scene]}])
    s = build_script(ad, LLM(fb, cfg))
    assert "1. she bakes" in fb.prompts[0] and "2. she wins" in fb.prompts[0]
    assert "plot beats [2] are not dramatised" in fb.prompts[1]
    assert [x.id for x in s.scenes] == ["scene_01", "scene_02"]
