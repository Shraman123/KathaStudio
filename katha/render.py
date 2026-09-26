"""Stage 5: plan and synthesize TTS requests for each scene.

Request planning (a pure function, unit-tested) follows the API rules in API_NOTES.md:
- voice_mode=designed: designed/replicated voices can't share a request, so each run of
  consecutive lines by one speaker is one request.
- voice_mode=prebuilt: up to 2 speakers per request (prebuilt voices only). Narrator
  lines always get their own request.
"""
from __future__ import annotations

import hashlib
import json
import logging
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Config
from .log import kv
from .models import NARRATOR_ID, Script
from .runs import RunDir
from .tts import Segment

if TYPE_CHECKING:
    from .voices import VoiceCast

log = logging.getLogger(__name__)

MAX_WORDS_PER_REQUEST = 600  # far below the 8,192-token input / ~655 s output limits


@dataclass
class TTSRequest:
    scene_id: str
    index: int                      # position within the scene
    line_idx: list[int] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable filename stem, used to resume."""
        return f"{self.scene_id}_{self.index:03d}"


def plan_requests(script: Script, voice_mode: str, max_scenes: int | None = None,
                  max_words: int = MAX_WORDS_PER_REQUEST) -> list[TTSRequest]:
    reqs: list[TTSRequest] = []
    for scene in script.scenes[:max_scenes]:
        cur: TTSRequest | None = None
        cur_words = 0
        n = 0
        for i, line in enumerate(scene.lines):
            w = len(line.spoken_text.split())
            if cur is not None:
                if voice_mode == "designed":
                    fits = cur.speakers == [line.speaker]
                else:
                    is_narr = line.speaker == NARRATOR_ID
                    cur_narr = cur.speakers == [NARRATOR_ID]
                    speakers = set(cur.speakers) | {line.speaker}
                    fits = (is_narr and cur_narr) or (not is_narr and not cur_narr and len(speakers) <= 2)
                if not fits or cur_words + w > max_words:
                    reqs.append(cur)
                    cur, cur_words = None, 0
            if cur is None:
                cur = TTSRequest(scene.id, n)
                n += 1
            cur.line_idx.append(i)
            if line.speaker not in cur.speakers:
                cur.speakers.append(line.speaker)
            cur_words += w
        if cur is not None:
            reqs.append(cur)
    return reqs


@dataclass
class Estimate:
    requests: int
    audio_seconds: float
    input_tokens: int
    audio_tokens: int
    tts_cost_usd: float
    lines: int
    scenes: int

    def summary(self, cfg: Config) -> str:
        m, s = divmod(round(self.audio_seconds), 60)
        tier = "free tier (cost $0 while free_tier: true)" if cfg.pricing.free_tier else "paid tier"
        return (f"{self.scenes} scenes · {self.lines} lines · {self.requests} TTS requests "
                f"({cfg.tts.voice_mode} voices)\n"
                f"estimated audio ≈ {m}m{s:02d}s → {self.audio_tokens:,} audio tokens\n"
                f"estimated TTS cost at paid rates ≈ ${self.tts_cost_usd:.4f} ({cfg.tts.model}); {tier}\n"
                f"at {cfg.rate_limit.requests_per_minute:g} RPM, synthesis takes ≥ "
                f"{self.requests / cfg.rate_limit.requests_per_minute:.1f} min"
                + self._quota_note(cfg))

    def _quota_note(self, cfg: Config) -> str:
        rpd = cfg.pricing.free_tier_rpd.get(cfg.tts.model)
        if not (cfg.pricing.free_tier and rpd and self.requests > rpd):
            return ""
        days = -(-self.requests // rpd)
        tip = (" Try --voice-mode prebuilt (2-speaker requests) to cut the count."
               if cfg.tts.voice_mode == "designed" else "")
        return (f"\n⚠ {self.requests} requests exceed the free tier's ~{rpd}/day for {cfg.tts.model}: "
                f"about {days} days of quota (render resumes each day), or enable billing.{tip}")


def estimate_script(script: Script, cfg: Config, max_scenes: int | None = None) -> Estimate:
    reqs = plan_requests(script, cfg.tts.voice_mode, max_scenes)
    scenes = script.scenes[:max_scenes]
    wpm = cfg.estimate.words_per_minute.get(script.lang, 150)
    words = sum(len(ln.spoken_text.split()) for s in scenes for ln in s.lines)
    tags = sum(len(ln.tags) for s in scenes for ln in s.lines)
    lines = sum(len(s.lines) for s in scenes)
    secs = (words / wpm * 60
            + tags * 0.6
            + lines * cfg.stitch.pause_between_lines_ms / 1000
            + len(scenes) * cfg.stitch.pause_between_scenes_ms / 1000)
    # Input tokens: transcript + style text, about 4 chars per token for English (more for Indic scripts)
    chars = sum(len(ln.text) + len(ln.style) for s in scenes for ln in s.lines)
    in_tok = int(chars / (4 if script.lang == "en" else 2.5))
    audio_tok = int(secs * cfg.pricing.audio_tokens_per_second)
    price = cfg.price_for(cfg.tts.model)
    cost = in_tok / 1e6 * price.input + audio_tok / 1e6 * price.output
    return Estimate(len(reqs), secs, in_tok, audio_tok, cost, lines, len(scenes))


def estimate(run: RunDir, cfg: Config, *, max_scenes: int | None = None) -> Estimate:
    script = Script.model_validate_json(run.require(run.script, "script").read_text(encoding="utf-8"))
    est = estimate_script(script, cfg, max_scenes)
    print(est.summary(cfg))
    return est


def wav_seconds(path: Path) -> float | None:
    """Duration of a valid WAV, or None if it's missing or broken (a crash mid-write)."""
    try:
        with wave.open(str(path), "rb") as w:
            n, rate = w.getnframes(), w.getframerate()
        return n / rate if n > 0 else None
    except (FileNotFoundError, wave.Error, EOFError):
        return None


def request_fingerprint(req: TTSRequest, script: Script, cast: "VoiceCast", model: str) -> str:
    scene = next(s for s in script.scenes if s.id == req.scene_id)
    parts = [model] + [f"{scene.lines[i].speaker}|{cast.voices[scene.lines[i].speaker].voice}|"
                       f"{scene.lines[i].style}|{scene.lines[i].text}" for i in req.line_idx]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def load_manifest(run: RunDir) -> dict:
    p = run.audio_dir / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def render_script(run: RunDir, script: Script, cast: "VoiceCast", cfg: Config, tts,
                  max_scenes: int | None = None, on_progress=None, upgrade: bool = False) -> dict:
    """Render every planned request that isn't already on disk and up to date.

    on_progress(done, total, key), if given, is called after each chunk (the UI uses it).
    If the primary TTS model's daily quota runs out, the rest render on tts.fallback_models
    (same voice ids). The manifest records each chunk's model. `upgrade=True` re-renders
    fallback chunks on the primary model.
    """
    from .retry import QuotaExhausted

    run.ensure()
    primary = cfg.tts.model
    allowed = [primary, *cfg.tts.fallback_models]
    reqs = plan_requests(script, cfg.tts.voice_mode, max_scenes)
    manifest = load_manifest(run)
    stats = {"requests": len(reqs), "synthesized": 0, "skipped": 0}
    scenes = {s.id: s for s in script.scenes}
    for n, req in enumerate(reqs, 1):
        out = run.audio_dir / f"{req.key}.wav"
        entry = manifest.get(req.key)
        if entry and wav_seconds(out):
            made_with = entry.get("model", primary)
            fresh = entry.get("fingerprint") == request_fingerprint(req, script, cast, made_with)
            if fresh and made_with in allowed and not (upgrade and made_with != primary):
                stats["skipped"] += 1
                if on_progress:
                    on_progress(n, len(reqs), req.key)
                continue
        scene = scenes[req.scene_id]
        segments = [Segment(scene.lines[i].speaker, scene.lines[i].text, scene.lines[i].style)
                    for i in req.line_idx]
        voices = {sp: cast.voices[sp].voice for sp in req.speakers}
        multi = len(req.speakers) == 2
        while True:
            try:
                wav = tts.synthesize(segments, voices, multi_speaker=multi)
                break
            except QuotaExhausted:
                old = getattr(tts, "model", primary)
                fallback = getattr(tts, "fallback", None)
                if upgrade or not (fallback and fallback()):
                    raise
                log.warning("TTS daily quota exhausted; continuing on the fallback model with the same voices",
                            extra=kv(from_model=old, to_model=tts.model))
        model_used = getattr(tts, "model", primary)
        tmp = out.with_suffix(".part")
        tmp.write_bytes(wav)
        tmp.replace(out)
        secs = wav_seconds(out)
        if secs is None:
            out.unlink(missing_ok=True)
            raise RuntimeError(f"{req.key}: the API returned audio that is not a valid WAV")
        manifest[req.key] = {"scene": req.scene_id, "lines": req.line_idx, "speakers": req.speakers,
                             "model": model_used,
                             "fingerprint": request_fingerprint(req, script, cast, model_used),
                             "seconds": round(secs, 2)}
        _save_manifest(run, manifest)  # after every chunk, so a crash loses at most one
        stats["synthesized"] += 1
        if on_progress:
            on_progress(n, len(reqs), req.key)
        log.info("chunk rendered", extra=kv(n=f"{n}/{len(reqs)}", key=req.key, model=model_used,
                                           speakers="+".join(req.speakers), secs=f"{secs:.1f}"))
    # Keep only the keys in the current plan, in order
    stats["keys"] = [r.key for r in reqs]
    stats["seconds"] = sum(manifest[k]["seconds"] for k in stats["keys"])
    stats["models"] = {k: manifest[k].get("model", primary) for k in stats["keys"]}
    return stats


def _save_manifest(run: RunDir, manifest: dict) -> None:
    p = run.audio_dir / "manifest.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(p)


def render(run: RunDir, cfg: Config, *, max_scenes: int | None = None, tts=None, on_progress=None,
           upgrade: bool = False) -> dict:
    from .voices import load_cast

    script = Script.model_validate_json(run.require(run.script, "script").read_text(encoding="utf-8"))
    run.require(run.voices, "voices")
    cast = load_cast(run, cfg)
    missing = [c.id for c in script.characters if c.id not in cast.voices]
    if missing:
        raise FileNotFoundError(f"No voices cast for {missing}. Run `katha voices {run.root}` first.")
    if tts is None:
        from .tts import GeminiTTS
        tts = GeminiTTS(cfg)
    stats = render_script(run, script, cast, cfg, tts, max_scenes, on_progress, upgrade)
    stats["max_scenes"] = max_scenes
    (run.root / "render_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("render done", extra=kv(synthesized=stats["synthesized"], skipped=stats["skipped"],
                                    audio=f"{stats['seconds']:.1f}s"))
    return stats
