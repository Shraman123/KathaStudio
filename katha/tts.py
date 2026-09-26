"""Thin wrapper over the Gemini 3.8 TTS + Voices APIs (shapes verified in API_NOTES.md).

Everything that touches the network lives here, so voices.py and render.py can be
tested with a fake client.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Config, require_key
from .retry import RateLimiter, call_with_retries

log = logging.getLogger(__name__)

LANG_CODES = {"en": "en-IN", "hi": "hi-IN", "bn": "bn-IN"}


@dataclass
class Segment:
    speaker: str  # character id
    text: str
    style: str = ""


class GeminiTTS:
    def __init__(self, cfg: Config):
        from google import genai

        self.cfg = cfg
        self.model = cfg.tts.model
        self.client = genai.Client(api_key=require_key("GEMINI_API_KEY"))
        self.limiter = RateLimiter(cfg.rate_limit.requests_per_minute)
        self.calls = 0

    def _call(self, fn, what: str):
        self.calls += 1
        return call_with_retries(fn, self.cfg.rate_limit, self.limiter, what=what)

    # ---------- voices ----------

    def design_voice(self, display_name: str, description: str, gender: str, lang: str) -> tuple[str, bytes | None]:
        voice = {
            "model": self.model, "type": "prompted", "display_name": display_name[:60],
            "language_code": LANG_CODES[lang], "prompted": {"input": description},
        }
        if gender in ("male", "female"):
            voice["gender"] = gender
        v = self._call(lambda: self.client.voices.create(store=True, voice=voice), f"design:{display_name}")
        sample = base64.b64decode(v.sample_audio.data) if getattr(v, "sample_audio", None) and v.sample_audio.data else None
        return v.id, sample

    def replicate_voice(self, display_name: str, source_wav: Path, consent_wav: Path) -> str:
        def b64(p: Path) -> str:
            return base64.b64encode(Path(p).read_bytes()).decode()

        voice = {
            "model": self.model, "type": "replicated", "display_name": display_name[:60],
            "replicated": {
                "source_audio": {"mime_type": "audio/wav", "data": b64(source_wav)},
                "consent_audio": {"mime_type": "audio/wav", "data": b64(consent_wav)},
            },
        }
        v = self._call(lambda: self.client.voices.create(store=True, voice=voice), "replicate:narrator")
        return v.id

    def library_voices(self, lang: str, gender: str | None, search: str | None = None) -> list[str]:
        """Prebuilt voices from the Extended Voice Library for this language (e.g. en-IN)."""
        kwargs = {"language_code": [LANG_CODES[lang]], "type_": ["prebuilt"], "page_size": 50}
        if search:
            kwargs["search"] = search
        if gender in ("male", "female"):
            kwargs["gender"] = [gender]
        r = self._call(lambda: self.client.voices.list(**kwargs), f"voices.list:{lang}:{gender}")
        return [v.id for v in (r.voices or [])]

    # ---------- speech ----------

    def synthesize(self, segments: list[Segment], voices: dict[str, str], multi_speaker: bool) -> bytes:
        """Return WAV bytes (24 kHz mono 16-bit with a RIFF header, the default for unary requests)."""
        content = []
        for s in segments:
            ann = {"type": "speech_metadata"}
            if multi_speaker:
                ann["speaker"] = s.speaker
            if s.style:
                ann["style"] = s.style
            item = {"type": "text", "text": s.text}
            if len(ann) > 1:
                item["annotations"] = [ann]
            content.append(item)

        if multi_speaker:
            speakers = list(dict.fromkeys(s.speaker for s in segments))
            speech_config = {"mode": "conversational",
                             "speakers": [{"speaker": sp, "voice": voices[sp]} for sp in speakers]}
        else:
            speech_config = [{"voice": voices[segments[0].speaker]}]

        i = self._call(lambda: self.client.interactions.create(
            model=self.model,
            input=[{"type": "user_input", "content": content}],
            response_format={"type": "audio"},
            generation_config={"speech_config": speech_config},
        ), f"tts:{'+'.join(dict.fromkeys(s.speaker for s in segments))}")
        audio = getattr(i, "output_audio", None)
        if not audio or not audio.data:
            raise RuntimeError("TTS response contained no audio")
        return base64.b64decode(audio.data)
