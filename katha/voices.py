"""Stage 4: cast a voice for each character and cache the ids in 04_voices.json.

- voice_mode=designed: one designed voice per character, built from its voice_description.
  If voice design fails for a character, that character falls back to a library voice.
- voice_mode=prebuilt: library voices for the target locale (e.g. en-IN), with distinct
  voices per gender where possible. These can share 2-speaker requests.
- --narrator-clip/--consent-clip: replicate YOUR OWN voice for the narrator.

Reruns never recreate a voice. An entry is reused as long as the character's description,
the TTS model and the mode are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import wave
from pathlib import Path

from pydantic import BaseModel

from .config import Config
from .log import kv
from .models import NARRATOR_ID, Character, Script
from .runs import RunDir, write_json

log = logging.getLogger(__name__)

CONSENT_WARNING = (
    "VOICE REPLICATION: only replicate YOUR OWN voice, or a voice whose owner has given "
    "explicit, recorded consent. The consent clip must be the same person reading Google's "
    "consent statement. Cloning someone's voice without consent is harmful and violates "
    "Google's terms."
)
CONSENT_STATEMENT = ("I am the owner of this voice and I consent to Google using this voice "
                     "to create a synthetic voice model.")

# Last-resort fallback if the voice library can't be queried. Genders are approximate:
# the 3.8 docs list these voices without genders, so this is based on how they sound.
PREBUILT_FALLBACK = {
    "female": ["Kore", "Aoede", "Leda", "Despina", "Sulafat", "Achernar", "Gacrux", "Vindemiatrix"],
    "male": ["Charon", "Puck", "Orus", "Iapetus", "Algieba", "Schedar", "Sadaltager", "Achird"],
}


class VoiceEntry(BaseModel):
    voice: str
    kind: str                 # designed | library | prebuilt | replicated
    fingerprint: str          # hash of whatever the voice was built from
    preview: str | None = None


class VoiceCast(BaseModel):
    model: str
    voice_mode: str
    voices: dict[str, VoiceEntry] = {}


def _fp(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def character_fingerprint(c: Character, lang: str) -> str:
    return _fp(c.name, c.gender, c.age, c.voice_description, lang)


def design_prompt(c: Character) -> str:
    # Age goes first and is stated plainly: a live check showed a "late 60s" character designed
    # from a description-led prompt came out sounding like someone in their 30s.
    age = f"{c.age} years old" if c.age.isdigit() else c.age
    who = c.gender if c.gender in ("male", "female") else "person"
    return f"A {who} who sounds {age}: {c.voice_description} Character: {c.name}."


def is_minor(c: Character) -> bool:
    digits = "".join(ch for ch in c.age if ch.isdigit())
    return bool(digits) and int(digits[:2]) < 18


def check_wav(path: Path, min_s: float = 10, max_s: float = 30) -> float:
    """Check a replication clip against the documented requirements. Returns its duration."""
    try:
        with wave.open(str(path), "rb") as w:
            dur = w.getnframes() / w.getframerate()
            ch, rate = w.getnchannels(), w.getframerate()
    except (wave.Error, EOFError) as e:
        raise ValueError(f"{path.name} is not a readable PCM WAV file ({e}). Export as 24 kHz mono 16-bit WAV") from e
    if not (min_s <= dur <= max_s):
        raise ValueError(f"{path.name} is {dur:.1f}s long; the API needs {min_s:.0f}-{max_s:.0f}s")
    if ch != 1 or rate != 24000:
        log.warning("clip is not 24 kHz mono (recommended)", extra=kv(file=path.name, channels=ch, rate=rate))
    return dur


def load_cast(run: RunDir, cfg: Config) -> VoiceCast:
    if run.voices.exists():
        cast = VoiceCast.model_validate_json(run.voices.read_text(encoding="utf-8"))
        if cast.model == cfg.tts.model and cast.voice_mode == cfg.tts.voice_mode:
            return cast
        log.warning("voice cache is for a different model/mode; recasting",
                    extra=kv(cached=f"{cast.model}/{cast.voice_mode}"))
    return VoiceCast(model=cfg.tts.model, voice_mode=cfg.tts.voice_mode)


class _LibraryPicker:
    """Hands out distinct voices per gender: first from the locale library, then the static fallback."""

    def __init__(self, tts, lang: str, used: set[str]):
        self.tts, self.lang, self.used = tts, lang, set(used)
        self._pools: dict[str, list[tuple[str, str]]] = {}

    def _pool(self, gender: str, young: bool = False) -> list[tuple[str, str]]:
        key = f"{gender}:{young}"
        if key not in self._pools:
            pool: list[tuple[str, str]] = []
            try:
                if young:
                    pool += [(v, "library") for v in self.tts.library_voices(self.lang, gender, search="young")]
                pool += [(v, "library") for v in self.tts.library_voices(self.lang, gender)]
            except Exception as e:  # noqa: BLE001 (the library is optional; fall back quietly)
                log.warning("voice library unavailable; using the prebuilt list", extra=kv(error=str(e)[:120]))
            g = gender if gender in PREBUILT_FALLBACK else "female"
            pool += [(v, "prebuilt") for v in PREBUILT_FALLBACK[g]]
            self._pools[key] = pool
        return self._pools[key]

    def pick(self, gender: str, preferred: str | None = None, young: bool = False) -> tuple[str, str]:
        pool = self._pool(gender, young)
        if preferred and preferred not in self.used:
            self.used.add(preferred)
            return preferred, "prebuilt"
        for v, kind in pool:
            if v not in self.used:
                self.used.add(v)
                return v, kind
        return pool[0]  # more characters than voices: reuse one


def cast_script(script: Script, cast: VoiceCast, cfg: Config, tts, previews_dir: Path | None = None,
                narrator_clip: Path | None = None, consent_clip: Path | None = None) -> VoiceCast:
    lang = script.lang
    picker = _LibraryPicker(tts, lang, {e.voice for e in cast.voices.values()})

    for c in script.characters:
        if c.id == NARRATOR_ID and narrator_clip:
            fp = _fp("replicated", hashlib.sha256(Path(narrator_clip).read_bytes()).hexdigest())
        else:
            fp = character_fingerprint(c, lang)
        cached = cast.voices.get(c.id)
        if cached and cached.fingerprint == fp:
            log.info("voice cached", extra=kv(character=c.id, kind=cached.kind, voice=cached.voice))
            continue

        if c.id == NARRATOR_ID and narrator_clip:
            log.warning(CONSENT_WARNING)
            check_wav(Path(narrator_clip))
            check_wav(Path(consent_clip), min_s=2, max_s=30)
            entry = VoiceEntry(voice=tts.replicate_voice(f"{script.title} narrator", narrator_clip, consent_clip),
                               kind="replicated", fingerprint=fp)
        elif cfg.tts.voice_mode == "designed":
            entry = _design_or_fallback(c, lang, fp, tts, picker, previews_dir, cfg)
        else:
            pref = cfg.tts.narrator_prebuilt_voice if c.id == NARRATOR_ID and lang == "en" else None
            voice, kind = picker.pick(c.gender, pref, young=is_minor(c))
            entry = VoiceEntry(voice=voice, kind=kind, fingerprint=fp)
        cast.voices[c.id] = entry
        log.info("voice cast", extra=kv(character=c.id, kind=entry.kind, voice=entry.voice))
    return cast


def _design_or_fallback(c, lang, fp, tts, picker, previews_dir, cfg) -> VoiceEntry:
    try:
        vid, sample = tts.design_voice(c.name, design_prompt(c), c.gender, lang)
    except Exception as e:  # noqa: BLE001 (any design failure falls back to a library voice)
        from .retry import QuotaExhausted
        if isinstance(e, QuotaExhausted):
            raise
        # Designing child voices is refused by Google's safety policy ("Voice prompt was blocked").
        # We respect that and fall back to a young-sounding library voice; we never re-prompt around it.
        voice, kind = picker.pick(c.gender, cfg.tts.narrator_prebuilt_voice if c.id == NARRATOR_ID else None,
                                  young=is_minor(c))
        log.warning("voice design failed; using a library voice",
                    extra=kv(character=c.id, error=str(e)[:160], fallback=voice))
        return VoiceEntry(voice=voice, kind=kind, fingerprint=fp)
    preview = None
    if sample and previews_dir:
        previews_dir.mkdir(parents=True, exist_ok=True)
        p = previews_dir / f"{c.id}.wav"
        p.write_bytes(sample)
        preview = str(p)
    return VoiceEntry(voice=vid, kind="designed", fingerprint=fp, preview=preview)


def cast_voices(run: RunDir, cfg: Config, *, narrator_clip: Path | None = None,
                consent_clip: Path | None = None, tts=None) -> VoiceCast:
    script = Script.model_validate_json(run.require(run.script, "script").read_text(encoding="utf-8"))
    if tts is None:
        from .tts import GeminiTTS
        tts = GeminiTTS(cfg)
    cast = load_cast(run, cfg)
    try:
        cast = cast_script(script, cast, cfg, tts, run.root / "voices", narrator_clip, consent_clip)
    finally:
        # Save after every run, including partial ones, so voices that were created are never recreated
        write_json(run.voices, cast)
    return cast


def voice_table(cast: VoiceCast) -> str:
    return json.dumps({k: f"{v.kind}:{v.voice}" for k, v in cast.voices.items()}, indent=2)
