"""Stage 3: turn the adapted story into an audio-drama script (characters + scenes + verbatim lines).

Each adapted chunk becomes one or more scenes. The characters found so far are passed to
the next chunk, so ids and voices stay stable across a long story.
"""
from __future__ import annotations

import logging

from .config import Config
from .llm import LLM, get_llm
from .log import kv
from .models import NARRATOR_ID, Adaptation, Character, Scene, Script, ScriptChunkOut
from .runs import RunDir, write_json

log = logging.getLogger(__name__)

LANG_RULES = {
    "en": "Write all spoken lines in natural Indian English, the way people in the target culture "
          "really speak it. A few native words are welcome where natural (forms of address, food, "
          "exclamations), but the lines must stay clear to an English listener.",
    "hi": "Write all spoken lines in natural, conversational Hindi in Devanagari script (not "
          "Sanskritised, not romanised). Common English loanwords are fine where people really use them.",
    "bn": "Write all spoken lines in natural, conversational Bengali (Kolkata colloquial, cholito bhasha) "
          "in Bengali script, not romanised. Common English loanwords are fine where people really use them.",
}

SYSTEM = """You are an award-winning audio-drama writer. You turn prose into a script \
that will be performed by a text-to-speech engine, one line per turn.

Format rules (strict):
- Every line has a `speaker` (a character id) and `text` that will be spoken VERBATIM.
- `text` must contain ONLY words to be spoken. No speaker names, no stage directions, \
no parentheses, square brackets, or asterisks.
- Put sustained delivery (emotion, pace, volume) in `style`, briefly: e.g. "whispering, scared", \
"brisk and proud". Leave it empty when neutral. Never put fixed traits (age, gender, accent) in style.
- For momentary sounds, use inline tags inside text, SPARINGLY (roughly one in every five lines at most), \
only from: <laugh> <chuckle> <giggle> <sigh> <gasp> <breath> <cough> <sob> <cry> <groan> <tsk> <yawn> \
<throat-clearing> <exhales> <short pause> <long pause>. Tags are always in English.
- The narrator (id "narrator") carries description, setting, and transitions. Keep narration \
vivid but lean. Let dialogue carry the story wherever possible, and convert reported speech into \
dialogue when it plays better.
- Keep lines short enough to speak in one breath or a few (usually under 40 words). Split long \
narration into several narrator lines.
- A scene is a continuous stretch in one place and time. Give each a short evocative title.
- Keep every plot beat from the prose, in order, and keep the memorable dialogue lines. Do not invent new plot. Prefer dramatising a moment as dialogue over summarising it in narration.

Characters:
- Declare each speaking character once, the first time they speak, in `new_characters`.
- `voice_description` is used to DESIGN a synthetic voice. Describe timbre, pitch, pace, energy, \
accent (e.g. "Kolkata Bengali-accented English"), and personality in 1–2 sentences. \
Do not name real people.
- Never declare the narrator. It already exists."""

PROMPT = """LANGUAGE: {lang_rule}

KNOWN CHARACTERS (reuse these ids; do not redeclare them):
{characters}

PLOT BEATS (every one must be dramatised, in order; report their numbers in covered_beats):
{beats}

ADAPTED STORY, PART {n}/{total}:
\"\"\"
{text}
\"\"\"

Write the scenes for this part."""


def default_narrator(culture: str, lang: str) -> Character:
    lang_name = {"en": "English", "hi": "Hindi", "bn": "Bengali"}[lang]
    return Character(
        id=NARRATOR_ID, name="Narrator", age="40s", gender="male",
        voice_description=(
            f"A warm, rich storyteller's voice for {culture} listeners, speaking {lang_name} with a "
            "gentle regional accent. Unhurried and intimate, like an elder telling a story at dusk, "
            "with a quiet smile in the voice."
        ),
    )


def _characters_text(chars: list[Character]) -> str:
    return "\n".join(f"- {c.id}: {c.name}, {c.gender}, {c.age}" for c in chars)


def _check_speakers(out: ScriptChunkOut, known: set[str]) -> None:
    declared = known | {c.id for c in out.new_characters}
    if NARRATOR_ID in {c.id for c in out.new_characters}:
        raise ValueError("do not declare the narrator in new_characters")
    missing = sorted({ln.speaker for s in out.scenes for ln in s.lines} - declared)
    if missing:
        raise ValueError(f"speakers {missing} are used but not declared in new_characters")


def build_script(adaptation: Adaptation, llm: LLM) -> Script:
    characters = [default_narrator(adaptation.culture, adaptation.lang)]
    scenes: list[Scene] = []
    total = len(adaptation.chunks)
    for ch in adaptation.chunks:
        prompt = PROMPT.format(
            lang_rule=LANG_RULES[adaptation.lang], characters=_characters_text(characters),
            n=ch.index + 1, total=total, text=ch.adapted_text,
            beats="\n".join(f"{i}. {b}" for i, b in enumerate(ch.plot_beats, 1)) or "(none listed)",
        )
        known = {c.id for c in characters}
        n_beats = len(ch.plot_beats)

        def check(o: ScriptChunkOut, known=known, n_beats=n_beats) -> None:
            _check_speakers(o, known)
            missing = sorted(set(range(1, n_beats + 1)) - set(o.covered_beats))
            if missing:
                raise ValueError(f"plot beats {missing} are not dramatised. Add scenes or lines for them")

        out = llm.structured(SYSTEM, prompt, ScriptChunkOut, stage=f"script[{ch.index}]", check=check)
        for c in out.new_characters:
            if c.id not in known:
                characters.append(c)
                known.add(c.id)
        scenes.extend(out.scenes)
        log.info("chunk scripted", extra=kv(chunk=ch.index, scenes=len(out.scenes),
                                           lines=sum(len(s.lines) for s in out.scenes)))
    for i, s in enumerate(scenes, 1):
        s.id = f"scene_{i:02d}"
    # Script validation catches any speaker the model forgot to declare
    return Script(title=adaptation.adapted_title, lang=adaptation.lang, characters=characters, scenes=scenes)


def write_script(run: RunDir, cfg: Config, *, llm: str) -> Script:
    adaptation = Adaptation.model_validate_json(run.require(run.adaptation, "adapt").read_text(encoding="utf-8"))
    model = get_llm(llm, cfg, usage_log=run.root / "llm_usage.jsonl")
    script = build_script(adaptation, model)
    write_json(run.script, script)
    log.info("wrote script", extra=kv(path=run.script, characters=len(script.characters),
                                     scenes=len(script.scenes),
                                     lines=sum(len(s.lines) for s in script.scenes)))
    return script
