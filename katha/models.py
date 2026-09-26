"""Pydantic schemas for every structured artifact in a run.

The LLM gets these schemas directly, and its output is validated against them again
after it comes back. Validation errors are fed back to the model for one repair attempt.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Inline vocal tags that Gemini 3.8 TTS supports (API_NOTES.md → prompting rules).
ALLOWED_TAGS = frozenset({
    "argh", "breath", "heavy breath", "exhales", "cackle", "cheer", "chuckle", "cough",
    "cry", "gasp", "giggle", "groan", "growl", "grunt", "grr", "hiss", "laugh", "moan",
    "pant", "pff", "scream", "shout", "shriek", "sigh", "sneeze", "snicker", "snort", "sob",
    "throat-clearing", "tsk", "whimper", "yawn", "short pause", "long pause",
})
TAG_RE = re.compile(r"<([^<>]+)>")
# Stage directions that must never be spoken: (whispers) [door slams] *laughs*
STAGE_DIRECTION_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]|\*[^*]+\*")
NARRATOR_ID = "narrator"


# ---------- Stage 1: ingest ----------

class Chunk(BaseModel):
    index: int
    heading: str | None = None
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


class Source(BaseModel):
    title: str
    chunks: list[Chunk]


# ---------- Stage 2: adapt ----------

ElementKind = Literal[
    "character", "place", "food", "festival", "custom", "idiom", "object", "social_context", "other"
]


class BibleEntry(BaseModel):
    """A fixed mapping between an original element and its adapted version. Later chunks must reuse it."""
    kind: ElementKind
    original: str
    adapted: str
    note: str = ""


class ChangeLogEntry(BaseModel):
    kind: ElementKind
    original: str = Field(description="The element as it appears in the source")
    adapted: str = Field(description="What it became in the adaptation")
    reason: str = Field(description="Why this change makes the story feel native")


class AdaptChunkOut(BaseModel):
    """What the LLM returns for one chunk."""
    adapted_title: str = Field("", description="A title for the whole story that feels native to the target audience")
    adapted_text: str = Field(description="The full adapted prose for this chunk, in English")
    changes: list[ChangeLogEntry]
    bible_updates: list[BibleEntry] = Field(
        description="New recurring elements (names, places…) introduced in this chunk"
    )
    plot_beats: list[str] = Field(
        description="The plot beats of this chunk, in order. They must be identical to the source's beats"
    )

    @field_validator("adapted_text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if len(v.split()) < 20:
            raise ValueError("adapted_text is too short. Adapt the whole chunk, don't summarise it")
        return v


class AdaptedChunk(AdaptChunkOut):
    index: int
    source_words: int


class Adaptation(BaseModel):
    title: str
    adapted_title: str
    culture: str
    lang: str
    llm: str
    chunks: list[AdaptedChunk]
    bible: list[BibleEntry]

    @property
    def change_log(self) -> list[ChangeLogEntry]:
        return [c for ch in self.chunks for c in ch.changes]

    @property
    def text(self) -> str:
        return "\n\n".join(c.adapted_text for c in self.chunks)


# ---------- Stage 3: script ----------

class Character(BaseModel):
    id: str = Field(description="lowercase snake_case id, e.g. 'rina_di'. The narrator is always 'narrator'")
    name: str
    age: str = Field(description="e.g. '12', 'late 60s'")
    gender: Literal["male", "female", "nonbinary", "unspecified"]
    voice_description: str = Field(
        description="Timbre, pitch, pace, accent and personality, for designing a TTS voice. No celebrity names"
    )

    @field_validator("id")
    @classmethod
    def _slug_id(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", v):
            raise ValueError(f"character id '{v}' must be lowercase snake_case")
        return v


class Line(BaseModel):
    speaker: str = Field(description="A character id")
    text: str = Field(description="Exactly what is spoken. Only inline vocal tags like <sigh> are allowed; no stage directions")
    style: str = Field("", description="Short delivery direction, e.g. 'whispering, afraid'. Empty if neutral")
    tags: list[str] = Field(default_factory=list, description="The inline tags used in text (filled in automatically)")

    @model_validator(mode="after")
    def _check_text(self) -> "Line":
        spoken = TAG_RE.sub("", self.text).strip()
        if not spoken.replace("|", "").strip() and not TAG_RE.search(self.text):
            raise ValueError("line text is empty")
        m = STAGE_DIRECTION_RE.search(self.text)
        if m:
            raise ValueError(
                f"stage direction {m.group(0)!r} inside spoken text. Move it to `style` or use an allowed inline tag"
            )
        found = [t.strip().lower() for t in TAG_RE.findall(self.text)]
        bad = [t for t in found if t not in ALLOWED_TAGS]
        if bad:
            raise ValueError(f"unsupported inline tag(s) {bad}. Allowed: {sorted(ALLOWED_TAGS)}")
        self.tags = found
        self.style = self.style.strip()
        return self

    @property
    def spoken_text(self) -> str:
        """The text without tags or backchannel pipes. Used for word counts."""
        return re.sub(r"\s+", " ", TAG_RE.sub(" ", self.text).replace("|", " ")).strip()


class Scene(BaseModel):
    id: str = ""
    title: str
    lines: list[Line] = Field(min_length=1)


class ScriptChunkOut(BaseModel):
    """What the LLM returns for one adapted chunk."""
    covered_beats: list[int] = Field(
        default_factory=list, description="Numbers of the PLOT BEATS dramatised in these scenes. Must include every beat"
    )
    new_characters: list[Character] = Field(
        description="Characters that first appear in this chunk (don't repeat known ones)"
    )
    scenes: list[Scene]


class Script(BaseModel):
    title: str
    lang: str
    characters: list[Character]
    scenes: list[Scene]

    @model_validator(mode="after")
    def _consistent(self) -> "Script":
        ids = [c.id for c in self.characters]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate character ids: {ids}")
        if NARRATOR_ID not in ids:
            raise ValueError("script must include a 'narrator' character")
        known = set(ids)
        for s in self.scenes:
            for ln in s.lines:
                if ln.speaker not in known:
                    raise ValueError(f"scene '{s.title}': unknown speaker '{ln.speaker}'. Known: {sorted(known)}")
        scene_ids = [s.id for s in self.scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("duplicate scene ids")
        return self

    def character(self, cid: str) -> Character:
        return next(c for c in self.characters if c.id == cid)
