"""Stage 2: rewrite the story for a target culture. Plot beats stay fixed and every change is logged.

Chunks are adapted in order. A story bible (original → adapted mappings) is passed from
chunk to chunk, so a name or place chosen in chunk 1 stays the same in chunk 9.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .ingest import ingest
from .llm import LLM, get_llm
from .log import kv
from .models import Adaptation, AdaptChunkOut, AdaptedChunk, BibleEntry, Source
from .runs import RunDir, write_json

log = logging.getLogger(__name__)

SYSTEM = """You are a senior story editor who adapts fiction for new audiences. \
The principle is ADAPTATION, NOT TRANSLATION: the story should feel as if it were \
written for the target audience, while the plot and emotions stay exactly the same.

Change, where it helps the story feel native:
- character names (fitting age, region, religion/community and era; keep the same gender)
- places, landmarks, climate, and everyday geography
- food, drink, festivals, holidays, rituals, clothing, money, units
- idioms, jokes, forms of address, and kinship terms (e.g. dada, didi, mashi, kaku, jethu)
- social context: family structure, school or work life, neighbourhood life

Never change:
- the plot beats, their order, or who does what to whom
- the emotional arc, the stakes, or the ending
- the number of characters or their relationships

Rules:
- Anything foreign that is CENTRAL to the plot (a signature dish, the festival the story builds towards, a key object) MUST become a native equivalent that serves the SAME plot function. Example: a prize-winning apple pie might become a prize-winning nolen gurer payesh or pitha. Follow the ripple effects through (ingredients, rivals' dishes, the secret ingredient).
- Keep every piece of dialogue and every event. Adapt their wording, but never drop them.
- Write the adapted prose in English (a later stage handles the spoken language).
- Keep a similar length to the source chunk. This is a full rewrite, not a summary.
- Be specific and authentic rather than generic. Avoid stereotypes and caricature.
- Reuse every mapping in the STORY BIBLE exactly. Add a bible entry only for new recurring elements.
- Log every meaningful substitution in `changes`, with a concrete reason."""

PROMPT = """TARGET CULTURE: {culture}

STORY BIBLE (fixed mappings, which you must reuse):
{bible}

{previous}SOURCE CHUNK {n}/{total}{heading}:
\"\"\"
{text}
\"\"\"

STORY TITLE: {title}

Return the adapted title (a natural title for this audience), the adapted chunk, the change log, any new bible entries, and this chunk's plot beats in order."""


def _bible_text(bible: list[BibleEntry]) -> str:
    if not bible:
        return "(empty: this is the first chunk)"
    return "\n".join(f"- [{b.kind}] {b.original} → {b.adapted}" + (f" ({b.note})" if b.note else "")
                     for b in bible)


def merge_bible(bible: list[BibleEntry], updates: list[BibleEntry]) -> list[BibleEntry]:
    """The first mapping wins, so later chunks can't rename someone by accident."""
    seen = {b.original.strip().lower() for b in bible}
    merged = list(bible)
    for u in updates:
        key = u.original.strip().lower()
        if key and key not in seen:
            merged.append(u)
            seen.add(key)
    return merged


def adapt_source(source: Source, culture: str, lang: str, llm: LLM) -> Adaptation:
    bible: list[BibleEntry] = []
    chunks: list[AdaptedChunk] = []
    for ch in source.chunks:
        prev = ""
        if chunks:
            tail = " ".join(chunks[-1].adapted_text.split()[-250:])
            prev = f"END OF THE PREVIOUS ADAPTED CHUNK (for continuity):\n\"\"\"…{tail}\"\"\"\n\n"
        prompt = PROMPT.format(
            culture=culture, title=source.title, bible=_bible_text(bible), previous=prev, n=ch.index + 1,
            total=len(source.chunks), heading=f" ({ch.heading})" if ch.heading else "", text=ch.text,
        )
        out = llm.structured(SYSTEM, prompt, AdaptChunkOut, stage=f"adapt[{ch.index}]")
        bible = merge_bible(bible, out.bible_updates)
        chunks.append(AdaptedChunk(**out.model_dump(), index=ch.index, source_words=ch.words))
        log.info("chunk adapted", extra=kv(chunk=ch.index, changes=len(out.changes),
                                          words_in=ch.words, words_out=len(out.adapted_text.split())))
    first = chunks[0].adapted_title.strip() if chunks else ""
    adapted_title = first or _adapt_title(source.title, bible)
    return Adaptation(title=source.title, adapted_title=adapted_title, culture=culture, lang=lang,
                      llm=llm.label, chunks=chunks, bible=bible)


def _adapt_title(title: str, bible: list[BibleEntry]) -> str:
    for b in sorted(bible, key=lambda b: -len(b.original)):
        if b.original and b.original in title:
            title = title.replace(b.original, b.adapted)
    return title


def adapt(story: Path, run: RunDir, cfg: Config, *, culture: str, lang: str, llm: str) -> Adaptation:
    run.ensure()
    source = ingest(story, cfg.llm.max_chunk_words)
    run.source.write_text(Path(story).read_text(encoding="utf-8-sig"), encoding="utf-8")
    log.info("ingested", extra=kv(title=source.title, chunks=len(source.chunks),
                                 words=sum(c.words for c in source.chunks)))
    model = get_llm(llm, cfg, usage_log=run.root / "llm_usage.jsonl")
    adaptation = adapt_source(source, culture, lang, model)
    write_json(run.adaptation, adaptation)
    log.info("wrote adaptation", extra=kv(path=run.adaptation, changes=len(adaptation.change_log),
                                         bible=len(adaptation.bible)))
    return adaptation

