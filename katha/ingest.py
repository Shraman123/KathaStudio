"""Stage 1: read the story and split it into context-sized chunks.

Chunk boundaries fall on chapter headings first, then paragraphs. A paragraph is only
split (by sentence) if it is bigger than a whole chunk on its own.
"""
from __future__ import annotations

import re
from pathlib import Path

from .models import Chunk, Source

HEADING_RE = re.compile(
    r"^\s*(#{1,3}\s+.+|(chapter|part|book)\s+([0-9]+|[ivxlc]+|[a-z]+)\b.*)\s*$", re.IGNORECASE
)
SENTENCE_RE = re.compile(r"(?<=[.!?…।])[\"'”’]?\s+")


def read_story(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ValueError(f"Story file is empty: {path}")
    return text


def detect_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
        if s:
            break
    return fallback.replace("_", " ").replace("-", " ").strip().title()


RULE_RE = re.compile(r"^\s*([-*_]\s*){3,}$")  # --- / *** scene-break rules


def _paragraphs(text: str) -> list[str]:
    pars = (re.sub(r"[ \t]*\n[ \t]*", " ", p).strip() for p in re.split(r"\n\s*\n", text))
    return [p for p in pars if p and not RULE_RE.match(p)]


def _split_long(par: str, max_words: int) -> list[str]:
    if len(par.split()) <= max_words:
        return [par]
    out, cur = [], []
    for sent in SENTENCE_RE.split(par):
        if cur and len(" ".join(cur + [sent]).split()) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        out.append(" ".join(cur))
    return out


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split on chapter headings. Returns (heading, body) pairs."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.splitlines():
        if HEADING_RE.match(line) and len(line.split()) <= 12:
            sections.append((line.strip().lstrip("#").strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def chunk_text(text: str, max_words: int) -> list[Chunk]:
    chunks: list[Chunk] = []

    def flush(heading, pars):
        if pars:
            chunks.append(Chunk(index=len(chunks), heading=heading, text="\n\n".join(pars)))

    for heading, body in split_sections(text):
        cur: list[str] = []
        cur_words = 0
        first = True
        for par in _paragraphs(body):
            for piece in _split_long(par, max_words):
                w = len(piece.split())
                if cur and cur_words + w > max_words:
                    flush(heading if first else None, cur)
                    first, cur, cur_words = False, [], 0
                cur.append(piece)
                cur_words += w
        flush(heading if first else None, cur)
    return chunks


def ingest(story_path: Path, max_words: int) -> Source:
    text = read_story(story_path)
    title = detect_title(text, Path(story_path).stem)
    # Drop the title heading from the body so it doesn't become a chapter of its own
    lines = text.splitlines()
    if lines and lines[0].strip() == f"# {title}":
        text = "\n".join(lines[1:])
    return Source(title=title, chunks=chunk_text(text, max_words))
