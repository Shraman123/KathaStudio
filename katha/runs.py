"""Run-directory layout. Each stage reads and writes files here, so any stage can be rerun."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def slugify(text: str, max_len: int = 60) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (text[:max_len].rstrip("-")) or "story"


@dataclass(frozen=True)
class RunDir:
    root: Path

    @classmethod
    def for_story(cls, runs_dir: Path, story_path: Path) -> "RunDir":
        return cls(Path(runs_dir) / slugify(Path(story_path).stem))

    @property
    def source(self) -> Path:
        return self.root / "01_source.txt"

    @property
    def adaptation(self) -> Path:
        return self.root / "02_adaptation.json"

    @property
    def script(self) -> Path:
        return self.root / "03_script.json"

    @property
    def voices(self) -> Path:
        return self.root / "04_voices.json"

    @property
    def audio_dir(self) -> Path:
        return self.root / "05_audio"

    @property
    def episode_mp3(self) -> Path:
        return self.root / "episode.mp3"

    @property
    def episode_wav(self) -> Path:
        return self.root / "episode.wav"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    def ensure(self) -> "RunDir":
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        return self

    def require(self, path: Path, produced_by: str) -> Path:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name} in {self.root}. Run `katha {produced_by}` first."
            )
        return path
