import json
from pathlib import Path

import pytest

from katha.config import load_config
from katha.models import Character, Line, Scene, Script

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "config.yaml")
    c.paths.runs_dir = tmp_path / "runs"
    c.rate_limit.requests_per_minute = 1e6  # no real waiting in tests
    return c


def ch(cid, name=None, gender="female"):
    return Character(id=cid, name=name or cid.title(), age="30s", gender=gender,
                     voice_description=f"voice of {cid}")


def make_script(scene_speakers: list[list[str]], lang="en") -> Script:
    ids = sorted({s for sc in scene_speakers for s in sc} | {"narrator"})
    scenes = [
        Scene(id=f"scene_{i:02d}", title=f"S{i}",
              lines=[Line(speaker=s, text=f"{s} says something short here.") for s in sc])
        for i, sc in enumerate(scene_speakers, 1)
    ]
    return Script(title="T", lang=lang, characters=[ch(i) for i in ids], scenes=scenes)


class FakeBackend:
    """Returns queued JSON strings in order. Records the prompts it was sent."""
    name = "fake"
    model = "fake-1"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete_json(self, system, prompt, schema):
        self.prompts.append(prompt)
        r = self.responses.pop(0)
        return (r if isinstance(r, str) else json.dumps(r)), 10, 20
