"""Swappable text LLM: Gemini (default, free tier) or Claude, both behind one interface."""
from __future__ import annotations


def get_llm(*args, **kwargs):
    raise NotImplementedError("LLM backends land in M2")
