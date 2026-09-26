"""Swappable text LLM: Gemini (default, free tier) or Claude, both behind one interface.

    llm = get_llm("gemini", cfg, usage_log=run.root / "llm_usage.jsonl")
    out = llm.structured(system, prompt, AdaptChunkOut)

`structured()` asks for JSON that follows the pydantic schema, validates it, and if
validation fails, re-asks once with the errors attached. It logs token usage so the
report can include LLM cost.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Config, ConfigError, require_key
from .log import kv
from .retry import QuotaExhausted, RateLimiter, call_with_retries

log = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)


class LLMOutputError(RuntimeError):
    pass


class Backend(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, prompt: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        """Return (raw_json_text, input_tokens, output_tokens)."""


class GeminiBackend:
    name = "gemini"

    def __init__(self, cfg: Config):
        from google import genai

        self.models = [cfg.llm.gemini_model, *cfg.llm.gemini_fallback_models]
        self.model = self.models[0]
        self.temperature = cfg.llm.temperature
        # Pass the key explicitly. Otherwise the SDK prefers GOOGLE_API_KEY if that's also set.
        self.client = genai.Client(api_key=require_key("GEMINI_API_KEY"))

    def complete_json(self, system, prompt, schema):
        i = self.client.interactions.create(
            model=self.model,
            system_instruction=system,
            input=prompt,
            generation_config={"temperature": self.temperature},
            response_format={"type": "text", "mime_type": "application/json",
                             "schema": schema.model_json_schema()},
        )
        u = getattr(i, "usage", None)
        tin = getattr(u, "total_input_tokens", 0) or 0
        tout = (getattr(u, "total_output_tokens", 0) or 0) + (getattr(u, "total_thought_tokens", 0) or 0)
        return i.output_text or "", tin, tout

    def fallback(self) -> bool:
        """Move to the next configured model after a daily-quota error. Returns False if none are left."""
        idx = self.models.index(self.model) + 1
        if idx >= len(self.models):
            return False
        self.model = self.models[idx]
        return True


class ClaudeBackend:
    name = "claude"

    def __init__(self, cfg: Config):
        try:
            import anthropic
        except ImportError:
            raise ConfigError("--llm claude needs the anthropic package: pip install -e .[claude]") from None
        self._anthropic = anthropic
        self.model = cfg.llm.claude_model
        # Our own retry loop handles backoff, so SDK retries are off.
        self.client = anthropic.Anthropic(api_key=require_key("ANTHROPIC_API_KEY"), max_retries=0)

    def complete_json(self, system, prompt, schema):
        # messages.create + transform_schema (instead of messages.parse) gives us the raw JSON,
        # so validation failures go through the same repair loop as Gemini.
        r = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema",
                                      "schema": self._anthropic.transform_schema(schema)}},
        )
        if r.stop_reason == "refusal":
            raise LLMOutputError(f"Claude declined the request: {getattr(r.stop_details, 'explanation', '')}")
        if r.stop_reason == "max_tokens":
            raise LLMOutputError("Claude hit max_tokens. Lower llm.max_chunk_words in config.yaml")
        text = next((b.text for b in r.content if b.type == "text"), "")
        return text, r.usage.input_tokens, r.usage.output_tokens


class LLM:
    def __init__(self, backend: Backend, cfg: Config, usage_log: Path | None = None):
        self.backend = backend
        self.cfg = cfg
        self.usage_log = usage_log
        self.limiter = RateLimiter(cfg.rate_limit.requests_per_minute)

    @property
    def label(self) -> str:
        return f"{self.backend.name}:{self.backend.model}"

    def structured(self, system: str, prompt: str, schema: type[M], stage: str = "llm",
                   check: Callable[[M], None] | None = None) -> M:
        """`check` can raise ValueError for cross-object rules the schema can't express."""
        attempt_prompt = prompt
        last_err = None
        for attempt in (1, 2):
            t0 = time.monotonic()
            raw, tin, tout = self._call(system, attempt_prompt, schema, stage)
            self._record(stage, tin, tout, time.monotonic() - t0)
            try:
                out = schema.model_validate_json(raw)
                if check:
                    check(out)
                return out
            except (ValidationError, ValueError) as e:
                last_err = _short_errors(e) if isinstance(e, ValidationError) else f"- {e}"
                log.warning("llm output failed validation; asking the model to repair it",
                            extra=kv(stage=stage, attempt=attempt))
                attempt_prompt = (
                    f"{prompt}\n\n---\nYour previous answer was invalid:\n{last_err}\n"
                    f"Previous answer:\n{raw[:20000]}\n\nReturn a corrected, complete JSON answer."
                )
        raise LLMOutputError(f"{stage}: model output still invalid after a repair attempt:\n{last_err}")

    def _call(self, system, prompt, schema, stage):
        while True:
            try:
                return call_with_retries(
                    lambda: self.backend.complete_json(system, prompt, schema),
                    self.cfg.rate_limit, self.limiter, what=f"{stage}:{self.backend.model}",
                )
            except QuotaExhausted:
                old = self.backend.model
                fallback = getattr(self.backend, "fallback", None)
                if not (fallback and fallback()):
                    raise
                log.warning("daily quota exhausted; falling back to the next model",
                            extra=kv(stage=stage, from_model=old, to_model=self.backend.model))

    def _record(self, stage: str, tin: int, tout: int, secs: float) -> None:
        log.info("llm call", extra=kv(stage=stage, model=self.backend.model, in_tok=tin, out_tok=tout,
                                     secs=f"{secs:.1f}"))
        if self.usage_log:
            self.usage_log.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"stage": stage, "model": self.backend.model,
                                    "input_tokens": tin, "output_tokens": tout}) + "\n")


def _short_errors(e: ValidationError | None, limit: int = 12) -> str:
    if e is None:
        return ""
    out = []
    for err in e.errors()[:limit]:
        loc = ".".join(str(p) for p in err["loc"])
        out.append(f"- {loc}: {err['msg']}")
    return "\n".join(out)


def get_llm(provider: str, cfg: Config, usage_log: Path | None = None) -> LLM:
    backends = {"gemini": GeminiBackend, "claude": ClaudeBackend}
    if provider not in backends:
        raise ConfigError(f"Unknown LLM provider '{provider}'. Choose from {sorted(backends)}")
    return LLM(backends[provider](cfg), cfg, usage_log)
