"""Client-side rate limiting and exponential backoff with jitter. Shared by the LLM and TTS calls."""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable, TypeVar

from .config import RateLimitConfig
from .log import kv

log = logging.getLogger(__name__)
T = TypeVar("T")

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class RateLimiter:
    """Spaces calls at least 60/rpm seconds apart. One limiter is shared per API."""

    def __init__(self, requests_per_minute: float, clock=time.monotonic, sleep=time.sleep):
        self.interval = 60.0 / requests_per_minute
        self._next = 0.0
        self._lock = threading.Lock()
        self._clock, self._sleep = clock, sleep

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            self._sleep(delay)


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    return None


def _retry_after(exc: BaseException) -> float | None:
    resp = getattr(exc, "raw_response", None) or getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


class QuotaExhausted(RuntimeError):
    """A per-day quota was hit. Retrying within minutes can't help."""


DAILY_QUOTA_HINTS = ("per day", "requests per day", "rpd", "daily")


def is_daily_quota(exc: BaseException) -> bool:
    return _status(exc) == 429 and any(h in str(exc).lower() for h in DAILY_QUOTA_HINTS)


def is_retryable(exc: BaseException) -> bool:
    if is_daily_quota(exc):
        return False
    status = _status(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    # Connection resets, timeouts and empty responses carry no status code
    name = type(exc).__name__
    return any(k in name for k in ("Timeout", "Connect", "NoResponse", "RemoteProtocol", "ReadError"))


def call_with_retries(
    fn: Callable[[], T],
    cfg: RateLimitConfig,
    limiter: RateLimiter | None = None,
    what: str = "api call",
    sleep=time.sleep,
) -> T:
    attempt = 0
    while True:
        if limiter:
            limiter.wait()
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 (we re-raise anything that isn't retryable)
            if is_daily_quota(e):
                raise QuotaExhausted(
                    f"{what}: daily free-tier quota exhausted. Wait for the reset (midnight "
                    f"US Pacific), switch model in config.yaml, or enable billing. API said: {e}"
                ) from e
            if not is_retryable(e) or attempt >= cfg.max_retries:
                raise
            backoff = min(cfg.backoff_max_seconds, cfg.backoff_base_seconds * 2 ** attempt)
            delay = max(_retry_after(e) or 0.0, backoff * random.uniform(0.5, 1.0))  # jitter
            attempt += 1
            log.warning("retrying", extra=kv(what=what, attempt=attempt, status=_status(e),
                                            error=type(e).__name__, sleep=f"{delay:.1f}s"))
            sleep(delay)
