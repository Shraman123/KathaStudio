"""Structured logging: one key=value line per event, so logs are easy to grep."""
from __future__ import annotations

import logging
import sys


class KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<5} {record.name}: {record.getMessage()}"
        extra = getattr(record, "kv", None)
        if extra:
            base += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(verbose: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(KVFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Keep third-party HTTP noise out of the logs
    for noisy in ("httpx", "httpcore", "google_genai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def kv(**fields) -> dict:
    """Usage: log.info("chunk rendered", extra=kv(scene=3, secs=4.2))"""
    return {"kv": fields}


# The SDK warns whenever both GOOGLE_API_KEY and GEMINI_API_KEY are set, even though an
# explicit api_key (which we always pass) takes precedence. Silence that one warning.
class _DropBothKeysWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Both GOOGLE_API_KEY and GEMINI_API_KEY" not in record.getMessage()


logging.getLogger("google_genai._api_client").addFilter(_DropBothKeysWarning())
