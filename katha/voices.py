"""Stage 4: cast a voice for each character (designed, or prebuilt as a fallback) and cache the ids."""
from __future__ import annotations

CONSENT_WARNING = (
    "VOICE REPLICATION: only replicate YOUR OWN voice, or a voice whose owner has given "
    "explicit, recorded consent. The consent clip must be the same person reading Google's "
    "consent statement. Cloning someone's voice without consent is harmful and violates "
    "Google's terms."
)


def cast_voices(*args, **kwargs):
    raise NotImplementedError("voice casting lands in M3")
