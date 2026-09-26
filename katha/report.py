"""Stage 7: report.md covering what was adapted, voices, durations, API calls, cost, and free-tier fit.

Costs use the prices in config.yaml → pricing, never hardcoded values. TTS output cost is
based on the real episode length at `audio_tokens_per_second`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .config import Config
from .models import Adaptation, Script
from .runs import RunDir


def _fmt_secs(s: float) -> str:
    m, s = divmod(round(s), 60)
    return f"{m}m {s:02d}s"


def _llm_usage(path: Path) -> dict[str, dict[str, int]]:
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                m = by_model[r["model"]]
                m["calls"] += 1
                m["input_tokens"] += r["input_tokens"]
                m["output_tokens"] += r["output_tokens"]
    return dict(by_model)


def build_report(run: RunDir, cfg: Config, stitched: dict | None = None) -> str:
    adaptation = Adaptation.model_validate_json(run.adaptation.read_text(encoding="utf-8"))
    script = Script.model_validate_json(run.script.read_text(encoding="utf-8"))
    stats = json.loads((run.root / "render_stats.json").read_text(encoding="utf-8"))
    voices = json.loads(run.voices.read_text(encoding="utf-8"))["voices"] if run.voices.exists() else {}
    seconds = (stitched or {}).get("seconds") or stats["seconds"]
    rendered_scenes = sorted({k.rsplit("_", 1)[0] for k in stats["keys"]})
    scenes = [s for s in script.scenes if s.id in rendered_scenes]

    # ---- cost ----
    tts_price = cfg.price_for(cfg.tts.model)
    audio_tokens = int(seconds * cfg.pricing.audio_tokens_per_second)
    chars = sum(len(ln.text) + len(ln.style) for s in scenes for ln in s.lines)
    tts_in_tokens = int(chars / (4 if script.lang == "en" else 2.5))
    tts_cost = audio_tokens / 1e6 * tts_price.output + tts_in_tokens / 1e6 * tts_price.input
    llm_rows, llm_cost, llm_unpriced = [], 0.0, []
    for model, u in _llm_usage(run.root / "llm_usage.jsonl").items():
        price = cfg.pricing.models.get(model)
        if price:
            c = u["input_tokens"] / 1e6 * price.input + u["output_tokens"] / 1e6 * price.output
            llm_cost += c
            cost_s = f"${c:.4f}"
        else:
            llm_unpriced.append(model)
            cost_s = "no price in config"
        llm_rows.append(f"| {model} | {u['calls']} | {u['input_tokens']:,} | {u['output_tokens']:,} | {cost_s} |")

    design_calls = sum(1 for v in voices.values() if v["kind"] in ("designed", "replicated"))
    tts_requests = stats["requests"]

    # ---- free-tier fit ----
    fit_lines = []
    llm_calls = defaultdict(int)
    for model, u in _llm_usage(run.root / "llm_usage.jsonl").items():
        llm_calls[model] += u["calls"]
    for model, n in [*llm_calls.items(), (cfg.tts.model, tts_requests + design_calls)]:
        rpd = cfg.pricing.free_tier_rpd.get(model)
        if rpd:
            verdict = "fits" if n <= rpd else f"**exceeds** ({n - rpd} over; spread across days or enable billing)"
            fit_lines.append(f"- `{model}`: {n} requests vs {rpd}/day free → {verdict}")
        else:
            fit_lines.append(f"- `{model}`: {n} requests. Free-tier daily limit unknown; check "
                             f"https://aistudio.google.com/rate-limit")

    change_rows = "\n".join(f"| {c.kind} | {c.original} | {c.adapted} | {c.reason} |" for c in adaptation.change_log)
    voice_rows = "\n".join(
        f"| {c.name} (`{c.id}`) | {c.gender}, {c.age} | {voices.get(c.id, {}).get('kind', '—')} | "
        f"{c.voice_description} |" for c in script.characters)
    scene_rows = "\n".join(f"| {s.id} | {s.title} | {len(s.lines)} |" for s in scenes)
    beats = "\n".join(f"{i}. {b}" for ch in adaptation.chunks for i, b in enumerate(ch.plot_beats, 1))
    partial = (f"\n> Partial render: only {len(scenes)} of {len(script.scenes)} scenes (--max-scenes).\n"
               if len(scenes) < len(script.scenes) else "")
    free = cfg.pricing.free_tier

    return f"""# {script.title}

*Adapted from "{adaptation.title}" for **{adaptation.culture}** · spoken language `{script.lang}` · \
adaptation LLM `{adaptation.llm}` · TTS `{cfg.tts.model}` ({cfg.tts.voice_mode} voices)*
{partial}
## Episode

| | |
|---|---|
| Duration | **{_fmt_secs(seconds)}** |
| Scenes / lines | {len(scenes)} / {sum(len(s.lines) for s in scenes)} |
| Audio chunks | {tts_requests} |
| Files | `episode.wav`{" · `episode.mp3`" if (stitched or {}).get("mp3") else " (MP3 skipped: ffmpeg not installed)"} |

| Scene | Title | Lines |
|---|---|---|
{scene_rows}

## What was adapted

| Kind | Original | Adapted | Why |
|---|---|---|---|
{change_rows}

### Plot beats (kept identical to the source)
{beats}

## Cast

| Character | | Voice | Voice description |
|---|---|---|---|
{voice_rows}

## API calls

- TTS synthesis requests: **{tts_requests}** in the plan ({stats['synthesized']} made on the last run, \
{stats['skipped']} resumed from disk)
- Voice creation calls: **{design_calls}** (cached in `04_voices.json`, so reruns never recreate voices)

| LLM model | Calls | Input tokens | Output tokens (incl. thinking) | Est. cost |
|---|---|---|---|---|
{chr(10).join(llm_rows) or "| — | 0 | 0 | 0 | $0 |"}

## Cost estimate (prices from config.yaml)

| Item | Basis | Paid-tier cost |
|---|---|---|
| TTS audio output | {_fmt_secs(seconds)} × {cfg.pricing.audio_tokens_per_second:g} tok/s = {audio_tokens:,} tok × ${tts_price.output}/1M | ${audio_tokens / 1e6 * tts_price.output:.4f} |
| TTS text input | ≈{tts_in_tokens:,} tok × ${tts_price.input}/1M | ${tts_in_tokens / 1e6 * tts_price.input:.4f} |
| Adaptation + script LLM | token usage above | ${llm_cost:.4f}{" (+ unpriced: " + ", ".join(llm_unpriced) + ")" if llm_unpriced else ""} |
| **Total** | | **${tts_cost + llm_cost:.4f}** |

{"You are on the **free tier** (`pricing.free_tier: true`), so the actual charge is $0. The figures above are what this episode would cost at paid rates." if free else "Billed at paid-tier rates."}
Voice design and replication pricing is not published, so it isn't included.

## Free-tier fit
{chr(10).join(fit_lines)}
"""


def write_report(run: RunDir, cfg: Config, stitched: dict | None = None) -> Path:
    run.report.write_text(build_report(run, cfg, stitched), encoding="utf-8")
    return run.report
