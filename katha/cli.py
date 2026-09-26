"""Katha Studio CLI.

    katha adapt story.txt --culture "Bengali, West Bengal" --lang en
    katha script <run_dir>
    katha voices <run_dir> [--narrator-clip my.wav --consent-clip consent.wav]
    katha render <run_dir>
    katha all story.txt --culture "..." --lang en [--dry-run] [--max-scenes 2]
    katha doctor
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from . import adapt as adapt_mod
from . import render as render_mod
from . import report as report_mod
from . import script as script_mod
from . import stitch as stitch_mod
from . import voices as voices_mod
from .config import Config, ConfigError, load_config, load_env, mask
from .llm import LLMOutputError
from .log import setup_logging
from .retry import QuotaExhausted
from .runs import RunDir

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
log = logging.getLogger("katha")


class Lang(str, Enum):
    en = "en"
    hi = "hi"
    bn = "bn"


class LLMChoice(str, Enum):
    gemini = "gemini"
    claude = "claude"


class State:
    cfg: Config


state = State()


@app.callback()
def main(
    config: Path = typer.Option(None, "--config", help="Path to config.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    voice_mode: Optional[str] = typer.Option(
        None, "--voice-mode", help="designed (best quality, 1 request per turn) | prebuilt (2-speaker requests, fewer calls)"),
):
    # Windows consoles default to cp1252; stories and scripts contain Bengali/Devanagari and symbols
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    setup_logging(verbose)
    load_env()
    try:
        state.cfg = load_config(config)
        if voice_mode:
            state.cfg.tts = state.cfg.tts.model_copy(update={"voice_mode": voice_mode})
            type(state.cfg.tts).model_validate(state.cfg.tts.model_dump())
    except (ConfigError, ValueError) as e:
        typer.secho(f"Config error: {e}", fg="red", err=True)
        raise typer.Exit(1)


def _run(stage: str, fn, *args, **kwargs):
    """Run a stage and turn the expected failures into clean messages without tracebacks."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        typer.secho(f"[{stage}] not implemented yet: {e}", fg="yellow", err=True)
        raise typer.Exit(2)
    except (ConfigError, FileNotFoundError, ValueError) as e:
        typer.secho(f"[{stage}] {e}", fg="red", err=True)
        raise typer.Exit(1)
    except QuotaExhausted as e:
        typer.secho(f"[{stage}] {str(e).split(' API said:')[0]}", fg="yellow", err=True)
        typer.secho("Everything finished so far is saved. Rerun the same command later to resume.", err=True)
        raise typer.Exit(3)
    except LLMOutputError as e:
        typer.secho(f"[{stage}] {e}", fg="red", err=True)
        raise typer.Exit(1)


def _existing_file(p: Path, label: str) -> Path:
    if not p.is_file():
        typer.secho(f"{label} not found: {p}", fg="red", err=True)
        raise typer.Exit(1)
    return p


@app.command()
def adapt(
    story: Path = typer.Argument(..., help="Story file (.txt / .md)"),
    culture: Optional[str] = typer.Option(None, help="Target culture (default from config)"),
    lang: Optional[Lang] = typer.Option(None, help="Output language of the script"),
    llm: Optional[LLMChoice] = typer.Option(None, help="Adaptation LLM (default from config)"),
):
    """Stages 1-2: ingest and culturally adapt a story."""
    _existing_file(story, "Story")
    run = RunDir.for_story(state.cfg.paths.runs_dir, story)
    _run("adapt", adapt_mod.adapt, story, run, state.cfg,
         culture=culture or state.cfg.defaults.culture,
         lang=(lang.value if lang else state.cfg.defaults.lang),
         llm=(llm.value if llm else state.cfg.llm.provider))


@app.command()
def script(
    run_dir: Path = typer.Argument(..., help="runs/<story_slug>"),
    llm: Optional[LLMChoice] = typer.Option(None),
):
    """Stage 3: turn the adapted story into an audio-drama script."""
    _run("script", script_mod.write_script, RunDir(run_dir), state.cfg,
         llm=(llm.value if llm else state.cfg.llm.provider))


@app.command()
def voices(
    run_dir: Path = typer.Argument(...),
    narrator_clip: Optional[Path] = typer.Option(None, help="10-30s WAV of YOUR voice"),
    consent_clip: Optional[Path] = typer.Option(None, help="Same speaker reading the consent statement"),
):
    """Stage 4: cast a voice for each character (cached in 04_voices.json)."""
    if bool(narrator_clip) != bool(consent_clip):
        typer.secho("--narrator-clip and --consent-clip must be given together.", fg="red", err=True)
        raise typer.Exit(1)
    if narrator_clip:
        _existing_file(narrator_clip, "Narrator clip")
        _existing_file(consent_clip, "Consent clip")
        typer.secho(voices_mod.CONSENT_WARNING, fg="yellow", err=True)
    _run("voices", voices_mod.cast_voices, RunDir(run_dir), state.cfg,
         narrator_clip=narrator_clip, consent_clip=consent_clip)


@app.command()
def render(
    run_dir: Path = typer.Argument(...),
    max_scenes: Optional[int] = typer.Option(None, min=1),
):
    """Stages 5-7: synthesize, stitch and write the report."""
    run = RunDir(run_dir)
    _run("render", render_mod.render, run, state.cfg, max_scenes=max_scenes)
    stitched = _run("stitch", stitch_mod.stitch, run, state.cfg)
    report = _run("report", report_mod.write_report, run, state.cfg, stitched)
    typer.secho(f"episode: {stitched['mp3'] or stitched['wav']}  ({stitched['seconds']:.1f}s)", fg="green")
    typer.echo(f"report:  {report}")


@app.command("all")
def all_(
    story: Path = typer.Argument(...),
    culture: Optional[str] = typer.Option(None),
    lang: Optional[Lang] = typer.Option(None),
    llm: Optional[LLMChoice] = typer.Option(None),
    dry_run: bool = typer.Option(False, help="Stages 1-3 only; print estimated audio minutes and cost"),
    max_scenes: Optional[int] = typer.Option(None, min=1),
    narrator_clip: Optional[Path] = typer.Option(None),
    consent_clip: Optional[Path] = typer.Option(None),
    force: bool = typer.Option(False, help="Redo the adapt/script stages even if their output exists"),
):
    """Run the whole pipeline end to end. Stages whose output already exists are reused (saves quota)."""
    run = RunDir.for_story(state.cfg.paths.runs_dir, story)
    want_culture = culture or state.cfg.defaults.culture
    want_lang = lang.value if lang else state.cfg.defaults.lang
    redo = force or not adapt_mod.adaptation_matches(run, want_culture, want_lang)
    if redo:
        adapt(story, culture, lang, llm)
    else:
        typer.echo(f"[adapt] reusing {run.adaptation} (use --force to redo)")
    if redo or not run.script.exists():
        script(run.root, llm)
    else:
        typer.echo(f"[script] reusing {run.script} (use --force to redo)")
    run_dir = run.root
    if dry_run:
        _run("estimate", render_mod.estimate, RunDir(run_dir), state.cfg, max_scenes=max_scenes)
        return
    voices(run_dir, narrator_clip, consent_clip)
    render(run_dir, max_scenes)


@app.command()
def ui(port: int = typer.Option(7860), share: bool = typer.Option(False, help="Public gradio.live link")):
    """Launch the Gradio studio UI."""
    try:
        from .ui import launch
    except ImportError:
        typer.secho("The UI needs gradio: pip install -e .[ui]", fg="red", err=True)
        raise typer.Exit(1)
    launch(port=port, share=share)


@app.command()
def doctor():
    """Check config, API keys (masked) and ffmpeg."""
    cfg = state.cfg
    typer.echo(f"config        llm={cfg.llm.provider} tts={cfg.tts.model} voice_mode={cfg.tts.voice_mode} "
               f"rpm={cfg.rate_limit.requests_per_minute}")
    typer.echo(f"runs_dir      {cfg.paths.runs_dir}")
    typer.echo(f"GEMINI_API_KEY    {mask(os.environ.get('GEMINI_API_KEY'))}")
    typer.echo(f"ANTHROPIC_API_KEY {mask(os.environ.get('ANTHROPIC_API_KEY'))}")
    from .stitch import find_ffmpeg
    ff = find_ffmpeg()
    typer.echo(f"ffmpeg        {ff or 'NOT FOUND (needed for MP3 export: pip install imageio-ffmpeg)'}")


if __name__ == "__main__":
    app()
