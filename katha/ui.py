"""Gradio UI: upload a story, pick culture/language, preview voices, render, play and download.

    katha ui            # or: python -m katha.ui

The layout follows the Claude Design canvas (Story → Adaptation → Cast & render). Every
button calls the same stage functions as the CLI, so a run started here can be finished with
`katha render runs/<slug>` and the other way round.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy (via gradio) crashes on this machine without it

import html
import json
import logging
from pathlib import Path

import gradio as gr

from . import adapt as adapt_mod
from . import render as render_mod
from . import report as report_mod
from . import script as script_mod
from . import stitch as stitch_mod
from . import voices as voices_mod
from .config import Config, ConfigError, load_config, load_env
from .llm import LLMOutputError
from .models import Adaptation, Script
from .retry import QuotaExhausted
from .runs import RunDir

log = logging.getLogger(__name__)

LANGS = {"English": "en", "हिन्दी": "hi", "বাংলা": "bn"}
MAX_VOICES = 8
SPEAKER_COLORS = ["#8A7B6C", "#B8412C", "#17494A", "#8C5A14", "#5B3F8C", "#2F6B3A", "#9E3522", "#3A5A8C"]

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600'
        '&family=Noto+Serif+Bengali:wght@600&display=swap" rel="stylesheet">')

CSS = """
.gradio-container {max-width: 1440px !important}
#katha-header {display:flex; align-items:baseline; gap:12px; padding:4px 0 8px; border-bottom:1px solid #D9CDB8}
#katha-header .bn {font-family:'Noto Serif Bengali',serif; font-size:30px; color:#B8412C; font-weight:600}
#katha-header .name {font-family:'Fraunces',serif; font-size:26px; font-weight:600}
#katha-header .tag {color:#6B5E52; font-size:14px}
.step h2 {font-family:'Fraunces',serif !important; font-weight:600 !important; margin:4px 0 !important}
.title-swap {font-family:'Fraunces',serif; font-size:30px; font-weight:600}
.title-swap s {color:#8A7B6C; font-size:20px; font-weight:500; margin-right:10px}
.script {background:#1F1A17; color:#F6F1E7; border-radius:12px; padding:16px 18px; max-height:520px; overflow:auto}
.script h4 {color:#CDBFA9; font-size:12px; letter-spacing:.08em; text-transform:uppercase; margin:14px 0 6px}
.script .ln {display:grid; grid-template-columns:110px 1fr; gap:12px; font-size:14px; line-height:1.5; margin:4px 0}
.script em {color:#CDBFA9}
.script code {background:#3A322C; color:#F2C9A0; border-radius:4px; padding:1px 6px; font-size:12px}
.estimate {background:#EDE4D3; border-radius:12px; padding:12px 14px}
.cast {display:flex; gap:12px; align-items:flex-start; justify-content:space-between; padding:10px 12px;
       border:1px solid #D9CDB8; border-radius:12px; background:#fff; margin-bottom:8px}
.cast .meta {color:#6B5E52; font-size:12px; margin-left:6px}
.cast .desc {color:#4A3F36; font-size:13px; line-height:1.4; margin-top:2px}
.cast .badge {flex-shrink:0; font-size:12px; font-weight:600; padding:3px 8px; border-radius:6px; text-transform:capitalize}
.note {color:#6B5E52; font-size:12px}
"""


# The design is a light "paper" look, so pin Gradio to light mode even when the OS is dark.
FORCE_LIGHT_JS = """() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); }
}"""


def make_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.colors.orange, neutral_hue=gr.themes.colors.stone,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    ).set(
        body_background_fill="#F6F1E7", block_background_fill="#FFFFFF", block_border_color="#D9CDB8",
        body_text_color="#1F1A17", button_primary_background_fill="#B8412C",
        button_primary_background_fill_hover="#9E3522", button_primary_text_color="#FFFFFF",
        button_secondary_background_fill="#F6F1E7", button_secondary_border_color="#1F1A17",
    )


# ---------------- pure helpers (unit-tested) ----------------

def script_html(script: Script, max_scenes: int | None = None) -> str:
    colors = {c.id: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, c in enumerate(script.characters)}
    colors["narrator"] = "#CDBFA9"
    names = {c.id: c.name for c in script.characters}
    out = ['<div class="script">']
    for s in script.scenes[:max_scenes]:
        out.append(f"<h4>{html.escape(s.id.replace('_', ' ').title())} · {html.escape(s.title)}</h4>")
        for ln in s.lines:
            text = html.escape(ln.text)
            for t in ln.tags:  # render inline tags as pills
                text = text.replace(html.escape(f"<{t}>"), f"<code>&lt;{html.escape(t)}&gt;</code>")
            style = f"<em>{html.escape(ln.style)} · </em>" if ln.style else ""
            out.append(f'<div class="ln"><b style="color:{colors.get(ln.speaker, "#fff")}">'
                       f'{html.escape(names.get(ln.speaker, ln.speaker))}</b><span>{style}{text}</span></div>')
    out.append("</div>")
    return "".join(out)


KIND_BADGE = {"designed": ("#E3EEEC", "#17494A"), "replicated": ("#F3E3DE", "#9E3522")}


def cast_card(name: str, meta: str, kind: str, desc: str) -> str:
    bg, fg = KIND_BADGE.get(kind, ("#F1E4C8", "#6A4A0E"))
    return (f'<div class="cast"><div><b>{html.escape(name)}</b> <span class="meta">{html.escape(meta)}</span>'
            f'<div class="desc">{html.escape(desc)}</div></div>'
            f'<span class="badge" style="background:{bg};color:{fg}">{html.escape(kind)}</span></div>')


def estimate_md(summary: str) -> str:
    """The CLI summary, reworded for the UI (one line per fact, no CLI flags)."""
    return summary.replace("Try --voice-mode prebuilt (2-speaker requests)", "Pick Prebuilt voices").replace("\n", "  \n")


def change_rows(a: Adaptation) -> list[list[str]]:
    return [[c.kind, c.original, c.adapted, c.reason] for c in a.change_log]


def title_html(a: Adaptation) -> str:
    beats = sum(len(c.plot_beats) for c in a.chunks)
    return (f'<div class="title-swap"><s>{html.escape(a.title)}</s>→ {html.escape(a.adapted_title)}</div>'
            f'<p>{html.escape(a.culture)} · {len(a.change_log)} changes · {beats} plot beats kept · '
            f'<code>{html.escape(a.llm)}</code></p>')


def list_runs(cfg: Config) -> list[str]:
    d = cfg.paths.runs_dir
    return sorted(p.name for p in d.iterdir() if (p / "03_script.json").exists()) if d.exists() else []


# ---------------- app ----------------

def build_app(base_cfg: Config | None = None) -> gr.Blocks:
    load_env()
    base_cfg = base_cfg or load_config()

    def cfg_for(voice_mode: str) -> Config:
        c = base_cfg.model_copy(deep=True)
        c.tts.voice_mode = voice_mode.lower()
        return c

    def fail(e: Exception):
        if isinstance(e, QuotaExhausted):
            raise gr.Error(str(e).split(" API said:")[0] + " Progress is saved; press the button again later to resume.")
        raise gr.Error(str(e))

    # ----- handlers -----
    def run_view(run_dir: str, voice_mode: str):
        run = RunDir(Path(run_dir))
        a = Adaptation.model_validate_json(run.adaptation.read_text(encoding="utf-8"))
        s = Script.model_validate_json(run.script.read_text(encoding="utf-8"))
        est = render_mod.estimate_script(s, cfg_for(voice_mode))
        return title_html(a), change_rows(a), script_html(s), estimate_md(est.summary(cfg_for(voice_mode)))

    def on_adapt(file, culture, lang_label, llm, voice_mode, redo, progress=gr.Progress()):
        if not file:
            raise gr.Error("Upload a .txt or .md story first.")
        cfg = cfg_for(voice_mode)
        story = Path(file)
        run = RunDir.for_story(cfg.paths.runs_dir, story)
        lang = LANGS[lang_label]
        provider = "claude" if llm.startswith("Claude") else "gemini"
        try:
            if redo or not adapt_mod.adaptation_matches(run, culture, lang):
                progress(0.1, desc="Adapting the story (1 LLM call per chunk)…")
                adapt_mod.adapt(story, run, cfg, culture=culture, lang=lang, llm=provider)
                progress(0.55, desc="Writing the drama script…")
                script_mod.write_script(run, cfg, llm=provider)
            elif not run.script.exists():
                script_mod.write_script(run, cfg, llm=provider)
        except (QuotaExhausted, LLMOutputError, ConfigError, ValueError) as e:
            fail(e)
        return (str(run.root), *run_view(str(run.root), voice_mode),
                gr.update(choices=list_runs(base_cfg), value=run.root.name))

    def on_open(name, voice_mode):
        if not name:
            raise gr.Error("Pick a run.")
        run_dir = str(base_cfg.paths.runs_dir / name)
        cast = _cast_outputs(run_dir) if RunDir(Path(run_dir)).voices.exists() else _empty_cast()
        ep = _episode_outputs(run_dir)
        return (run_dir, *run_view(run_dir, voice_mode), *cast, *ep)

    def _empty_cast():
        return ["", *[gr.update(visible=False, value=None) for _ in range(MAX_VOICES)]]

    def _cast_outputs(run_dir: str):
        run = RunDir(Path(run_dir))
        s = Script.model_validate_json(run.script.read_text(encoding="utf-8"))
        cast = json.loads(run.voices.read_text(encoding="utf-8"))["voices"]
        rows, audios = [], []
        for c in s.characters[:MAX_VOICES]:
            v = cast.get(c.id, {})
            kind = v.get("kind", "—")
            rows.append(cast_card(c.name, f"{c.gender}, {c.age}", kind, c.voice_description))
            prev = v.get("preview")
            audios.append(gr.update(visible=bool(prev), value=prev, label=f"{c.name} · {kind} voice preview"))
        note = ("<p class='note'>Designing child voices is refused by Google's safety policy, so they use a "
                "library voice.</p>"
                if any(cast.get(c.id, {}).get("kind") == "library" and voices_mod.is_minor(c) for c in s.characters) else "")
        md = "".join(rows) + note
        audios += [gr.update(visible=False, value=None)] * (MAX_VOICES - len(audios))
        return [md, *audios]

    def on_cast(run_dir, voice_mode, narrator_clip, consent_clip, consent_ok):
        if not run_dir:
            raise gr.Error("Adapt a story (or open a run) first.")
        if bool(narrator_clip) != bool(consent_clip):
            raise gr.Error("Upload both the narrator clip and the consent clip, or neither.")
        if narrator_clip and not consent_ok:
            raise gr.Error("Tick the consent box: only replicate your own voice or one you have explicit consent for.")
        try:
            voices_mod.cast_voices(RunDir(Path(run_dir)), cfg_for(voice_mode),
                                   narrator_clip=Path(narrator_clip) if narrator_clip else None,
                                   consent_clip=Path(consent_clip) if consent_clip else None)
        except (QuotaExhausted, ConfigError, ValueError) as e:
            fail(e)
        return _cast_outputs(run_dir)

    def _episode_outputs(run_dir: str):
        run = RunDir(Path(run_dir))
        files = [str(p) for p in (run.episode_mp3, run.episode_wav, run.report) if p.exists()]
        audio = str(run.episode_mp3 if run.episode_mp3.exists() else run.episode_wav) if run.episode_wav.exists() else None
        rep = run.report.read_text(encoding="utf-8") if run.report.exists() else ""
        return audio, files or None, rep

    def on_render(run_dir, voice_mode, max_scenes, progress=gr.Progress()):
        if not run_dir:
            raise gr.Error("Adapt a story (or open a run) first.")
        run = RunDir(Path(run_dir))
        cfg = cfg_for(voice_mode)
        if not run.voices.exists():
            raise gr.Error("Cast the voices first.")
        progress(0, desc="Synthesising…")
        try:
            render_mod.render(run, cfg, max_scenes=int(max_scenes) or None,
                              on_progress=lambda d, t, k: progress(d / t, desc=f"Chunk {d}/{t} · {k}"))
            progress(1, desc="Stitching…")
            stitched = stitch_mod.stitch(run, cfg)
            report_mod.write_report(run, cfg, stitched)
        except (QuotaExhausted, ConfigError, FileNotFoundError, ValueError) as e:
            fail(e)
        return _episode_outputs(run_dir)

    # ----- layout -----
    with gr.Blocks(title="Katha Studio") as app:
        run_state = gr.State("")
        gr.HTML('<div id="katha-header"><span class="bn">কথা</span><span class="name">Katha Studio</span>'
                '<span class="tag">Adaptation, not translation: story → culturally adapted → voiced audio drama</span></div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes="step"):
                gr.Markdown("## 1 · Story")
                story = gr.File(label="Story (.txt / .md)", file_types=[".txt", ".md"], type="filepath")
                culture = gr.Textbox(label="Target culture", value=base_cfg.defaults.culture)
                lang = gr.Radio(list(LANGS), value="English", label="Spoken language")
                with gr.Row():
                    llm = gr.Dropdown(["Gemini 3.8 Flash", "Claude"], value="Gemini 3.8 Flash", label="Writer LLM")
                    voice_mode = gr.Dropdown(["Designed", "Prebuilt"], value=base_cfg.tts.voice_mode.title(),
                                             label="Voices", info="Prebuilt = fewer TTS requests")
                redo = gr.Checkbox(label="Redo adaptation even if one exists", value=False)
                adapt_btn = gr.Button("Adapt & write script", variant="primary")
                estimate = gr.Markdown(elem_classes="estimate")
                with gr.Accordion("Open an earlier run", open=bool(list_runs(base_cfg))):
                    runs = gr.Dropdown(list_runs(base_cfg), label="Runs", value=None)
                    open_btn = gr.Button("Open run")

            with gr.Column(scale=5, elem_classes="step"):
                gr.Markdown("## 2 · Adaptation")
                title = gr.HTML()
                with gr.Tabs():
                    with gr.Tab("Change log"):
                        changes = gr.Dataframe(headers=["Kind", "Original", "Adapted", "Why"], wrap=True,
                                               interactive=False, column_widths=["13%", "22%", "22%", "43%"])
                    with gr.Tab("Script"):
                        script_view = gr.HTML()
                    with gr.Tab("Report"):
                        report_md = gr.Markdown()

            with gr.Column(scale=4, elem_classes="step"):
                gr.Markdown("## 3 · Cast & render")
                with gr.Accordion("Narrate in your own voice (optional)", open=False):
                    gr.Markdown(f"⚠️ **{voices_mod.CONSENT_WARNING}**\n\nConsent clip must say: "
                                f"*“{voices_mod.CONSENT_STATEMENT}”*. Source clip: 10–30 s, 24 kHz mono WAV.")
                    narrator_clip = gr.Audio(label="Your voice (10–30 s)", type="filepath", sources=["upload", "microphone"])
                    consent_clip = gr.Audio(label="Consent statement, same speaker", type="filepath",
                                            sources=["upload", "microphone"])
                    consent_ok = gr.Checkbox(label="This is my own voice, or I have the speaker's explicit consent")
                cast_btn = gr.Button("Cast voices")
                cast_md = gr.HTML()
                previews = [gr.Audio(visible=False, interactive=False) for _ in range(MAX_VOICES)]
                max_scenes = gr.Slider(0, 20, value=0, step=1, label="Max scenes (0 = all)")
                render_btn = gr.Button("Render episode", variant="primary")
                episode = gr.Audio(label="Episode", interactive=False)
                downloads = gr.File(label="Downloads", file_count="multiple", interactive=False)

        view_outputs = [title, changes, script_view, estimate]
        adapt_btn.click(on_adapt, [story, culture, lang, llm, voice_mode, redo],
                        [run_state, *view_outputs, runs])
        open_btn.click(on_open, [runs, voice_mode],
                       [run_state, *view_outputs, cast_md, *previews, episode, downloads, report_md])
        cast_btn.click(on_cast, [run_state, voice_mode, narrator_clip, consent_clip, consent_ok],
                       [cast_md, *previews])
        render_btn.click(on_render, [run_state, voice_mode, max_scenes], [episode, downloads, report_md])
    return app


def launch(port: int = 7860, share: bool = False) -> None:
    cfg = load_config()
    build_app(cfg).queue().launch(server_port=port, share=share, theme=make_theme(), css=CSS, head=HEAD, js=FORCE_LIGHT_JS,
                                  allowed_paths=[str(cfg.paths.runs_dir)])


if __name__ == "__main__":
    launch()
