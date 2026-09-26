"""Stage 6: join chunks with natural pauses, normalise loudness, export WAV (+ MP3 if ffmpeg exists).

Loudness: each chunk is gain-matched to `stitch.target_dbfs` (RMS), so voices from
different requests sit at the same level. The episode is then peak-limited to -1 dBFS.
This is simple RMS matching, not full LUFS metering, and it's enough for speech-only audio.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .config import Config, StitchConfig
from .log import kv
from .runs import RunDir

log = logging.getLogger(__name__)

PEAK_CEILING_DBFS = -1.0
FADE_MS = 8  # removes clicks at chunk edges


def _level(seg, target_dbfs: float):
    if seg.dBFS == float("-inf"):
        return seg  # silence: nothing to normalise
    return seg.apply_gain(target_dbfs - seg.dBFS)


def stitch_segments(chunks: list[tuple[str, "AudioSegment"]], cfg: StitchConfig, frame_rate: int = 24000):
    """chunks: (scene_id, AudioSegment) in play order."""
    from pydub import AudioSegment

    line_gap = AudioSegment.silent(cfg.pause_between_lines_ms, frame_rate=frame_rate)
    scene_gap = AudioSegment.silent(cfg.pause_between_scenes_ms, frame_rate=frame_rate)
    episode = AudioSegment.silent(0, frame_rate=frame_rate)
    prev_scene = None
    for scene_id, seg in chunks:
        seg = _level(seg.set_frame_rate(frame_rate).set_channels(1), cfg.target_dbfs).fade_in(FADE_MS).fade_out(FADE_MS)
        if prev_scene is not None:
            episode += scene_gap if scene_id != prev_scene else line_gap
        episode += seg
        prev_scene = scene_id
    if len(episode) and episode.max_dBFS > PEAK_CEILING_DBFS:
        episode = episode.apply_gain(PEAK_CEILING_DBFS - episode.max_dBFS)
    return episode


def find_ffmpeg() -> str | None:
    """System ffmpeg first, then the binary bundled in the imageio-ffmpeg wheel (no system install needed)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def ffmpeg_available() -> bool:
    exe = find_ffmpeg()
    if exe:
        from pydub import AudioSegment
        AudioSegment.converter = exe
    return exe is not None


def stitch(run: RunDir, cfg: Config, keys: list[str] | None = None) -> dict:
    import json

    from pydub import AudioSegment

    if keys is None:
        stats_path = run.require(run.root / "render_stats.json", "render")
        keys = json.loads(stats_path.read_text(encoding="utf-8"))["keys"]
    manifest = json.loads((run.audio_dir / "manifest.json").read_text(encoding="utf-8"))
    chunks = []
    for k in keys:
        p = run.audio_dir / f"{k}.wav"
        if not p.exists():
            raise FileNotFoundError(f"Missing chunk {p.name}. Rerun `katha render {run.root}` (it resumes).")
        chunks.append((manifest[k]["scene"], AudioSegment.from_wav(p)))
    episode = stitch_segments(chunks, cfg.stitch, cfg.tts.sample_rate)
    episode.export(run.episode_wav, format="wav")
    out = {"wav": str(run.episode_wav), "mp3": None, "seconds": len(episode) / 1000, "chunks": len(chunks)}
    if ffmpeg_available():
        episode.export(run.episode_mp3, format="mp3", bitrate=cfg.stitch.mp3_bitrate,
                       tags={"title": run.root.name, "artist": "Katha Studio"})
        out["mp3"] = str(run.episode_mp3)
    else:
        log.warning("ffmpeg not found, so only WAV was exported. For MP3: pip install imageio-ffmpeg")
    log.info("episode stitched", extra=kv(seconds=f"{out['seconds']:.1f}", chunks=len(chunks),
                                         mp3=bool(out["mp3"])))
    return out
