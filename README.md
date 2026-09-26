# Katha Studio · কথা

**Story → culturally adapted → voiced audio drama.**

Katha Studio takes a long-form English story and gives it to a new audience the way a good editor would: through **adaptation, not translation**. Names, places, food, festivals, idioms and social texture change so the story feels native. The plot beats, the emotional arc and the ending stay exactly the same. The adapted story then becomes a multi-character audio-drama script and is performed by **Gemini 3.8 Flash TTS**, with a designed voice for every character.

The default target is a Bengali-speaking audience in West Bengal. Spoken output can be Indian English, Hindi or Bengali.

```
story.md ─▶ ingest ─▶ adapt ─▶ script ─▶ cast voices ─▶ render ─▶ stitch ─▶ episode.mp3
             chunks    change log  scenes +   designed /    TTS chunks  pauses +     + report.md
             + bible   + plot beats verbatim  library /     (resumable) loudness
                                    lines     your voice*
```
<sub>*only with a recorded consent clip; see [Voice consent](#voice-consent).</sub>

---

## Demo: *The Last Pie* → *The Last Bowl*

`samples/the_last_pie.md` is an original ~750-word story set in small-town Vermont. A widow nearly skips the fair's pie contest after her husband dies, then her grandson finds his handwritten secret ingredient. Here is what the pipeline did with it (real output, `runs/the-last-pie/`):

| Kind | Original | Adapted | Why (the model's reason) |
|---|---|---|---|
| food | Apple pie | Nolen gurer payesh | Iconic Bengal winter dessert built on seasonal date-palm jaggery |
| place | Maple Hollow, Vermont · Harvest Fair | Shantiniketan · Poush Mela | A town known for its arts and its winter fair |
| food | Pecan pie (the rival's) | Chaler payesh | The rival dessert in a local contest |
| character | Martha / Walt / Sam | Protima / Dadu / Shom | Bengali names and kinship terms (Didima, Dadu, Haren-kaku) |
| object | Black pepper (the secret) | Crushed green fennel | A surprising aromatic that cuts richness in Bengali sweets |

A few script lines (`03_script.json`). Spoken text is verbatim; delivery goes in `style`; momentary sounds are sparing inline tags:

```
narrator            November in Shantiniketan. The Poush Mela is coming, but for Protima,
                    the house feels hollow since her husband passed in March.
shom     [excited]  Didima, look what I found in this old ledger. Dadu wrote a recipe here.
protima             Oh. It was always ours, Shom. He kept the proportions. I just did the tasting. <sigh>
haren               I almost did not enter. It did not feel right, winning against you this year.
protima             Dadu would have hated that, Haren. He loved watching you lose!
```

**Audio.** 4 voices were cast (3 designed, 1 library) and 13 of the episode's 27 chunks are rendered: about 91 s of audio at 24 kHz mono. The rest is waiting on the free-tier daily TTS quota (see [costs](#costs--free-tier)); `katha render runs/the-last-pie` picks up where it stopped. As a spot check, Gemini transcribed two rendered chunks back to exactly the scripted lines.

> **Studio UI.** The Gradio app (`katha ui`) was designed first on a Claude Design canvas. Its three columns follow the pipeline: **Story** (upload, culture, language, estimate) → **Adaptation** (change log, script, report) → **Cast & render** (voice cards with previews, render progress, player and downloads).

---

## Setup

Requires Python 3.11+ (pydub uses `audioop`, which was removed in 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,ui]"        # add ,claude to use Claude as the writer LLM
copy .env.example .env            # then put your GEMINI_API_KEY in .env
katha doctor                      # checks config, keys (masked) and ffmpeg
```

MP3 export uses the ffmpeg binary bundled with `imageio-ffmpeg`, so you don't need a system install. A system `ffmpeg` on PATH takes precedence.

Keys are read only from `.env` or the environment. They are never logged or printed (`katha doctor` shows only their length), and `.env` is gitignored.

## Usage

```bash
# everything, but stop before TTS and print audio minutes + cost
katha all samples/the_last_pie.md --culture "Bengali, West Bengal" --lang en --dry-run

# stage by stage (each stage writes to runs/<slug>/ and can be rerun on its own)
katha adapt samples/the_last_pie.md --culture "Bengali, West Bengal" --lang en [--llm claude]
katha script runs/the-last-pie
katha voices runs/the-last-pie [--narrator-clip me.wav --consent-clip consent.wav]
katha render runs/the-last-pie [--max-scenes 1]

# fewer TTS requests: prebuilt voices, 2 speakers per request
katha --voice-mode prebuilt all story.md

katha ui                          # the studio in your browser (http://127.0.0.1:7860)
```

`katha all` reuses finished stages when the culture and language match, which saves quota. Pass `--force` to redo them.

### What a run contains

```
runs/<story_slug>/
  01_source.txt          the original story
  02_adaptation.json     adapted text · change log (original → adapted → reason) · story bible · plot beats
  03_script.json         characters (with voice descriptions) · scenes · lines {speaker, text, style, tags}
  04_voices.json         character → voice id cache (never recreated on rerun)
  voices/<id>.wav        designed-voice previews
  05_audio/*.wav         one WAV per TTS request + manifest.json (fingerprints for resume)
  episode.wav / .mp3     stitched episode
  report.md              what was adapted, cast, durations, API calls, cost, free-tier fit
  llm_usage.jsonl        token usage per LLM call
```

## How it works

**Ingest.** The story is split on chapter headings, then paragraphs, into chunks that fit the model's context. A paragraph is only split, at sentence boundaries, when it is bigger than a whole chunk.

**Adapt.** Each chunk is rewritten against a **story bible**, the list of original → adapted mappings from earlier chunks. That keeps "Martha" as "Protima" in chapter 9, not just in chapter 1. The prompt requires that anything foreign that is *central to the plot* (the signature dish, the festival the story builds towards) becomes a native equivalent that serves the same plot function. Output is pydantic-validated JSON with a change log and the chunk's plot beats.

**Script.** The adapted prose becomes scenes of `{speaker, text, style, tags}`. Validators reject:
- stage directions inside spoken text, such as `(whispers)`, `[door slams]` or `*laughs*`
- inline tags the TTS doesn't support
- speakers that were never declared
- scripts that skip a plot beat

When a check fails, the model gets its own errors back once and is asked to repair its answer.

**Cast.**
- **`designed` mode** (default): one designed voice per character, built from its description.
- **`prebuilt` mode**: distinct voices from the `en-IN`/`hi-IN`/`bn-IN` voice library.

Voices are cached by a fingerprint of the character's description, so reruns never recreate them.

**Render.** The API only allows 2-speaker requests with prebuilt voices, and designed voices must be synthesised turn by turn (see [API_NOTES.md](API_NOTES.md)). The planner groups lines accordingly. Each chunk is written atomically and fingerprinted, so:
- interrupted renders resume
- a broken file is redone
- editing one line re-renders only that chunk

**Stitch.**
- 300 ms between lines and 1 s between scenes
- each chunk RMS-matched to −16 dBFS so voices sit at the same level
- a −1 dBFS peak ceiling
- 8 ms fades to prevent clicks

**LLM choice.** `gemini-3.8-flash` (free tier) by default. If its daily quota runs out, it falls back to `gemini-3.1-flash-lite`. Claude is available with `--llm claude`.

## Costs & free tier

Prices live in `config.yaml` (from the Gemini pricing page, September 2026) and every figure in `report.md` is computed from them.

| | Price | Per minute of audio |
|---|---|---|
| `gemini-3.8-flash-tts` output | $9.00 / 1M audio tokens · 25 tokens/s | **≈ $0.0135** |
| `gemini-3.8-flash-lite-tts` output | $6.00 / 1M | ≈ $0.009 |
| TTS text input | $0.50 / 1M tokens | negligible |

So a 30-minute episode costs about **$0.40** at paid rates. The pricing page says prices double on 2027-01-01. Voice design and replication prices are not published.

**Free tier (what we actually hit, 2026-09-26):**
- `gemini-3.8-flash` (text): **20 requests/day**.
- `gemini-3.8-flash-tts`: about **10 requests/day**. The error says 10, but 12 got through that day.

That shapes how the tool behaves:
- A 750-word story needs **27 TTS requests with designed voices** but **11 with prebuilt voices**. `--dry-run` warns when a plan exceeds the daily quota and suggests prebuilt mode.
- A daily-quota error is **not** retried (retrying can't help). The command exits with code 3, everything finished so far stays on disk, and rerunning it the next day resumes.
- Per-minute limits and 5xx errors use exponential backoff with jitter and honour `Retry-After`. A client-side limiter spaces calls at `rate_limit.requests_per_minute`.

## Voice consent

> ⚠️ **Only replicate your own voice, or a voice whose owner has given explicit, recorded consent.** Cloning someone's voice without consent is harmful and violates Google's terms.

`--narrator-clip` / `--consent-clip` (or the UI's "Narrate in your own voice", behind a consent checkbox) create a replicated voice for the narrator:
- **source clip:** 10–30 s, ideally 24 kHz mono 16-bit WAV, in a quiet room
- **consent clip:** the **same speaker**, same mic and room, reading:
  *"I am the owner of this voice and I consent to Google using this voice to create a synthetic voice model."*

The tool checks clip length and format. It cannot check *who* is speaking; that responsibility is yours.

Voice design for child characters is refused by Google's safety policy ("Voice prompt was blocked by safety policies"). Katha respects that: those characters get a library voice, and it never re-prompts to get around the policy.

## Tests

```bash
pytest -q        # 55 tests, all APIs mocked, ~15 s
```

The tests cover:
- chunking
- schema validation (stage directions, tags, speakers, plot beats)
- the validation-repair loop
- speaker-chunk planning in both voice modes
- resume and re-render
- stitching (pause lengths, loudness matching)
- cost maths from config
- quota fallback
- report contents
- the UI's HTML escaping

## Layout

```
katha/
  cli.py      typer CLI                    ingest.py  chunking
  config.py   config.yaml + .env           adapt.py   cultural adaptation + story bible
  llm.py      Gemini / Claude + repair     script.py  drama script + beat coverage
  models.py   pydantic schemas             voices.py  casting, cache, consent-gated replication
  retry.py    backoff, rate & quota        tts.py     Gemini TTS + Voices API wrapper
  runs.py     run-dir layout               render.py  request planning, synthesis, resume, estimate
  log.py      key=value logging            stitch.py  pauses, loudness, WAV/MP3
  ui.py       Gradio studio                report.py  report.md
API_NOTES.md  what was verified against the live API
```

## Known limitations

- The cheaper `gemini-3.1-flash-lite` fallback adapts well but sometimes compresses the final scene. Prefer `gemini-3.8-flash` or Claude for the final pass.
- Loudness matching is RMS-based, not full LUFS metering. That's fine for speech-only drama, but not broadcast-spec.
- Designed voices can come out younger than described. The design prompt now leads with the character's age to counter this.
