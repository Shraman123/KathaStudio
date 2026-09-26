# API_NOTES — Gemini 3.8 TTS (checked 2026-09-26)

Sources: official Gemini API docs. Everything below comes from those pages. Anything the docs don't cover is marked **UNKNOWN**.
- Speech generation (includes the prompting guide and the 3.1 → 3.8 migration guide): https://ai.google.dev/gemini-api/docs/speech-generation
- Voice design: https://ai.google.dev/gemini-api/docs/voice-design
- Voice replication: https://ai.google.dev/gemini-api/docs/voice-replication
- Model cards: https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash-tts , .../gemini-3.8-flash-lite-tts , .../gemini-3.8-flash
- Pricing: https://ai.google.dev/gemini-api/docs/pricing · Rate limits: https://ai.google.dev/gemini-api/docs/rate-limits

## SDK
- Package: `google-genai` **>= 2.25.0** (the latest on PyPI today is 2.25.0; the docs name it as the minimum for the voices API). Import with `from google import genai`, then `client = genai.Client()`, which reads `GEMINI_API_KEY`.
- TTS uses `client.interactions.create(...)`. Voices use `client.voices.create / get / list`.

## Models
| model | status | input limit | output limit | languages |
|---|---|---|---|---|
| `gemini-3.8-flash-tts` | GA (replaces `gemini-3.1-flash-tts-preview`) | 8,192 tok | 16,384 tok (~655 s audio at 25 tok/s) | 130+ |
| `gemini-3.8-flash-lite-tts` | GA | — | — | ~100 |
| `gemini-3.8-flash` (adaptation LLM) | stable, structured output supported | 1,048,576 | 65,536 | — |

The docs list **Hindi and Bengali as supported on both TTS models**. Language is auto-detected from the text. Designed voices also take a `language_code`.

## Single-speaker request (confirmed shape)
```python
interaction = client.interactions.create(
    model="gemini-3.8-flash-tts",
    input=[{"type": "user_input", "content": [{
        "type": "text", "text": "Have a wonderful day! <laugh>",
        "annotations": [{"type": "speech_metadata", "style": "cheerful and friendly"}],
    }]}],
    response_format={"type": "audio"},
    generation_config={"speech_config": [{"voice": "Kore"}]},   # LIST here; voice = prebuilt name or voice_... id
)
wav_bytes = base64.b64decode(interaction.output_audio.data)
```

## Multi-speaker request (confirmed shape)
- In this form `speech_config` is a **dict**, not a list: `{"mode": "conversational", "speakers": [{"speaker": "Joe", "voice": "Puck"}, {"speaker": "Jane", "voice": "Kore"}]}`
- One `content` item per turn. Every turn **must** set `speech_metadata.speaker`.
- **Up to 2 speakers per request, and only with prebuilt voices.**
- ⚠️ **Designed or replicated voices cannot go in `speakers`.** Quoting the docs: "synthesize each speaker's turn individually."

## Output audio
- Unary requests return **WAV with a RIFF header by default**: 24 kHz, 16-bit signed LE, mono. You write the bytes straight to `.wav`, with no manual header. (This changed from 3.1, which returned raw PCM.)
- Streaming returns `audio/l16`, headerless. Other options: `audio/mulaw`, `audio/alaw`.

## Voice design
```python
v = client.voices.create(store=True, voice={
    "model": "gemini-3.8-flash-tts", "type": "prompted",
    "display_name": "...", "gender": "male", "language_code": "en-IN",
    "prompted": {"input": "A warm, thoughtful ... in his late 60s ..."},
})
v.id                 # "voice_..." — persistent
v.sample_audio.data  # base64 WAV preview (also returned by voices.get; voices.list omits it)
```
- `store=True`: up to 200 voices per project, 1-year TTL. `store=False`: returns a `voicekey_...` that lasts 7 days.
- Voice design pricing: **UNKNOWN** (not on the pricing page).
- Extended Voice Library: `client.voices.list()` has hundreds of voices and can filter by language, region, accent, gender and so on. This is a good fallback for Indian accents.

## Voice replication (narrator = my own voice)
```python
client.voices.create(store=True, voice={
    "model": "gemini-3.8-flash-tts", "type": "replicated", "display_name": "...",
    "replicated": {
        "source_audio":  {"mime_type": "audio/wav", "data": <b64>},
        "consent_audio": {"mime_type": "audio/wav", "data": <b64>},
    }})
```
- The source clip must be **10–30 s** (the launch screenshot said 15–20 s; the docs say 10–30 s). The docs recommend 24 kHz mono 16-bit WAV.
- Both clips must be the same adult speaker, recorded in a quiet room, with matching recording conditions.
- Required consent sentence (English; the docs also give 30 other languages): *"I am the owner of this voice and I consent to Google using this voice to create a synthetic voice model."*
- Free-tier eligibility, allowlisting and regional limits: **UNKNOWN**. The docs don't say.

## Prompting rules (these apply directly to the script stage)
- The text is a **verbatim transcript**. No stage directions and no speaker labels go in the text.
- `speech_metadata.style` holds sustained delivery (emotion, pace). Keep it short, and try an empty style first.
- Inline tags are for point-in-time events only. Tags: `<argh> <breath> <heavy breath> <exhales> <cackle> <cheer> <chuckle> <cough> <cry> <gasp> <giggle> <groan> <growl> <grunt> <grr> <hiss> <laugh> <moan> <pant> <pff> <scream> <shout> <shriek> <sigh> <sneeze> <snicker> <snort> <sob> <throat-clearing> <tsk> <whimper> <yawn> <short pause> <long pause>`. Tags stay in English even when the transcript is Hindi or Bengali.
- Backchannels go in pipes inside a turn, e.g. `|hmm|`.
- Don't put fixed traits (age, gender, accent) in `style`. Put them in the voice. Avoid meta-instructions such as "don't switch speaker", which make drift worse.

## Pricing (standard tier, per 1M tokens) → goes into config.yaml
| model | input text | output audio | from 2027-01-01 |
|---|---|---|---|
| flash-tts | $0.50 | $9.00 | $1.00 / $18.00 |
| flash-lite-tts | $0.50 | $6.00 | $1.00 / $12.00 |
| 3.8-flash (text) | $0.75 | $3.75 out | $1.50 / $7.50 |
- **25 audio tokens per second is confirmed.** So flash-tts costs about **$0.0135 per minute of audio**, and 30 minutes costs about $0.41.
- Batch API is 50% off. Free tier: the docs say input and output are "free of charge" for all three models.
- **Free-tier RPM/RPD: UNKNOWN.** The docs only say to check AI Studio (https://aistudio.google.com/rate-limit). The pipeline will take `requests_per_minute` from config, with a conservative default of 10. Please read your real numbers from AI Studio.

## Design implications for Katha
1. **Casting trade-off.** Designed voices sound better but need one call per turn. Prebuilt voices allow 2-speaker requests, so fewer calls. Plan: a `voice_mode: designed | prebuilt` setting.
   - `designed` (default): one call per run of consecutive lines from the same speaker.
   - `prebuilt`: scenes split into chunks of ≤2 speakers. Narrator lines are always their own calls.
2. Keep each request well under 8,192 input tokens and ~10 min of audio. Scene chunks will be far smaller than that anyway.
3. No manual header handling: stitching reads the WAVs directly.
4. The `--narrator-clip` check is 10–30 s. The consent clip is checked for length and format only; we can't verify who is speaking. The consent warning goes in the code, the CLI and the README.
5. **Local env: `ffmpeg` is not installed.** It's needed for pydub's MP3 export. Install with `winget install Gyan.FFmpeg` before M4.

## Confirmed live (2026-09-26, with a real key)
- `interactions.create` with a `response_format` JSON schema works on `gemini-3.8-flash`; the result is in `.output_text`. Usage fields: `usage.total_input_tokens`, `total_output_tokens`, `total_thought_tokens`.
- **Free-tier daily limit for `gemini-3.8-flash` is 20 requests/day.** Found from the 429 text: "limit: 20 requests per day on Free Tier". `gemini-3.1-flash-lite` has a separate quota and works as a fallback, but it's slow (about 2–3 min per call).
- `gemini-3.8-flash-lite` does **not** exist (404). The lite text model is `gemini-3.1-flash-lite`.
- The SDK logs "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY" at client init, but an explicit `api_key=` still takes precedence (`self.api_key = api_key or env_api_key`).
- `voices.create(type="prompted")` works and returns `voice_…` ids plus a 40–50 s preview WAV.
- **Voice design for a 12-year-old character was refused**: `400 "Voice prompt was blocked by safety policies."` We fall back to a library voice and never re-prompt around the policy.
- `voices.list(language_code=["en-IN"], gender=[…], type_=["prebuilt"])` returns library ids such as `en-in-advisor-1`.
- TTS unary responses are WAV at 24 kHz, mono, 16-bit, as documented. 7 single-speaker requests took 5–60 s each. Gemini transcribed the rendered chunks back to exactly the scripted lines.
- Designed voices sounded younger than described ("late 60s" came out like 30s), so the design prompt now leads with the age.
