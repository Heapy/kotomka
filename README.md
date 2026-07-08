# Kotomka

Local web service that turns a video URL into a readable web presentation with
summary, detailed notes, key frames, timestamps, full transcript, and PDF export.

## Quick Start

```bash
uv sync --extra dev
uv run kotomka serve
```

On macOS, add `--extra ocr` to enable Apple Vision OCR of slide frames
(smarter dedupe of bullet-build slides and OCR-grounded captions); without it
the pipeline runs the same minus OCR.

Open <http://127.0.0.1:8000>.

The service works without external AI keys by using fake providers. For live
processing, configure:

- `ASSEMBLYAI_API_KEY` for speaker-labeled transcription.
- Either `OPENAI_API_KEY` for the public OpenAI Platform API, or run
  `uv run kotomka codex-login` to use the ChatGPT/Codex subscription route for
  report generation and frame scoring.

The Codex subscription route is not an OpenAI Platform API key. It stores a
separate OAuth state under `data/codex_subscription_auth.json` by default and
uses the ChatGPT Codex backend for Responses-style text/vision requests. It
does not provide audio transcription; STT remains a separate pluggable provider.

## YouTube Cookies

Some YouTube videos fail before media processing with "Sign in to confirm you're
not a bot." For those, set either:

- `Cookies browser`: a yt-dlp browser source such as `firefox`, `chrome`, or
  `safari`. The web form uses a browser select with `firefox` selected by
  default; API callers may pass yt-dlp profile syntax such as
  `chrome:Profile 1`.
- `Cookies file`: a fresh Netscape-format `cookies.txt` export.

Use only one of those fields per job. If yt-dlp says the YouTube account cookies
are no longer valid, refresh the YouTube login in that browser or export a new
cookies file, then retry the job.

## Pipeline

1. `yt-dlp` downloads the video plus metadata (description, chapters, tags,
   upload date, language); `ffmpeg` extracts mono 16 kHz FLAC audio.
2. STT provider returns a normalized speaker-labeled transcript (AssemblyAI
   requests current speech models with keyterm boosting derived from the video
   metadata; the raw payload is kept as `transcript_raw.json`).
3. Candidate frames come from slide-aware plateau detection, scene detection,
   and gap filling; on macOS with the `ocr` extra they are OCR-annotated and
   bullet-build slide sequences collapse to the final slide.
4. The LLM scores frame batches against matching transcript windows, selection
   guarantees at least one frame per chapter, and the winners are re-captioned
   at high image detail.
5. The report is generated in one pass for short videos or map-reduced through
   chapter-aligned structured notes for long ones, with the selected frame
   images attached. Citations are then snapped to real transcript timestamps.
6. An assessment pass critiques originality, freshness (anchored to the upload
   date, optionally web-grounded on the OpenAI provider), audience,
   actionability, and whether the report replaces watching.
7. FastAPI renders HTML and exports a cached PDF.

## Provider Defaults

`KOTOMKA_LLM_PROVIDER=auto` resolves in this order:

1. `codex_subscription` if a Codex OAuth auth file exists.
2. `openai` if `OPENAI_API_KEY` exists.
3. `fake`.

`KOTOMKA_STT_PROVIDER` defaults to `fake`. Set it to `assemblyai` for live
speaker-labeled transcription, or to `whisper` for offline transcription with
faster-whisper (`uv sync --extra whisper`; first run downloads model weights;
no speaker diarization).

## Configuration

Settings come from `.env.local` / environment variables with the `KOTOMKA_`
prefix. Everything has a sensible default; the most useful knobs:

| Setting | Default | Purpose |
| --- | --- | --- |
| `KOTOMKA_STT_PROVIDER` | `fake` | `fake`, `assemblyai`, or `whisper` |
| `KOTOMKA_LLM_PROVIDER` | `auto` | `auto`, `fake`, `openai`, `codex_subscription` |
| `KOTOMKA_OPENAI_MODEL` / `KOTOMKA_CODEX_MODEL` | `gpt-4.1` / `gpt-5.4` | report + assessment model |
| `KOTOMKA_OPENAI_SCORING_MODEL` / `KOTOMKA_CODEX_SCORING_MODEL` | unset | cheaper model for frame scoring (falls back to the main model) |
| `KOTOMKA_REPORT_MAX_IMAGES` | `16` | selected frame images attached to the report call |
| `KOTOMKA_REPORT_SINGLE_PASS_MAX_CHARS` | `24000` | transcripts longer than this are map-reduced |
| `KOTOMKA_REPORT_CHUNK_TARGET_SECONDS` | `600` | map-reduce chunk size |
| `KOTOMKA_ASSESSMENT_ENABLED` | `true` | originality/freshness/usefulness pass |
| `KOTOMKA_ASSESSMENT_WEB_SEARCH` | `false` | ground the assessment with OpenAI web search (openai provider only) |
| `KOTOMKA_RECAPTION_SELECTED_FRAMES` | `true` | high-detail re-caption of selected frames |
| `KOTOMKA_FRAME_MAX_GAP_SECONDS` | `60` | guaranteed candidate-frame coverage |
| `KOTOMKA_FRAME_PLATEAU_MIN_DWELL_SECONDS` | `3.0` | minimum slide dwell to count as a plateau |
| `KOTOMKA_FRAME_BLUR_THRESHOLD` | `0` | blur gate for candidates (0 = off) |
| `KOTOMKA_FRAME_OCR_ENABLED` | `true` | OCR annotation when the `ocr` extra is installed |
| `KOTOMKA_STT_KEYTERMS_MAX` | `200` | keyterm boost cap for AssemblyAI |
| `KOTOMKA_WHISPER_MODEL` | `large-v3` | faster-whisper model size |

Example `.env.local`:

```dotenv
ASSEMBLYAI_API_KEY=...
OPENAI_API_KEY=...
KOTOMKA_STT_PROVIDER=assemblyai
KOTOMKA_LLM_PROVIDER=auto
KOTOMKA_ASSESSMENT_WEB_SEARCH=1
```

## Tests

```bash
uv run pytest
```

Integration tests use fake providers and do not call external APIs.
