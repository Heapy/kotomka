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

## Pipeline

1. `yt-dlp` downloads the YouTube video and metadata.
2. `ffmpeg` extracts audio and candidate frames.
3. STT provider returns normalized speaker-labeled transcript.
4. LLM/vision provider scores frames and generates the report JSON.
5. FastAPI renders HTML and exports a cached PDF.

## Provider Defaults

`KOTOMKA_LLM_PROVIDER=auto` resolves in this order:

1. `codex_subscription` if a Codex OAuth auth file exists.
2. `openai` if `OPENAI_API_KEY` exists.
3. `fake`.

`KOTOMKA_STT_PROVIDER` defaults to `fake`. Set it to `assemblyai` for live
speaker-labeled transcription.

## Tests

```bash
uv run pytest
```

Integration tests use fake providers and do not call external APIs.

