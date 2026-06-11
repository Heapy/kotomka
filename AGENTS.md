# Kotomka Agent Notes

## Architecture

Kotomka is a local FastAPI service that converts a video URL into a web report and PDF.
The default runtime is local-first for media processing and pluggable for AI providers.

Main flow:

1. `POST /api/jobs` or the `/` form creates a SQLite-backed job.
2. `JobWorker` downloads or copies media through `SourceProvider`.
3. `ffmpeg` extracts audio and candidate frames.
4. `SttProvider` returns a normalized speaker-labeled `Transcript`.
5. `LlmProvider` scores frames across the full timeline in batches and builds a structured `Report`.
6. `normalize_report` deterministically snaps citation timestamps to transcript segments, clamps out-of-range values, and drops unknown frame references before the report is saved.
7. FastAPI renders status, report, filtered job list, assets, retry/reprocess/delete, read-state, and PDF endpoints.

Important modules:

- `src/kotomka/app.py`: FastAPI routes, templates, citation-link rendering, job actions.
- `src/kotomka/worker.py`: end-to-end pipeline orchestration and frame selection policy.
- `src/kotomka/storage.py`: SQLite job records plus artifact lifecycle.
- `src/kotomka/source.py`: YouTube/yt-dlp and local file source providers.
- `src/kotomka/media.py`: ffmpeg audio/frame extraction and perceptual-hash dedupe.
- `src/kotomka/models.py`: normalized schemas for jobs, transcript, frames, and reports.
  `VideoMetadata` carries yt-dlp metadata (description, tags, upload date, language, channel, chapters) used by STT keyterms, report grounding, and frame selection.
- `src/kotomka/providers/stt/`: `fake` and `assemblyai` STT providers.
- `src/kotomka/providers/llm/`: `fake`, OpenAI Platform, and Codex subscription providers.
- `src/kotomka/pdf.py`: PDF export; ReportLab is the default renderer.

## Data And Artifacts

Generated data is intentionally local and ignored by git:

- SQLite DB: `data/app.db`
- Per-job artifacts: `data/jobs/{job_id}/`
- Downloaded media: `media/source.*`, `media/audio.flac`, `media/source.info.json`
  (jobs processed before the FLAC switch may still contain a legacy `media/audio.mp3`)
- Frame candidates: `frames/*.png`
- Structured artifacts: `transcript.json`, `transcript_raw.json`, `frames.json`, `selected_frames.json`, `report.json`, `report.pdf`

Deleting a terminal job removes both its SQLite record and `data/jobs/{job_id}`.
Active jobs are not deletable from the UI to avoid racing the worker.

Jobs also have an `is_read` state stored in SQLite. New and retried jobs are
unread by default. `/jobs` hides read jobs unless `show_read=1` is present, and
`POST /jobs/{job_id}/read` toggles the state from the list or report page.

## Providers

STT providers:

- `fake`: offline test transcript.
- `assemblyai`: live speaker-labeled transcription, requires `ASSEMBLYAI_API_KEY`.
  Requests `speech_models: ["universal-3-pro", "universal-2"]`, entity detection,
  an explicit `language_code` when video metadata carries one (else language
  detection), keyterms extracted from title/chapters/tags/description
  (`KOTOMKA_STT_KEYTERMS_MAX`, default 200), and `speakers_expected` when the
  job provides it. A 400 naming an optional parameter triggers one retry with a
  minimal request body. The raw completed payload is saved to
  `transcript_raw.json`.

LLM providers:

- `fake`: offline report and frame scoring.
- `openai`: OpenAI Platform Responses API, requires `OPENAI_API_KEY`.
- `codex_subscription`: ChatGPT/Codex OAuth route; run `uv run kotomka codex-login`.

Provider defaults are configured through `.env.local` and `KOTOMKA_*` settings.
Do not print secret values in logs, tests, or terminal output.

## Frame Selection

Frame extraction first uses ffmpeg scene detection:

```text
select=gt(scene\,0.35),showinfo
```

If too few scene frames are found, periodic extraction is used as a fallback.
Frames are deduplicated with perceptual hash.

LLM frame scoring is batched across the full timeline:

- `KOTOMKA_MAX_FRAMES_FOR_LLM`: batch size for one scoring request.
- `KOTOMKA_MAX_SELECTED_FRAMES`: final selected frame limit.
- `KOTOMKA_SELECTED_FRAME_MIN_GAP_SECONDS`: preferred time gap between selected frames.

The final selected frames are returned in chronological order.
If LLM scoring returns nothing, fallback selection samples frames evenly across the timeline.

## PDF

ReportLab is the default PDF renderer because launching system Chrome from Codex/macOS sandbox can crash Chrome.
Set `KOTOMKA_PDF_RENDERER=browser` only when intentionally running outside the sandbox and accepting that risk.

PDF cache is regenerated when missing, smaller than 4 KB, older than `report.json`, or requested with `?force=1`.

## Commands

A `Makefile` wraps the common tasks (uses uv's default global cache):

```bash
make         # same as `make serve`
make serve   # uv run kotomka serve --port 8000
make sync    # uv sync --extra dev
make test    # uv run pytest
```

Override the port with `make serve PORT=8001`.

Or run the underlying commands directly:

```bash
uv sync --extra dev
uv run pytest
uv run kotomka serve
```

When running in a sandbox where the global uv cache path (under `$HOME`) is not
writable, redirect it to a workspace-local cache:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev
UV_CACHE_DIR=.uv-cache make serve
```

Default URL:

```text
http://127.0.0.1:8000
```

If a server already owns port 8000 and cannot be stopped from sandbox, use another port:

```bash
uv run kotomka serve --port 8001
```

## Maintenance Rule

When changing architecture, provider behavior, artifact layout, job lifecycle, frame-selection policy, PDF rendering strategy, or public routes, update this `AGENTS.md` in the same change.
