# Kotomka Agent Notes

## Architecture

Kotomka is a local FastAPI service that converts a video URL into a web report and PDF.
The default runtime is local-first for media processing and pluggable for AI providers.

Main flow:

1. `POST /api/jobs` or the `/` form creates a SQLite-backed job.
2. `JobWorker` downloads or copies media through `SourceProvider`; `ffmpeg` extracts
   mono 16 kHz FLAC audio, and yt-dlp metadata (description, tags, upload date,
   language, chapters) is carried into `VideoMetadata`.
3. `SttProvider` returns a normalized speaker-labeled `Transcript`; the raw provider
   payload is saved as `transcript_raw.json`.
4. Candidate frames come from plateau detection, scene detection, and gap filling
   (see Frame Selection below); with the `ocr` extra they are OCR-annotated and
   bullet-build duplicates collapse.
5. `LlmProvider` scores frame batches against time-windowed transcript excerpts;
   selection guarantees chapter coverage; the winners are optionally re-captioned
   at high image detail.
6. `LlmProvider` builds the `Report`. Long transcripts (over
   `KOTOMKA_REPORT_SINGLE_PASS_MAX_CHARS`) are map-reduced: chapter-aligned chunks
   are distilled into structured notes (saved as `notes.json`), then synthesized
   together with the selected frame images. The report pass can map diarization
   labels to real speaker names, applied to the report's embedded transcript copy
   (`transcript.json` keeps raw labels and word-level data; the report copy drops words).
7. `normalize_report` deterministically snaps citation timestamps to transcript
   segments, clamps out-of-range values, and drops unknown frame references.
8. An assessment pass (`KOTOMKA_ASSESSMENT_ENABLED`, default on) critiques the
   finished report: originality, freshness anchored to the upload date (with
   stale-claim flags), audience, actionability, insight density, and a verdict.
   `KOTOMKA_ASSESSMENT_WEB_SEARCH=1` adds the OpenAI web_search tool to this call;
   the Codex transport has no tools support and silently ignores the flag.
   Assessment failures never fail the job.
9. FastAPI renders status, report, filtered job list, assets, retry/reprocess/delete, read-state, and PDF endpoints.

Important modules:

- `src/kotomka/app.py`: FastAPI routes, templates, citation-link rendering, job actions.
- `src/kotomka/worker.py`: end-to-end pipeline orchestration and frame selection policy.
- `src/kotomka/storage.py`: SQLite job records plus artifact lifecycle.
- `src/kotomka/source.py`: YouTube/yt-dlp and local file source providers.
- `src/kotomka/media.py`: ffmpeg audio/frame extraction and perceptual-hash dedupe.
- `src/kotomka/models.py`: normalized schemas for jobs, transcript, frames, and reports.
  `VideoMetadata` carries yt-dlp metadata (description, tags, upload date, language, channel, chapters) used by STT keyterms, report grounding, and frame selection.
- `src/kotomka/providers/stt/`: `fake`, `assemblyai`, and optional `whisper` STT providers.
- `src/kotomka/providers/llm/`: `fake`, OpenAI Platform, and Codex subscription providers.
  Shared prompt/schema orchestration lives in `json_base.py` (`JsonLlmProviderBase`);
  OpenAI and Codex implement only the `_request_json` transport (OpenAI: strict
  json_schema + tools; Codex: schema-as-prompt-hint, streaming, no tools).
- `src/kotomka/transcripts.py`: compact transcript formatting and time-windowed excerpts for LLM input.
- `src/kotomka/pdf.py`: PDF export; ReportLab is the default renderer.

## Data And Artifacts

Generated data is intentionally local and ignored by git:

- SQLite DB: `data/app.db`
- Per-job artifacts: `data/jobs/{job_id}/`
- Downloaded media: `media/source.*`, `media/audio.flac`, `media/source.info.json`
  (jobs processed before the FLAC switch may still contain a legacy `media/audio.mp3`)
- Frame candidates: `frames/*.png`
- Structured artifacts: `transcript.json`, `transcript_raw.json`, `frames.json`, `selected_frames.json`, `notes.json` (map-reduce runs only), `report.json`, `report.pdf`

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
- `whisper`: offline faster-whisper transcription, available only when the
  `whisper` extra is installed (`uv sync --extra whisper`; first run downloads
  model weights, `KOTOMKA_WHISPER_MODEL`, default `large-v3`). No diarization:
  all segments are `Speaker 1`. Useful as an A/B baseline for languages outside
  AssemblyAI's best-model coverage (e.g. Russian).

LLM providers:

- `fake`: offline report and frame scoring.
- `openai`: OpenAI Platform Responses API, requires `OPENAI_API_KEY`.
- `codex_subscription`: ChatGPT/Codex OAuth route; run `uv run kotomka codex-login`.

Provider defaults are configured through `.env.local` and `KOTOMKA_*` settings.
Do not print secret values in logs, tests, or terminal output.

YouTube downloads can use either `JobCreate.cookies_from_browser` (yt-dlp
`--cookies-from-browser`, e.g. `firefox` or `chrome:Profile 1`) or
`JobCreate.cookies_file` (yt-dlp `--cookies` with a Netscape-format
`cookies.txt` export). The web form exposes browser cookies as a select with
`firefox` selected by default; API callers may still pass yt-dlp profile syntax.
Use only one cookie source per job. If yt-dlp reports rotated or invalid YouTube
account cookies, the job should fail with a recovery hint telling the user to
refresh the browser login or export a fresh cookies file.

## Frame Selection

Candidate frames come from three sources, merged and deduplicated:

1. Plateau detection (slide-aware): grayscale thumbnails are sampled at 1 fps and
   hashed; stable runs of at least `KOTOMKA_FRAME_PLATEAU_MIN_DWELL_SECONDS` (hash
   distance ≤ `KOTOMKA_FRAME_PLATEAU_HASH_DISTANCE`) yield one full-resolution frame
   near the run's end, after slide builds/animations have finished. Dwell time is
   recorded on the candidate (`dwell_s`).
2. ffmpeg scene detection (`select=gt(scene\,0.35),showinfo`) for camera cuts.
3. Gap filling: any stretch longer than `KOTOMKA_FRAME_MAX_GAP_SECONDS` without a
   candidate is filled at `KOTOMKA_FRAME_INTERVAL_SECONDS` strides, so the whole
   timeline always has coverage (this replaces the old "<3 scene frames" fallback).

An optional blur gate (`KOTOMKA_FRAME_BLUR_THRESHOLD`, 0 = disabled) drops
transition-blurred plateau/scene candidates before LLM scoring. Perceptual-hash
dedupe runs in source-priority order (plateau, then scene, then periodic), so the
post-animation plateau frame wins over a mid-transition scene duplicate.

When the `ocr` extra is installed (ocrmac, macOS Apple Vision) and
`KOTOMKA_FRAME_OCR_ENABLED` is on (default), candidates are OCR-annotated after
extraction: bullet-build predecessors whose text is contained in a time-adjacent
later slide are dropped (`src/kotomka/ocr.py`), and the recognized text is passed
to the frame-scoring prompt and into `FrameSelection.ocr_text` for the report.
Without ocrmac the step is a silent no-op.

LLM frame scoring is batched across the full timeline, with each batch scored
against the transcript window covering its time range. Frame labels carry dwell
time and OCR text as scoring evidence.

- `KOTOMKA_MAX_FRAMES_FOR_LLM`: batch size for one scoring request.
- `KOTOMKA_MAX_SELECTED_FRAMES`: final selected frame limit.
- `KOTOMKA_SELECTED_FRAME_MIN_GAP_SECONDS`: preferred time gap between selected frames.

Selection guarantees at least one frame per video chapter when a scored candidate
exists in that chapter (best-scored chapter picks are reserved first, then the
usual greedy score/gap selection fills the rest). The final selected frames are
returned in chronological order. If LLM scoring returns nothing, fallback
selection samples frames evenly across the timeline.

After selection, an optional re-caption pass (`KOTOMKA_RECAPTION_SELECTED_FRAMES`,
default on) sends only the winners at high image detail to refresh captions and
`ocr_text`; failures fall back to the scoring-pass captions.

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

The launchd user agent (`launchd/dev.kotomka.plist`, label `dev.kotomka`) keeps
the server running in the background; its logs go to `data/launchd.out.log` and
`data/launchd.err.log`. Manage it with:

```bash
make launchd-install    # copy the plist to ~/Library/LaunchAgents and (re)load it
make launchd-restart    # restart the running service (launchctl kickstart -k)
make launchd-status     # launchctl print for the service
make launchd-uninstall  # unload the service and remove the installed plist
```

`launchd-install` is also the update path: after editing the repo plist, run it
to reinstall and restart the service. Note it restarts the server, which kills
any in-flight job.

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
