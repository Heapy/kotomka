# Kotomka Agent Notes

## Architecture

Kotomka is a local FastAPI service that converts a video URL into a web report and PDF.
The default runtime is local-first for media processing and pluggable for AI providers.

Main flow:

1. `POST /api/jobs` or the `/` form creates a SQLite-backed job.
2. `JobWorker` runs a pool of `KOTOMKA_WORKER_POOL_SIZE` threads (default 3)
   pulling jobs off a shared queue, so multiple jobs' pipelines run
   concurrently. Each job downloads or copies media through `SourceProvider`;
   this step is serialized across the whole pool (one download at a time) to
   avoid tripping YouTube's bot detection and saturating bandwidth, while
   transcription, frame extraction, and LLM calls for other jobs proceed
   unblocked. `SourceProvider.fetch` returns the video, metadata, and intended
   audio path without transcoding. After releasing the download permit, the worker
   checks the duration limit before `ffmpeg` extracts mono 16 kHz FLAC audio. yt-dlp metadata
   (description, tags, upload date, language, chapters) is carried into
   `VideoMetadata`.
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
   Individual notes-request failures are tolerated, but if all fail, the job
   fails before final synthesis instead of generating an ungrounded report.
   Gaps in chapter metadata extend the preceding chapter's context window,
   so intervening speech stays in chronological order rather than moving to the
   final chunk.
7. `normalize_report` deterministically snaps citation timestamps to transcript
   segments, clamps out-of-range values, and drops unknown frame references.
   Video duration defines the bounds (transcript duration is only a fallback),
   preserving silent visual content after the last speech segment.
   Inline citations use explicit `[t=123.4]` markup outside code. Plain numeric
   brackets are never inferred as timestamps or rewritten, including in old
   reports. Structured section citations remain clickable for legacy reports.
8. An assessment pass (`KOTOMKA_ASSESSMENT_ENABLED`, default on) critiques the
   finished report: originality, freshness anchored to the upload date (with
   stale-claim flags), audience, actionability, insight density, and a verdict.
   `KOTOMKA_ASSESSMENT_WEB_SEARCH=1` adds the OpenAI web_search tool to this call;
   the report is marked web-checked only when the response contains a completed
   web search call, not merely when the tool was offered.
   The Codex transport has no tools support and silently ignores the flag.
   Assessment failures never fail the job.
9. FastAPI renders status, report, filtered job list, assets, retry/reprocess/delete, read-state, and PDF endpoints.
   A polled transition to failed reloads the status page once to expose recovery
   actions; an already-failed page stops polling without a reload loop.

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
- Structured artifacts: `transcript.json`, `transcript_raw.json`, `frames.json`, `selected_frames.json`, `notes.json` (map-reduce runs only), `report.json`, `report-<sha256>.pdf` (legacy `report.pdf` may still exist)

After a report is saved and the job completes, unused PNG candidates are removed;
every image referenced by `report.frames` is retained, including images not used
in a section. `frames.json` remains a diagnostic manifest and can reference pruned
candidates. Cleanup validates all retained images before deleting anything and
holds a SQLite write transaction to exclude concurrent retries. Invalid/incomplete
reports are skipped. Existing completed jobs can be cleaned with
`uv run kotomka cleanup-frames` (`--dry-run` previews counts and bytes).
Cleanup failures, including a busy database, are logged without changing job
status after completion; they cannot fail a job concurrently queued for retry.

Each extraction attempt uses a fresh temporary directory and publishes surviving
frames with unique filename prefixes. Retries cannot reuse stale scene suffixes
or overwrite images of the previous report. Scene files require matching ffmpeg
timestamps. Downloads also run in a fresh staging directory and require exactly
one resulting video, so an older, larger media file cannot become the new source.

Deleting a terminal job removes both its SQLite record and `data/jobs/{job_id}`.
Deletion is conditional on terminal status in the same SQL statement, so a
concurrent retry cannot have its queued job or artifacts removed. Workers skip
queue entries whose records no longer exist; unexpected job exceptions are
logged without terminating the worker thread.
Job updates write only supplied fields in one `UPDATE ... RETURNING` statement;
omitted error/result values are preserved, while explicit `None` clears them.
This avoids stale read/modify/write snapshots overwriting unrelated updates.

Jobs also have an `is_read` state stored in SQLite. New and retried jobs are
unread by default. `/jobs` hides read jobs unless `show_read=1` is present, and
`POST /jobs/{job_id}/read` toggles the state from the list or report page.
The job list reads titles from the small `source.json` metadata first, falling
back to `report.json` and then the URL for older or incomplete artifacts.

## Providers

STT providers:

- `fake`: offline test transcript.
- `assemblyai`: live speaker-labeled transcription, requires `ASSEMBLYAI_API_KEY`.
  Requests `speech_models: ["universal-3-5-pro", "universal-3-pro", "universal-2"]`, entity detection,
  an explicit `language_code` when video metadata carries one (else language
  detection), keyterms extracted from title/chapters/tags/description
  (`KOTOMKA_STT_KEYTERMS_MAX`, default 200), and `speakers_expected` when the
  job provides it. A 400 naming an optional parameter triggers one retry with a
  minimal request body. The raw completed payload is saved to
  `transcript_raw.json`.
  Polling has a total deadline (`KOTOMKA_ASSEMBLYAI_MAX_POLL_SECONDS`, default
  14400 = 4 hours); unknown statuses fail immediately. Poll request timeouts and
  sleep intervals are capped by the remaining budget.
- `whisper`: offline faster-whisper transcription, available only when the
  `whisper` extra is installed (`uv sync --extra whisper`; first run downloads
  model weights, `KOTOMKA_WHISPER_MODEL`, default `large-v3`). No diarization:
  all segments are `Speaker 1`. Useful as an A/B baseline for languages outside
  AssemblyAI's best-model coverage (e.g. Russian).
  One model configuration is cached per process across jobs; local inference is
  serialized through completion of the lazy segment iterator to bound resource use.

LLM providers:

- `fake`: offline report and frame scoring.
- `openai`: OpenAI Platform Responses API, requires `OPENAI_API_KEY`.
- `codex_subscription`: ChatGPT/Codex OAuth route; run `uv run kotomka codex-login`.
  Defaults to `gpt-6-astra` for reports, assessment, and frame scoring;
  `KOTOMKA_CODEX_SCORING_MODEL` can override the scoring model.
  Credential refresh is serialized across worker threads, with expiry rechecked
  under the lock before every request. A 401 triggers one retry with refreshed
  credentials; a token already rotated by another request is reused. Client
  credentials and account headers are local to each request. Auth files are
  atomically replaced using unique private temp
  files; separate server processes must not share the same auth store.

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

One ffmpeg analysis pass splits the decoded video into scene detection and a
stream of 1 fps grayscale thumbnails. Thumbnails are hashed in memory and never
written as PNGs. Candidate timestamps are budgeted before a second sequential
ffmpeg pass extracts full-resolution images; no per-candidate seeks are needed.
Several targets mapping to one VFR frame share that image and use its actual PTS.
An enabled blur gate can reserve one extra emergency image, used only if all
normal picks fail the gate; the scoring budget remains unchanged.

1. Plateau detection (slide-aware): grayscale thumbnails are sampled at 1 fps and
   streamed at up to 640 pixels per side; stable runs of at least
   `KOTOMKA_FRAME_PLATEAU_MIN_DWELL_SECONDS` require both hash distance
   ≤ `KOTOMKA_FRAME_PLATEAU_HASH_DISTANCE` and a maximum grayscale pixel difference
   ≤ 12 from the run's anchor. This catches small text changes and accumulated
   drift that perceptual hashes alone miss. Runs yield one full-resolution frame
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
Perceptual hashes only shortlist duplicates: deletion also requires pixelwise
agreement (at most 8 levels per channel). Bounded codec ringing can pass a local
1.5-pixel blur check; differences above 128 levels always survive. This preserves
changed slide text, numbers, and builds while collapsing repeated slides after
video compression, before OCR and scoring inspect the surviving frames.
Full-resolution decoding is reused through a 64 MiB LRU cache per extraction.
Compact 256-pixel previews of the locally blurred images reject most hash
collisions before full-image comparison; previews alone never authorize deletion.
The extraction candidate budget also bounds the number of pair comparisons.

When the `ocr` extra is installed (ocrmac, macOS Apple Vision) and
`KOTOMKA_FRAME_OCR_ENABLED` is on (default), candidates are OCR-annotated after
extraction: bullet-build predecessors whose text is contained in a time-adjacent
later slide are dropped only when every OCR token (including numbers, negations,
and repeated tokens) survives and the existing visual foreground is unchanged
on a flat background (`src/kotomka/ocr.py`). Charts or uncertain builds are kept.
The recognized text is passed
to the frame-scoring prompt and into `FrameSelection.ocr_text` for the report.
Without ocrmac the step is a silent no-op.

LLM frame scoring is batched across the full timeline. Scoring and re-captioning
share their fixed transcript budget across the frames, prioritizing nearby speech
for each timestamp and deduplicating overlapping excerpts. Oversized segments are
explicitly truncated rather than excluding later frames. Frame labels carry dwell
time and OCR text as scoring evidence.

Independent scoring batches and transcript-notes chunks share one process-wide
pool of four threads. Results are merged in input order (and scores by frame ID).
Both live transports also share a four-request semaphore, including synchronous
report/recaption/assessment calls, so concurrent jobs cannot multiply the HTTP
limit. A failed stage cancels pending work and joins in-flight calls before exit;
notes retain the existing partial-failure policy and fail if every chunk fails.

- `KOTOMKA_MAX_FRAMES_FOR_LLM`: batch size for one scoring request.
- `KOTOMKA_MAX_CANDIDATE_FRAMES`: candidate budget before full-resolution extraction
  and again after OCR, before scoring (default 150).
  Chapter representatives are reserved, then time buckets prefer plateau over scene
  over periodic candidates, with longer dwell winning within a source. If chapter
  count exceeds the budget, chapter picks are also spread across time.
- `KOTOMKA_MAX_SELECTED_FRAMES`: final selected frame limit.
- `KOTOMKA_SELECTED_FRAME_MIN_GAP_SECONDS`: preferred time gap between selected frames.

Selection guarantees at least one frame per video chapter when a scored candidate
exists in that chapter (best-scored chapter picks are reserved first, then the
usual greedy score/gap selection fills the rest). The final selected frames are
returned in chronological order. If LLM scoring returns nothing, fallback
selection samples frames evenly across the timeline.
Fallback sampling uses actual timestamps: reserve the ends, then fill the largest
uncovered time gaps. Dense early cuts cannot crowd out sparse late candidates.

After selection, an optional re-caption pass (`KOTOMKA_RECAPTION_SELECTED_FRAMES`,
default on) sends only the winners at high image detail to refresh captions and
`ocr_text`; failures fall back to the scoring-pass captions.

The final report's image budget (`KOTOMKA_REPORT_MAX_IMAGES`, default 16) also
spans the timeline instead of truncating to the earliest frames. Available
chapter representatives are reserved by score, remaining slots fill time gaps,
and missing files do not consume the budget. All selected captions/OCR remain
in the report's text context even when an image is outside this visual budget.

## PDF

ReportLab is the default PDF renderer because launching system Chrome from Codex/macOS sandbox can crash Chrome.
Set `KOTOMKA_PDF_RENDERER=browser` only when intentionally running outside the sandbox and accepting that risk.

Frame images fit both the available width and page height, including portrait
video. A PDF is published atomically only after full rendering succeeds; rendering
errors preserve the previous cache and propagate instead of returning a truncated
placeholder document.

The ordinary PDF button uses the cache. PDF cache is regenerated when missing,
smaller than 4 KB, older than `report.json`, or explicitly requested with `?force=1`.
The cache filename includes the hash of the exact report snapshot being rendered;
an older concurrent export cannot overwrite a newer generation's PDF. Publication
rechecks the report and job state, retrying once if the report changed. Report JSON
is also atomically replaced, so concurrent readers cannot see half-written JSON.

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
