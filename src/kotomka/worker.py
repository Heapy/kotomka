from __future__ import annotations

import traceback
from queue import Empty, Queue
from threading import Event, Thread

from .config import Settings
from .media import extract_candidate_frames
from .models import CandidateFrame, FrameSelection, Transcript
from .providers.llm.base import LlmProvider
from .providers.llm import get_llm_provider
from .providers.stt import get_stt_provider
from .reporting import normalize_report, save_report
from .source import SourceProvider, YtDlpSourceProvider
from .storage import JobStore
from .utils import write_json


class JobWorker:
    def __init__(self, *, store: JobStore, settings: Settings, source_provider: SourceProvider | None = None) -> None:
        self.store = store
        self.settings = settings
        self.source_provider = source_provider or YtDlpSourceProvider()
        self._queue: Queue[str] = Queue()
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = Thread(target=self._run, name="kotomka-worker", daemon=True)
        self._thread.start()
        for job_id in self.store.list_requeueable_jobs():
            self.enqueue(job_id)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self.process(job_id)
            finally:
                self._queue.task_done()

    def process(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        try:
            self.store.update_job(job_id, status="running", progress=5, message="Downloading video")
            source = self.source_provider.fetch(job.input, job.artifact_dir)
            if source.metadata.duration_s > self.settings.max_video_duration_seconds:
                raise RuntimeError("Video is longer than the configured 2 hour MVP limit")
            write_json(job.artifact_dir / "source.json", source.model_dump())

            self.store.update_job(job_id, progress=25, message="Transcribing audio")
            stt = get_stt_provider(job.input.stt_provider)
            transcript = stt.transcribe(
                source.audio_path,
                source.metadata,
                speakers_expected=job.input.speakers_expected,
                raw_path=job.artifact_dir / "transcript_raw.json",
            )
            write_json(job.artifact_dir / "transcript.json", transcript.model_dump())

            self.store.update_job(job_id, progress=50, message="Extracting and deduplicating frames")
            frames = extract_candidate_frames(
                source.video_path,
                job.artifact_dir / "frames",
                duration_s=source.metadata.duration_s,
                interval_seconds=self.settings.frame_interval_seconds,
            )
            write_json(job.artifact_dir / "frames.json", [frame.model_dump() for frame in frames])

            self.store.update_job(job_id, progress=65, message="Scoring useful frames")
            llm = get_llm_provider(job.input.llm_provider or self.settings.llm_provider)
            selected_frames = _score_frames_across_timeline(
                llm,
                frames,
                transcript,
                batch_size=self.settings.max_frames_for_llm,
                max_selected=self.settings.max_selected_frames,
                min_gap_seconds=self.settings.selected_frame_min_gap_seconds,
            )
            if not selected_frames and frames:
                selected_frames = _fallback_frame_selection(frames, max_selected=min(6, self.settings.max_selected_frames))
            write_json(job.artifact_dir / "selected_frames.json", [frame.model_dump() for frame in selected_frames])

            self.store.update_job(job_id, progress=80, message="Building report")
            report = llm.build_report(
                source=source,
                transcript=transcript,
                frames=selected_frames,
                output_language=job.input.output_language,
                work_dir=job.artifact_dir,
            )
            report = normalize_report(report, tolerance_s=self.settings.citation_snap_tolerance_seconds)
            if self.settings.assessment_enabled:
                self.store.update_job(job_id, progress=90, message="Assessing report")
                try:
                    assessment = llm.assess_report(
                        report=report,
                        metadata=source.metadata,
                        output_language=job.input.output_language,
                    )
                except Exception:
                    # The report is complete and useful without an assessment.
                    traceback.print_exc()
                    assessment = None
                if assessment is not None:
                    report = report.model_copy(update={"assessment": assessment})
            report_path = job.artifact_dir / "report.json"
            save_report(report, report_path)
            self.store.update_job(
                job_id,
                status="completed",
                progress=100,
                message="Completed",
                error=None,
                result={"report_path": str(report_path)},
            )
        except Exception as exc:
            traceback.print_exc()
            self.store.update_job(job_id, status="failed", progress=100, message="Failed", error=str(exc))


def _score_frames_across_timeline(
    llm: LlmProvider,
    frames: list[CandidateFrame],
    transcript: Transcript,
    *,
    batch_size: int,
    max_selected: int,
    min_gap_seconds: int,
) -> list[FrameSelection]:
    if not frames or max_selected <= 0:
        return []
    batch_size = max(1, int(batch_size))
    by_id = {frame.frame_id: frame for frame in frames}
    scored: dict[str, FrameSelection] = {}
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        for selection in llm.score_frames(batch, transcript):
            frame = by_id.get(selection.frame_id)
            if frame is None:
                continue
            normalized = selection.model_copy(
                update={
                    "timestamp_s": frame.timestamp_s,
                    "image_path": frame.path.name,
                }
            )
            prior = scored.get(normalized.frame_id)
            if prior is None or normalized.score > prior.score:
                scored[normalized.frame_id] = normalized
    return _select_diverse_frames(
        list(scored.values()),
        max_selected=max_selected,
        min_gap_seconds=max(0, int(min_gap_seconds)),
    )


def _select_diverse_frames(
    selections: list[FrameSelection],
    *,
    max_selected: int,
    min_gap_seconds: int,
) -> list[FrameSelection]:
    if max_selected <= 0:
        return []
    selected: list[FrameSelection] = []
    for candidate in sorted(selections, key=lambda item: (-item.score, item.timestamp_s)):
        if len(selected) >= max_selected:
            break
        if min_gap_seconds and any(abs(candidate.timestamp_s - item.timestamp_s) < min_gap_seconds for item in selected):
            continue
        selected.append(candidate)
    if len(selected) < max_selected:
        selected_ids = {item.frame_id for item in selected}
        for candidate in sorted(selections, key=lambda item: (-item.score, item.timestamp_s)):
            if len(selected) >= max_selected:
                break
            if candidate.frame_id not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate.frame_id)
    return sorted(selected, key=lambda item: item.timestamp_s)


def _fallback_frame_selection(frames: list[CandidateFrame], *, max_selected: int) -> list[FrameSelection]:
    if max_selected <= 0:
        return []
    timeline_frames = _pick_evenly_spaced_frames(frames, max_selected=max_selected)
    return [
        FrameSelection(
            frame_id=frame.frame_id,
            timestamp_s=frame.timestamp_s,
            image_path=frame.path.name,
            content_type="representative",
            score=0.5,
            caption="Representative deduplicated frame",
            reason="Fallback selection used because the LLM provider returned no useful frames.",
        )
        for frame in timeline_frames
    ]


def _pick_evenly_spaced_frames(frames: list[CandidateFrame], *, max_selected: int) -> list[CandidateFrame]:
    if len(frames) <= max_selected:
        return frames
    if max_selected <= 1:
        return [frames[0]]
    last_index = len(frames) - 1
    indexes = {round(index * last_index / (max_selected - 1)) for index in range(max_selected)}
    return [frames[index] for index in sorted(indexes)]
