from kotomka.models import Chapter, Transcript, TranscriptSegment
from kotomka.transcripts import chunk_transcript, format_segment_line, format_transcript, frame_excerpts, window_excerpt


def make_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=300,
        speakers=["Speaker A"],
        segments=[
            TranscriptSegment(start_s=0, end_s=10, speaker="Speaker A", text="intro words", confidence=0.9),
            TranscriptSegment(start_s=100, end_s=110, speaker="Speaker A", text="middle words", confidence=0.3),
            TranscriptSegment(start_s=200, end_s=210, speaker="Speaker A", text="late words"),
        ],
    )


def test_format_segment_line_flags_low_confidence() -> None:
    segment = TranscriptSegment(start_s=1.25, end_s=4.5, speaker="Speaker B", text="hello", confidence=0.2)
    assert format_segment_line(segment) == "[1.2-4.5] Speaker B: hello"
    assert format_segment_line(segment, low_confidence_below=0.5) == "[1.2-4.5] Speaker B [low-confidence]: hello"
    no_confidence = segment.model_copy(update={"confidence": None})
    assert "[low-confidence]" not in format_segment_line(no_confidence, low_confidence_below=0.5)


def test_format_transcript_joins_and_truncates_whole_lines() -> None:
    transcript = make_transcript()
    full = format_transcript(transcript)
    assert full.splitlines() == [
        "[0.0-10.0] Speaker A: intro words",
        "[100.0-110.0] Speaker A: middle words",
        "[200.0-210.0] Speaker A: late words",
    ]
    first_line = full.splitlines()[0]
    truncated = format_transcript(transcript, max_chars=len(first_line) + 5)
    assert truncated == first_line


def test_window_excerpt_selects_overlapping_segments_with_margin() -> None:
    transcript = make_transcript()
    excerpt = window_excerpt(transcript, start_s=100, end_s=110, margin_s=30)
    assert "middle words" in excerpt
    assert "intro words" not in excerpt
    assert "late words" not in excerpt
    wide = window_excerpt(transcript, start_s=100, end_s=110, margin_s=95)
    assert "intro words" in wide
    assert "late words" in wide


def test_frame_excerpts_deduplicate_shared_context_and_keep_confidence() -> None:
    excerpt = frame_excerpts(make_transcript(), [100, 101, 102, 101], low_confidence_below=0.5)
    assert excerpt.count("middle words") == 1
    assert "[low-confidence]" in excerpt
    assert "intro words" not in excerpt
    assert "late words" not in excerpt
    assert frame_excerpts(make_transcript(), []) == ""
    assert frame_excerpts(make_transcript(), [100], max_chars=0) == ""


def test_frame_excerpts_keep_a_bounded_piece_of_an_oversized_segment() -> None:
    transcript = Transcript(segments=[TranscriptSegment(start_s=0, end_s=60, text="relevant words " * 1000)])
    excerpt = frame_excerpts(transcript, [30], max_chars=100)
    assert "relevant words" in excerpt
    assert excerpt.endswith("…")
    assert len(excerpt) == 100


def segments_every(step_s: float, *, duration_s: float) -> list[TranscriptSegment]:
    segments = []
    start = 0.0
    while start < duration_s:
        end = min(start + step_s, duration_s)
        segments.append(TranscriptSegment(start_s=start, end_s=end, speaker="Speaker A", text=f"at {start:.0f}"))
        start = end
    return segments


def test_chunk_transcript_without_chapters_uses_fixed_windows() -> None:
    transcript = Transcript(duration_s=1500, segments=segments_every(50, duration_s=1500))
    chunks = chunk_transcript(transcript, [], target_seconds=600, duration_s=1500)
    assert [(chunk.start_s, chunk.end_s) for chunk in chunks] == [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1500.0)]
    assert sum(len(chunk.segments) for chunk in chunks) == len(transcript.segments)
    assert chunks[0].segments[0].text == "at 0"
    assert chunks[-1].segments[-1].text == "at 1450"


def test_chunk_transcript_merges_short_chapters() -> None:
    transcript = Transcript(duration_s=700, segments=segments_every(50, duration_s=700))
    chapters = [
        Chapter(title="Hello", start_s=0, end_s=30),
        Chapter(title="Main", start_s=30, end_s=700),
    ]
    chunks = chunk_transcript(transcript, chapters, target_seconds=600, min_seconds=120, duration_s=700)
    assert len(chunks) == 1
    assert chunks[0].title == "Hello"
    assert chunks[0].start_s == 0.0
    assert chunks[0].end_s == 700.0


def test_chunk_transcript_splits_long_chapters() -> None:
    transcript = Transcript(duration_s=1500, segments=segments_every(50, duration_s=1500))
    chapters = [Chapter(title="One big talk", start_s=0, end_s=1500)]
    chunks = chunk_transcript(transcript, chapters, target_seconds=600, duration_s=1500)
    assert len(chunks) == 3
    assert chunks[0].title == "One big talk"
    assert chunks[1].title == "One big talk (cont.)"
    assert sum(len(chunk.segments) for chunk in chunks) == len(transcript.segments)


def test_chunk_transcript_assigns_tail_segments_to_last_chunk() -> None:
    segments = segments_every(50, duration_s=650)
    transcript = Transcript(duration_s=600, segments=segments)
    chunks = chunk_transcript(transcript, [], target_seconds=300, duration_s=600)
    assert chunks[-1].segments[-1].text == "at 600"
    assert sum(len(chunk.segments) for chunk in chunks) == len(segments)


def test_chapter_gaps_stay_with_the_preceding_chapter() -> None:
    chapters = [
        Chapter(title="A", start_s=0, end_s=200),
        Chapter(title="B", start_s=300, end_s=500),
        Chapter(title="C", start_s=600, end_s=900),
    ]
    transcript = Transcript(duration_s=900, segments=[
        TranscriptSegment(start_s=ts, end_s=ts + 5, text=f"at {ts}") for ts in [50, 250, 350, 550, 650]
    ])
    chunks = chunk_transcript(transcript, chapters)
    assert [(chunk.title, [segment.start_s for segment in chunk.segments]) for chunk in chunks] == [
        ("A", [50, 250]), ("B", [350, 550]), ("C", [650]),
    ]
    assert sum(len(chunk.segments) for chunk in chunks) == len(transcript.segments)
    assert all(chunk.start_s <= segment.start_s < chunk.end_s for chunk in chunks for segment in chunk.segments)


def test_binary_lookup_preserves_first_window_precedence_for_overlapping_chapters():
    transcript = Transcript(duration_s=1000, segments=[
        TranscriptSegment(start_s=ts, end_s=ts + 1, text=str(ts)) for ts in [50, 150, 600, 700, 1100]
    ])
    chapters = [Chapter(title="A", start_s=0, end_s=600), Chapter(title="B", start_s=100, end_s=1000)]
    chunks = chunk_transcript(transcript, chapters, target_seconds=100, min_seconds=0)
    assert [(chunk.start_s, [segment.start_s for segment in chunk.segments]) for chunk in chunks] == [
        (0, [50]), (100, [150]), (600, [600]), (700, [700]), (900, [1100]),
    ]


def test_window_lookup_uses_logarithmic_number_of_boundary_reads():
    from kotomka.transcripts import _window_index
    class CountingBounds(list):
        reads = 0
        def __getitem__(self, index):
            self.reads += 1
            return super().__getitem__(index)
    boundaries = CountingBounds(range(1, 65537))
    assert _window_index(boundaries, 32768) == 32768
    assert boundaries.reads <= 17
