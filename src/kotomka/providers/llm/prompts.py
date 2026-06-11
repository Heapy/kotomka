FRAME_SCORE_INSTRUCTIONS = """You score video frames for a knowledge-preserving report.

Keep frames whose content a reader would want to study: slides, diagrams, charts,
tables, code, terminal output, whiteboards, documents, product or app screens, and
concrete visual examples. Reject mood shots, audience shots, speaker close-ups,
transition blur, and frames whose information already appears in a clearly better
frame.

For each frame return:
- score: usefulness for the report from 0.0 to 1.0 (use the full range; anything
  below 0.45 is discarded).
- content_type: one of slide, diagram, chart, code, terminal, whiteboard,
  document, screen, demo, photo, other.
- caption: one factual sentence naming what the frame shows; mention concrete
  titles, numbers, and labels that are visible.
- reason: why this frame earns a place in the report.
- ocr_text: the exact readable text when it is legible, else null. Never guess
  text you cannot actually read.

Use the transcript excerpt to judge how strongly each frame supports what is being
said around its timestamp. Return only the requested JSON."""

NOTES_INSTRUCTIONS = """You extract structured notes from one chunk of a video transcript.

Capture, each with the timestamp where it is said:
- claim: assertions, conclusions, recommendations, opinions
- number: every concrete figure, metric, benchmark, price, version
- definition: terms and concepts the speaker explains
- quote: short verbatim quotes worth preserving exactly
- step: instructions or sequences of actions

Rules: no commentary or interpretation; keep exact numbers, names, versions, and
units as spoken; write note text in the transcript's own language; set importance
from 0.0 to 1.0; treat lines marked [low-confidence] with caution. chunk_summary
is one or two sentences on what this chunk covers. Return only the requested
JSON."""

ASSESSMENT_INSTRUCTIONS = """You critically assess a video based on its report and metadata. Write in the
requested output_language, keeping technical terms untranslated. Be a fair but
skeptical reviewer: the reader uses this assessment to decide whether the content
deserves their trust and time.

- originality / originality_score (0 = textbook rehash, 1 = genuinely novel):
  which parts are the author's own experience, data, or argument versus restated
  common knowledge? Name the specific claims that drive the score.
- freshness / freshness_score (0 = clearly outdated, 1 = current): anchor on the
  video's upload_date and today's date. List version-sensitive or time-sensitive
  claims in stale_claims, each with its timestamp when known, the risk of acting
  on it today, and your confidence from 0.0 to 1.0. If web search is available,
  verify the most consequential claims; otherwise reason from your own knowledge
  and say so explicitly.
- audience / prerequisites: who actually benefits, and what they must already
  know.
- actionability: can a reader apply this directly? What is missing before they
  could?
- insight_density: is the runtime justified by the substance, given the video
  duration?
- verdict: one or two sentences - does reading the report replace watching, and
  is the content worth trusting?

Return only the requested JSON."""

REPORT_INSTRUCTIONS = """You write a knowledge-preserving report on a video so a reader can skip
watching it. Write in the requested output_language, but keep established
technical terms, product names, and code in their original language instead of
translating them.

summary: open with one thesis sentence saying what the video is about and what it
concludes, then 3-7 takeaway sentences carrying the most load-bearing facts.
Never write filler like "this video discusses".

sections: follow the video's actual topic shifts (chapters are hints, not a
template), each section covering roughly 2-6 minutes. For every section provide:
- title: specific and informative ("Sharding by user_id", not "Main part").
- start_s / end_s: the time range the section covers, in seconds.
- body: the substance. Preserve exact numbers, names, versions, units, and
  comparisons. Render code, commands, and configuration (from frame OCR or
  speech) as fenced code blocks. Quote memorable phrasing sparingly. Where the
  transcript is marked [low-confidence], hedge explicitly instead of asserting.
- citations: timestamps in seconds for the important claims; also embed each
  citation inline in the body as [123.4] right after the claim it supports.
- frame_ids: the provided frames whose content belongs to this section.

speaker_names: map diarization labels such as "Speaker A" to real names whenever
the transcript or metadata reveals them (introductions, "my name is", channel
name). Use the real names inside section bodies as well. Leave the array empty
when unsure.

Return only the requested JSON."""
