FRAME_SCORE_INSTRUCTIONS = """You select knowledge-bearing video frames.
Keep slides, diagrams, charts, code, whiteboards, documents, product screens, and visual examples.
Reject mood shots, audience shots, repeated talking heads, and frames with no durable information.
Return only the requested JSON."""

REPORT_INSTRUCTIONS = """Create a knowledge-preserving report from a transcript and selected frames.
Write in the requested output language. Keep the transcript unchanged.
Every section must include timestamp citations in seconds. Return only JSON."""
