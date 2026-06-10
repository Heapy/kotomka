from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    return payload


FRAME_SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "frame_id": {"type": "string"},
                    "score": {"type": "number"},
                    "content_type": {"type": "string"},
                    "caption": {"type": "string"},
                    "reason": {"type": "string"},
                    "ocr_text": {"type": ["string", "null"]},
                },
                "required": ["frame_id", "score", "content_type", "caption", "reason", "ocr_text"],
            },
        }
    },
    "required": ["frames"],
}


REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                    "body": {"type": "string"},
                    "frame_ids": {"type": "array", "items": {"type": "string"}},
                    "citations": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["title", "start_s", "end_s", "body", "frame_ids", "citations"],
            },
        },
    },
    "required": ["summary", "sections"],
}

