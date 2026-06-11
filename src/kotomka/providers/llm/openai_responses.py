from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from ...config import get_settings
from .json_base import ImageInput, JsonLlmProviderBase, image_data_url
from .json_helpers import parse_json_object


class OpenAiResponsesProvider(JsonLlmProviderBase):
    name = "openai"

    def __init__(self, *, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for llm_provider=openai")
        self.client = OpenAI(api_key=api_key)
        settings = get_settings()
        self.model = model or settings.openai_model
        self.scoring_model = settings.openai_scoring_model or self.model

    def _scoring_model(self) -> str | None:
        return self.scoring_model

    def _request_json(
        self,
        *,
        instructions: str,
        text: str,
        images: list[ImageInput],
        image_detail: str,
        schema_name: str,
        schema: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        for image in images:
            content.append({"type": "input_text", "text": image.label})
            content.append({"type": "input_image", "image_url": image_data_url(image.path), "detail": image_detail})
        extra: dict[str, Any] = {"tools": tools} if tools else {}
        response = self.client.responses.create(
            model=model or self.model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            **extra,
        )
        return parse_json_object(getattr(response, "output_text", "") or "")
