from __future__ import annotations
import json
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from mcp_client.utils.logger import logger


class IntentRoute(BaseModel):
    intent: Literal["kak_analyzer", "generate_document", "other"]
    confidence_score: float = Field(ge=0, le=1)


# -------- classifier -------------------------------------------
async def classify_intent(llm, query: str, model: str = "gpt-4o") -> IntentRoute:
    """
    Kembalikan IntentRoute; jika model gagal, default => other, score 0.0
    """
    system_msg = (
        "Anda adalah *Router Agent*.\n"
        "Balas **hanya** JSON sesuai skema: "
        '{"intent":"<kak_analyzer|generate_document|other>", "confidence_score":0.xx}'
    )

    # Few-shot sebagai dialog — 4 contoh “golden”
    fewshot = [
        # 1
        {"role": "user", "content": "Analisa proyek Bank Sumsel Babel"},
        {
            "role": "assistant",
            "content": '{"intent":"kak_analyzer","confidence_score":0.95}',
        },
        # 2
        {
            "role": "user",
            "content": "Apa saja barang dan jasa di dalam proyek bank sumsel babel?",
        },
        {
            "role": "assistant",
            "content": '{"intent":"kak_analyzer","confidence_score":0.92}',
        },
        # 3
        {"role": "user", "content": "Buatkan proposal implementasi Switch Core"},
        {
            "role": "assistant",
            "content": '{"intent":"generate_document","confidence_score":0.9}',
        },
        # 4
        {"role": "user", "content": "Berapa harga Bitcoin hari ini?"},
        {"role": "assistant", "content": '{"intent":"other","confidence_score":0.88}'},
    ]

    messages = [
        {"role": "system", "content": system_msg},
        *fewshot,
        {"role": "user", "content": query},
    ]

    try:
        resp = await llm.chat.completions.parse(
            model=model,
            temperature=0,
            top_p=0,
            messages=messages,
            response_format=IntentRoute,
        )
        raw_json = resp.choices[0].message.content
        logger.info(f"Raw router output: {raw_json}")

        return IntentRoute.model_validate_json(raw_json)
    except (ValidationError, json.JSONDecodeError) as ve:
        logger.info(f"Router JSON parse error: {ve}")
    except Exception as e:
        logger.info(f"Router LLM error: {e}")

    # Fallback aman
    return IntentRoute(intent="other", confidence_score=0.0)
