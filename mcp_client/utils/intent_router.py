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
        # ------------------------------------------------------------------
        # KONTEXT & PERAN
        # ------------------------------------------------------------------
        "Anda adalah *AI ProjectWise* assistant cerdas untuk tim Presales & Project Manager.\n"
        "Tugas: menganalisis setiap pesan user, lalu memilih SATU dari tiga intent:\n"
        "  • kak_analyzer        → user MEMINTA analisis atau ringkasan KAK/TOR/proyek.\n"
        "  • generate_document   → user MEMINTA pembuatan dokumen/proposal.\n"
        "  • other               → di luar dua kategori di atas.\n"
        " Jika user MEMINTA analisis proyek tetapi tidak memberikan informasi nama proyeknya, "
        " Mintalah klafirikasi."
        "\n"
        # ------------------------------------------------------------------
        # FORMAT KELUARAN WAJIB
        # ------------------------------------------------------------------
        "Kembalikan *hanya* JSON valid persis sesuai skema:\n"
        '  {"intent":"<kak_analyzer|generate_document|other>", "confidence_score":0.xx}\n'
        "JANGAN menambah properti lain.\n"
        "\n"
        # ------------------------------------------------------------------
        # ATURAN KLASIFIKASI
        # ------------------------------------------------------------------
        "• Gunakan *kata kunci pemicu* berikut:\n"
        '  – kak_analyzer: "analisa", "analisis", "summary", "summaries", "analyze", '
        '"analyzer", "analisa ruang lingkup", "ringkas proyek", "analisa proyek".\n'
        '  – generate_document: "buatkan dokumen", "buat proposal", "proposal teknis", '
        '"proposal teknis dan penawaran", "proposal harga", "generate dokument", '
        '"generate document", "buatkan document".\n'
        "• Bila pesan HANYA pertanyaan/informasi tanpa permintaan aksi ⇒ intent = *other*.\n"
        "• Jika kata kunci pemicu terdeteksi untuk intent dan nama proyek tidak diberikan,"
        "dengan jelas. Anda WAJIB KLARIFIKASI.\n"
        "• confidence_score selalu 0‑1; gunakan penilaian sendiri, tidak ada ambang tetap.\n"
        "\n"
        # ------------------------------------------------------------------
        # PERILAKU SESUDAH KLASIFIKASI
        # ------------------------------------------------------------------
        "• Jika intent = kak_analyzer *atau* generate_document → "
        "KEMBALIKAN JSON saja (jangan jawab isi permintaan).\n"
        "• Jika intent = other → Anda boleh langsung menjawab pertanyaan user "
        "tanpa memanggil tool *kecuali* Anda menilai tool diperlukan.\n"
        "\n"
    )

    # Few-shot sebagai dialog — 4 contoh “golden”
    fewshot = [
        {"role": "user", "content": "Analisa proyek Bank Sumsel Babel"},
        {
            "role": "assistant",
            "content": '{"intent":"kak_analyzer","confidence_score":0.95}',
        },
        {"role": "user", "content": "Buat summaries/summary proyek Bank Sumsel Babel"},
        {
            "role": "assistant",
            "content": '{"intent":"kak_analyzer","confidence_score":0.95}',
        },
        {
            "role": "user",
            "content": "Apa saja barang dan jasa di dalam proyek bank sumsel babel?",
        },
        {
            "role": "assistant",
            "content": '{"intent":"other","confidence_score":0.92}',
        },
        {
            "role": "user",
            "content": "Berapa SLA di dalam proyek bank sumsel babel?",
        },
        {
            "role": "assistant",
            "content": '{"intent":"other","confidence_score":0.92}',
        },
        {
            "role": "user",
            "content": "Berikan informasi terkait proyek bank sumsel babel?",
        },
        {
            "role": "assistant",
            "content": '{"intent":"other","confidence_score":0.92}',
        },
        {"role": "user", "content": "Buatkan proposal implementasi Switch Core"},
        {
            "role": "assistant",
            "content": '{"intent":"generate_document","confidence_score":0.9}',
        },
        {"role": "user", "content": "Berapa harga Bitcoin hari ini?"},
        {"role": "assistant", "content": '{"intent":"other","confidence_score":0.88}'},
        {"role": "user", "content": "Bantu saya analisa proyek dong."},
        {"role": "assistant", "content": '{"intent":"other","confidence_score":0.80}'},
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
