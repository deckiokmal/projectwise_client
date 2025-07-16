from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import asyncio

from dotenv import load_dotenv
from mem0 import AsyncMemory
from mcp_client.settings import Settings

settings = Settings()  # type: ignore


load_dotenv()

# -----------------------------------------------------
#  Konfigurasi default – dapat dioverride via env/file
# -----------------------------------------------------


def _default_config() -> Dict[str, Any]:
    """Bangun konfigurasi default mem0.

    Nilai dapat dioverride via ENV:
    • MEM0_VECTOR_HOST, MEM0_VECTOR_PORT
    • OPENAI_API_KEY, MEM0_LLM_MODEL, MEM0_EMBED_MODEL
    """

    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": os.getenv("MEM0_VECTOR_HOST", "localhost"),
                "port": int(os.getenv("MEM0_VECTOR_PORT", "6333")),
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "model": os.getenv("MEM0_LLM_MODEL", settings.llm_model),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "model": os.getenv("MEM0_EMBED_MODEL", settings.embed_model),
            },
        },
        # Opsional – boleh ditambahkan jika memakai graph store / history DB
        # "graph_store": {...},
        # "history_db_path": mcp_client/qdrant_storage/history.db,
        "version": "v1.1",
    }


class Mem0Manager:
    """Wrapper asinkron untuk mem0 AsyncMemory agar lebih modular."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or _default_config()
        self._memory: Optional[AsyncMemory] = None
        # lock sederhana agar init hanya terjadi sekali
        self._init_lock = asyncio.Lock()

    # ------------- lifecycle -----------------------------------------
    async def init(self) -> None:
        """Inisialisasi *lazily*; aman dipanggil berkali‑kali."""
        if self._memory is None:
            async with self._init_lock:
                if self._memory is None:  # cek ulang di dalam lock
                    self._memory = await AsyncMemory.from_config(self._config)

    @property
    def memory(self) -> AsyncMemory:
        if self._memory is None:
            raise RuntimeError(
                "Mem0Manager belum di‑init. Panggil await init() dahulu."
            )
        return self._memory

    # ------------- operasi utama -------------------------------------
    async def get_memories(
        self, query: str, *, user_id: str = "default", limit: int = 5
    ) -> List[str]:
        """Cari memori relevan untuk *query* dan kembalikan list string."""
        await self.init()
        try:
            result = await self.memory.search(query=query, user_id=user_id, limit=limit)
            return [item["memory"] for item in result.get("results", [])]
        except Exception as e:
            # Jangan memutus alur chatbot – cukup log & kembalikan list kosong
            print(f"[Mem0] Gagal search memory: {e}")
            return []

    async def add_conversation(
        self, messages: List[Dict[str, str]], *, user_id: str = "default"
    ) -> None:
        """Simpan *messages* (urutan dialog) ke memori."""
        await self.init()
        try:
            await self.memory.add(messages=messages, user_id=user_id)
        except Exception as e:
            print(f"[Mem0] Gagal menambah memori: {e}")

    # Convenience helper ------------------------------------------------
    async def chat_with_memories(
        self, llm_client, *, user_message: str, user_id: str = "default"
    ) -> str:
        """Contoh util satu‑pintu: ambil memories, panggil LLM, simpan hasil."""
        memories = await self.get_memories(user_message, user_id=user_id)
        memories_block = "\n".join(f"- {m}" for m in memories) or "[Tidak ada]"
        system_prompt = (
            "Anda adalah ProjectWise, asisten AI presales & PM.\n"
            f"Memori relevan:\n{memories_block}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        response = await llm_client.chat.completions.create(
            model=self._config["llm"]["config"]["model"],
            messages=messages,  # type: ignore[arg-type]
        )
        assistant_reply = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_reply})  # type: ignore
        await self.add_conversation(messages, user_id=user_id)
        return assistant_reply  # type: ignore
