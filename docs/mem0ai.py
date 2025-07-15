import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import AsyncMemory
from typing import List, Dict

load_dotenv()

# OpenAI client setup
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Memory config
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333
        }
    },
    "llm": {
        "provider": "openai",
        "config": {"api_key": openai_api_key, "model": "gpt-4o-mini"},
    },
    "embedder": {
        "provider": "openai",
        "config": {"api_key": openai_api_key, "model": "text-embedding-3-small"},
    },
    # "graph_store": {
    #     "provider": "neo4j",
    #     "config": {
    #         "url": "neo4j+s://your-instance",
    #         "username": "neo4j",
    #         "password": "password"
    #     }
    # },
    # "history_db_path": os.path.abspath("mcp_client/vectordb/history.db"),
    # "version": "v1.1",
    # "custom_fact_extraction_prompt": "Optional custom prompt for fact extraction for memory",
    # "custom_update_memory_prompt": "Optional custom prompt for update memory"
}


async def init_memory() -> AsyncMemory:
    return await AsyncMemory.from_config(config)


async def chat_with_memory(
    message: str, memory: AsyncMemory, user_id: str = "default_user"
) -> str:
    """
    Fungsi utama untuk percakapan dengan memori historis menggunakan mem0.
    """
    try:
        relevant_memories = await memory.search(query=message, user_id=user_id, limit=5)
        memory_items = relevant_memories.get("results", [])
        memories_str = (
            "\n".join(f"- {entry['memory']}" for entry in memory_items)
            or "[Tidak ada memori ditemukan]"
        )
    except Exception as e:
        memories_str = "[Gagal mengambil memori]"
        print(f"[Warning] Gagal search memory: {e}")

    # Prompt sistem
    system_prompt = (
        "Anda adalah ProjectWise, asisten AI untuk presales dan project manager.\n"
        "Jawab berdasarkan konteks user dan memori historis.\n"
        f"User Memories:\n{memories_str}"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # type: ignore
        )
        assistant_response = response.choices[0].message.content
    except Exception as e:
        assistant_response = "[Gagal mendapatkan respons dari LLM]"
        print(f"[Error] Gagal panggil OpenAI: {e}")

    messages.append({"role": "assistant", "content": assistant_response})  # type: ignore

    try:
        await memory.add(messages=messages, user_id=user_id)
    except Exception as e:
        print(f"[Warning] Gagal menambahkan ke memori: {e}")

    return assistant_response  # type: ignore


async def main():
    memory = await init_memory()
    print("ProjectWise Memory Chat. Ketik 'exit' untuk keluar.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = await chat_with_memory(user_input, memory)
        print("AI:", response)


if __name__ == "__main__":
    asyncio.run(main())
