# cli_chat.py
"""
Sesi interaktif terminal untuk ProjectWise MCP-Client
=====================================================
• Terhubung ke MCP-Server via SSE / stdio (otomatis).
• Prompt URL server sekali di awal (bisa diketik langsung atau Enter
  untuk default http://localhost:5000/sse).
• Setelah koneksi sukses, REPL sederhana:
      > ketik pertanyaan / perintah
      > tulis :quit  / :exit  untuk keluar
• Mendukung pemanggilan MCP tool secara otomatis via OpenAI Function-Calling,
  sesuai implementasi di MCPClient.process_query().
------------------------------------------------------
"""

import asyncio
import signal
import sys
from datetime import datetime

from mcp_client.client import MCPClient
from mcp_client.settings import Settings


settings = Settings()  # type: ignore

DEFAULT_SERVER = settings.mcp_server_url


async def interactive():
    """REPL utama"""
    print("╔═ ProjectWise Terminal ───────────────────────────────═╗")
    print("║ Tekan Ctrl-C / ketik :quit untuk keluar               ║")
    print("╚═══════════════════════════════════════════════════════╝")

    # 1. Inisialisasi MCPClient
    client = MCPClient()

    # 2. Minta URL server
    url = input(f"URL MCP Server [{DEFAULT_SERVER}]: ").strip() or DEFAULT_SERVER

    # 3. Coba koneksi
    if not await client.connect(url):
        print("Gagal terhubung ke server. Periksa URL atau jalankan server dulu.")
        return

    # 4. Masuk REPL
    try:
        while True:
            query = input("\nAnda > ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue

            # Proses query
            try:
                started = datetime.now()
                result = await client.process_query(query)
                dur = (datetime.now() - started).total_seconds()
                print(f"\nProjectWise [{dur:.2f}s] > {result}")
            except Exception as e:
                print(f"Error: {e}")

    except (EOFError, KeyboardInterrupt):
        # Tangani Ctrl-C dengan rapi
        pass
    finally:
        await client.cleanup()
        print("\nSesi berakhir. Sampai jumpa!")


def main():
    # Agar Ctrl-C langsung mematikan event-loop Windows juga
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    asyncio.run(interactive())


if __name__ == "__main__":
    main()
