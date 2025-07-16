import sys
import io
import json


from mcp_client.utils.safe_args import _safe_args, _truncate_by_tokens
from mcp_client.settings import Settings
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from mcp_client.utils.intent_router import classify_intent
from mcp_client.utils.slug_kak import infer_kak_md, best_match
from mcp_client.utils.mem0_utils import Mem0Manager

import traceback
import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

from mcp_client.utils.pipeline_kak import run as run_kak_pipeline
from mcp_client.utils.pipeline_docgen import run as run_docgen_pipeline

import asyncio
import uuid
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from mcp_client.utils.logger import logger  # noqa: E402

TOOL_TIMEOUT_SEC = 30
PIPE_TIMEOUT_SEC = 180


nest_asyncio.apply()

load_dotenv()

settings = Settings()  # type: ignore


class MCPClient:
    """Client for interacting with OpenAI models using MCP tools."""

    def __init__(self, model: str = settings.llm_model):
        """Initialize the OpenAI MCP client.

        Args:
            model: The OpenAI model to use.
        """
        # Initialize session and client object
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None
        self.llm = AsyncOpenAI()
        self.memory_mgr = Mem0Manager()
        self.model = model
        self.tools = []  # Populated by MCP Server available tools
        self.messages = []  # Chain of thoug store
        self.logger = logger

    # TODO: connect to the MCP Server
    async def connect(self, server_endpoint: str = settings.mcp_server_url):
        """Connect to an MCP server

        Args:
            str:
                server_endpoint: url_endpoint to mcp server with sse transport. default 'http://127.0.0.1:5000/sse'
        """

        try:
            if server_endpoint.startswith("http"):
                self.logger.info("Menghubungkan ke MCP Server via SSE...")
                read_stream, write_stream = await self.exit_stack.enter_async_context(
                    sse_client(server_endpoint)  # type: ignore
                )
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
            else:
                self.logger.info("Menghubungkan ke MCP Server via STDIN/STDOUT...")
                server_params = StdioServerParameters(
                    command="python",
                    args=[server_endpoint],
                )
                # Connect to the server
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                self.stdio, self.write = stdio_transport
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(self.stdio, self.write)
                )

            # Inisialisasi koneksi
            await self.session.initialize()
            await self.memory_mgr.init()
            self.logger.info("Berhasil terhubung ke MCP Server.")

            # List available tools
            mcp_tools = await self.get_tools()
            self.tools = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"],
                }
                for tool in mcp_tools
            ]
            self.logger.info(f"Tools tersedia: {[tool['name'] for tool in self.tools]}")
            return True

        except Exception as e:
            self.logger.error(f"Gagal terhubung ke MCP Server: {e}")
            traceback.print_exc()
            # memanggil cleanup jika koneksi gagal di tengah jalan
            await self.cleanup()
            return False

    # TODO: call a mcp tool
    async def call_tool(self, name: str, args: Dict[str, Any]) -> str:
        """_summary_

        Args:
            name (str): _description_
            args (Dict[str, Any]): _description_

        Returns:
            str: _description_
        """
        try:
            # Call our tool
            result = await self.session.call_tool(name, args)  # type: ignore
            return f"{result.content[0].text}"  # type: ignore
        except Exception as e:
            self.logger.error(f"Gagal memanggil MCP tool: {e}")
            raise

    # TODO: get mcp tool list
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server in OpenAI format.

        Returns:
            list: A list of tool objects provided by the server in OpenAI format.

        Raises:
            Exception: Re-raises any exception encountered during the API call after logging an error message.
        """
        try:
            # List available tools
            tools_result = await self.session.list_tools()  # type: ignore
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in tools_result.tools
            ]

        except Exception as e:
            self.logger.error(f"Gagal mendapatkan MCP Tools: {e}")
            raise

    # TODO: proses query with chat memory mem0
    async def process_query(
        self, query: str, user_id: str = "default", max_turns: int = 20
    ) -> str:
        trace_id = uuid.uuid4().hex[:8]
        tic = time.perf_counter()
        self.logger.info(f"[{trace_id}] > Memproses query: {query!r}")

        # ---------- 1. Intent Classification (dengan retry) -------------------- #
        intent = "other"
        for attempt in range(3):
            try:
                route = await classify_intent(self.llm, query, self.model)
                intent = route.intent if route.confidence_score >= 0.7 else "other"
                self.logger.info(
                    f"[{trace_id}] Intent: {intent} (conf={route.confidence_score:.2f})"
                )
                break
            except Exception as e:
                self.logger.error(
                    f"[{trace_id}] classify_intent gagal (try {attempt + 1}/3): {e}"
                )
                if attempt == 2:
                    self.logger.warning(f"[{trace_id}] Fallback ke intent 'other'")
                await asyncio.sleep(2**attempt)

        # ---------- 2. Jalankan pipeline khusus -------------------------------- #
        try:
            if intent == "kak_analyzer":
                return await self._run_kak(trace_id, query, user_id, max_turns)
            elif intent == "generate_document":
                return await self._run_docgen(trace_id, query, user_id, max_turns)
            else:
                return await self._run_other(trace_id, query, user_id, max_turns)

        finally:
            toc = time.perf_counter() - tic
            self.logger.info(f"[{trace_id}] -- Total latency: {toc:0.2f}s")

    # ======================= HELPER – PIPELINE SPESIFIK ========================= #
    async def _run_kak(self, trace_id: str, query: str, user_id: str, max_turns: int):
        slug = infer_kak_md(query)

        # list_kak_files bisa gagal; aman-kan
        try:
            files_json = await self.call_tool("list_kak_files", {})
            all_files = json.loads(files_json)
        except Exception:
            all_files = []
        kak_md = best_match(all_files, slug) or slug  # type: ignore

        try:
            result = await asyncio.wait_for(
                run_kak_pipeline(
                    client=self,
                    user_query=query,
                    prompt_instruction_name="kak_analyzer",
                    kak_tor_md_name=kak_md,  # type: ignore
                    max_turns=max_turns,
                ),
                timeout=PIPE_TIMEOUT_SEC,
            )
            reply = result
        except asyncio.TimeoutError:
            self.logger.error(f"[{trace_id}] run_kak_pipeline TIMEOUT")
            reply = "Maaf, analisis KAK memerlukan waktu lebih lama dari batas sistem."
        except Exception as e:
            self.logger.error(f"[{trace_id}] run_kak_pipeline error: {e}")
            reply = f"Terjadi kesalahan saat analisis KAK: {e}"

        # commit memori
        await self.memory_mgr.add_conversation(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": reply},
            ],
            user_id=user_id,
        )
        return reply

    async def _run_docgen(
        self, trace_id: str, query: str, user_id: str, max_turns: int
    ):
        slug = infer_kak_md(query)

        try:
            files_json = await self.call_tool("list_kak_files", {})
            all_files = json.loads(files_json)
        except Exception:
            all_files = []
        kak_md = best_match(all_files, slug) or slug  # type: ignore

        try:
            result = await asyncio.wait_for(
                run_docgen_pipeline(
                    client=self,
                    project_name=kak_md,  # type: ignore
                    user_query=query,
                    override_template=None,
                    max_turns=max_turns,
                ),
                timeout=PIPE_TIMEOUT_SEC,
            )
            reply = f"Proposal berhasil dibuat untuk proyek “{kak_md}”.\n\nLokasi file: {result}"
        except asyncio.TimeoutError:
            self.logger.error(f"[{trace_id}] run_docgen_pipeline TIMEOUT")
            reply = "Maaf, pembuatan proposal melebihi batas waktu."
        except Exception as e:
            self.logger.error(f"[{trace_id}] run_docgen_pipeline error: {e}")
            reply = f"Terjadi kesalahan saat generate proposal: {e}"

        await self.memory_mgr.add_conversation(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": reply},
            ],
            user_id=user_id,
        )
        return reply

    # ----------------- Fallback chat dengan Tool-Calling ------------------------ #
    async def _run_other(self, trace_id: str, query: str, user_id: str, max_turns: int):
        # ambil memori relevan
        try:
            memories = await self.memory_mgr.get_memories(query, limit=5)
        except Exception as e:
            self.logger.error(f"[{trace_id}] mem0 search error: {e}")
            memories = []

        mem_block = (
            "\n".join(f"- {_truncate_by_tokens(m)}" for m in memories) or "[Tidak ada]"
        )
        system_mem = {
            "role": "system",
            "content": (
                "Memori historis relevan:\n"
                f"{mem_block}\n\n"
                "Gunakan memori di atas jika membantu."
            ),
        }
        messages = [
            system_mem,
            {
                "role": "system",
                "content": "Anda adalah “ProjectWise”, asisten virtual untuk tim Presales & PM.",
            },
            {"role": "user", "content": query},
        ]
        tools = await self.get_tools()
        final_answer = None

        try:
            for turn in range(max_turns):
                self.logger.info(f"[{trace_id}] - Turn {turn + 1}/{max_turns}")
                response = await self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    tools=tools,  # type: ignore
                    tool_choice="auto",
                )
                assistant_msg = response.choices[0].message
                messages.append(assistant_msg.model_dump())

                if not assistant_msg.tool_calls:  # ▶ Jawaban final
                    final_answer = assistant_msg.content or "Tidak ada jawaban."
                    break

                # ─ Jalankan setiap tool call (parallel → gather) ─
                async def _exec(tc):
                    fname = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                        self.logger.info(
                            f"[{trace_id}]  · tool '{fname}' args={_safe_args(args)}"
                        )
                        return await asyncio.wait_for(
                            self.call_tool(fname, args), timeout=TOOL_TIMEOUT_SEC
                        )
                    except asyncio.TimeoutError:
                        self.logger.error(f"[{trace_id}] tool {fname} TIMEOUT")
                        return f"TIMEOUT executing {fname}"
                    except Exception as e:
                        self.logger.error(f"[{trace_id}] tool {fname} error: {e}")
                        return f"Error executing {fname}: {e}"

                results = await asyncio.gather(
                    *[_exec(tc) for tc in assistant_msg.tool_calls]
                )
                # masukkan hasil ke messages
                for tc, out in zip(assistant_msg.tool_calls, results):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": out,
                        }
                    )

            if not final_answer:
                self.logger.warning(f"[{trace_id}] Batas {max_turns} turn tercapai.")
                final_answer = (
                    "Maaf, saya belum bisa menyelesaikan permintaan dalam batas waktu."
                )

        finally:
            answer_to_save = final_answer or "Maaf, terjadi kegagalan internal."
            # commit memori apa pun hasilnya
            await self.memory_mgr.add_conversation(
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer_to_save},
                ],
                user_id=user_id,
            )

        return answer_to_save

    # TODO: cleanup
    async def cleanup(self):
        """
        Asynchronously clean up and close all managed resources.

        This method ensures all resources within the exit stack are properly closed. It will log the outcome of the operation.

        :raises Exception: Re-raises any exception that occurs during the cleanup process.
        """
        try:
            await self.exit_stack.aclose()
            self.logger.info("Terputus dari MCP Server.")

        except Exception as e:
            self.logger.error(f"Gagal saat cleanup: {e}")
            traceback.print_exc()
            raise
