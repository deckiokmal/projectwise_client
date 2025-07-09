import json
from mcp_client.utils.logger import logger
from mcp_client.settings import Settings
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import traceback
import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI


nest_asyncio.apply()

load_dotenv()

settings = Settings()  # type: ignore


class MCPClient:
    """Client for interacting with OpenAI models using MCP tools."""

    def __init__(self, model: str = "gpt-4o"):
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
                    sse_client(server_endpoint)
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

    # TODO: proses query
    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available MCP tools.

        Args:
            query (str): The usre query.

        Returns:
            str: The response from OpenAI.

        Raises:
            Exception: Re-raises any exception that occurs during the LLM call or tool execution after logging the error.
        """
        self.logger.info("memproses query")
        try:
            # Ambil tools yang tersedia
            tools = await self.get_tools()

            # Initial OpenAI API call
            response = await self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                tools=tools,  # type: ignore
                tool_choice="auto",
            )

            # Get assistant's response
            assistant_message = response.choices[0].message

            # Initialize conversation with user query and assistant response
            messages = [
                {
                    "role": "user",
                    "content": query,
                },
                assistant_message,
            ]

            # Handle tool call jika tersedia
            if assistant_message.tool_calls:
                # Proses setiap tool call yg ada
                for tool_call in assistant_message.tool_calls:
                    # Execute tool call
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    result = await self.call_tool(name, args)

                    if name != "build_summary_tender_payload":
                        # Add tool response to conversation
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )
                    else:
                        try:
                            payload = json.loads(result)
                            instruksi = payload["instruction"]
                            konteks = payload["context"]
                        except Exception:
                            raise ValueError(
                                "Output tool tidak valid.\n"
                                f"Harus JSON {'instruction', 'context'}, dapat: {result[:200]}"
                            )
                        messages = [
                            {"role": "system", "content": instruksi},
                            {"role": "user", "content": konteks},
                        ]

                # Get final response from OpenAI with tool results
                final_response = await self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    tools=tools,  # type: ignore
                    tool_choice="none",
                )

                return f"{final_response.choices[0].message.content}"

            # No tool calls, just return the direct response
            return f"{assistant_message.content}"

        except Exception as e:
            self.logger.error(f"Gagal memproses query: {e}")
            raise

    # TODO: call llm
    async def call_llm(self):
        """
        Asynchronously calls the Large Language Model (LLM) with the current message history.

        This method sends the conversation history stored in `self.messages` and the available tools defined in `self.tools` to the specified GPT model. It allows the model to automatically choose whether to call a tool or respond with a message.

        Returns:
            The awaited response object from the LLM API. The structure of this object depends on the specific LLM client library being used, but it typically contains the model's output, such as text content or a request to use a tool.

        Raises:
            Exception: Propagates any exception that occurs during the API call after logging the error.
        """
        try:
            # Initial OpenAI API call
            response = self.llm.responses.create(
                model="gpt-4.1",
                input=self.messages,  # type: ignore
                tool_choice="auto",
                tools=self.tools,  # type: ignore
            )

            return await response  # type: ignore

        except Exception as e:
            self.logger.error(f"Gagal call_llm: {e}")
            raise

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
