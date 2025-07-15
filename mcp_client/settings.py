from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


# .env absolute path
env_path = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    # Konfigurasi Model
    model_config = SettingsConfigDict(env_file=str(env_path), env_file_encoding="utf-8")

    # MCP Server Endpoint
    mcp_server_url: str = "http://localhost:5000/sse"

    # Kunci API dan host model
    openai_api_key: str
    ollama_host: str = "http://localhost:11434"

    # Model dan parameter LLM
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.0

    # Direktori penyimpanan dan dokumen
    prompt_base_path: str = "mcp_client/prompts"
    vectordb_base_path: str = "mcp_client/vectordb"
