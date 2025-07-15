from tiktoken import encoding_for_model
from mcp_client.settings import Settings

settings = Settings()  # type: ignore


def _safe_args(d: dict, redact_keys=("api_key", "password", "token")) -> dict:
    """Redact sensitive keys for logging."""
    return {k: ("***" if k in redact_keys else v) for k, v in d.items()}


ENC = encoding_for_model(settings.llm_model)  # sesuaikan
MAX_MEM_TOKENS = 150


def _truncate_by_tokens(text: str, max_tokens: int = MAX_MEM_TOKENS) -> str:
    """Potong string agar ≤ max_tokens."""
    ids = ENC.encode(text)
    return ENC.decode(ids[:max_tokens])
