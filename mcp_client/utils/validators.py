# mcp_client/validators.py
from pydantic import ValidationError
from schemas import BaseToolResponse

def validate_tool_output(raw: dict, model) -> BaseToolResponse:
    try:
        validated = model.model_validate(raw)
    except ValidationError as e:
        # log & naikkan error keras; LLM tak boleh lanjut
        raise RuntimeError(f"Invalid tool schema: {e}") from e
    return validated
