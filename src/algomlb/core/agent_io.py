import sys
from typing import Any
from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Encapsulates the structured output required by automated agents."""

    status: str  # "success" | "error" | "warning"
    command: str  # e.g., "db.init"
    duration_ms: int = 0
    data: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def emit_agent_result(result: AgentResult) -> None:
    """Formats and writes the structured result to stdout and flushes."""
    sys.stdout.write(result.model_dump_json() + "\n")
    sys.stdout.flush()
