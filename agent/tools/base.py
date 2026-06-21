"""Abstract base tool class and shared data models for the agent's tool system."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a request to invoke a tool."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolResult(BaseModel):
    """Represents the outcome of a tool execution."""

    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


class BaseTool(ABC):
    """Abstract base class that every tool must implement.

    Provides a consistent interface so the evals framework can intercept,
    mock, and score tool usage without knowing the concrete tool.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Natural-language description so the LLM knows when to use this tool."""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema describing the tool's input parameters."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool and return a structured result."""

    async def safe_execute(self, **kwargs: Any) -> ToolResult:
        """Wrapper that catches exceptions and measures duration."""
        start = time.perf_counter()
        try:
            result = await self.execute(**kwargs)
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
            )

    def to_llm_schema(self) -> dict[str, Any]:
        """Return the tool definition in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
