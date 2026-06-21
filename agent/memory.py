"""Conversation memory management for the agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation history."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_results: Optional[list[dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationMemory:
    """Stores and manages the full conversation history.

    Provides methods for adding messages, retrieving the history in LLM-friendly
    format, and serializing state for eval logging.
    """

    def __init__(self, system_prompt: str = "") -> None:
        self._messages: list[Message] = []
        self._system_prompt = system_prompt
        self._turn_count: int = 0

    @property
    def turn_count(self) -> int:
        """Number of user turns in this conversation."""
        return self._turn_count

    @property
    def message_count(self) -> int:
        """Total messages in history."""
        return len(self._messages)

    def add_user_message(self, content: str) -> None:
        """Append a user message and increment the turn counter."""
        self._messages.append(Message(role="user", content=content))
        self._turn_count += 1

    def add_assistant_message(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append an assistant message, optionally with tool call metadata."""
        self._messages.append(
            Message(role="assistant", content=content, tool_calls=tool_calls)
        )

    def add_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        """Append a tool result message."""
        self._messages.append(
            Message(
                role="tool",
                content=f"Tool '{tool_name}' returned: {result}",
                tool_results=[{"tool_name": tool_name, **result}],
            )
        )

    def get_messages(self) -> list[dict[str, Any]]:
        """Return the conversation history formatted for the LLM API.

        Includes the system prompt as the first message, followed by
        all user/assistant/tool messages.
        """
        messages: list[dict[str, Any]] = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        for msg in self._messages:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            messages.append(entry)

        return messages

    def get_last_n_messages(self, n: int) -> list[dict[str, Any]]:
        """Return the last n messages (useful for context window management)."""
        messages = self.get_messages()
        # Always include the system prompt if present
        if messages and messages[0].get("role") == "system":
            return [messages[0]] + messages[-(n):]
        return messages[-n:]

    def get_summary(self) -> str:
        """Return a brief text summary of the conversation so far."""
        if not self._messages:
            return "No conversation history."

        user_msgs = [m for m in self._messages if m.role == "user"]
        tool_uses = [m for m in self._messages if m.role == "tool"]

        lines = [
            f"Conversation with {self._turn_count} user turn(s) and {len(self._messages)} total messages.",
        ]
        if user_msgs:
            lines.append(f"Last user message: '{user_msgs[-1].content[:100]}...'")
        if tool_uses:
            tool_names = [
                m.tool_results[0]["tool_name"]
                for m in tool_uses
                if m.tool_results
            ]
            lines.append(f"Tools used: {', '.join(tool_names)}")

        return " ".join(lines)

    def clear(self) -> None:
        """Reset the conversation history."""
        self._messages.clear()
        self._turn_count = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full memory state for eval logging."""
        return {
            "system_prompt": self._system_prompt,
            "turn_count": self._turn_count,
            "message_count": len(self._messages),
            "messages": [msg.model_dump(mode="json") for msg in self._messages],
        }
