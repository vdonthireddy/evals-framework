"""Task planner that uses the LLM to decide the agent's next action."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel

from agent.memory import ConversationMemory
from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)

# ── System prompt template ──────────────────────────────────────────────

_PLANNER_SYSTEM_PROMPT = """\
You are an intelligent assistant that helps users by using available tools. \
Your job is to decide the NEXT single action to take.

## Available Tools
{tool_descriptions}

## Instructions
1. Analyze the conversation so far and the user's latest request.
2. Decide the SINGLE next action to take. Choose exactly one:
   - "use_tool": Call one of the available tools.
   - "respond": Provide a final answer to the user (only when you have enough information).
   - "clarify": Ask the user a clarifying question (only when the request is genuinely ambiguous).

3. Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{
    "action": "use_tool" | "respond" | "clarify",
    "tool_name": "<name of tool to call, or null if not using a tool>",
    "tool_args": {{<arguments for the tool, or null>}},
    "response": "<your response text if action is 'respond' or 'clarify', otherwise null>",
    "reasoning": "<brief explanation of why you chose this action>"
}}

## Rules
- Use tools when you need external information. Do NOT make up information.
- If a tool returned results, use them to formulate your response.
- If a tool returned an error, explain the issue or try an alternative approach.
- Be concise and helpful in your responses.
- When you have all the information needed, choose "respond" and provide the final answer.
"""


class PlanStep(BaseModel):
    """Represents the planner's decision for the next action."""

    action: str  # "use_tool", "respond", "clarify"
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    response: Optional[str] = None
    reasoning: str = ""


class TaskPlanner:
    """Uses an LLM to decide the agent's next action given conversation state."""

    def __init__(
        self,
        llm_client: Any,
        tools: list[BaseTool],
        provider: str = "openai",
        model: str = "gpt-4o-mini",
    ) -> None:
        self._llm_client = llm_client
        self._tools = {tool.name: tool for tool in tools}
        self._provider = provider
        self._model = model

    def _build_tool_descriptions(self) -> str:
        """Format all tool schemas into a readable block for the system prompt."""
        descriptions = []
        for tool in self._tools.values():
            schema = tool.parameters_schema
            params_str = json.dumps(schema.get("properties", {}), indent=2)
            required = schema.get("required", [])
            descriptions.append(
                f"### {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Parameters:\n{params_str}\n"
                f"Required: {required}"
            )
        return "\n\n".join(descriptions)

    def _build_system_prompt(self) -> str:
        """Construct the full system prompt with tool descriptions."""
        return _PLANNER_SYSTEM_PROMPT.format(
            tool_descriptions=self._build_tool_descriptions()
        )

    async def plan_next_step(self, memory: ConversationMemory) -> PlanStep:
        """Ask the LLM to decide the next action.

        Returns a PlanStep with the chosen action. On parse failure, retries
        once. If still malformed, returns a safe fallback response.
        """
        system_prompt = self._build_system_prompt()

        # Build messages: system + conversation history
        messages = [{"role": "system", "content": system_prompt}]
        for msg in memory.get_messages():
            if msg.get("role") != "system":
                messages.append(msg)

        for attempt in range(2):
            try:
                raw_response = await self._call_llm(messages)
                plan = self._parse_response(raw_response)
                return plan
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning(
                    "Failed to parse planner response (attempt %d): %s",
                    attempt + 1,
                    exc,
                )
                if attempt == 0:
                    # Nudge the LLM to output valid JSON on retry
                    messages.append({
                        "role": "user",
                        "content": "Your previous response was not valid JSON. "
                        "Please respond with ONLY a JSON object matching "
                        "the required format.",
                    })

        # Fallback: respond with an apology
        logger.error("Planner failed to produce valid output after retries.")
        return PlanStep(
            action="respond",
            response="I apologize, but I'm having trouble processing your request. "
            "Could you please rephrase it?",
            reasoning="Fallback due to repeated parse failures.",
        )

    async def _call_llm(self, messages: list[dict[str, Any]]) -> str:
        """Call the LLM provider and return the raw text response."""
        if self._provider == "openai":
            response = await self._llm_client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""

        elif self._provider == "anthropic":
            # Anthropic expects system prompt separately
            system_msg = ""
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    chat_messages.append(msg)
            response = await self._llm_client.messages.create(
                model=self._model,
                system=system_msg,
                messages=chat_messages,
                temperature=0.0,
                max_tokens=1024,
            )
            return response.content[0].text

        elif self._provider == "gemini":
            # Google GenAI
            response = await self._llm_client.aio.models.generate_content(
                model=self._model,
                contents=[msg["content"] for msg in messages if msg["role"] != "system"],
                config={
                    "system_instruction": next(
                        (msg["content"] for msg in messages if msg["role"] == "system"),
                        "",
                    ),
                    "temperature": 0.0,
                    "max_output_tokens": 1024,
                },
            )
            return response.text or ""

        else:
            raise ValueError(f"Unsupported LLM provider: {self._provider}")

    def _parse_response(self, raw: str) -> PlanStep:
        """Parse the raw LLM response into a PlanStep."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            # Remove opening fence (possibly ```json)
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()

        data = json.loads(text)

        action = data["action"]
        if action not in ("use_tool", "respond", "clarify"):
            raise ValueError(f"Invalid action: {action}")

        if action == "use_tool":
            if not data.get("tool_name"):
                raise ValueError("'use_tool' action requires 'tool_name'")
            if data["tool_name"] not in self._tools:
                raise ValueError(f"Unknown tool: {data['tool_name']}")

        return PlanStep(
            action=action,
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args"),
            response=data.get("response"),
            reasoning=data.get("reasoning", ""),
        )
