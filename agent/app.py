"""Main agent orchestration loop.

Ties together the planner, tools, memory, and safety filter into a single
`Agent` class that returns a fully structured `AgentTrace` for every run.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from agent.memory import ConversationMemory
from agent.planner import TaskPlanner
from agent.safety import SafetyFilter
from agent.tools.base import BaseTool, ToolResult
from agent.tools.calculator import CalculatorTool
from agent.tools.knowledge_base import KnowledgeBaseTool
from agent.tools.weather import WeatherTool
from agent.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

# ── System prompt for the agent ─────────────────────────────────────────

_AGENT_SYSTEM_PROMPT = """\
You are a helpful research assistant. You can search the web, perform \
calculations, check the weather, and look up information in the company \
knowledge base. Always provide accurate, well-sourced responses. If you \
are unsure about something, say so rather than making up information.\
"""


# ── Data models ─────────────────────────────────────────────────────────


class TraceStep(BaseModel):
    """One step in the agent's execution trajectory."""

    step_number: int
    action: str  # "use_tool", "respond", "clarify", "safety_block"
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Optional[dict[str, Any]] = None
    reasoning: str = ""
    response: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentTrace(BaseModel):
    """Complete execution trace for a single agent run.

    This is the primary artifact that the evals framework evaluates.
    It captures everything — every tool call, every LLM decision,
    every token count — so the evals framework never needs to look
    inside the agent's internals.
    """

    input: str
    output: str
    steps: list[TraceStep] = Field(default_factory=list)
    total_steps: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    safety_triggered: bool = False
    safety_reason: Optional[str] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── LLM client factory ─────────────────────────────────────────────────


def _create_llm_client(provider: str, api_key: str) -> Any:
    """Create an async LLM client for the specified provider."""
    if provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=api_key)

    elif provider == "anthropic":
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=api_key)

    elif provider == "gemini":
        from google import genai
        return genai.Client(api_key=api_key)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# ── Agent class ─────────────────────────────────────────────────────────


class Agent:
    """Multi-tool research assistant with planning, memory, and safety.

    Usage::

        agent = Agent(provider="openai", model="gpt-4o-mini", api_key="sk-...")
        trace = await agent.run("What's the weather in San Francisco?")
        print(trace.output)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        max_steps: int = 10,
        temperature: float = 0.0,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_steps = max_steps
        self._temperature = temperature

        # LLM client
        self._llm_client = _create_llm_client(provider, api_key)

        # Tools registry
        self._default_tools: list[BaseTool] = [
            WebSearchTool(),
            CalculatorTool(),
            WeatherTool(),
            KnowledgeBaseTool(),
        ]
        self._tools: dict[str, BaseTool] = {t.name: t for t in self._default_tools}

        # Memory
        self.memory = ConversationMemory(system_prompt=_AGENT_SYSTEM_PROMPT)

        # Planner
        self._planner = TaskPlanner(
            llm_client=self._llm_client,
            tools=self._default_tools,
            provider=provider,
            model=model,
        )

        # Safety
        self._safety = SafetyFilter(system_prompt=_AGENT_SYSTEM_PROMPT)

    def _set_tools(self, tools: list[BaseTool]) -> None:
        """Replace the tool registry (used by evals to inject mocks)."""
        self._tools = {t.name: t for t in tools}
        self._planner = TaskPlanner(
            llm_client=self._llm_client,
            tools=tools,
            provider=self._provider,
            model=self._model,
        )

    async def run(self, user_input: str) -> AgentTrace:
        """Execute the agent on a user input and return the full trace."""
        start_time = time.perf_counter()
        steps: list[TraceStep] = []
        step_number = 0

        # ── Safety check on input ───────────────────────────────────
        is_safe, rejection_reason = self._safety.check_input(user_input)
        if not is_safe:
            step_number += 1
            steps.append(TraceStep(
                step_number=step_number,
                action="safety_block",
                reasoning=f"Input blocked: {rejection_reason}",
                response="I'm sorry, but I can't process that request. "
                "Please rephrase your question in a way that "
                "I can safely help with.",
            ))
            latency = (time.perf_counter() - start_time) * 1000
            return AgentTrace(
                input=user_input,
                output=steps[-1].response or "",
                steps=steps,
                total_steps=step_number,
                total_latency_ms=latency,
                safety_triggered=True,
                safety_reason=rejection_reason,
                metadata={
                    "provider": self._provider,
                    "model": self._model,
                    "safety_triggered": True,
                },
            )

        # ── Add user message to memory ──────────────────────────────
        self.memory.add_user_message(user_input)

        # ── Planning and execution loop ─────────────────────────────
        final_output = ""
        total_tokens = 0

        for _ in range(self._max_steps):
            step_number += 1

            try:
                plan = await self._planner.plan_next_step(self.memory)
            except Exception as exc:
                logger.error("Planner error: %s", exc)
                error_msg = f"Planning error: {exc}"
                steps.append(TraceStep(
                    step_number=step_number,
                    action="error",
                    reasoning=error_msg,
                ))
                latency = (time.perf_counter() - start_time) * 1000
                return AgentTrace(
                    input=user_input,
                    output="I encountered an error while processing your request.",
                    steps=steps,
                    total_steps=step_number,
                    total_tokens=total_tokens,
                    total_latency_ms=latency,
                    error=error_msg,
                    metadata={
                        "provider": self._provider,
                        "model": self._model,
                        "safety_triggered": False,
                    },
                )

            # ── Action: respond ─────────────────────────────────────
            if plan.action == "respond":
                response_text = plan.response or ""

                # Safety check on output
                is_safe, reason = self._safety.check_output(response_text)
                if not is_safe:
                    response_text = (
                        "I generated a response but it was flagged by "
                        "safety checks. Let me try again with a safer response."
                    )
                    steps.append(TraceStep(
                        step_number=step_number,
                        action="safety_block",
                        reasoning=f"Output blocked: {reason}",
                        response=response_text,
                    ))
                else:
                    steps.append(TraceStep(
                        step_number=step_number,
                        action="respond",
                        reasoning=plan.reasoning,
                        response=response_text,
                    ))

                self.memory.add_assistant_message(response_text)
                final_output = response_text
                break

            # ── Action: clarify ─────────────────────────────────────
            elif plan.action == "clarify":
                clarification = plan.response or "Could you please clarify your request?"
                steps.append(TraceStep(
                    step_number=step_number,
                    action="clarify",
                    reasoning=plan.reasoning,
                    response=clarification,
                ))
                self.memory.add_assistant_message(clarification)
                final_output = clarification
                break

            # ── Action: use_tool ────────────────────────────────────
            elif plan.action == "use_tool":
                tool_name = plan.tool_name or ""
                tool_args = plan.tool_args or {}
                tool = self._tools.get(tool_name)

                if tool is None:
                    # Unknown tool — log and continue
                    steps.append(TraceStep(
                        step_number=step_number,
                        action="use_tool",
                        tool_name=tool_name,
                        tool_args=tool_args,
                        reasoning=f"Tool '{tool_name}' not found.",
                    ))
                    self.memory.add_tool_result(
                        tool_name,
                        {"success": False, "error": f"Unknown tool: {tool_name}"},
                    )
                    continue

                # Execute the tool
                result: ToolResult = await tool.safe_execute(**tool_args)

                result_dict = result.model_dump(mode="json")
                steps.append(TraceStep(
                    step_number=step_number,
                    action="use_tool",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=result_dict,
                    reasoning=plan.reasoning,
                ))

                self.memory.add_tool_result(tool_name, result_dict)

        else:
            # Loop exhausted — fallback response
            final_output = (
                "I wasn't able to complete your request within the "
                f"maximum number of steps ({self._max_steps}). "
                "Please try simplifying your request."
            )
            step_number += 1
            steps.append(TraceStep(
                step_number=step_number,
                action="respond",
                reasoning="Max steps exhausted.",
                response=final_output,
            ))

        latency = (time.perf_counter() - start_time) * 1000

        return AgentTrace(
            input=user_input,
            output=final_output,
            steps=steps,
            total_steps=step_number,
            total_tokens=total_tokens,
            total_latency_ms=latency,
            safety_triggered=False,
            metadata={
                "provider": self._provider,
                "model": self._model,
                "safety_triggered": False,
            },
        )

    async def execute(
        self,
        input: str,
        tools: list[BaseTool] | None = None,
    ) -> AgentTrace:
        """Evals-friendly interface: allows overriding the tool registry.

        This is the method the evals framework calls. It lets the framework
        inject mocked tools without touching the agent's internals.
        """
        if tools is not None:
            self._set_tools(tools)
        return await self.run(input)
