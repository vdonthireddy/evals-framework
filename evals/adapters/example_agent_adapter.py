"""Adapter that bridges the Phase 1 example agent to the evals framework.

This is the ONLY file in the evals package that imports from ``agent/``.
Everything else in the framework is completely decoupled from any
specific agent implementation.
"""

from __future__ import annotations

from typing import Any

from evals.core.interfaces import AgentAdapter, AgentOutput, TraceStep

from agent.app import Agent, AgentTrace


class ExampleAgentAdapter(AgentAdapter):
    """Adapter for the Phase 1 multi-tool research assistant.

    Usage::

        adapter = ExampleAgentAdapter(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-...",
        )
        output = await adapter.execute("What's the weather in Tokyo?")
        print(output.output)
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
        self._agent = Agent(
            provider=provider,
            model=model,
            api_key=api_key,
            max_steps=max_steps,
            temperature=temperature,
        )

    async def execute(self, input: str) -> AgentOutput:
        """Run the example agent and convert its AgentTrace to AgentOutput."""
        trace: AgentTrace = await self._agent.run(input)
        return self._convert_trace(trace)

    def reset(self) -> None:
        """Clear the agent's conversation memory between eval cases."""
        self._agent.memory.clear()

    def get_info(self) -> dict[str, Any]:
        """Return metadata about the example agent."""
        return {
            "name": "ExampleResearchAssistant",
            "provider": self._provider,
            "model": self._model,
            "version": "1.0.0",
        }

    # ── Private helpers ─────────────────────────────────────────────

    @staticmethod
    def _convert_trace(trace: AgentTrace) -> AgentOutput:
        """Map the agent's native AgentTrace to the framework's AgentOutput."""
        steps = [
            TraceStep(
                step_number=step.step_number,
                action=step.action,
                tool_name=step.tool_name,
                tool_args=step.tool_args,
                tool_result=step.tool_result,
                reasoning=step.reasoning or "",
                timestamp=step.timestamp,
            )
            for step in trace.steps
        ]

        return AgentOutput(
            input=trace.input,
            output=trace.output,
            steps=steps,
            total_steps=trace.total_steps,
            total_tokens=trace.total_tokens,
            total_latency_ms=trace.total_latency_ms,
            metadata={
                "safety_triggered": trace.safety_triggered,
                "safety_reason": trace.safety_reason,
                "error": trace.error,
                "provider": trace.metadata.get("provider"),
                "model": trace.metadata.get("model"),
            },
        )
