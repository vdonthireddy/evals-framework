"""Adapter for bridging NativeFunctionCallingAgent to the evals framework."""

from __future__ import annotations

import asyncio
from typing import Any

from agent.native_agent import NativeFunctionCallingAgent
from evals.adapters.example_agent_adapter import ExampleAgentAdapter
from evals.core.interfaces import AgentAdapter, AgentOutput, TraceStep


class NativeFunctionCallingAdapter(AgentAdapter):
    """Adapter for zero-dependency native OpenAI tools function calling agent."""

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
        self._api_key = api_key
        self._max_steps = max_steps
        self._temperature = temperature
        self._agent = self._create_agent()

    def _create_agent(self) -> NativeFunctionCallingAgent:
        return NativeFunctionCallingAgent(
            provider=self._provider,
            model=self._model,
            api_key=self._api_key,
            max_steps=self._max_steps,
            temperature=self._temperature,
        )

    async def execute(self, input: str) -> AgentOutput:
        """Run the native function calling agent and convert its trace to AgentOutput."""
        agent = self._create_agent()
        trace = await agent.run(input)
        return ExampleAgentAdapter._convert_trace(trace)

    def reset(self) -> None:
        """Clear agent state between evaluation cases."""
        self._agent = self._create_agent()

    def get_info(self) -> dict[str, Any]:
        """Return metadata about the NativeFunctionCallingAgent."""
        return {
            "name": "NativeFunctionCallingAgent",
            "provider": self._provider,
            "model": self._model,
            "framework": "NativeOpenAIFunctionCalling",
            "version": "1.0.0",
        }
