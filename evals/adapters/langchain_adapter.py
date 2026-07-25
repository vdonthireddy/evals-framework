"""Generic adapter for bridging LangChain agents to the evals framework.

This adapter allows evaluating any LangChain AgentExecutor, Runnable, or custom
agent chain using the framework's standard `AgentAdapter` contract.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional

from evals.core.interfaces import AgentAdapter, AgentOutput, TraceStep


class LangChainAdapter(AgentAdapter):
    """Adapter for LangChain agents, runnables, and agentic workflows.

    Usage::

        from langchain.agents import AgentExecutor
        from evals.adapters.langchain_adapter import LangChainAdapter

        adapter = LangChainAdapter(
            agent=agent_executor,
            name="MyLangChainAssistant",
            reset_fn=lambda: memory.clear(),
        )
        output = await adapter.execute("Find the capital of France")
    """

    def __init__(
        self,
        agent: Any,
        name: str = "LangChainAgent",
        version: str = "1.0.0",
        reset_fn: Optional[Callable[[], None]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the LangChain adapter.

        Args:
            agent: A LangChain AgentExecutor, Runnable, or callable.
            name: Human-readable identifier for the agent.
            version: Agent version string.
            reset_fn: Optional callback invoked during reset() to clear memory/state.
            metadata: Additional metadata dictionary describing the agent.
        """
        self._agent = agent
        self._name = name
        self._version = version
        self._reset_fn = reset_fn
        self._metadata = metadata or {}

    async def execute(self, input: str) -> AgentOutput:
        """Execute the LangChain agent and map its response to `AgentOutput`."""
        start_time = time.perf_counter()

        raw_result = await self._run_agent(input)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return self._convert_result(input, raw_result, latency_ms)

    def reset(self) -> None:
        """Reset agent memory or conversation state between evaluation cases."""
        if self._reset_fn is not None:
            self._reset_fn()
            return

        # Attempt automatic memory clearance if a standard LangChain memory object exists
        memory = getattr(self._agent, "memory", None)
        if memory is not None and hasattr(memory, "clear"):
            try:
                memory.clear()
            except Exception:
                pass

    def get_info(self) -> dict[str, Any]:
        """Return metadata about the underlying LangChain agent."""
        info = {
            "name": self._name,
            "version": self._version,
            "framework": "LangChain",
            "agent_class": self._agent.__class__.__name__,
        }
        info.update(self._metadata)
        return info

    # ── Internal Helpers ───────────────────────────────────────────────────

    async def _run_agent(self, input_text: str) -> Any:
        """Invoke the agent asynchronously or synchronously depending on interface."""
        input_payload = {"input": input_text}

        # Case 1: Standard LangChain `ainvoke` (Async Runnable / AgentExecutor)
        if hasattr(self._agent, "ainvoke") and callable(self._agent.ainvoke):
            try:
                return await self._agent.ainvoke(input_payload)
            except TypeError:
                # Fallback if ainvoke expects a single string input
                return await self._agent.ainvoke(input_text)

        # Case 2: Async callable / coroutine
        if asyncio.iscoroutinefunction(self._agent):
            return await self._agent(input_payload)

        # Case 3: Synchronous `invoke`
        if hasattr(self._agent, "invoke") and callable(self._agent.invoke):
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(
                    None, self._agent.invoke, input_payload
                )
            except TypeError:
                return await loop.run_in_executor(None, self._agent.invoke, input_text)

        # Case 4: Synchronous `run` (Classic LangChain AgentExecutor method)
        if hasattr(self._agent, "run") and callable(self._agent.run):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._agent.run, input_text)

        # Case 5: Standard synchronous callable
        if callable(self._agent):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._agent, input_text)

        raise TypeError(
            f"Object of type '{type(self._agent).__name__}' is not a valid LangChain agent or callable."
        )

    def _convert_result(
        self, input_text: str, raw_result: Any, latency_ms: float
    ) -> AgentOutput:
        """Convert raw LangChain response into standard `AgentOutput`."""
        output_text = ""
        steps: list[TraceStep] = []
        metadata: dict[str, Any] = {}

        if isinstance(raw_result, dict):
            # Parse output string from dict keys commonly used by LangChain
            output_text = str(
                raw_result.get("output")
                or raw_result.get("text")
                or raw_result.get("result")
                or ""
            )

            # Parse intermediate steps (AgentAction, observation tuples)
            intermediate_steps = raw_result.get("intermediate_steps") or []
            steps = self._parse_intermediate_steps(intermediate_steps)

            # Extract token metadata if available
            if "token_usage" in raw_result:
                metadata["token_usage"] = raw_result["token_usage"]

        elif isinstance(raw_result, str):
            output_text = raw_result

        return AgentOutput(
            input=input_text,
            output=output_text,
            steps=steps,
            total_steps=len(steps),
            total_latency_ms=latency_ms,
            metadata=metadata,
        )

    def _parse_intermediate_steps(
        self, intermediate_steps: list[Any]
    ) -> list[TraceStep]:
        """Convert LangChain intermediate_steps into framework `TraceStep` items."""
        steps: list[TraceStep] = []

        for idx, item in enumerate(intermediate_steps, start=1):
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                action_obj, observation = item[0], item[1]

                # Extract tool name, args, and reasoning log from AgentAction object or dict
                tool_name = (
                    getattr(action_obj, "tool", None)
                    or (action_obj.get("tool") if isinstance(action_obj, dict) else None)
                    or "unknown_tool"
                )
                tool_args = getattr(action_obj, "tool_input", None) or (
                    action_obj.get("tool_input") if isinstance(action_obj, dict) else {}
                )
                reasoning = (
                    getattr(action_obj, "log", "")
                    or (action_obj.get("log") if isinstance(action_obj, dict) else "")
                )

                steps.append(
                    TraceStep(
                        step_number=idx,
                        action="use_tool",
                        tool_name=str(tool_name),
                        tool_args=tool_args if isinstance(tool_args, dict) else {"input": tool_args},
                        tool_result=observation,
                        reasoning=str(reasoning),
                    )
                )

        return steps
