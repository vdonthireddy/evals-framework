"""Unit tests for the generic LangChainAdapter."""

from dataclasses import dataclass
from typing import Any
import pytest

from evals.adapters.langchain_adapter import LangChainAdapter
from evals.core.interfaces import AgentOutput


@dataclass
class MockAgentAction:
    tool: str
    tool_input: dict[str, Any]
    log: str


class MockLangChainAgentExecutor:
    """Simulates a LangChain AgentExecutor returning dict results with intermediate steps."""

    def __init__(self) -> None:
        self.reset_called = False

    async def ainvoke(self, input_data: dict[str, Any]) -> dict[str, Any]:
        query = input_data.get("input", "")
        if "weather" in query:
            action = MockAgentAction(
                tool="weather_lookup",
                tool_input={"location": "Tokyo"},
                log="I need to check the weather in Tokyo.",
            )
            return {
                "input": query,
                "output": "The weather in Tokyo is sunny and 22°C.",
                "intermediate_steps": [(action, {"temperature": "22C", "condition": "Sunny"})],
            }
        return {"input": query, "output": f"Processed: {query}", "intermediate_steps": []}

    def reset(self) -> None:
        self.reset_called = True


class MockSimpleSyncRunnable:
    """Simulates a simple synchronous LangChain runnable returning a string."""

    def invoke(self, input_data: dict[str, Any]) -> str:
        return f"Echo: {input_data['input']}"


@pytest.mark.asyncio
async def test_langchain_adapter_agent_executor():
    mock_agent = MockLangChainAgentExecutor()
    adapter = LangChainAdapter(agent=mock_agent, name="TestLangChainAgent")

    output: AgentOutput = await adapter.execute("What's the weather in Tokyo?")

    assert output.input == "What's the weather in Tokyo?"
    assert "Tokyo is sunny" in output.output
    assert output.total_steps == 1
    assert output.steps[0].tool_name == "weather_lookup"
    assert output.steps[0].tool_args == {"location": "Tokyo"}
    assert output.steps[0].tool_result == {"temperature": "22C", "condition": "Sunny"}
    assert "check the weather" in output.steps[0].reasoning

    # Check info
    info = adapter.get_info()
    assert info["name"] == "TestLangChainAgent"
    assert info["framework"] == "LangChain"
    assert info["agent_class"] == "MockLangChainAgentExecutor"


@pytest.mark.asyncio
async def test_langchain_adapter_sync_runnable():
    mock_runnable = MockSimpleSyncRunnable()
    adapter = LangChainAdapter(agent=mock_runnable, name="SimpleRunnable")

    output: AgentOutput = await adapter.execute("Hello LangChain")

    assert output.input == "Hello LangChain"
    assert output.output == "Echo: Hello LangChain"
    assert output.total_steps == 0


@pytest.mark.asyncio
async def test_langchain_adapter_reset_callback():
    was_reset = False

    def custom_reset():
        nonlocal was_reset
        was_reset = True

    mock_agent = MockLangChainAgentExecutor()
    adapter = LangChainAdapter(agent=mock_agent, reset_fn=custom_reset)

    adapter.reset()
    assert was_reset is True
