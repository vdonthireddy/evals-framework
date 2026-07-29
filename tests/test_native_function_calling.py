"""Unit tests for NativeFunctionCallingAgent and NativeFunctionCallingAdapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.native_agent import NativeFunctionCallingAgent
from evals.adapters.native_agent_adapter import NativeFunctionCallingAdapter


@pytest.mark.asyncio
async def test_native_agent_execution_openai():
    agent = NativeFunctionCallingAgent(provider="openai", model="gpt-4o-mini", api_key="dummy")

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"location": "Tokyo"}'

    mock_msg_1 = MagicMock()
    mock_msg_1.content = None
    mock_msg_1.tool_calls = [mock_tool_call]
    mock_msg_1.model_dump.return_value = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}}],
    }

    mock_msg_2 = MagicMock()
    mock_msg_2.content = "The weather in Tokyo is 32°C."
    mock_msg_2.tool_calls = None

    mock_res_1 = MagicMock()
    mock_res_1.choices = [MagicMock(message=mock_msg_1)]
    mock_res_1.usage.total_tokens = 45

    mock_res_2 = MagicMock()
    mock_res_2.choices = [MagicMock(message=mock_msg_2)]
    mock_res_2.usage.total_tokens = 30

    with patch.object(agent.client.chat.completions, "create", new_callable=AsyncMock, side_effect=[mock_res_1, mock_res_2]):
        trace = await agent.run("What's the weather in Tokyo?")

    assert trace.output == "The weather in Tokyo is 32°C."
    assert trace.total_steps == 2
    assert trace.steps[0].action == "use_tool"
    assert trace.steps[0].tool_name == "get_weather"
    assert trace.steps[0].tool_args == {"location": "Tokyo"}


@pytest.mark.asyncio
async def test_native_agent_execution_anthropic():
    agent = NativeFunctionCallingAgent(provider="anthropic", model="claude-3-5-sonnet", api_key="dummy")

    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.id = "toolu_999"
    mock_block.name = "calculator"
    mock_block.input = {"expression": "25 * 4"}

    mock_res_1 = MagicMock()
    mock_res_1.content = [mock_block]
    mock_res_1.usage.input_tokens = 20
    mock_res_1.usage.output_tokens = 15

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "25 * 4 is 100."

    mock_res_2 = MagicMock()
    mock_res_2.content = [mock_text_block]
    mock_res_2.usage.input_tokens = 30
    mock_res_2.usage.output_tokens = 10

    with patch.object(agent.client.messages, "create", new_callable=AsyncMock, side_effect=[mock_res_1, mock_res_2]):
        trace = await agent.run("Calculate 25 * 4")

    assert trace.output == "25 * 4 is 100."
    assert trace.steps[0].action == "use_tool"
    assert trace.steps[0].tool_name == "calculator"


@pytest.mark.asyncio
async def test_native_adapter():
    adapter = NativeFunctionCallingAdapter(provider="openai", model="gpt-4o-mini", api_key="dummy")
    info = adapter.get_info()

    assert info["name"] == "NativeFunctionCallingAgent"
    assert info["framework"] == "NativeOpenAIFunctionCalling"
