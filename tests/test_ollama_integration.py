"""Integration test for verifying direct connectivity to local Ollama (llama3.2)."""

import pytest
import httpx
from openai import AsyncOpenAI

from evals.adapters.example_agent_adapter import ExampleAgentAdapter


async def check_ollama_online() -> bool:
    """Check if local Ollama server is running on localhost:11434."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.get("http://localhost:11434/api/tags")
            return res.status_code == 200
    except Exception:
        return False


@pytest.mark.asyncio
async def test_direct_ollama_completion():
    """Test raw completions directly against local Ollama OpenAI endpoint."""
    if not await check_ollama_online():
        pytest.skip("Ollama is not running locally on port 11434.")

    client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    
    response = await client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "user", "content": "What is the square root of 256?"}
        ],
        temperature=0.0
    )
    
    output = response.choices[0].message.content
    print("\n[Direct Ollama Raw Response]:", output)
    assert output is not None
    assert len(output) > 0


@pytest.mark.asyncio
async def test_adapter_ollama_agent_execution():
    """Test executing a prompt through ExampleAgentAdapter with provider='ollama'."""
    if not await check_ollama_online():
        pytest.skip("Ollama is not running locally on port 11434.")

    adapter = ExampleAgentAdapter(
        provider="ollama",
        model="llama3.2",
    )
    
    output = await adapter.execute("What is the square root of 256?")
    print("\n[Adapter Ollama Final Response]:", output.output)
    print("[Adapter Steps]:", output.steps)
    
    assert output.output is not None
    assert len(output.output) > 0
