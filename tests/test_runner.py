import pytest
import asyncio
from typing import Any

from evals.core.runner import EvalConfig, EvalRunner
from evals.core.dataset import EvalDataset
from evals.core.interfaces import AgentAdapter, AgentOutput, EvalCase


class MockAdapter(AgentAdapter):
    def __init__(self, delay: float = 0.0, error: bool = False):
        self.delay = delay
        self.error = error
        self.reset_count = 0

    async def execute(self, input: str) -> AgentOutput:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
            
        if self.error:
            raise RuntimeError("Mock adapter error")
            
        return AgentOutput(input=input, output="mock output", steps=[], total_steps=1)

    def reset(self) -> None:
        self.reset_count += 1

    def get_info(self) -> dict[str, Any]:
        return {"name": "mock"}


@pytest.mark.asyncio
async def test_runner_happy_path(tmp_path):
    cases = [
        EvalCase(id="1", input="t1", tags=["test"], category="unit"),
        EvalCase(id="2", input="t2", tags=["test"], category="unit")
    ]
    ds = EvalDataset(cases)
    config = EvalConfig(output_dir=str(tmp_path))
    adapter = MockAdapter()
    
    runner = EvalRunner(adapter, ds, config)
    report = await runner.run()
    
    assert report.summary["total_cases"] == 2
    assert adapter.reset_count == 2
    assert (tmp_path / f"{report.run_id}.json").exists()


@pytest.mark.asyncio
async def test_runner_timeout(tmp_path):
    cases = [EvalCase(id="1", input="t1", category="unit")]
    ds = EvalDataset(cases)
    
    # Timeout after 1s
    config = EvalConfig(output_dir=str(tmp_path), timeout_seconds=1, retry_on_error=False)
    
    # Adapter takes 2s
    adapter = MockAdapter(delay=2.0)
    
    runner = EvalRunner(adapter, ds, config)
    report = await runner.run()
    
    # Should be an error
    assert report.summary["error_count"] == 1
    assert "Timeout" in report.results[0].error or "TimeoutError" in report.results[0].error


@pytest.mark.asyncio
async def test_runner_adapter_error(tmp_path):
    cases = [EvalCase(id="1", input="t1", category="unit")]
    ds = EvalDataset(cases)
    
    # No retries to speed up test
    config = EvalConfig(output_dir=str(tmp_path), retry_on_error=False)
    
    # Adapter throws
    adapter = MockAdapter(error=True)
    
    runner = EvalRunner(adapter, ds, config)
    report = await runner.run()
    
    assert report.summary["error_count"] == 1
    assert "Mock adapter error" in report.results[0].error
    assert report.results[0].overall_passed is False
