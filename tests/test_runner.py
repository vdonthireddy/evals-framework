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


@pytest.mark.asyncio
async def test_runner_multi_turn():
    from evals.core.interfaces import AgentOutput, TraceStep, Turn

    class StatefulMockAdapter(AgentAdapter):
        def __init__(self):
            self.turn_count = 0
            
        def reset(self):
            self.turn_count = 0
            
        def get_info(self):
            return {"name": "stateful"}
            
        async def execute(self, input: str):
            self.turn_count += 1
            return AgentOutput(
                input=input,
                output=f"response {self.turn_count}",
                steps=[TraceStep(step_number=1, action="test")],
                total_tokens=10,
                total_latency_ms=100.0
            )

    multi_turn_case = EvalCase(
        id="multi_1",
        input="ignore",
        turns=[
            Turn(input="turn 1"),
            Turn(input="turn 2", expected_outcome="response 2"),
        ]
    )
    
    dataset = EvalDataset([multi_turn_case])
    config = EvalConfig(presets=["regression"])
    runner = EvalRunner(StatefulMockAdapter(), dataset, config)
    
    report = await runner.run()
    
    # We should have aggregated 2 steps and combined latencies/tokens
    res = report.results[0]
    out = res.agent_output
    assert len(out.steps) == 2
    assert out.total_tokens == 20
    assert out.total_latency_ms == 200.0
    assert "response 1" in out.output
    assert "response 2" in out.output
