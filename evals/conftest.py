"""Pytest integration for running eval cases as standard tests."""

import asyncio
from typing import Any, Iterator

import pytest
from dotenv import load_dotenv

from evals.adapters.example_agent import ExampleAgentAdapter
from evals.core.dataset import EvalDataset
from evals.core.interfaces import AgentOutput, EvalCase
from evals.scorers.composite import CompositeScorer

# Load env vars (for API keys)
load_dotenv()


@pytest.fixture(scope="session")
def agent_adapter() -> ExampleAgentAdapter:
    """Provide the agent adapter for testing."""
    import os
    
    # We use the example agent for these tests
    provider = os.getenv("AGENT_LLM_PROVIDER", "openai")
    model = os.getenv("AGENT_MODEL_NAME", "gpt-4o-mini")
    api_key = os.getenv("AGENT_API_KEY", "dummy-key-for-unit-tests")
    
    return ExampleAgentAdapter(
        provider=provider,
        model=model,
        api_key=api_key,
    )


@pytest.fixture(scope="session")
def composite_scorer() -> CompositeScorer:
    """Provide the default deterministic scorer."""
    return CompositeScorer.default()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically generate a pytest test for each case in the dataset.
    
    This hook looks for test functions that request `eval_case`.
    """
    if "eval_case" in metafunc.fixturenames:
        # Load the full dataset
        # In a real setup, we might parameterize the dataset path
        try:
            dataset = EvalDataset("evals/datasets")
            
            # Create a list of test parameters (case object, and an ID for pytest output)
            cases = list(dataset)
            ids = [f"{case.category}-{case.id}" for case in cases]
            
            metafunc.parametrize("eval_case", cases, ids=ids)
        except Exception as e:
            # If dataset fails to load, create a dummy failing case to report the error
            metafunc.parametrize(
                "eval_case", 
                [EvalCase(id="error", input=f"Failed to load dataset: {e}", category="error")],
                ids=["dataset-load-error"]
            )


@pytest.fixture
def run_eval(agent_adapter: ExampleAgentAdapter, composite_scorer: CompositeScorer) -> Any:
    """Fixture that returns a function to evaluate a specific case."""
    
    async def _run(case: EvalCase) -> None:
        agent_adapter.reset()
        
        # Execute the agent
        if case.is_multi_turn:
            from evals.core.interfaces import TraceStep
            all_steps: list[TraceStep] = []
            combined_output_text = []
            total_tokens = 0
            total_latency = 0.0
            aggregated_expected_tools = []
            last_metadata = {}
            
            if not case.turns:
                raise ValueError("Case is marked multi-turn but has no turns.")
                
            for turn_idx, turn in enumerate(case.turns):
                turn_output = await agent_adapter.execute(turn.input)
                
                base_step = len(all_steps)
                for step in turn_output.steps:
                    step.step_number += base_step
                    all_steps.append(step)
                    
                combined_output_text.append(f"Turn {turn_idx+1} Output: {turn_output.output}")
                
                if turn_output.total_tokens is not None:
                    total_tokens += turn_output.total_tokens
                if turn_output.total_latency_ms is not None:
                    total_latency += turn_output.total_latency_ms
                    
                if turn.expected_tool_calls:
                    aggregated_expected_tools.extend(turn.expected_tool_calls)
                    
                last_metadata = turn_output.metadata
                
            case.expected_tool_calls = aggregated_expected_tools
            if case.turns[-1].expected_outcome:
                case.expected_outcome = case.turns[-1].expected_outcome
                
            output = AgentOutput(
                input=case.input,
                output="\n".join(combined_output_text),
                steps=all_steps,
                total_steps=len(all_steps),
                total_tokens=total_tokens if total_tokens > 0 else None,
                total_latency_ms=total_latency if total_latency > 0 else None,
                metadata=last_metadata
            )
        else:
            output = await agent_adapter.execute(case.input)
        
        # Score the output
        result = await composite_scorer.score(case, output)
        
        # If it failed, construct a detailed error message
        if not result.passed:
            failed_scorers = [s for s in result.details.get("individual_results", []) if not s.get("passed")]
            
            msg = [
                f"\nEval Case Failed: {case.id}",
                f"Input: {case.input}",
                f"Output: {output.output}",
                "\nFailed Scorers:"
            ]
            
            for fs in failed_scorers:
                msg.append(f"  - {fs['scorer_name']} ({fs['score']:.2f}): {fs['reasoning']}")
                if fs.get("details"):
                    msg.append(f"    Details: {fs['details']}")
                    
            pytest.fail("\n".join(msg))
            
    return _run
