import pytest
import asyncio
from evals.core.interfaces import AgentOutput, EvalCase, TraceStep
from evals.scorers.deterministic import (
    ToolSelectionScorer,
    ToolArgumentScorer,
    TrajectoryEfficiencyScorer,
    SafetyScorer,
    ExactMatchScorer,
    ContainsKeywordsScorer
)


@pytest.mark.asyncio
async def test_tool_selection_scorer():
    scorer = ToolSelectionScorer()
    
    # Exact match
    case = EvalCase(id="1", input="", expected_tool_calls=[{"tool_name": "web_search"}])
    out = AgentOutput(input="", output="", steps=[TraceStep(step_number=1, action="use_tool", tool_name="web_search")])
    res = await scorer.score(case, out)
    assert res.score == 1.0
    
    # Partial match
    case = EvalCase(id="2", input="", expected_tool_calls=[{"tool_name": "web_search"}, {"tool_name": "calculator"}])
    out = AgentOutput(input="", output="", steps=[TraceStep(step_number=1, action="use_tool", tool_name="web_search")])
    res = await scorer.score(case, out)
    assert 0.4 < res.score < 0.6  # 1 out of 2 matched
    
    # Empty expected (returns 1.0)
    case = EvalCase(id="3", input="", expected_tool_calls=None)
    out = AgentOutput(input="", output="", steps=[TraceStep(step_number=1, action="use_tool", tool_name="web_search")])
    res = await scorer.score(case, out)
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_tool_argument_scorer():
    scorer = ToolArgumentScorer()
    
    # Exact and fuzzy match
    case = EvalCase(id="1", input="", expected_tool_calls=[
        {"tool_name": "calc", "arguments": {"x": 5, "op": "add "}}
    ])
    out = AgentOutput(input="", output="", steps=[
        TraceStep(step_number=1, action="use_tool", tool_name="calc", tool_args={"x": 5, "op": "ADD"})
    ])
    res = await scorer.score(case, out)
    assert res.score == 1.0
    
    # Missing arg
    out2 = AgentOutput(input="", output="", steps=[
        TraceStep(step_number=1, action="use_tool", tool_name="calc", tool_args={"x": 5})
    ])
    res2 = await scorer.score(case, out2)
    assert res2.score == 0.5


@pytest.mark.asyncio
async def test_trajectory_efficiency_scorer():
    scorer = TrajectoryEfficiencyScorer()
    
    case = EvalCase(id="1", input="", max_steps=3)
    
    # Optimal
    out = AgentOutput(input="", output="", total_steps=2)
    res = await scorer.score(case, out)
    assert res.score == 1.0
    
    # Over steps
    out2 = AgentOutput(input="", output="", total_steps=6)
    res2 = await scorer.score(case, out2)
    assert res2.score == 0.5


@pytest.mark.asyncio
async def test_safety_scorer():
    scorer = SafetyScorer()
    
    # Triggered correctly
    case = EvalCase(id="1", input="", expected_safety_trigger=True)
    out = AgentOutput(input="", output="", metadata={"safety_triggered": True})
    res = await scorer.score(case, out)
    assert res.score == 1.0
    
    # Falsely triggered
    case2 = EvalCase(id="2", input="", expected_safety_trigger=False)
    res2 = await scorer.score(case2, out)
    assert res2.score == 0.0


@pytest.mark.asyncio
async def test_exact_match_scorer():
    scorer = ExactMatchScorer()
    
    case = EvalCase(id="1", input="", expected_output="Hello World!")
    
    # Case insensitivity and whitespace normalization
    out = AgentOutput(input="", output=" hello  world! ")
    res = await scorer.score(case, out)
    assert res.score == 1.0
    
    # Mismatch
    out2 = AgentOutput(input="", output="Hi World")
    res2 = await scorer.score(case, out2)
    assert res2.score == 0.0


@pytest.mark.asyncio
async def test_contains_keywords_scorer():
    scorer = ContainsKeywordsScorer()
    
    case = EvalCase(id="1", input="", expected_outcome="Tokyo weather is raining heavily")
    
    # Should match Tokyo, weather, raining, heavily
    out = AgentOutput(input="", output="It's currently raining quite heavily in the city of tokyo.")
    res = await scorer.score(case, out)
    assert res.score > 0.7


@pytest.mark.asyncio
async def test_llm_judge_scorer_ollama():
    from unittest.mock import AsyncMock, MagicMock, patch
    from evals.scorers.llm_judge import LLMJudgeScorer

    judge = LLMJudgeScorer(provider="ollama", model_name="llama3.2", api_key="dummy")
    mock_msg = MagicMock()
    mock_msg.content = '{"scores": {"correctness": 5, "helpfulness": 5, "safety": 5, "efficiency": 5}, "overall": 5, "reasoning": "Excellent"}'
    mock_res = MagicMock(choices=[MagicMock(message=mock_msg)])

    case = EvalCase(id="1", input="What is 2+2?", expected_outcome="4")
    out = AgentOutput(input="What is 2+2?", output="4")

    with patch.object(judge.client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_res):
        res = await judge.score(case, out)

    assert res.passed is True
    assert res.score == 1.0

