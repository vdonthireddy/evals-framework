"""Unit tests for verified bug fixes."""

import pytest
import json
from agent.memory import ConversationMemory
from agent.planner import TaskPlanner, PlanStep
from evals.core.interfaces import EvalCase, Turn, AgentOutput, TraceStep
from evals.core.runner import EvalRunner, EvalConfig
from evals.core.dataset import EvalDataset
from evals.scorers.deterministic import ToolSelectionScorer
from evals.scorers.llm_judge import LLMJudgeScorer


def test_get_last_n_messages_no_duplication():
    memory = ConversationMemory(system_prompt="You are an assistant.")
    memory.add_user_message("Hello")
    memory.add_assistant_message("Hi there!")

    # Total messages: 3 (system, user, assistant). Request n=5 (greater than length)
    last_5 = memory.get_last_n_messages(5)
    
    roles = [m["role"] for m in last_5]
    assert roles == ["system", "user", "assistant"], f"System prompt was duplicated: {roles}"
    assert len(last_5) == 3


def test_greedy_json_parsing_in_planner():
    class DummyClient:
        pass

    planner = TaskPlanner(llm_client=DummyClient(), tools=[], provider="openai")
    
    # LLM response with markdown and nested JSON braces in tool_args
    raw_llm_response = (
        "Here is the plan:\n"
        "```json\n"
        "{\n"
        '  "action": "use_tool",\n'
        '  "tool_name": "calculator",\n'
        '  "tool_args": {"expression": "42 * 10"},\n'
        '  "reasoning": "calculate 42 * 10"\n'
        "}\n"
        "```"
    )
    
    planner._tools = {"calculator": None}
    step = planner._parse_response(raw_llm_response)
    assert step.action == "use_tool"
    assert step.tool_name == "calculator"
    assert step.tool_args == {"expression": "42 * 10"}


def test_greedy_json_parsing_in_llm_judge():
    class DummyScorer(LLMJudgeScorer):
        def __init__(self):
            self.provider = "openai"
            self.model_name = "gpt-4o"

    scorer = DummyScorer()
    
    raw_judge_response = (
        "```json\n"
        "{\n"
        '  "scores": {"correctness": 5, "details": {"sub_score": 10}},\n'
        '  "reasoning": "Excellent work",\n'
        '  "overall": 5\n'
        "}\n"
        "```"
    )
    
    result = scorer._parse_json_response(raw_judge_response)
    assert result["scores"]["correctness"] == 5
    assert result["scores"]["details"]["sub_score"] == 10


def test_is_multi_turn_single_turn_case():
    case = EvalCase(
        id="single_turn_in_list",
        input="test",
        turns=[Turn(input="turn 1")]
    )
    assert case.is_multi_turn is True


@pytest.mark.asyncio
async def test_tool_selection_no_bonus_on_empty_expected():
    scorer = ToolSelectionScorer()
    
    # Case with empty expected tool calls (not None)
    case = EvalCase(id="empty_expected", input="", expected_tool_calls=[])
    out = AgentOutput(
        input="",
        output="",
        steps=[TraceStep(step_number=1, action="use_tool", tool_name="web_search")]
    )
    
    res = await scorer.score(case, out)
    # Expected 0 tools, actual 1 tool -> matches 0/1 -> score should be 0.0, NOT 0.1
    assert res.score == 0.0


@pytest.mark.asyncio
async def test_multi_turn_output_newline_joining():
    from evals.adapters.example_agent import AgentAdapter
    
    class MockStatefulAdapter(AgentAdapter):
        def reset(self): pass
        def get_info(self): return {"name": "mock"}
        async def execute(self, input_str: str) -> AgentOutput:
            return AgentOutput(
                input=input_str,
                output=f"Answer to {input_str}",
                steps=[TraceStep(step_number=1, action="respond", response=f"Answer to {input_str}")],
                total_tokens=5,
                total_latency_ms=10.0
            )

    case = EvalCase(
        id="multi_turn_lines",
        input="first",
        turns=[Turn(input="first"), Turn(input="second")]
    )
    
    dataset = EvalDataset([case])
    config = EvalConfig(scorer_config="safety_only")
    runner = EvalRunner(MockStatefulAdapter(), dataset, config)
    
    report = await runner.run()
    output_text = report.results[0].agent_output.output
    
    # Must use actual newline character, not literal string \n
    assert "\n" in output_text
    assert "\\n" not in output_text
    assert "Turn 1 Output: Answer to first\nTurn 2 Output: Answer to second" in output_text


def test_short_system_prompt_fragments():
    from agent.safety import SafetyFilter
    filter_inst = SafetyFilter(system_prompt="Short prompt test.")
    assert len(filter_inst._system_prompt_fragments) == 1
    assert filter_inst._system_prompt_fragments[0] == "short prompt test."


def test_reporter_extract_sub_scores():
    from evals.core.reporter import EvalReporter
    from evals.core.interfaces import ScoreResult
    
    comp_score = ScoreResult(
        scorer_name="composite",
        score=0.8,
        passed=True,
        threshold=0.7,
        details={
            "individual_results": [
                {"scorer_name": "tool_selection", "score": 1.0, "passed": True, "threshold": 0.8},
                {"scorer_name": "safety", "score": 0.6, "passed": False, "threshold": 1.0},
            ]
        }
    )
    
    unwrapped = EvalReporter._extract_sub_scores([comp_score])
    assert len(unwrapped) == 2
    assert unwrapped[0].scorer_name == "tool_selection"
    assert unwrapped[1].scorer_name == "safety"
    assert unwrapped[1].passed is False


def test_reporter_compare_filtered_cases():
    from evals.core.reporter import EvalReporter
    from evals.core.runner import EvalRunReport, EvalConfig
    from evals.core.interfaces import EvalResult, AgentOutput
    
    res1 = EvalResult(case_id="1", case_input="a", agent_output=AgentOutput(input="a", output="b"), overall_passed=True, overall_score=1.0)
    res2 = EvalResult(case_id="2", case_input="x", agent_output=AgentOutput(input="x", output="y"), overall_passed=True, overall_score=1.0)
    
    baseline = EvalRunReport(
        run_id="r1", agent_info={}, dataset_info={}, config=EvalConfig(),
        results=[res1, res2], summary={"total_cases": 2, "passed": 2, "failed": 0, "error_count": 0, "average_score": 1.0}
    )
    
    # Current run evaluated ONLY case 1 (case 2 was filtered out)
    current = EvalRunReport(
        run_id="r2", agent_info={}, dataset_info={}, config=EvalConfig(),
        results=[res1], summary={"total_cases": 1, "passed": 1, "failed": 0, "error_count": 0, "average_score": 1.0}
    )
    
    diff = EvalReporter.compare(baseline, current)
    # Case 2 should NOT be flagged as a regression since it was not run in current
    assert "Regressions" not in diff


def test_planner_action_normalization():
    class DummyClient: pass
    planner = TaskPlanner(llm_client=DummyClient(), tools=[], provider="openai")
    planner._tools = {"calculator": None}
    
    # LLM outputs action: "calculator" directly instead of action: "use_tool"
    raw_llm_response = '{"action": "calculator", "tool_args": {"expression": "2+2"}}'
    step = planner._parse_response(raw_llm_response)
    assert step.action == "use_tool"
    assert step.tool_name == "calculator"
    assert step.tool_args == {"expression": "2+2"}


@pytest.mark.asyncio
async def test_contains_keywords_numeric_extraction():
    from evals.scorers.deterministic import ContainsKeywordsScorer
    scorer = ContainsKeywordsScorer()
    
    case = EvalCase(id="num_test", input="math", expected_outcome="16")
    output = AgentOutput(input="math", output="The answer is 16.")
    res = await scorer.score(case, output)
    assert res.passed is True
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_tool_argument_type_normalization():
    from evals.scorers.deterministic import ToolArgumentScorer
    scorer = ToolArgumentScorer()
    
    # Expected argument is integer 25, actual is string "25"
    case = EvalCase(
        id="arg_type_test",
        input="test",
        expected_tool_calls=[{"tool_name": "calc", "arguments": {"value": 25}}]
    )
    output = AgentOutput(
        input="test",
        output="done",
        steps=[TraceStep(step_number=1, action="use_tool", tool_name="calc", tool_args={"value": "25"})]
    )
    res = await scorer.score(case, output)
    assert res.passed is True
    assert res.score == 1.0

