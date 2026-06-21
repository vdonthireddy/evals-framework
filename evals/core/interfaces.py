"""Abstract interfaces that define the evals framework's contracts.

This is the most important file in the framework. By defining these
interfaces, the evals framework knows nothing about any specific agent.
Any agent can be evaluated as long as someone writes a small adapter
that implements `AgentAdapter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── 12a: TraceStep ──────────────────────────────────────────────────────


class TraceStep(BaseModel):
    """One step in an agent's execution trajectory."""

    step_number: int
    action: str  # e.g., "use_tool", "respond", "clarify", "safety_block"
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Optional[Any] = None
    reasoning: Optional[str] = None
    timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ── 12b: AgentOutput ───────────────────────────────────────────────────


class AgentOutput(BaseModel):
    """Complete output of any agent execution.

    This is provider-agnostic: it doesn't matter whether the underlying
    agent is built with OpenAI, LangChain, CrewAI, or raw Python — as long
    as the adapter converts the agent's native output into this model.
    """

    input: str
    output: str
    steps: list[TraceStep] = Field(default_factory=list)
    total_steps: int = 0
    total_tokens: Optional[int] = None
    total_latency_ms: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── 12c: EvalCase ──────────────────────────────────────────────────────


class EvalCase(BaseModel):
    """One eval test case loaded from a JSONL dataset."""

    id: str
    input: str
    expected_output: Optional[str] = None
    expected_tool_calls: Optional[list[dict[str, Any]]] = None
    expected_outcome: Optional[str] = None
    max_steps: Optional[int] = None
    expected_safety_trigger: Optional[bool] = None
    tags: list[str] = Field(default_factory=list)
    difficulty: str = "easy"
    category: str = "unit"


# ── 12d: ScoreResult ───────────────────────────────────────────────────


class ScoreResult(BaseModel):
    """Output of one scorer applied to one eval case."""

    scorer_name: str
    score: float  # 0.0 – 1.0
    passed: bool
    threshold: float
    reasoning: Optional[str] = None
    details: Optional[dict[str, Any]] = None


# ── 12e: EvalResult ────────────────────────────────────────────────────


class EvalResult(BaseModel):
    """Full result of evaluating one case (agent output + all scores)."""

    case_id: str
    case_input: str
    agent_output: AgentOutput
    scores: list[ScoreResult] = Field(default_factory=list)
    overall_passed: bool = False
    overall_score: float = 0.0
    error: Optional[str] = None


# ── 12f: AgentAdapter ──────────────────────────────────────────────────


class AgentAdapter(ABC):
    """Abstract base class that any agent must implement to be evaluable.

    To evaluate a new agent, write a class that inherits from this and
    implements the three methods below. That's it — everything else in the
    evals framework (datasets, scorers, runner, reporter) works unchanged.

    Example::

        class MyAgentAdapter(AgentAdapter):
            def __init__(self):
                self.agent = MyCustomAgent()

            async def execute(self, input: str) -> AgentOutput:
                result = await self.agent.run(input)
                return AgentOutput(
                    input=input,
                    output=result.answer,
                    steps=[...],
                    total_steps=len(result.steps),
                )

            def reset(self) -> None:
                self.agent.clear_history()

            def get_info(self) -> dict:
                return {"name": "MyAgent", "version": "1.0"}
    """

    @abstractmethod
    async def execute(self, input: str) -> AgentOutput:
        """Run the agent on the given input and return structured output."""

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state between eval cases (clear memory, etc.)."""

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return metadata about the agent (name, model, version, etc.)."""
