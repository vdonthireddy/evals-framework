"""Composite scorer that combines multiple scorers."""

import asyncio
from typing import Any, Optional

from evals.core.interfaces import AgentOutput, EvalCase, ScoreResult
from evals.scorers.base import BaseScorer
from evals.scorers.deterministic import (
    ContainsKeywordsScorer,
    CostLatencyScorer,
    ExactMatchScorer,
    SafetyScorer,
    ToolArgumentScorer,
    ToolSelectionScorer,
    TrajectoryEfficiencyScorer,
)
from evals.scorers.llm_judge import LLMJudgeScorer


class CompositeScorer(BaseScorer):
    """Combines multiple scorers with configurable weights."""

    def __init__(self, scorers_with_weights: list[tuple[BaseScorer, float]]):
        super().__init__()
        self._threshold = 0.7
        self._scorers_with_weights = scorers_with_weights
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weight for _, weight in self._scorers_with_weights)
        if total_weight > 0:
            self._scorers_with_weights = [
                (scorer, weight / total_weight)
                for scorer, weight in self._scorers_with_weights
            ]

    @property
    def name(self) -> str:
        return "composite"

    @property
    def description(self) -> str:
        names = [s.name for s, _ in self._scorers_with_weights]
        return f"Combines: {', '.join(names)}"

    async def score(self, case: EvalCase, output: AgentOutput) -> ScoreResult:
        if not self._scorers_with_weights:
            return ScoreResult(
                scorer_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                reasoning="No scorers configured.",
            )

        # Run all scorers concurrently
        scorers = [s for s, _ in self._scorers_with_weights]
        tasks = [scorer.score(case, output) for scorer in scorers]
        results: list[ScoreResult] = await asyncio.gather(*tasks)

        # Calculate weighted score
        final_score = 0.0
        all_passed = True
        
        for result, (_, weight) in zip(results, self._scorers_with_weights):
            final_score += result.score * weight
            if not result.passed:
                all_passed = False

        return ScoreResult(
            scorer_name=self.name,
            score=final_score,
            passed=all_passed,  # Must pass ALL individual thresholds
            threshold=self.threshold,
            details={"individual_results": [r.model_dump() for r in results]},
            reasoning=f"Passed {sum(1 for r in results if r.passed)}/{len(results)} individual scorers.",
        )

    # ── Factory methods ─────────────────────────────────────────────

    @classmethod
    def default(cls) -> "CompositeScorer":
        """All deterministic scorers with equal weight."""
        return cls([
            (ToolSelectionScorer(), 1.0),
            (ToolArgumentScorer(), 1.0),
            (TrajectoryEfficiencyScorer(), 1.0),
            (SafetyScorer(), 1.0),
            (ExactMatchScorer(), 1.0),
            (ContainsKeywordsScorer(), 1.0),
            (CostLatencyScorer(), 1.0),
        ])

    @classmethod
    def with_llm_judge(cls, llm_config: dict[str, Any]) -> "CompositeScorer":
        """Deterministic scorers (60%) + LLM judge (40%)."""
        deterministic = cls.default()
        judge = LLMJudgeScorer(
            provider=llm_config["provider"],
            model_name=llm_config["model"],
            api_key=llm_config["api_key"],
        )
        return cls([
            (deterministic, 0.6),
            (judge, 0.4),
        ])

    @classmethod
    def safety_only(cls) -> "CompositeScorer":
        """Only the safety scorer."""
        return cls([
            (SafetyScorer(), 1.0),
        ])

    @classmethod
    def regression(cls) -> "CompositeScorer":
        """Deterministic scorers, but requiring perfect scores."""
        scorers = [
            ToolSelectionScorer(),
            ToolArgumentScorer(),
            TrajectoryEfficiencyScorer(),
            SafetyScorer(),
            ExactMatchScorer(),
            ContainsKeywordsScorer(),
            CostLatencyScorer(),
        ]
        
        # Override thresholds to require perfect scores for regression tests
        for s in scorers:
            s.threshold = 1.0
            
        return cls([(s, 1.0) for s in scorers])
