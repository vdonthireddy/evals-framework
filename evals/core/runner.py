"""Evaluation execution engine."""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from evals.core.dataset import EvalDataset
from evals.core.interfaces import AgentAdapter, EvalCase, EvalResult, ScoreResult
from evals.scorers.composite import CompositeScorer

logger = logging.getLogger(__name__)


# ── Config & Models ───────────────────────────────────────────────────


class EvalConfig(BaseModel):
    """Configuration for an eval run."""

    max_concurrency: int = 5
    timeout_seconds: int = 60
    retry_on_error: bool = True
    scorer_config: str = "default"
    llm_judge_config: Optional[dict[str, Any]] = None
    output_dir: str = "evals/results"
    run_id: Optional[str] = None
    tags_filter: Optional[list[str]] = None
    category_filter: Optional[str] = None


class EvalRunReport(BaseModel):
    """Complete report of an eval run."""

    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_info: dict[str, Any]
    dataset_info: dict[str, Any]
    config: EvalConfig
    results: list[EvalResult]
    summary: dict[str, Any]

    def to_json(self) -> str:
        # Avoid pydantic datetime serialization issues by dumping via model_dump_json
        return self.model_dump_json(indent=2)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ── Runner ────────────────────────────────────────────────────────────


class EvalRunner:
    """Executes evaluation cases against an agent and scores the results."""

    def __init__(self, adapter: AgentAdapter, dataset: EvalDataset, config: EvalConfig):
        self.adapter = adapter
        self.dataset = dataset
        self.config = config

        if not self.config.run_id:
            self.config.run_id = f"run-{uuid.uuid4().hex[:8]}"

    async def run(self) -> EvalRunReport:
        """Run the evaluation."""
        logger.info(f"Starting eval run {self.config.run_id}")

        # Apply filters
        filtered_ds = self.dataset
        if self.config.category_filter:
            filtered_ds = filtered_ds.filter_by_category(self.config.category_filter)
        if self.config.tags_filter:
            filtered_ds = filtered_ds.filter_by_tags(self.config.tags_filter)

        total_cases = len(filtered_ds)
        logger.info(f"Loaded {total_cases} cases after filtering.")

        if total_cases == 0:
            logger.warning("No cases to evaluate!")
            return self._build_empty_report()

        # Initialize scorer
        scorer = self._get_scorer()

        # Concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        
        # Shared progress counter
        completed = 0
        
        async def evaluate_case_with_semaphore(case: EvalCase) -> EvalResult:
            nonlocal completed
            async with semaphore:
                result = await self._evaluate_case(case, scorer)
                completed += 1
                print(f"Evaluated {completed}/{total_cases} cases...")
                return result

        # Run all tasks
        tasks = [evaluate_case_with_semaphore(case) for case in filtered_ds]
        results = await asyncio.gather(*tasks)

        # Build report
        report = self._build_report(filtered_ds, results)

        # Save to disk
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.config.run_id}.json"
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
            
        logger.info(f"Report saved to {out_path}")
        return report

    def _get_scorer(self) -> CompositeScorer:
        """Instantiate the correct composite scorer based on config."""
        preset = self.config.scorer_config.lower()
        if preset == "with_llm_judge":
            if not self.config.llm_judge_config:
                raise ValueError("llm_judge_config is required when using 'with_llm_judge' scorer")
            return CompositeScorer.with_llm_judge(self.config.llm_judge_config)
        elif preset == "safety_only":
            return CompositeScorer.safety_only()
        elif preset == "regression":
            return CompositeScorer.regression()
        else:
            return CompositeScorer.default()

    async def _evaluate_case(self, case: EvalCase, scorer: CompositeScorer) -> EvalResult:
        """Evaluate a single case (with retries and timeouts)."""
        
        async def _run_once() -> EvalResult:
            self.adapter.reset()
            try:
                if case.is_multi_turn:
                    output = await asyncio.wait_for(
                        self._run_multi_turn(case),
                        timeout=self.config.timeout_seconds
                    )
                else:
                    output = await asyncio.wait_for(
                        self.adapter.execute(case.input),
                        timeout=self.config.timeout_seconds
                    )
                score_result = await scorer.score(case, output)
                return EvalResult(
                    case_id=case.id,
                    case_input=case.input,
                    agent_output=output,
                    scores=[score_result],
                    overall_passed=score_result.passed,
                    overall_score=score_result.score,
                    error=None
                )
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                logger.error(f"Error evaluating case {case.id}: {e}", exc_info=True)
                from evals.core.interfaces import AgentOutput
                empty_output = AgentOutput(
                    input=case.input, output="", steps=[], metadata={"error": str(e)}
                )
                return EvalResult(
                    case_id=case.id,
                    case_input=case.input,
                    agent_output=empty_output,
                    scores=[],
                    overall_passed=False,
                    overall_score=0.0,
                    error=str(e)
                )

        try:
            return await _run_once()
        except Exception as e:
            error_str = "TimeoutError" if isinstance(e, asyncio.TimeoutError) else str(e)
            if self.config.retry_on_error and not isinstance(e, asyncio.TimeoutError):
                logger.info(f"Retrying case {case.id} after error...")
                try:
                    return await _run_once()
                except Exception as e2:
                    error_str2 = "TimeoutError" if isinstance(e2, asyncio.TimeoutError) else str(e2)
                    return self._build_error_result(case, error_str2)
            return self._build_error_result(case, error_str)

    async def _run_multi_turn(self, case: EvalCase) -> 'AgentOutput':
        """Execute a multi-turn conversation case."""
        from evals.core.interfaces import AgentOutput, TraceStep
        
        all_steps: list[TraceStep] = []
        combined_output_text = []
        total_tokens = 0
        total_latency = 0.0
        
        # We aggregate all expected tool calls across turns so scorers like 
        # ToolSelectionScorer can see everything we expected the agent to do
        aggregated_expected_tools = []
        last_metadata = {}
        
        if not case.turns:
            raise ValueError("Case is marked multi-turn but has no turns.")
            
        for turn_idx, turn in enumerate(case.turns):
            # Execute turn
            turn_output = await self.adapter.execute(turn.input)
            
            # Combine steps (adjusting step numbers so they increment globally)
            base_step = len(all_steps)
            for step in turn_output.steps:
                step.step_number += base_step
                all_steps.append(step)
                
            combined_output_text.append(f"Turn {turn_idx+1} Output: {turn_output.output}")
            
            # Accumulate metrics
            if turn_output.total_tokens is not None:
                total_tokens += turn_output.total_tokens
            if turn_output.total_latency_ms is not None:
                total_latency += turn_output.total_latency_ms
                
            # Aggregate expectations
            if turn.expected_tool_calls:
                aggregated_expected_tools.extend(turn.expected_tool_calls)
                
            last_metadata = turn_output.metadata
            
        # Update the case object temporarily so scorers see the aggregated expectations
        case.expected_tool_calls = aggregated_expected_tools
        # The final expected outcome should be checked against the final output
        if case.turns[-1].expected_outcome:
            case.expected_outcome = case.turns[-1].expected_outcome
            
        return AgentOutput(
            input=case.input,  # Original first input
            output="\\n".join(combined_output_text),
            steps=all_steps,
            total_steps=len(all_steps),
            total_tokens=total_tokens if total_tokens > 0 else None,
            total_latency_ms=total_latency if total_latency > 0 else None,
            metadata=last_metadata
        )

    def _build_error_result(self, case: EvalCase, error_msg: str) -> EvalResult:
        """Helper to build a failed result when the agent crashes or times out."""
        from evals.core.interfaces import AgentOutput
        
        empty_output = AgentOutput(
            input=case.input, output="", steps=[], metadata={"error": error_msg}
        )
        return EvalResult(
            case_id=case.id,
            case_input=case.input,
            agent_output=empty_output,
            scores=[],
            overall_passed=False,
            overall_score=0.0,
            error=error_msg
        )

    def _build_report(self, dataset: EvalDataset, results: list[EvalResult]) -> EvalRunReport:
        """Aggregate results into a final report."""
        passed = sum(1 for r in results if r.overall_passed)
        failed = sum(1 for r in results if not r.overall_passed and not r.error)
        errors = sum(1 for r in results if r.error is not None)
        
        total = len(results)
        
        # Averages
        avg_score = sum(r.overall_score for r in results) / total if total > 0 else 0.0
        
        # By category/tag/difficulty
        scores_by_tag: dict[str, list[float]] = {}
        scores_by_cat: dict[str, list[float]] = {}
        scores_by_diff: dict[str, list[float]] = {}
        
        for r in results:
            case = dataset.get_case(r.case_id)
            if not case:
                continue
                
            cat = case.category
            scores_by_cat.setdefault(cat, []).append(r.overall_score)
            
            diff = case.difficulty
            scores_by_diff.setdefault(diff, []).append(r.overall_score)
            
            for tag in case.tags:
                scores_by_tag.setdefault(tag, []).append(r.overall_score)
                
        # Averages helpers
        def avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return EvalRunReport(
            run_id=self.config.run_id or "unknown",
            agent_info=self.adapter.get_info(),
            dataset_info=dataset.summary(),
            config=self.config,
            results=results,
            summary={
                "total_cases": total,
                "passed": passed,
                "failed": failed,
                "error_count": errors,
                "overall_pass_rate": passed / total if total > 0 else 0.0,
                "average_score": avg_score,
                "scores_by_category": {k: avg(v) for k, v in scores_by_cat.items()},
                "scores_by_difficulty": {k: avg(v) for k, v in scores_by_diff.items()},
                "scores_by_tag": {k: avg(v) for k, v in scores_by_tag.items()},
                "average_steps": avg([r.agent_output.total_steps for r in results])
            }
        )

    def _build_empty_report(self) -> EvalRunReport:
        return EvalRunReport(
            run_id=self.config.run_id or "unknown",
            agent_info=self.adapter.get_info(),
            dataset_info=self.dataset.summary(),
            config=self.config,
            results=[],
            summary={
                "total_cases": 0, "passed": 0, "failed": 0, "error_count": 0,
                "overall_pass_rate": 0.0, "average_score": 0.0
            }
        )
