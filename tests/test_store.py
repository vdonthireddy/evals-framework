"""Unit tests for SQLiteEvalStore and run comparison engine."""

import os
import tempfile
import pytest
from datetime import datetime, timezone

from evals.core.interfaces import AgentOutput, EvalCase, ScoreResult
from evals.core.runner import EvalConfig, EvalResult, EvalRunReport
from evals.store.sqlite_store import SQLiteEvalStore


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def create_sample_report(run_id: str, provider: str, model: str, pass_rate: float, score: float) -> EvalRunReport:
    now = datetime.now(timezone.utc)
    return EvalRunReport(
        run_id=run_id,
        timestamp=now,
        agent_info={"name": "TestAgent", "provider": provider, "model": model},
        dataset_info={"total_cases": 2},
        config=EvalConfig(run_id=run_id),
        results=[
            EvalResult(
                case_id="case_1",
                case_input="What's 2+2?",
                agent_output=AgentOutput(input="What's 2+2?", output="4", total_steps=1, total_latency_ms=10.0, total_tokens=10),
                scores=[ScoreResult(scorer_name="exact_match", score=score, passed=pass_rate >= 0.5, threshold=1.0)],
                overall_passed=pass_rate >= 0.5,
                overall_score=score,
            ),
            EvalResult(
                case_id="case_2",
                case_input="What's the capital of France?",
                agent_output=AgentOutput(input="What's the capital of France?", output="Paris", total_steps=1, total_latency_ms=15.0, total_tokens=12),
                scores=[ScoreResult(scorer_name="exact_match", score=score, passed=pass_rate >= 1.0, threshold=1.0)],
                overall_passed=pass_rate >= 1.0,
                overall_score=score,
            ),
        ],
        summary={
            "total_cases": 2,
            "passed": 2 if pass_rate >= 1.0 else (1 if pass_rate >= 0.5 else 0),
            "failed": 0 if pass_rate >= 1.0 else (1 if pass_rate >= 0.5 else 2),
            "error_count": 0,
            "overall_pass_rate": pass_rate,
            "average_score": score,
            "average_steps": 1.0,
        },
    )


def test_sqlite_store_save_and_get(temp_db):
    store = SQLiteEvalStore(db_path=temp_db)
    report = create_sample_report("run_001", "openai", "gpt-4o-mini", 1.0, 0.95)
    
    run_id = store.save_run(report)
    assert run_id == "run_001"
    
    retrieved = store.get_run("run_001")
    assert retrieved is not None
    assert retrieved.run_id == "run_001"
    assert retrieved.agent_info["model"] == "gpt-4o-mini"
    assert len(retrieved.results) == 2


def test_sqlite_store_list_and_delete(temp_db):
    store = SQLiteEvalStore(db_path=temp_db)
    report1 = create_sample_report("run_001", "openai", "gpt-4o-mini", 1.0, 0.95)
    report2 = create_sample_report("run_002", "anthropic", "claude-3-5-sonnet", 0.5, 0.70)
    
    store.save_run(report1)
    store.save_run(report2)
    
    runs = store.list_runs()
    assert len(runs) == 2
    
    deleted = store.delete_run("run_001")
    assert deleted is True
    
    runs_after = store.list_runs()
    assert len(runs_after) == 1
    assert runs_after[0]["run_id"] == "run_002"


def test_sqlite_store_compare_runs(temp_db):
    store = SQLiteEvalStore(db_path=temp_db)
    # Model A: gpt-4o-mini (Pass 1/2, score 0.5)
    report_a = create_sample_report("run_model_a", "openai", "gpt-4o-mini", 0.5, 0.50)
    # Model B: claude-3-5-sonnet (Pass 2/2, score 1.0)
    report_b = create_sample_report("run_model_b", "anthropic", "claude-3-5-sonnet", 1.0, 1.00)
    
    store.save_run(report_a)
    store.save_run(report_b)
    
    diff = store.compare_runs("run_model_a", "run_model_b")
    assert diff["run_a"]["model"] == "gpt-4o-mini"
    assert diff["run_b"]["model"] == "claude-3-5-sonnet"
    assert diff["deltas"]["pass_rate_delta"] == 0.5
    assert diff["deltas"]["avg_score_delta"] == 0.5
    assert diff["deltas"]["improvements_count"] == 1
    assert diff["deltas"]["regressions_count"] == 0
