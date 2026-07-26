"""SQLite-backed persistent evaluation store."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from evals.core.runner import EvalRunReport
from evals.store.base import BaseEvalStore


class SQLiteEvalStore(BaseEvalStore):
    """Stores evaluation run reports and detailed case results in SQLite."""

    def __init__(self, db_path: str = "evals/results/eval_results.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Create storage schema if tables do not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    agent_name TEXT,
                    total_cases INTEGER,
                    passed INTEGER,
                    failed INTEGER,
                    errors INTEGER,
                    pass_rate REAL,
                    average_score REAL,
                    average_steps REAL,
                    avg_latency_ms REAL,
                    avg_tokens REAL,
                    summary_json TEXT,
                    report_json TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS case_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    case_input TEXT,
                    overall_passed INTEGER,
                    overall_score REAL,
                    latency_ms REAL,
                    total_tokens INTEGER,
                    steps_count INTEGER,
                    error TEXT,
                    scores_json TEXT,
                    output_text TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_case_results_run_id ON case_results(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_case_results_case_id ON case_results(case_id)")
            conn.commit()

    def save_run(self, report: EvalRunReport) -> str:
        """Persist an evaluation run report and all case results into SQLite."""
        report_dict = report.model_dump(mode="json")
        summary = report.summary or {}
        agent_info = report.agent_info or {}

        # Calculate averages for tokens and latency
        latencies = [r.agent_output.total_latency_ms for r in report.results if r.agent_output and r.agent_output.total_latency_ms]
        tokens = [r.agent_output.total_tokens for r in report.results if r.agent_output and r.agent_output.total_tokens]
        
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
        avg_tokens = (sum(tokens) / len(tokens)) if tokens else 0.0

        run_id = report.run_id
        timestamp_str = report.timestamp.isoformat() if isinstance(report.timestamp, datetime) else str(report.timestamp)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs (
                    run_id, timestamp, provider, model, agent_name,
                    total_cases, passed, failed, errors, pass_rate,
                    average_score, average_steps, avg_latency_ms, avg_tokens,
                    summary_json, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                timestamp_str,
                agent_info.get("provider", "unknown"),
                agent_info.get("model", "unknown"),
                agent_info.get("name", "unknown"),
                summary.get("total_cases", len(report.results)),
                summary.get("passed", 0),
                summary.get("failed", 0),
                summary.get("error_count", 0),
                summary.get("overall_pass_rate", 0.0),
                summary.get("average_score", 0.0),
                summary.get("average_steps", 0.0),
                avg_latency,
                avg_tokens,
                json.dumps(summary),
                json.dumps(report_dict),
            ))

            # Clear old case results for this run_id if updating
            conn.execute("DELETE FROM case_results WHERE run_id = ?", (run_id,))

            for res in report.results:
                scores_data = [s.model_dump(mode="json") for s in res.scores]
                out = res.agent_output
                conn.execute("""
                    INSERT INTO case_results (
                        run_id, case_id, case_input, overall_passed, overall_score,
                        latency_ms, total_tokens, steps_count, error, scores_json, output_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    res.case_id,
                    res.case_input,
                    1 if res.overall_passed else 0,
                    res.overall_score,
                    out.total_latency_ms if out else None,
                    out.total_tokens if out else None,
                    out.total_steps if out else 0,
                    res.error,
                    json.dumps(scores_data),
                    out.output if out else "",
                ))

            conn.commit()

        return run_id

    def get_run(self, run_id: str) -> Optional[EvalRunReport]:
        """Retrieve a stored EvalRunReport by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT report_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if not row:
                return None
            data = json.loads(row["report_json"])
            return EvalRunReport.model_validate(data)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List stored evaluation runs ordered by timestamp descending."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT run_id, timestamp, provider, model, agent_name,
                       total_cases, passed, failed, errors, pass_rate,
                       average_score, average_steps, avg_latency_ms, avg_tokens
                FROM runs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and its associated case results."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM case_results WHERE run_id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        """Compare two runs side-by-side (e.g. Model A vs Model B)."""
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)

        if not run_a or not run_b:
            raise ValueError(f"One or both run IDs not found in store: {run_id_a}, {run_id_b}")

        a_info = run_a.agent_info or {}
        b_info = run_b.agent_info or {}
        a_sum = run_a.summary or {}
        b_sum = run_b.summary or {}

        # Case-by-case comparison map
        cases_a = {r.case_id: r for r in run_a.results}
        cases_b = {r.case_id: r for r in run_b.results}
        all_case_ids = sorted(list(set(cases_a.keys()) | set(cases_b.keys())))

        case_comparisons = []
        regressions = []  # Passed in A, Failed in B
        improvements = [] # Failed in A, Passed in B

        for cid in all_case_ids:
            ca = cases_a.get(cid)
            cb = cases_b.get(cid)

            passed_a = ca.overall_passed if ca else False
            passed_b = cb.overall_passed if cb else False
            score_a = ca.overall_score if ca else 0.0
            score_b = cb.overall_score if cb else 0.0

            if ca and cb:
                if passed_a and not passed_b:
                    regressions.append(cid)
                elif not passed_a and passed_b:
                    improvements.append(cid)

            case_comparisons.append({
                "case_id": cid,
                "input": ca.case_input if ca else (cb.case_input if cb else ""),
                "run_a": {
                    "passed": passed_a,
                    "score": score_a,
                    "output": ca.agent_output.output if (ca and ca.agent_output) else "",
                    "steps": ca.agent_output.total_steps if (ca and ca.agent_output) else 0,
                    "error": ca.error if ca else None,
                },
                "run_b": {
                    "passed": passed_b,
                    "score": score_b,
                    "output": cb.agent_output.output if (cb and cb.agent_output) else "",
                    "steps": cb.agent_output.total_steps if (cb and cb.agent_output) else 0,
                    "error": cb.error if cb else None,
                },
                "score_delta": score_b - score_a,
            })

        return {
            "run_a": {
                "run_id": run_a.run_id,
                "provider": a_info.get("provider", "unknown"),
                "model": a_info.get("model", "unknown"),
                "total_cases": a_sum.get("total_cases", 0),
                "passed": a_sum.get("passed", 0),
                "pass_rate": a_sum.get("overall_pass_rate", 0.0),
                "avg_score": a_sum.get("average_score", 0.0),
            },
            "run_b": {
                "run_id": run_b.run_id,
                "provider": b_info.get("provider", "unknown"),
                "model": b_info.get("model", "unknown"),
                "total_cases": b_sum.get("total_cases", 0),
                "passed": b_sum.get("passed", 0),
                "pass_rate": b_sum.get("overall_pass_rate", 0.0),
                "avg_score": b_sum.get("average_score", 0.0),
            },
            "deltas": {
                "pass_rate_delta": b_sum.get("overall_pass_rate", 0.0) - a_sum.get("overall_pass_rate", 0.0),
                "avg_score_delta": b_sum.get("average_score", 0.0) - a_sum.get("average_score", 0.0),
                "regressions_count": len(regressions),
                "improvements_count": len(improvements),
            },
            "regressions": regressions,
            "improvements": improvements,
            "cases": case_comparisons,
        }
