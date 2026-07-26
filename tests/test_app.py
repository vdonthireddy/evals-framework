"""Unit tests for the reporting application REST API endpoints."""

import os
import tempfile
import threading
import urllib.request
import json
import pytest
from datetime import datetime, timezone

from evals.app.server import EvalReportHTTPRequestHandler, ThreadingHTTPServer
from evals.core.interfaces import AgentOutput, ScoreResult
from evals.core.runner import EvalConfig, EvalResult, EvalRunReport
from evals.store.sqlite_store import SQLiteEvalStore


@pytest.fixture
def test_server():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    store = SQLiteEvalStore(db_path=db_path)
    now = datetime.now(timezone.utc)
    report = EvalRunReport(
        run_id="run_api_test",
        timestamp=now,
        agent_info={"name": "TestAgent", "provider": "openai", "model": "gpt-4o-mini"},
        dataset_info={"total_cases": 1},
        config=EvalConfig(run_id="run_api_test"),
        results=[
            EvalResult(
                case_id="c1",
                case_input="Hi",
                agent_output=AgentOutput(input="Hi", output="Hello", total_steps=1),
                scores=[ScoreResult(scorer_name="exact", score=1.0, passed=True, threshold=1.0)],
                overall_passed=True,
                overall_score=1.0,
            )
        ],
        summary={"total_cases": 1, "passed": 1, "failed": 0, "error_count": 0, "overall_pass_rate": 1.0, "average_score": 1.0},
    )
    store.save_run(report)
    
    EvalReportHTTPRequestHandler.store = store
    server = ThreadingHTTPServer(("localhost", 0), EvalReportHTTPRequestHandler)
    port = server.server_port
    
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    
    yield f"http://localhost:{port}", store
    
    server.shutdown()
    server.server_close()
    if os.path.exists(db_path):
        os.remove(db_path)


def test_api_list_runs(test_server):
    base_url, store = test_server
    req = urllib.request.urlopen(f"{base_url}/api/runs")
    assert req.status == 200
    
    data = json.loads(req.read().decode("utf-8"))
    assert len(data) == 1
    assert data[0]["run_id"] == "run_api_test"
    assert data[0]["model"] == "gpt-4o-mini"


def test_api_get_run_detail(test_server):
    base_url, store = test_server
    req = urllib.request.urlopen(f"{base_url}/api/runs/run_api_test")
    assert req.status == 200
    
    data = json.loads(req.read().decode("utf-8"))
    assert data["run_id"] == "run_api_test"
    assert len(data["results"]) == 1


def test_api_dashboard_html(test_server):
    base_url, store = test_server
    req = urllib.request.urlopen(f"{base_url}/")
    assert req.status == 200
    html = req.read().decode("utf-8")
    assert "<title>Evals Framework" in html
