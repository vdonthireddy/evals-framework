"""Unit tests for the Admin Panel manager and REST API endpoints."""

import json
import os
import tempfile
import time
import pytest

from evals.app.admin import AdapterRegistry, AsyncEvalManager, DatasetManager


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def test_dataset_manager_list_and_add(temp_dir):
    mgr = DatasetManager(datasets_dir=temp_dir)
    
    # Add case
    case_dict = {
        "id": "test-admin-01",
        "category": "unit",
        "input": "What is 10 + 10?",
        "expected_outcome": "20",
    }
    
    case = mgr.add_case("unit/custom.jsonl", case_dict)
    assert case.id == "test-admin-01"
    assert case.input == "What is 10 + 10?"
    
    # List datasets
    datasets = mgr.list_datasets()
    assert len(datasets) == 1
    assert "unit/custom.jsonl" in datasets[0]["name"]
    assert datasets[0]["case_count"] == 1


def test_adapter_registry(temp_dir):
    registry_file = os.path.join(temp_dir, "adapters.json")
    registry = AdapterRegistry(registry_file=registry_file)
    
    initial = registry.list_adapters()
    assert len(initial) >= 2
    
    # Register custom adapter
    new_adapter = {
        "id": "custom_agent",
        "name": "Custom Agent",
        "class_name": "CustomAgentAdapter",
        "provider": "openai",
        "model": "gpt-4o",
    }
    saved = registry.register_adapter(new_adapter)
    assert saved["id"] == "custom_agent"
    
    updated = registry.list_adapters()
    assert len(updated) == len(initial) + 1


def test_async_eval_manager(temp_dir):
    db_path = os.path.join(temp_dir, "evals.db")
    eval_mgr = AsyncEvalManager(db_path=db_path)
    
    # Start background job
    job_id = eval_mgr.start_job(
        adapter_id="example",
        provider="openai",
        model="gpt-4o-mini",
        dataset_path="evals/datasets/unit",
        concurrency=1,
    )
    
    assert job_id.startswith("job-")
    
    # Poll progress
    for _ in range(30):
        progress = eval_mgr.get_progress(job_id)
        if progress.get("status") in ("completed", "failed"):
            break
        time.sleep(0.2)
        
    final_progress = eval_mgr.get_progress(job_id)
    assert final_progress["status"] in ("completed", "failed")
    assert final_progress["total_cases"] > 0
