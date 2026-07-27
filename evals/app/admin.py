"""Admin backend manager for managing datasets, adapters, models, and async eval runs."""

import asyncio
import json
import logging
import uuid
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from evals.adapters.example_agent_adapter import ExampleAgentAdapter
from evals.adapters.langchain_adapter import LangChainAdapter
from evals.core.dataset import EvalDataset
from evals.core.interfaces import AgentAdapter, EvalCase
from evals.core.runner import EvalConfig, EvalRunner, EvalRunReport
from evals.store.sqlite_store import SQLiteEvalStore

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages reading and writing evaluation dataset JSONL files."""

    def __init__(self, datasets_dir: str = "evals/datasets") -> None:
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all dataset JSONL files across subdirectories."""
        files = list(self.datasets_dir.rglob("*.jsonl"))
        result = []
        for f in sorted(files):
            rel_path = str(f.relative_to(self.datasets_dir))
            cases_count = 0
            with open(f, "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        cases_count += 1
            result.append({
                "name": rel_path,
                "full_path": str(f),
                "case_count": cases_count,
            })
        return result

    def add_case(self, dataset_rel_path: str, case_dict: Dict[str, Any]) -> EvalCase:
        """Validate and append a new EvalCase entry to a dataset file."""
        case = EvalCase.model_validate(case_dict)
        target_file = self.datasets_dir / dataset_rel_path
        target_file.parent.mkdir(parents=True, exist_ok=True)

        with open(target_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(case.model_dump(mode="json")) + "\n")

        return case


class AdapterRegistry:
    """Manages custom agent adapter definitions and configuration templates."""

    def __init__(self, registry_file: str = "evals/results/adapters_registry.json") -> None:
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        self._default_adapters = [
            {
                "id": "example",
                "name": "Example ReAct Multi-Tool Agent",
                "class_name": "ExampleAgentAdapter",
                "description": "Default ReAct research assistant with search, calculator, weather, and KB tools.",
            },
            {
                "id": "langchain_agent",
                "name": "LangChain Agent Executor",
                "class_name": "LangChainAdapter",
                "description": "Adapter bridging LangChain AgentExecutors to Evals.",
            },
        ]
        self._init_registry()

    def _init_registry(self) -> None:
        if not self.registry_file.exists():
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self._default_adapters, f, indent=2)

    def list_adapters(self) -> List[Dict[str, Any]]:
        """Return all registered agent adapters."""
        if not self.registry_file.exists():
            return self._default_adapters
        with open(self.registry_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def register_adapter(self, adapter_info: Dict[str, Any]) -> Dict[str, Any]:
        """Save a new custom agent adapter definition."""
        adapters = self.list_adapters()
        adapter_id = adapter_info.get("id") or str(uuid.uuid4())[:8]
        adapter_info["id"] = adapter_id

        # Update if existing, else append
        existing = [i for i, a in enumerate(adapters) if a["id"] == adapter_id]
        if existing:
            adapters[existing[0]] = adapter_info
        else:
            adapters.append(adapter_info)

        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(adapters, f, indent=2)

        return adapter_info

    def get_adapter_instance(
        self,
        adapter_id: str,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = "",
    ) -> AgentAdapter:
        """Instantiate an agent adapter by ID."""
        if adapter_id == "example" or adapter_id == "ExampleAgentAdapter":
            return ExampleAgentAdapter(provider=provider, model=model, api_key=api_key)
        else:
            # Fallback to ExampleAgentAdapter for custom registrations
            return ExampleAgentAdapter(provider=provider, model=model, api_key=api_key)


class ModelRegistry:
    """Manages configurable evaluation LLM models."""

    def __init__(self, registry_file: str = "evals/results/models_registry.json") -> None:
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        self._default_models = [
            {"id": "gpt-4o-mini", "name": "OpenAI GPT-4o Mini", "provider": "openai", "server_url": "https://api.openai.com/v1", "api_key": ""},
            {"id": "gpt-4o", "name": "OpenAI GPT-4o", "provider": "openai", "server_url": "https://api.openai.com/v1", "api_key": ""},
            {"id": "claude-3-5-sonnet", "name": "Anthropic Claude 3.5 Sonnet", "provider": "anthropic", "server_url": "https://api.anthropic.com", "api_key": ""},
            {"id": "gemini-1.5-pro", "name": "Google Gemini 1.5 Pro", "provider": "gemini", "server_url": "https://generativelanguage.googleapis.com", "api_key": ""},
            {"id": "llama3.2", "name": "Ollama Llama 3.2 (Local)", "provider": "ollama", "server_url": "http://localhost:11434/v1", "api_key": ""},
            {"id": "gemma2:2b", "name": "Ollama Gemma 2 2B (Local)", "provider": "ollama", "server_url": "http://localhost:11434/v1", "api_key": ""},
            {"id": "qwen2.5-coder:7b", "name": "Ollama Qwen 2.5 Coder 7B (Local)", "provider": "ollama", "server_url": "http://localhost:11434/v1", "api_key": ""},
            {"id": "mistral", "name": "Ollama Mistral (Local)", "provider": "ollama", "server_url": "http://localhost:11434/v1", "api_key": ""},
        ]
        self._init_registry()

    def _init_registry(self) -> None:
        if not self.registry_file.exists():
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self._default_models, f, indent=2)

    def list_models(self) -> List[Dict[str, Any]]:
        """Return all configured LLM models."""
        if not self.registry_file.exists():
            return self._default_models
        with open(self.registry_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def register_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register or update a model configuration."""
        models = self.list_models()
        model_id = model_info.get("id") or str(uuid.uuid4())[:8]
        model_info["id"] = model_id

        existing = [i for i, m in enumerate(models) if m["id"] == model_id]
        if existing:
            models[existing[0]] = model_info
        else:
            models.append(model_info)

        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(models, f, indent=2)

        return model_info


class AsyncEvalManager:
    """Orchestrates asynchronous background evaluation runs with progress tracking."""

    def __init__(self, db_path: str = "evals/results/eval_results.db") -> None:
        self.db_path = db_path
        self.store = SQLiteEvalStore(db_path=db_path)
        self.registry = AdapterRegistry()
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def start_job(
        self,
        adapter_id: str = "example",
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        dataset_path: str = "evals/datasets",
        concurrency: int = 2,
    ) -> str:
        """Launch an evaluation job in a background thread."""
        job_id = f"job-{str(uuid.uuid4())[:8]}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "adapter_id": adapter_id,
            "provider": provider,
            "model": model,
            "dataset_path": dataset_path,
            "total_cases": 0,
            "completed_cases": 0,
            "passed_cases": 0,
            "current_case_id": "Initializing...",
            "run_id": None,
            "error": None,
        }

        thread = threading.Thread(
            target=self._run_job_thread,
            args=(job_id, adapter_id, provider, model, dataset_path, concurrency),
            daemon=True,
        )
        thread.start()
        return job_id

    def get_progress(self, job_id: str) -> Dict[str, Any]:
        """Return the current progress of a background job."""
        return self.jobs.get(job_id, {"job_id": job_id, "status": "not_found"})

    def _run_job_thread(
        self,
        job_id: str,
        adapter_id: str,
        provider: str,
        model: str,
        dataset_path: str,
        concurrency: int,
    ) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._execute_eval(job_id, adapter_id, provider, model, dataset_path, concurrency)
            )
        except Exception as e:
            logger.error(f"Error in job {job_id}: {e}", exc_info=True)
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
        finally:
            loop.close()

    async def _execute_eval(
        self,
        job_id: str,
        adapter_id: str,
        provider: str,
        model: str,
        dataset_path: str,
        concurrency: int,
    ) -> None:
        import os
        api_key = os.getenv("AGENT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or "dummy-key"
        adapter = self.registry.get_adapter_instance(adapter_id, provider=provider, model=model, api_key=api_key)

        dataset = EvalDataset(dataset_path)
        cases = list(dataset)
        self.jobs[job_id]["total_cases"] = len(cases)

        config = EvalConfig(
            run_id=f"run-{job_id}",
            max_concurrency=concurrency,
            timeout_seconds=180 if provider == "ollama" else 120,
            output_dir="evals/results",
        )

        def handle_case_complete(res: Any, completed_count: int, total_count: int):
            self.jobs[job_id]["total_cases"] = total_count
            self.jobs[job_id]["completed_cases"] = completed_count
            if res.overall_passed:
                self.jobs[job_id]["passed_cases"] = self.jobs[job_id].get("passed_cases", 0) + 1
            self.jobs[job_id]["current_case_id"] = res.case_id

        runner = EvalRunner(adapter, dataset, config)
        report: EvalRunReport = await runner.run(on_case_complete=handle_case_complete)

        # Update job stats from report
        self.jobs[job_id]["status"] = "completed"
        self.jobs[job_id]["completed_cases"] = len(report.results)
        self.jobs[job_id]["passed_cases"] = report.summary.get("passed", 0)
        self.jobs[job_id]["run_id"] = report.run_id
        self.jobs[job_id]["current_case_id"] = "Finished"

        # Persist report to SQLite
        self.store.save_run(report)
