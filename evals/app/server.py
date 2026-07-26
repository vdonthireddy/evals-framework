"""Web server and REST API for the evaluation reporting and admin application."""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from evals.app.admin import AdapterRegistry, AsyncEvalManager, DatasetManager, ModelRegistry
from evals.store.sqlite_store import SQLiteEvalStore

logger = logging.getLogger(__name__)


class EvalReportHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles REST API requests and serves static dashboard assets."""

    store: SQLiteEvalStore = None
    dataset_mgr: DatasetManager = DatasetManager()
    adapter_registry: AdapterRegistry = AdapterRegistry()
    model_registry: ModelRegistry = ModelRegistry()
    eval_mgr: AsyncEvalManager = None
    static_dir: Path = Path(__file__).parent / "static"

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_content: str, status: int = 200) -> None:
        body = html_content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_post_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length).decode("utf-8")
        return json.loads(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # ── REST API Endpoints ──────────────────────────────────────────────
        if path == "/api/runs":
            runs = self.store.list_runs()
            self._send_json(runs)
            return

        elif path.startswith("/api/runs/"):
            run_id = path.replace("/api/runs/", "").strip()
            report = self.store.get_run(run_id)
            if not report:
                self._send_json({"error": f"Run '{run_id}' not found"}, status=404)
            else:
                self._send_json(report.model_dump(mode="json"))
            return

        elif path == "/api/compare":
            run_a = query.get("run_a", [""])[0]
            run_b = query.get("run_b", [""])[0]
            if not run_a or not run_b:
                self._send_json({"error": "Parameters 'run_a' and 'run_b' are required"}, status=400)
                return

            try:
                comparison = self.store.compare_runs(run_a, run_b)
                self._send_json(comparison)
            except ValueError as ve:
                self._send_json({"error": str(ve)}, status=404)
            except Exception as e:
                self._send_json({"error": f"Failed to compare: {e}"}, status=500)
            return

        # ── Admin API Endpoints (GET) ──────────────────────────────────────
        elif path == "/api/admin/datasets":
            datasets = self.dataset_mgr.list_datasets()
            self._send_json(datasets)
            return

        elif path == "/api/admin/adapters":
            adapters = self.adapter_registry.list_adapters()
            self._send_json(adapters)
            return

        elif path == "/api/admin/models":
            models = self.model_registry.list_models()
            self._send_json(models)
            return

        elif path.startswith("/api/admin/evals/progress/"):
            job_id = path.replace("/api/admin/evals/progress/", "").strip()
            if self.eval_mgr:
                progress = self.eval_mgr.get_progress(job_id)
                self._send_json(progress)
            else:
                self._send_json({"error": "Eval manager not initialized"}, status=500)
            return

        # ── Static Dashboard Asset ─────────────────────────────────────────
        elif path == "/" or path == "/dashboard":
            dash_file = self.static_dir / "dashboard.html"
            if dash_file.exists():
                with open(dash_file, "r", encoding="utf-8") as f:
                    self._send_html(f.read())
            else:
                self._send_html("<h1>Dashboard UI not found</h1>", status=404)
            return

        else:
            self._send_json({"error": "Not Found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            payload = self._read_post_json()
        except Exception as e:
            self._send_json({"error": f"Invalid JSON payload: {e}"}, status=400)
            return

        if path == "/api/admin/datasets/cases":
            dataset_name = payload.get("dataset_name", "unit/custom.jsonl")
            case_data = payload.get("case", {})
            if not case_data or "id" not in case_data or "input" not in case_data:
                self._send_json({"error": "Case data must include 'id' and 'input'"}, status=400)
                return

            try:
                new_case = self.dataset_mgr.add_case(dataset_name, case_data)
                self._send_json({"success": True, "case": new_case.model_dump(mode="json")})
            except Exception as e:
                self._send_json({"error": f"Failed to add case: {e}"}, status=500)
            return

        elif path == "/api/admin/adapters":
            if "id" not in payload or "name" not in payload:
                self._send_json({"error": "Adapter must include 'id' and 'name'"}, status=400)
                return

            saved = self.adapter_registry.register_adapter(payload)
            self._send_json({"success": True, "adapter": saved})
            return

        elif path == "/api/admin/models":
            if "id" not in payload or "name" not in payload or "provider" not in payload:
                self._send_json({"error": "Model must include 'id', 'name', and 'provider'"}, status=400)
                return

            saved = self.model_registry.register_model(payload)
            self._send_json({"success": True, "model": saved})
            return

        elif path == "/api/admin/evals/start":
            adapter_id = payload.get("adapter_id", "example")
            provider = payload.get("provider", "openai")
            model = payload.get("model", "gpt-4o-mini")
            dataset_path = payload.get("dataset_path", "evals/datasets")
            concurrency = int(payload.get("concurrency", 2))

            if not self.eval_mgr:
                self._send_json({"error": "Eval manager not initialized"}, status=500)
                return

            job_id = self.eval_mgr.start_job(
                adapter_id=adapter_id,
                provider=provider,
                model=model,
                dataset_path=dataset_path,
                concurrency=concurrency,
            )
            self._send_json({"success": True, "job_id": job_id})
            return

        else:
            self._send_json({"error": "Not Found"}, status=404)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/runs/"):
            run_id = path.replace("/api/runs/", "").strip()
            if not run_id:
                self._send_json({"error": "Run ID required"}, status=400)
                return
            deleted = self.store.delete_run(run_id)
            if deleted:
                self._send_json({"success": True, "run_id": run_id})
            else:
                self._send_json({"error": f"Run '{run_id}' not found"}, status=404)
            return
        else:
            self._send_json({"error": "Not Found"}, status=404)


def run_report_server(
    host: str = "localhost",
    port: int = 8000,
    db_path: str = "evals/results/eval_results.db"
) -> None:
    """Launch the evaluation reporting web application server."""
    store = SQLiteEvalStore(db_path=db_path)
    eval_mgr = AsyncEvalManager(db_path=db_path)

    EvalReportHTTPRequestHandler.store = store
    EvalReportHTTPRequestHandler.eval_mgr = eval_mgr

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, EvalReportHTTPRequestHandler)
    print(f"\n🚀 Evaluation Reporting & Admin Dashboard running at: http://{host}:{port}/")
    print("Press Ctrl+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down reporting server...")
        httpd.server_close()
