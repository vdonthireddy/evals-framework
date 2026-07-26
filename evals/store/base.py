"""Abstract base class for persistent evaluation stores."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from evals.core.runner import EvalRunReport


class BaseEvalStore(ABC):
    """Abstract interface for local evaluation storage engines."""

    @abstractmethod
    def save_run(self, report: EvalRunReport) -> str:
        """Persist an evaluation run report and return its run_id."""
        pass

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[EvalRunReport]:
        """Retrieve a stored evaluation run report by ID."""
        pass

    @abstractmethod
    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return a list of stored run metadata summaries ordered by timestamp descending."""
        pass

    @abstractmethod
    def delete_run(self, run_id: str) -> bool:
        """Delete a stored run by ID."""
        pass

    @abstractmethod
    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        """Compare two evaluation runs and return structured head-to-head metrics."""
        pass
