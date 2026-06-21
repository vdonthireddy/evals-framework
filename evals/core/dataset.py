"""Dataset loading and management."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import ValidationError

from evals.core.interfaces import EvalCase

logger = logging.getLogger(__name__)


class EvalDataset:
    """Loads and manages a collection of EvalCases from JSONL files."""

    def __init__(self, path_or_cases: str | Path | list[EvalCase]) -> None:
        """Initialize the dataset.

        Args:
            path_or_cases: Can be a path to a single JSONL file, a directory
                containing JSONL files, or a pre-populated list of EvalCase objects.
        """
        self._cases: list[EvalCase] = []

        if isinstance(path_or_cases, list):
            self._cases = list(path_or_cases)
        elif isinstance(path_or_cases, (str, Path)):
            path = Path(path_or_cases)
            if not path.exists():
                raise FileNotFoundError(f"Dataset path not found: {path}")

            if path.is_file():
                self._load_file(path)
            elif path.is_dir():
                for jsonl_file in sorted(path.rglob("*.jsonl")):
                    self._load_file(jsonl_file)
        else:
            raise TypeError("path_or_cases must be a path string, Path object, or list of cases")

    def _load_file(self, filepath: Path) -> None:
        """Load cases from a single JSONL file."""
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    data = json.loads(line)
                    # Automatically set category based on parent directory if not specified
                    if "category" not in data:
                        data["category"] = filepath.parent.name
                    case = EvalCase.model_validate(data)
                    self._cases.append(case)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSON at %s:%d - %s", filepath, line_num, e)
                except ValidationError as e:
                    logger.warning("Skipping invalid EvalCase at %s:%d - %s", filepath, line_num, e)

    def filter_by_tags(self, tags: list[str]) -> EvalDataset:
        """Return a new dataset containing only cases with at least one matching tag."""
        target_tags = set(tags)
        filtered_cases = [
            case for case in self._cases if set(case.tags).intersection(target_tags)
        ]
        return EvalDataset(filtered_cases)

    def filter_by_category(self, category: str) -> EvalDataset:
        """Return a new dataset containing only cases in the specified category."""
        filtered_cases = [case for case in self._cases if case.category == category]
        return EvalDataset(filtered_cases)

    def filter_by_difficulty(self, difficulty: str) -> EvalDataset:
        """Return a new dataset containing only cases with the specified difficulty."""
        filtered_cases = [case for case in self._cases if case.difficulty == difficulty]
        return EvalDataset(filtered_cases)

    def sample(self, n: int, seed: int = 42) -> EvalDataset:
        """Return a new dataset containing a random sample of n cases."""
        if n >= len(self._cases):
            return EvalDataset(self._cases.copy())
        
        # Use a local random instance to avoid side effects
        rng = random.Random(seed)
        sampled_cases = rng.sample(self._cases, n)
        return EvalDataset(sampled_cases)

    def get_case(self, case_id: str) -> Optional[EvalCase]:
        """Look up a case by ID."""
        for case in self._cases:
            if case.id == case_id:
                return case
        return None

    def summary(self) -> dict[str, Any]:
        """Return summary statistics about the dataset."""
        categories: dict[str, int] = {}
        difficulties: dict[str, int] = {}
        tags: dict[str, int] = {}

        for case in self._cases:
            categories[case.category] = categories.get(case.category, 0) + 1
            difficulties[case.difficulty] = difficulties.get(case.difficulty, 0) + 1
            for tag in case.tags:
                tags[tag] = tags.get(tag, 0) + 1

        return {
            "total_cases": len(self._cases),
            "by_category": categories,
            "by_difficulty": difficulties,
            "by_tag": tags,
        }

    def __len__(self) -> int:
        """Return the number of cases in the dataset."""
        return len(self._cases)

    def __iter__(self) -> Iterable[EvalCase]:
        """Iterate over the cases in the dataset."""
        return iter(self._cases)
