"""Integration of eval framework with pytest.

This file provides a single test definition that is dynamically
parameterized by the conftest.py plugin, running once per case.
"""

import pytest
from typing import Any

from evals.core.interfaces import EvalCase


@pytest.mark.asyncio
async def test_agent_eval(eval_case: EvalCase, run_eval: Any) -> None:
    """Dynamically generated test for a single eval case."""
    await run_eval(eval_case)
