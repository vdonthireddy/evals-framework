"""Abstract base class for all evals scorers."""

from abc import ABC, abstractmethod

from evals.core.interfaces import AgentOutput, EvalCase, ScoreResult


class BaseScorer(ABC):
    """Abstract interface for all evaluators/scorers.
    
    A scorer takes an EvalCase and the actual AgentOutput, and evaluates
    whether the output meets the expectations defined in the case.
    """

    def __init__(self) -> None:
        self._threshold: float = 0.7

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this scorer."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this scorer evaluates."""

    @property
    def threshold(self) -> float:
        """Minimum score required to pass (0.0 to 1.0). Default is 0.7."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    @abstractmethod
    async def score(self, case: EvalCase, output: AgentOutput) -> ScoreResult:
        """Evaluate the agent's output against the test case.
        
        Args:
            case: The evaluation case containing the input and expectations.
            output: The complete execution trajectory and final output of the agent.
            
        Returns:
            A ScoreResult with a score between 0.0 and 1.0.
        """
