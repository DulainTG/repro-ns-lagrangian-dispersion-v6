from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseExperiment(ABC):
    """Abstract base for reproduction experiments (EXP1, EXP2, EXP3).

    Provides a consistent lifecycle for preparing data, executing calculations,
    and exporting artifacts as defined in the paper reproduction requirements.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def prepare(self) -> None:
        """Load snapshots, initialize grids, or set up tracer ensembles."""
        pass

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute the core numerical or statistical procedure."""
        pass

    @abstractmethod
    def save_artifacts(self, results: Dict[str, Any]) -> None:
        """Write required CSV tables and summary reports to disk."""
        pass