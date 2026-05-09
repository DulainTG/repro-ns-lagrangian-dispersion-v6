import numpy as np
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TrajectoryEnsemble:
    """Container for Lagrangian tracer trajectories and associated flow properties.

    Attributes:
        times: Array of snapshot time indices or physical times, shape (T,).
        positions: Array of tracer coordinates at each time, shape (N, T, 3).
        sampled_properties: Dictionary mapping property names (e.g., 'q_criterion') 
            to arrays of values sampled along the trajectories, shape (N, T) for scalars.
    """
    times: np.ndarray
    positions: np.ndarray
    sampled_properties: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def num_tracers(self) -> int:
        return self.positions.shape[0]

    @property
    def num_steps(self) -> int:
        return self.positions.shape[1]