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


def calculate_trajectory_displacements(ensemble: TrajectoryEnsemble, extent: float=1.0, unwrap: bool=True) -> np.ndarray:
    """Calculates displacement vectors Delta_x(t) = x(t) - x(0) for a tracer ensemble.

    In periodic domains, researchers require 'unwrapped' displacements where coordinate 
    jumps across domain boundaries (e.g., from L to 0) are treated as continuous travel 
    to properly calculate Mean-Square Displacement (MSD) as per EXP1 Equation 3.

    Args:
        ensemble: Container housing tracer positions over time.
        extent: The characteristic length L of the cubic periodic domain (default 1.0).
        unwrap: If True, corrects for periodic boundary crossings to provide 
            continuous trajectories. If False, returns raw coordinate differences.

    Returns:
        A numpy array of displacement vectors with shape (num_tracers, num_steps, 3).

    Raises:
        ValueError: If the ensemble contains no position data.
    """
    if ensemble.positions is None or ensemble.positions.size == 0:
        raise ValueError("Ensemble contains no position data.")
    
    if not unwrap:
        # Simple coordinate difference without considering periodicity
        return ensemble.positions - ensemble.positions[:, 0:1, :]
    
    # Calculate incremental jumps across time steps along axis 1 (time axis)
    jumps = np.diff(ensemble.positions, axis=1)
    
    # Unwrap jumps: mapping them to the range [-extent/2, extent/2]
    # to handle crossings of the periodic boundary.
    unwrapped_jumps = (jumps + extent / 2) % extent - extent / 2
    
    # Initial displacement at t=0 is zero for all tracers
    num_tracers = ensemble.num_tracers
    zero_displacement = np.zeros((num_tracers, 1, 3), dtype=ensemble.positions.dtype)
    
    # Concatenate t=0 displacement with the subsequent unwrapped increments
    all_increments = np.concatenate([zero_displacement, unwrapped_jumps], axis=1)
    
    # Cumulative sum along the time axis gives total unwrapped displacement Delta_x(t)
    return np.cumsum(all_increments, axis=1)


def estimate_lagrangian_velocities(ensemble: TrajectoryEnsemble) -> np.ndarray:
    """Estimates Lagrangian tracer velocities using finite time differences of positions.

    Applies a central or forward difference scheme to the trajectory coordinates 
    x(t) to approximate v(t) = dx/dt as defined in EXP1 Equation 2.

    Args:
        ensemble: Container housing tracer positions and snapshot times.

    Returns:
        A numpy array of velocity vectors with shape (num_tracers, num_steps, 3).
        The final step velocity is typically handled via backward difference.

    Raises:
        RuntimeError: If trajectory temporal resolution is insufficient for differencing.
    """
    if ensemble.num_steps < 2:
        raise RuntimeError("Trajectory temporal resolution is insufficient (need at least 2 steps).")
    
    # Use unwrapped displacements to avoid spurious velocity spikes at boundary crossings.
    # Assumes L=1 periodic domain as specified for these experiments.
    displacements = calculate_trajectory_displacements(ensemble, extent=1.0, unwrap=True)
    
    # Calculate the numerical gradient of displacements along the time axis (axis 1)
    # This provides the numerator for v = dx/dt.
    grad_d = np.gradient(displacements, axis=1)
    
    # Calculate the numerical gradient of the snapshot times.
    # This provides the denominator for v = dx/dt.
    grad_t = np.gradient(ensemble.times)
    
    # Expand grad_t to allow broadcasting over (N, T, 3)
    # v_i = grad_d_i / grad_t_i
    return grad_d / grad_t[np.newaxis, :, np.newaxis]


class PropertyHistoryBuffer:
    """Manages access and extraction of time-series data for properties sampled along trajectories.

    This interface facilitates the retrieval of scalar fields (like the Q-criterion for EXP3)
    or vector fields sampled for each tracer at every snapshot, preparing them for 
    statistical analysis such as Lagrangian autocorrelation.

    Attributes:
        ensemble: The trajectory data source containing sampled properties.
    """

    def __init__(self, ensemble: TrajectoryEnsemble):
        """Initialize the buffer with a trajectory ensemble."""
        self.ensemble = ensemble

    def get_property_series(self, property_name: str) -> np.ndarray:
        """Retrieves the full temporal history of a specific property for all tracers.

        Args:
            property_name: The key in the ensemble's sampled_properties dictionary 
                (e.g., 'q_criterion').

        Returns:
            Array of shape (num_tracers, num_steps) for scalars, 
            or (num_tracers, num_steps, D) for tensors.

        Raises:
            KeyError: If the property_name does not exist in the ensemble.
        """
        if property_name not in self.ensemble.sampled_properties:
            raise KeyError(f"Property '{property_name}' was not found in the trajectory ensemble.")
        return self.ensemble.sampled_properties[property_name]

    def get_flattened_history(self, property_name: str) -> np.ndarray:
        """Extracts all samples of a property as a single flattened collection.

        Useful for calculating ensemble-wide distributions or global thresholds.

        Returns:
            One-dimensional array of all sampled values.
        """
        return self.get_property_series(property_name).flatten()
