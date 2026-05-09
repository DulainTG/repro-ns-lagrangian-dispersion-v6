import numpy as np
from typing import List
from src.dynamics.interpolation import EightPointTrilinearKernel
from src.dynamics.domain import DomainBox, wrap_to_periodic_domain

class RungeKutta4Solver:
    """Numerical solver implementing the 4th-order Runge-Kutta scheme for trajectory integration.

    Coordinates the multi-stage evaluation of the velocity field at intermediate steps
    to calculate high-accuracy displacements in solenoidal turbulence.

    Args:
        kernel: The trilinear interpolation kernel for spatial field sampling.
        domain: The periodic domain box for coordinate wrapping at intermediate stages.
    """

    def __init__(self, kernel: EightPointTrilinearKernel, domain: DomainBox) -> None:
        self.kernel = kernel
        self.domain = domain

    def calculate_stage(self, positions: np.ndarray, field: np.ndarray) -> np.ndarray:
        """Compute an intermediate RK4 stage vector (k_n).

        Args:
            positions: Current tracer coordinates (N, 3).
            field: The velocity field grid to sample from.

        Returns:
            Velocity vector at the given coordinates.
        """
        return self.kernel.interpolate(field, positions)

    def evaluate_step(self, positions: np.ndarray, v_start: np.ndarray, v_end: np.ndarray, dt: float) -> np.ndarray:
        """Perform a full multi-stage RK4 displacement evaluation over a time interval dt.

        This handles the linear temporal interpolation of the velocity field between 
        the start and end snapshots.

        Args:
            positions: Current tracer coordinates (N, 3).
            v_start: Velocity field grid at the start of the interval.
            v_end: Velocity field grid at the end of the interval.
            dt: The integration time step.

        Returns:
            The calculated displacement vector for the full step.
        """
        # k1 stage: velocity at the beginning of the step
        k1 = self.calculate_stage(positions, v_start)

        # k2 stage: velocity at the midpoint of the step
        # Linear interpolation of field at midpoint: (v_start + v_end) / 2
        pos2 = self.update_position(positions, 0.5 * dt * k1)
        k2_v_start = self.calculate_stage(pos2, v_start)
        k2_v_end = self.calculate_stage(pos2, v_end)
        k2 = 0.5 * (k2_v_start + k2_v_end)

        # k3 stage: velocity at the midpoint of the step
        pos3 = self.update_position(positions, 0.5 * dt * k2)
        k3_v_start = self.calculate_stage(pos3, v_start)
        k3_v_end = self.calculate_stage(pos3, v_end)
        k3 = 0.5 * (k3_v_start + k3_v_end)

        # k4 stage: velocity at the end of the step
        pos4 = self.update_position(positions, dt * k3)
        k4 = self.calculate_stage(pos4, v_end)

        # Combine stages using RK4 coefficients
        displacement = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return displacement

    def update_position(self, positions: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        """Apply the calculated displacement to positions and enforce periodic boundaries.

        Args:
            positions: Current tracer coordinates (N, 3).
            displacement: Calculated RK4 displacement.

        Returns:
            New coordinates (N, 3) wrapped within the periodic domain.
        """
        return wrap_to_periodic_domain(positions + displacement, self.domain)


class TemporalSubStepper:
    """Orchestrator for temporal sub-stepping between discrete velocity snapshots.

    Implements the logic to divide a macro-interval (snapshot spacing) into 
    dense numerical integration steps to ensure convergence and capture small-scale
    tracer dynamics.

    Args:
        steps_per_interval: Number of sub-steps to perform (e.g., 10 for EXP1).
        total_interval_dt: Physical time duration between two consecutive snapshots.
    """

    def __init__(self, steps_per_interval: int, total_interval_dt: float) -> None:
        self.steps_per_interval = steps_per_interval
        self.total_interval_dt = total_interval_dt

    @property
    def sub_dt(self) -> float:
        """The physical time increment for a single sub-step."""
        return self.total_interval_dt / self.steps_per_interval

    def get_sub_step_times(self) -> List[float]:
        """Calculate the normalized time offsets [0, 1] for each sub-step stage."""
        return [float(i) / self.steps_per_interval for i in range(self.steps_per_interval)]
