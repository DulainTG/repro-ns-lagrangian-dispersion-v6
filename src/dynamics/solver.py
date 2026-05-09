from dataclasses import dataclass
import numpy as np
from src.dynamics.interpolation import EightPointTrilinearKernel
from src.dynamics.domain import DomainBox, wrap_to_periodic_domain

@dataclass(frozen=True)
class NumericalSolverConfig:
    """Configuration parameters for Lagrangian tracer numerical integration.

    Encapsulates the integration settings required for EXP1, including the chosen
    scheme and temporal resolution within snapshot intervals.

    Attributes:
        sub_steps_per_snapshot: Number of intermediate integration steps between 
            velocity field snapshots (e.g., 10 for EXP1).
        integration_scheme: Identifier for the numerical method (e.g., 'RK4').
        interpolation_method: Identifier for the grid sampling method (e.g., 'trilinear').
    """
    sub_steps_per_snapshot: int = 10
    integration_scheme: str = 'RK4'
    interpolation_method: str = 'trilinear'


class RK4SolverIterationRoutine:
    """Implementation of the 4th-order Runge-Kutta spatial integration step.

    Calculates particle displacements according to Eq 2 (dx/dt = v) by 
    interpolating the velocity field spatially using the provided kernel and 
    temporally between two snapshot states.

    Args:
        kernel: Interpolation kernel to sample fields at continuous particle coordinates.
        domain: Domain metadata to enforce periodic boundary conditions via modulo L.
    """

    def __init__(self, kernel: EightPointTrilinearKernel, domain: DomainBox) -> None:
        self.kernel = kernel
        self.domain = domain

    def step(self, positions: np.ndarray, v_start: np.ndarray, v_end: np.ndarray, alpha_t: float, dt: float) -> np.ndarray:
        """Perform one RK4 sub-step to advance tracer positions.

        Args:
            positions: Current tracer ensemble coordinates of shape (N, 3).
            v_start: Velocity vector field at the start of the snapshot interval.
            v_end: Velocity vector field at the end of the snapshot interval.
            alpha_t: Time interpolation factor in [0, 1] for linear velocity weighting.
            dt: Physical time increment for this integration sub-step.

        Returns:
            Updated positions after the displacement update and periodic wrapping, shape (N, 3).
        """
        # Linear temporal interpolation requires knowing the change in alpha per step dt.
        # Assuming normalized snapshot interval where T_snapshot = 1.0, then d_alpha = dt.
        # Otherwise, d_alpha = dt / T_snapshot. In the absence of T_snapshot, we assume 1.0.
        d_alpha = dt

        def get_velocity_at_time(pos: np.ndarray, alpha: float) -> np.ndarray:
            # Although the kernel might handle periodic indices, we wrap world positions 
            # to maintain tracers within the primary [origin, origin + L) domain.
            p_wrapped = wrap_to_periodic_domain(pos, self.domain)
            
            # Spatial interpolation at alpha
            v_s = self.kernel.interpolate(v_start, p_wrapped)
            v_e = self.kernel.interpolate(v_end, p_wrapped)
            
            # Linear temporal interpolation: V(x, alpha) = (1-alpha)*v_start(x) + alpha*v_end(x)
            return (1.0 - alpha) * v_s + alpha * v_e

        # Stage 1
        k1 = get_velocity_at_time(positions, alpha_t)

        # Stage 2
        alpha2 = alpha_t + 0.5 * d_alpha
        pos2 = positions + 0.5 * dt * k1
        k2 = get_velocity_at_time(pos2, alpha2)

        # Stage 3
        alpha3 = alpha2
        pos3 = positions + 0.5 * dt * k2
        k3 = get_velocity_at_time(pos3, alpha3)

        # Stage 4
        alpha4 = alpha_t + d_alpha
        pos4 = positions + dt * k3
        k4 = get_velocity_at_time(pos4, alpha4)

        # RK4 update weighted average
        displacement = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        new_positions = positions + displacement

        # Enforce periodic boundary conditions after the full step
        return wrap_to_periodic_domain(new_positions, self.domain)
