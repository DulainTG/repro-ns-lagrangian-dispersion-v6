from dataclasses import dataclass
from typing import Mapping
import numpy as np
from src.dynamics.interpolation import EightPointTrilinearKernel
from src.dynamics.domain import DomainBox, wrap_to_periodic_domain
from src.dynamics.tracking import TracerStateBuffer

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

    def step(self, positions: np.ndarray, v_start: np.ndarray, v_end: np.ndarray, alpha_t: float, dt: float, dt_interval: float = 1.0) -> np.ndarray:
        """Perform one RK4 sub-step to advance tracer positions.

        Args:
            positions: Current tracer ensemble coordinates of shape (N, 3).
            v_start: Velocity vector field at the start of the snapshot interval.
            v_end: Velocity vector field at the end of the snapshot interval.
            alpha_t: Time interpolation factor in [0, 1] for linear velocity weighting.
            dt: Physical time increment for this integration sub-step.
            dt_interval: Total physical time between velocity snapshots.

        Returns:
            Updated positions after the displacement update and periodic wrapping, shape (N, 3).
        """
        # Linear temporal interpolation requires knowing the change in alpha per step dt.
        # d_alpha is the fractional change in the snapshot interval.
        d_alpha = dt / dt_interval

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
class SnapshotIterationEngine:
    """Coordinator for tracer integration across a single snapshot time interval.

    Handles the sub-stepping loop defined in EXP1 and the auxiliary field 
    sampling required for EXP3 (Lagrangian Q-autocorrelation).

    Args:
        routine: The numerical routine (e.g., RK4) used for spatial integration.
        config: Configuration defining sub-step counts and integration logic.
    """

    def __init__(self, routine: 'RK4SolverIterationRoutine', config: 'NumericalSolverConfig') -> None:
        self.routine = routine
        self.config = config
        self._current_step_idx = 0

    def iterate_interval(self, buffer: TracerStateBuffer, v_start: np.ndarray, v_end: np.ndarray, dt_interval: float, aux_fields: Mapping[str, np.ndarray]) -> None:
        """Execute sub-stepped integration for a specific snapshot transition.

        Args:
            buffer: The state buffer tracking current positions and recording trajectory history.
            v_start: Velocity field at simulation time T.
            v_end: Velocity field at simulation time T + dt_interval.
            dt_interval: The total physical time between the two snapshots.
            aux_fields: Dictionary of additional fields (like 'q_criterion') to be sampled 
                along the trajectories at every sub-step for diagnostic analysis.

        Raises:
            RuntimeError: If integration Diverges or if field shapes are inconsistent.
        """
        # Verification of field consistency
        if v_start.shape != v_end.shape:
            raise RuntimeError(f"Velocity field shape mismatch: {v_start.shape} vs {v_end.shape}")

        K = self.config.sub_steps_per_snapshot
        if K <= 0:
             # Should not happen with default config, but for robustness:
             return

        dt = dt_interval / K
        d_alpha = 1.0 / K

        for k in range(K):
            alpha_t = k * d_alpha
            
            # EXP3: Sample auxiliary fields at every sub-step
            sampled_aux = {}
            for name, field in aux_fields.items():
                # Use the trilinear kernel from the routine for spatial interpolation
                sampled_aux[name] = self.routine.kernel.interpolate(field, buffer.current_positions)
            
            # Record current step positions and sampled properties
            buffer.commit_to_history(self._current_step_idx, samples=sampled_aux)
            self._current_step_idx += 1
            
            # Perform numerical integration to advance to next sub-step positions
            new_positions = self.routine.step(
                positions=buffer.current_positions,
                v_start=v_start,
                v_end=v_end,
                alpha_t=alpha_t,
                dt=dt,
                dt_interval=dt_interval
            )
            
            # Check for numerical divergence
            if np.any(np.isnan(new_positions)):
                raise RuntimeError(f"Integration diverged at step {self._current_step_idx} (encountered NaN).")
                
            buffer.update_positions(new_positions)
