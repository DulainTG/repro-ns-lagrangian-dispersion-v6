import numpy as np
from typing import List, Mapping, Optional
from src.dynamics.interpolation import EightPointTrilinearKernel
from src.dynamics.domain import DomainBox, wrap_to_periodic_domain
from src.dynamics.tracking import TracerStateBuffer
from src.fields.grid import GridMetadata

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
class LagrangianTracerEngine:
    """High-level engine for integrating Lagrangian tracer trajectories in DNS snapshots.

    Primary interface for performing the end-to-end integration required by EXP1.
    Coordinates the solvers, temporal steppers, and state buffers to simulate
    particle transport through turbulent flow fields.

    Args:
        solver: The RK4 solver for inner-step displacements.
        stepper: The sub-stepping logic controller.
        buffer: The buffer holding current tracer states and tracking history.
    """

    def __init__(self, solver: 'RungeKutta4Solver', stepper: 'TemporalSubStepper', buffer: TracerStateBuffer) -> None:
        self.solver = solver
        self.stepper = stepper
        self.buffer = buffer

    def integrate_snapshot_interval(self, v_start: np.ndarray, v_end: np.ndarray, snapshot_idx: int, aux_fields: Mapping[str, np.ndarray]) -> None:
        """Integrate the tracer ensemble between two snapshots.

        Performs N sub-steps using the RK4 solver, sampling auxiliary fields (like Q-criterion) 
        for EXP3 tracking, and commits results to the history buffer.

        Args:
            v_start: Velocity field at the starting snapshot.
            v_end: Velocity field at the ending snapshot.
            snapshot_idx: The index of the current temporal interval.
            aux_fields: Mapping of field names to grids for Lagrangian sampling.
        """
        positions = self.buffer.current_positions
        num_steps = self.stepper.steps_per_interval
        dt = self.stepper.sub_dt
        
        # Pre-calculate velocity difference for temporal interpolation within the interval
        dv = v_end - v_start
        
        for i in range(num_steps):
            alpha_start = i / num_steps
            alpha_end = (i + 1) / num_steps
            
            # Fields at start and end of this sub-step
            v_step_start = v_start + alpha_start * dv
            v_step_end = v_start + alpha_end * dv
            
            # Calculate displacement for this sub-step using RK4
            displacement = self.solver.evaluate_step(positions, v_step_start, v_step_end, dt)
            positions = self.solver.update_position(positions, displacement)
            
        # Update current positions in the buffer
        self.buffer.update_positions(positions)
        
        # Sample auxiliary fields (e.g., Q-criterion) at the end of the interval
        sampled_aux = {}
        if aux_fields:
            for name, grid in aux_fields.items():
                sampled_aux[name] = self.solver.kernel.interpolate(grid, positions)
        
        # Persist state at the end of the interval (snapshot_idx + 1)
        # Note: Initial snapshot at t=0 should be recorded during setup before first interval
        self.buffer.commit_to_history(snapshot_idx + 1, samples=sampled_aux)
class PassiveTracerIntegrator:
    """Integrator for passive tracers in an Eulerian velocity field frame.

    Provides a simplified interface for advecting particles through static or semi-static
    Eulerian fields where complex temporal sub-stepping is not the primary focus.

    Args:
        metadata: Metadata defining the Eulerian grid scale and topology.
    """

    def __init__(self, metadata: GridMetadata) -> None:
        self.metadata = metadata
        self.kernel = EightPointTrilinearKernel(metadata)
        self.domain = DomainBox(extent=metadata.extent, origin=metadata.origin)
        self.solver = RungeKutta4Solver(self.kernel, self.domain)

    def step_forward(self, positions: np.ndarray, velocity_field: np.ndarray, dt: float) -> np.ndarray:
        """Advect tracer positions by one step using the provided field.

        Args:
            positions: Current tracer coordinates.
            velocity_field: The stationary velocity field for this step.
            dt: Time delta for the advection.

        Returns:
            New advected and boundary-wrapped coordinates.
        """
        # For a stationary field, the field at the start and end of the step is the same.
        # RK4 handling of linear temporal interpolation will correctly use this stationary field.
        displacement = self.solver.evaluate_step(positions, velocity_field, velocity_field, dt)
        return self.solver.update_position(positions, displacement)
