import pytest
import numpy as np
from src.fields.grid import GridMetadata
from src.dynamics.domain import DomainBox
from src.dynamics.interpolation import EightPointTrilinearKernel
from src.dynamics.solver import (
    RK4SolverIterationRoutine, 
    SnapshotIterationEngine, 
    NumericalSolverConfig
)
from src.dynamics.integrators import (
    RungeKutta4Solver, 
    TemporalSubStepper, 
    LagrangianTracerEngine, 
    PassiveTracerIntegrator
)
from src.dynamics.initialization import ParticleSeeder
from src.dynamics.tracking import (
    TracerStateBuffer, 
    PositionCoordinateBuffer, 
    StateHistoryRecords
)

def test_dynamics_initialization_and_seeding():
    """Test the interaction between DomainBox, ParticleSeeder, and TracerStateBuffer."""
    extent = 1.0
    origin = (-0.5, -0.5, -0.5)
    domain = DomainBox(extent=extent, origin=origin)
    
    num_tracers = 100
    seeder = ParticleSeeder(domain, seed=42)
    initial_positions = seeder.generate_initial_positions(num_tracers)
    
    assert initial_positions.shape == (num_tracers, 3)
    assert np.all(initial_positions >= -0.5)
    assert np.all(initial_positions < 0.5)
    
    num_snapshots = 10
    coord_buffer = PositionCoordinateBuffer(num_tracers, num_snapshots)
    state_records = StateHistoryRecords(num_tracers, num_snapshots)
    
    buffer = TracerStateBuffer(initial_positions, coord_buffer, state_records)
    
    assert np.array_equal(buffer.current_positions, initial_positions % 1.0)
    
    # Test updating positions
    new_pos = initial_positions + 0.1
    buffer.update_positions(new_pos)
    assert np.allclose(buffer.current_positions, (initial_positions + 0.1) % 1.0)

def test_trilinear_interpolation_integration():
    """Test trilinear interpolation integrated with GridMetadata."""
    dims = (10, 10, 10)
    origin = (-0.5, -0.5, -0.5)
    spacing = (0.1, 0.1, 0.1)
    metadata = GridMetadata(dimensions=dims, origin=origin, spacing=spacing, is_periodic=True)
    
    kernel = EightPointTrilinearKernel(metadata)
    
    # Create a periodic field: f(x, y, z) = sin(2*pi*x) + sin(2*pi*y) + sin(2*pi*z)
    x = np.linspace(origin[0], origin[0] + (dims[0]-1)*spacing[0], dims[0])
    y = np.linspace(origin[1], origin[1] + (dims[1]-1)*spacing[1], dims[1])
    z = np.linspace(origin[2], origin[2] + (dims[2]-1)*spacing[2], dims[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    field = np.sin(2 * np.pi * X) + np.sin(2 * np.pi * Y) + np.sin(2 * np.pi * Z)
    
    # Test points
    test_points = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.05, 0.05],
        [-0.45, -0.45, -0.45],
        [0.44, 0.44, 0.44]
    ])
    
    interpolated_values = kernel.interpolate(field, test_points)
    expected_values = np.sum(np.sin(2 * np.pi * test_points), axis=1)
    
    assert np.allclose(interpolated_values, expected_values, atol=2e-1) # Increase atol for trilinear on sine

def test_lagrangian_tracer_engine_rk4_integration():
    """Test full integration loop using LagrangianTracerEngine and RK4Solver."""
    dims = (10, 10, 10)
    origin = (-0.5, -0.5, -0.5)
    spacing = (0.1, 0.1, 0.1)
    metadata = GridMetadata(dimensions=dims, origin=origin, spacing=spacing, is_periodic=True)
    domain = DomainBox(extent=1.0, origin=origin)
    kernel = EightPointTrilinearKernel(metadata)
    solver = RungeKutta4Solver(kernel, domain)
    
    stepper = TemporalSubStepper(steps_per_interval=5, total_interval_dt=0.5)
    
    num_tracers = 10
    initial_positions = np.zeros((num_tracers, 3))
    
    num_snapshots = 2
    coord_buffer = PositionCoordinateBuffer(num_tracers, num_snapshots)
    state_records = StateHistoryRecords(num_tracers, num_snapshots)
    buffer = TracerStateBuffer(initial_positions, coord_buffer, state_records)
    
    engine = LagrangianTracerEngine(solver, stepper, buffer)
    
    # Uniform velocity field: v = (1, 0, 0)
    v_start = np.zeros(dims + (3,))
    v_start[..., 0] = 1.0
    v_end = v_start.copy()
    
    # Record initial state
    buffer.commit_to_history(0)
    
    # Integrate one interval
    engine.integrate_snapshot_interval(v_start, v_end, snapshot_idx=0, aux_fields={})
    
    # After dt=0.5 with v=1.0, displacement should be 0.5 in x direction
    expected_positions = (initial_positions + [0.5, 0, 0])
    
    assert np.allclose(buffer.current_positions, expected_positions % 1.0)
    
    # Check history
    history = coord_buffer.get_data()
    assert np.allclose(history[:, 0, :], initial_positions)
    assert np.allclose(history[:, 1, :], initial_positions + [0.5, 0, 0])

def test_snapshot_iteration_engine_integration():
    """Test integration using SnapshotIterationEngine and RK4SolverIterationRoutine."""
    dims = (10, 10, 10)
    origin = (-0.5, -0.5, -0.5)
    spacing = (0.1, 0.1, 0.1)
    metadata = GridMetadata(dimensions=dims, origin=origin, spacing=spacing, is_periodic=True)
    domain = DomainBox(extent=1.0, origin=origin)
    kernel = EightPointTrilinearKernel(metadata)
    
    routine = RK4SolverIterationRoutine(kernel, domain)
    config = NumericalSolverConfig(sub_steps_per_snapshot=5)
    engine = SnapshotIterationEngine(routine, config)
    
    num_tracers = 10
    initial_positions = np.zeros((num_tracers, 3))
    
    # SnapshotIterationEngine records at every sub-step. 
    # With 5 sub-steps, we need at least 6 slots in buffer (initial + 5 steps)
    num_snapshots = 6 
    coord_buffer = PositionCoordinateBuffer(num_tracers, num_snapshots)
    state_records = StateHistoryRecords(num_tracers, num_snapshots)
    buffer = TracerStateBuffer(initial_positions, coord_buffer, state_records)
    
    # Uniform velocity field: v = (1, 0, 0)
    v_start = np.zeros(dims + (3,))
    v_start[..., 0] = 1.0
    v_end = v_start.copy()
    
    dt_interval = 0.5
    engine.iterate_interval(buffer, v_start, v_end, dt_interval=dt_interval, aux_fields={'q': v_start[..., 0]})
    
    # After dt_interval=0.5, total displacement is 0.5
    expected_final_positions = (initial_positions + [0.5, 0, 0])
    
    assert np.allclose(buffer.current_positions, expected_final_positions % 1.0)
    
    history_coords = coord_buffer.get_data()
    assert np.allclose(history_coords[:, 4, :], initial_positions + [0.4, 0, 0])

def test_passive_tracer_integrator():
    """Test the simplified PassiveTracerIntegrator."""
    dims = (10, 10, 10)
    origin = (-0.5, -0.5, -0.5)
    spacing = (0.1, 0.1, 0.1)
    metadata = GridMetadata(dimensions=dims, origin=origin, spacing=spacing, is_periodic=True, extent=1.0)
    
    integrator = PassiveTracerIntegrator(metadata)
    
    num_tracers = 5
    positions = np.array([
        [0, 0, 0],
        [0.1, 0.1, 0.1],
        [-0.4, 0.4, 0.0],
        [0.45, -0.45, 0.0],
        [0.0, 0.0, 0.45]
    ])
    
    velocity_field = np.zeros(dims + (3,))
    velocity_field[..., 1] = 2.0  # Constant velocity in Y
    
    dt = 0.1
    new_positions = integrator.step_forward(positions, velocity_field, dt)
    
    expected_positions = positions + np.array([0, 2.0 * dt, 0])
    # Periodic wrap using origin -0.5 and extent 1.0
    expected_positions = ((expected_positions - (-0.5)) % 1.0) + (-0.5)
    
    assert np.allclose(new_positions, expected_positions)

def test_tracer_state_buffer_ensemble_building():
    """Test building a TrajectoryEnsemble from TracerStateBuffer."""
    num_tracers = 5
    num_snapshots = 3
    initial_positions = np.random.rand(num_tracers, 3)
    coord_buffer = PositionCoordinateBuffer(num_tracers, num_snapshots)
    state_records = StateHistoryRecords(num_tracers, num_snapshots)
    buffer = TracerStateBuffer(initial_positions, coord_buffer, state_records)
    
    times = np.array([0.0, 0.5, 1.0])
    
    # Record snapshots
    buffer.commit_to_history(0, samples={'q': np.zeros(num_tracers)})
    buffer.update_positions(initial_positions + 0.1)
    buffer.commit_to_history(1, samples={'q': np.ones(num_tracers)})
    buffer.update_positions(initial_positions + 0.2)
    buffer.commit_to_history(2, samples={'q': np.ones(num_tracers) * 2})
    
    ensemble = buffer.build_ensemble(times)
    
    assert ensemble.num_tracers == num_tracers
    assert ensemble.num_steps == num_snapshots
    assert np.allclose(ensemble.times, times)
    assert 'q' in ensemble.sampled_properties
    assert ensemble.sampled_properties['q'].shape == (num_tracers, num_snapshots)
    assert np.allclose(ensemble.sampled_properties['q'][:, 1], 1.0)

def test_dynamics_error_handling():
    """Test error handling in dynamics components."""
    # DomainBox errors
    with pytest.raises(ValueError, match="Domain extent must be positive"):
        DomainBox(extent=0)
    
    # Trilinear interpolation out of bounds for non-periodic grid
    metadata_np = GridMetadata(dimensions=(10, 10, 10), origin=(0,0,0), spacing=(1,1,1), is_periodic=False)
    kernel_np = EightPointTrilinearKernel(metadata_np)
    field_np = np.zeros((10, 10, 10))
    
    with pytest.raises(ValueError, match="Positions are outside non-periodic bounds"):
        kernel_np.interpolate(field_np, np.array([[11, 0, 0]]))

    # wrap_to_periodic_domain errors
    domain = DomainBox(extent=1.0)
    from src.dynamics.domain import wrap_to_periodic_domain
    with pytest.raises(ValueError, match="must be an instance of DomainBox"):
        wrap_to_periodic_domain(np.zeros((1, 3)), None)
    with pytest.raises(ValueError, match="last dimension of size 3"):
        wrap_to_periodic_domain(np.zeros((1, 2)), domain)
