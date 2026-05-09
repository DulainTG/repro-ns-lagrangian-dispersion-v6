import pytest
import numpy as np
from src.fields.grid import GridMetadata, StructuredField, FieldComponentLayout, CoordinateMapper, PeriodicTopology
from src.fields.operators import GradientTensorOperator, decompose_strain_vorticity, VorticityOperator
from src.fields.diagnostics import VortexDiagnostic
from src.fields.utils import apply_velocity_scale
from src.fields.spectral.transforms import SpectralTransformer, WavenumberIndexMap, SpectralCoefficientBuffer
from src.fields.spectral.filters import WavenumberBandMask, apply_band_mask

@pytest.fixture
def sample_grid_metadata():
    return GridMetadata(
        dimensions=(16, 16, 16),
        origin=(-0.5, -0.5, -0.5),
        spacing=(1.0/16, 1.0/16, 1.0/16),
        is_periodic=True,
        extent=1.0
    )

def test_grid_and_coordinate_integration(sample_grid_metadata):
    """Integrates GridMetadata, CoordinateMapper, and PeriodicTopology."""
    mapper = CoordinateMapper(sample_grid_metadata)
    topology = PeriodicTopology(extent=sample_grid_metadata.extent, origin=sample_grid_metadata.origin[0])
    
    # 1. Coordinate mapping
    world_pos = np.array([0.0, 0.0, 0.0])
    grid_idx = mapper.to_index(world_pos)
    # Origin is at -0.5. Spacing is 1/16.
    # index = (world - origin) / spacing = (0 - (-0.5)) / (1/16) = 0.5 * 16 = 8.0
    expected_idx = np.array([8.0, 8.0, 8.0])
    np.testing.assert_allclose(grid_idx, expected_idx)
    
    back_to_world = mapper.to_world(grid_idx)
    np.testing.assert_allclose(back_to_world, world_pos)
    
    # 2. Periodic wrapping
    out_of_bounds_pos = np.array([0.6, -0.6, 1.1])
    # origin is -0.5, extent is 1.0. Valid range is [-0.5, 0.5)
    # 0.6 -> -0.4
    # -0.6 -> 0.4
    # 1.1 -> 0.1
    wrapped_pos = topology.wrap_positions(out_of_bounds_pos)
    expected_wrapped = np.array([-0.4, 0.4, 0.1])
    np.testing.assert_allclose(wrapped_pos, expected_wrapped)
    
    # 3. Shortest displacement
    p1 = np.array([-0.4, 0.0, 0.0])
    p2 = np.array([0.4, 0.0, 0.0])
    # Direct distance is 0.8
    # Periodic distance is -0.2 (wrap around the other way)
    displacement = topology.calculate_displacement(p1, p2)
    np.testing.assert_allclose(displacement, np.array([-0.2, 0.0, 0.0]), atol=1e-7)

def test_velocity_analysis_pipeline(sample_grid_metadata):
    """Integrates Operators and Diagnostics for Q-criterion calculation."""
    # Create a simple velocity field: v = (y, 0, 0) -> simple shear
    # dvx/dy = 1, all other gradients are 0.
    nx, ny, nz = sample_grid_metadata.dimensions
    velocity = np.zeros((3, nx, ny, nz))
    
    # We need to use world coordinates for y to set velocity
    mapper = CoordinateMapper(sample_grid_metadata)
    for j in range(ny):
        world_y = mapper.to_world(np.array([0, j, 0]))[1]
        velocity[0, :, j, :] = world_y
        
    operator = GradientTensorOperator(sample_grid_metadata)
    grad_tensor = operator.compute_gradient_tensor(velocity)
    
    # Verify grad_tensor: A_ij = dv_i / dx_j
    # i=0, j=1 -> dvx/dy should be approx 1.0
    # Note: finite difference for a linear field should be exact.
    # But wait, periodic BC might affect boundaries if field is not periodic.
    # v=(y, 0, 0) is NOT periodic in y. 
    # Metadata says is_periodic=True.
    
    # Let's use a periodic field instead: v = (sin(2*pi*y), 0, 0)
    L = sample_grid_metadata.extent
    for j in range(ny):
        world_y = mapper.to_world(np.array([0, j, 0]))[1]
        velocity[0, :, j, :] = np.sin(2.0 * np.pi * world_y / L)
        
    grad_tensor = operator.compute_gradient_tensor(velocity)
    # dvx/dy = (2*pi/L) * cos(2*pi*y/L)
    
    decomposed = decompose_strain_vorticity(grad_tensor)
    
    diagnostic = VortexDiagnostic()
    q_field = diagnostic.generate_q_criterion_field(decomposed)
    
    assert q_field.shape == (nx, ny, nz)
    # Q = 0.5 * (||Omega||^2 - ||S||^2)
    # For this field:
    # A = [[0, dvx/dy, 0], [0, 0, 0], [0, 0, 0]]
    # S = 0.5 * (A + A^T) = [[0, 0.5*dvx/dy, 0], [0.5*dvx/dy, 0, 0], [0, 0, 0]]
    # Omega = 0.5 * (A - A^T) = [[0, 0.5*dvx/dy, 0], [-0.5*dvx/dy, 0, 0], [0, 0, 0]]
    # ||S||^2 = (0.5*dvx/dy)^2 + (0.5*dvx/dy)^2 = 0.5 * (dvx/dy)^2
    # ||Omega||^2 = (0.5*dvx/dy)^2 + (-0.5*dvx/dy)^2 = 0.5 * (dvx/dy)^2
    # Q = 0.5 * (0.5 * (dvx/dy)^2 - 0.5 * (dvx/dy)^2) = 0
    
    # In simple shear, rotation and strain are balanced, Q = 0.
    np.testing.assert_allclose(q_field, 0.0, atol=1e-7)

def test_spectral_filtering_integration(sample_grid_metadata):
    """Integrates SpectralTransformer, IndexMap, and BandMask."""
    nx, ny, nz = sample_grid_metadata.dimensions
    # Field with a single mode: sin(2*pi*x/L) -> n=1
    L = sample_grid_metadata.extent
    mapper = CoordinateMapper(sample_grid_metadata)
    field = np.zeros((nx, ny, nz))
    for i in range(nx):
        world_x = mapper.to_world(np.array([i, 0, 0]))[0]
        field[i, :, :] = np.sin(2.0 * np.pi * world_x / L)
    
    transformer = SpectralTransformer(sample_grid_metadata)
    coeffs = transformer.forward_fft_3d(field)
    
    index_map = WavenumberIndexMap(sample_grid_metadata)
    
    # 1. Mask that keeps n=1
    mask_keep = WavenumberBandMask(k_min=0.5, k_max=1.5, is_sharp=True) # n is around 1
    # Check k values: k = (2*pi/L) * n. Here L=1 so k = 2*pi * n.
    # n=1 -> k = 2*pi approx 6.28
    
    mask_n1 = WavenumberBandMask(k_min=6.0, k_max=7.0, is_sharp=True)
    weights = mask_n1.compute_weights(index_map)
    
    filtered_coeffs = apply_band_mask(coeffs, weights)
    reconstructed = transformer.reconstruct_physical_field(filtered_coeffs)
    
    # Should be almost identical to original
    np.testing.assert_allclose(reconstructed, field, atol=1e-10)
    
    # 2. Mask that excludes n=1
    mask_exclude = WavenumberBandMask(k_min=10.0, k_max=20.0, is_sharp=True)
    weights_ex = mask_exclude.compute_weights(index_map)
    filtered_coeffs_ex = apply_band_mask(coeffs, weights_ex)
    reconstructed_ex = transformer.reconstruct_physical_field(filtered_coeffs_ex)
    
    # Should be zero
    np.testing.assert_allclose(reconstructed_ex, 0.0, atol=1e-10)

def test_vorticity_and_scaling_integration(sample_grid_metadata):
    """Integrates apply_velocity_scale and VorticityOperator."""
    nx, ny, nz = sample_grid_metadata.dimensions
    L = sample_grid_metadata.extent
    dx, dy, dz = sample_grid_metadata.spacing
    mapper = CoordinateMapper(sample_grid_metadata)
    
    # Create velocity field v = (0, A*sin(2*pi*x/L), 0)
    velocity = np.zeros((nx, ny, nz, 3)) # INTERLEAVED layout
    A = 1.0 # Amplitude
    for i in range(nx):
        world_x = mapper.to_world(np.array([i, 0, 0]))[0]
        velocity[i, :, :, 1] = A * np.sin(2.0 * np.pi * world_x / L)
        
    scale = 2.5
    scaled_velocity = apply_velocity_scale(velocity, scale)
    
    # Convert to FLAT layout for VorticityOperator 
    flat_velocity = np.transpose(scaled_velocity, (3, 0, 1, 2))
    
    vort_op = VorticityOperator(sample_grid_metadata)
    vorticity = vort_op.compute_vorticity(flat_velocity)
    
    # Index where x = 0
    # origin = -0.5, spacing = 1/16 -> index 8 is x=0
    target_idx = 8
    world_x_at_target = mapper.to_world(np.array([target_idx, 0, 0]))[0]
    assert abs(world_x_at_target) < 1e-10
    
    # Expected dvy/dx at x=0 using central difference
    # v_y(x) = scale * A * sin(2*pi*x/L)
    # dv_y/dx = scale * A * (sin(2*pi*(x+dx)/L) - sin(2*pi*(x-dx)/L)) / (2*dx)
    expected_dv_y_dx = scale * A * (np.sin(2.0 * np.pi * dx / L) - np.sin(2.0 * np.pi * (-dx) / L)) / (2.0 * dx)
    
    # vorticity[2] = dv_y/dx - dv_x/dy = dv_y/dx
    np.testing.assert_allclose(vorticity[2, target_idx, 0, 0], expected_dv_y_dx, atol=1e-7)
    assert vorticity.shape == (3, nx, ny, nz)

def test_structured_field_integration(sample_grid_metadata):
    """Integrates StructuredField and FieldComponentLayout."""
    nx, ny, nz = sample_grid_metadata.dimensions
    
    # Test INTERLEAVED
    data_interleaved = np.random.rand(nx, ny, nz, 3)
    field_interleaved = StructuredField(data_interleaved, FieldComponentLayout.INTERLEAVED)
    assert field_interleaved.shape == (nx, ny, nz, 3)
    for c in range(3):
        np.testing.assert_array_equal(field_interleaved.get_component(c), data_interleaved[..., c])
    
    # Test FLAT
    data_flat = np.random.rand(3, nx, ny, nz)
    field_flat = StructuredField(data_flat, FieldComponentLayout.FLAT)
    assert field_flat.shape == (3, nx, ny, nz)
    for c in range(3):
        np.testing.assert_array_equal(field_flat.get_component(c), data_flat[c])
        
    # Test SCALAR
    data_scalar = np.random.rand(nx, ny, nz)
    field_scalar = StructuredField(data_scalar, FieldComponentLayout.SCALAR)
    assert field_scalar.shape == (nx, ny, nz)
    np.testing.assert_array_equal(field_scalar.get_component(0), data_scalar)
    with pytest.raises(ValueError):
        field_scalar.get_component(1)

def test_operator_error_handling(sample_grid_metadata):
    """Tests error handling in operators when input dimensions are mismatched."""
    operator = GradientTensorOperator(sample_grid_metadata)
    
    # Wrong number of components
    velocity_wrong_components = np.zeros((2, 16, 16, 16))
    with pytest.raises(ValueError, match="Expected 3 components"):
        operator.compute_gradient_tensor(velocity_wrong_components)
        
    # Wrong spatial dimensions
    velocity_wrong_dims = np.zeros((3, 32, 16, 16))
    with pytest.raises(ValueError, match="Velocity field dimensions"):
        operator.compute_gradient_tensor(velocity_wrong_dims)

def test_vorticity_operator_error_handling(sample_grid_metadata):
    """Tests error handling in VorticityOperator."""
    vort_op = VorticityOperator(sample_grid_metadata)
    
    # Wrong number of components
    velocity_wrong_components = np.zeros((2, 16, 16, 16))
    with pytest.raises(ValueError, match="Expected 3 components"):
        vort_op.compute_vorticity(velocity_wrong_components)
        
    # Wrong spatial dimensions
    velocity_wrong_dims = np.zeros((3, 32, 16, 16))
    with pytest.raises(ValueError, match="Velocity field dimensions"):
        vort_op.compute_vorticity(velocity_wrong_dims)

def test_spectral_filtering_error_handling(sample_grid_metadata):
    """Tests error handling in spectral filtering."""
    data = np.zeros((16, 16, 16))
    buffer = SpectralCoefficientBuffer(data)
    
    # Incompatible mask shape
    incompatible_mask = np.zeros((8, 8, 8))
    with pytest.raises(ValueError, match="operands could not be broadcast together"):
        apply_band_mask(buffer, incompatible_mask)

