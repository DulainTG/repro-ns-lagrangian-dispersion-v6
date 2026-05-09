import numpy as np
from dataclasses import dataclass
from src.fields.grid import GridMetadata, CoordinateMapper

@dataclass(frozen=True)
class EightPointWeightSet:
    """Container for trilinear interpolation indices and weights for N points.

    Attributes:
        indices: Grid indices of the 8 neighbors for each point, shape (N, 8, 3).
        weights: Interpolation coefficients for each neighbor, shape (N, 8).
            Weights for each point in N must sum to 1.0.
    """
    indices: np.ndarray
    weights: np.ndarray

class TrilinearWeightCalculator:
    """Calculates grid indices and interpolation weights for 3D coordinates.

    Handles periodic wrapping of indices based on GridMetadata constraints.
    """

    def __init__(self, metadata: GridMetadata) -> None:
        """Initialize with grid spacing and dimensions.

        Args:
            metadata: Metadata defining the grid resolution and extent.
        """
        self.metadata = metadata
        self.mapper = CoordinateMapper(metadata)
        self.dims = np.array(metadata.dimensions)

    def compute_weights(self, positions: np.ndarray) -> EightPointWeightSet:
        """Compute 8-point weights and indices for given world positions.

        Args:
            positions: Array of (x, y, z) coordinates, shape (N, 3).

        Returns:
            An EightPointWeightSet containing neighbor indices and weights.

        Raises:
            ValueError: If positions are outside non-periodic bounds.
        """
        # Convert world coordinates to continuous grid indices
        idx_cont = self.mapper.to_index(positions)
        
        # Base grid indices (integer part)
        idx_0 = np.floor(idx_cont).astype(int)
        
        # Fractional parts for weight calculation
        d = idx_cont - idx_0
        
        # Corner offsets for trilinear interpolation (8 corners of a cube)
        offsets = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ])
        
        # All 8 neighbor indices for each position: shape (N, 8, 3)
        indices = idx_0[:, np.newaxis, :] + offsets[np.newaxis, :, :]
        
        # Handle periodic boundaries or check bounds
        if self.metadata.is_periodic:
            # For periodic, wrap indices using modulo of dimensions
            indices = indices % self.dims
        else:
            # Check bounds for non-periodic grid
            # Positions must be within [0, dimensions-1) to allow a surrounding cube
            # We use a small epsilon for the upper bound to handle precision
            if np.any(idx_cont < 0) or np.any(idx_cont > self.dims - 1):
                raise ValueError("Positions are outside non-periodic bounds for interpolation.")

        dx = d[:, 0]
        dy = d[:, 1]
        dz = d[:, 2]
        
        # Weights for the 8 neighbors
        # Weights follow the standard trilinear interpolation formula:
        # P = Σ (w_i * P_i)
        weights = np.empty((positions.shape[0], 8))
        weights[:, 0] = (1 - dx) * (1 - dy) * (1 - dz)
        weights[:, 1] = dx * (1 - dy) * (1 - dz)
        weights[:, 2] = (1 - dx) * dy * (1 - dz)
        weights[:, 3] = dx * dy * (1 - dz)
        weights[:, 4] = (1 - dx) * (1 - dy) * dz
        weights[:, 5] = dx * (1 - dy) * dz
        weights[:, 6] = (1 - dx) * dy * dz
        weights[:, 7] = dx * dy * dz
        
        return EightPointWeightSet(indices=indices, weights=weights)

def accumulate_trilinear_field_values(field: np.ndarray, weight_set: EightPointWeightSet) -> np.ndarray:
    """Performs weighted summation of field values at specified grid indices.

    Args:
        field: The source grid data. Shape (NX, NY, NZ) for scalars 
            or (NX, NY, NZ, C) for vectors.
        weight_set: The indices and coefficients prepared by the calculator.

    Returns:
        Interpolated values at the tracer positions. 
        Shape (N,) for scalars or (N, C) for vectors.

    Raises:
        IndexError: If indices in weight_set are out of field bounds.
    """
    indices = weight_set.indices
    weights = weight_set.weights
    
    # Use advanced indexing to gather neighbor values
    # neighbor_values shape: (N, 8) or (N, 8, C)
    try:
        # field is indexed as (NX, NY, NZ, ...)
        # indices is (N, 8, 3)
        neighbor_values = field[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]]
    except IndexError as e:
        raise IndexError(f"Indices in weight_set are out of field bounds: {e}")

    if field.ndim == 3:  # Scalar field (NX, NY, NZ)
        # weights is (N, 8), neighbor_values is (N, 8)
        return np.sum(neighbor_values * weights, axis=1)
    else:  # Vector field (NX, NY, NZ, C)
        # neighbor_values is (N, 8, C)
        # weights[:, :, np.newaxis] has shape (N, 8, 1)
        return np.sum(neighbor_values * weights[:, :, np.newaxis], axis=1)
