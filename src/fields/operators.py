import numpy as np
from typing import NamedTuple, Tuple
from .grid import GridMetadata

def compute_central_difference(field: np.ndarray, axis: int, spacing: float, periodic: bool=True) -> np.ndarray:
    """Computes the second-order central finite difference along a specific axis.

    Args:
        field: The input N-dimensional array (scalar or vector component).
        axis: The axis along which to compute the derivative (0, 1, or 2).
        spacing: The physical grid spacing (dx, dy, or dz).
        periodic: Whether to use periodic boundary conditions for shifts.

    Returns:
        np.ndarray: The partial derivative field of the same shape as input.

    Raises:
        ValueError: If the axis is out of bounds or spacing is non-positive.
    """
    if spacing <= 0:
        raise ValueError("Spacing must be positive.")
    if axis < 0 or axis >= field.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for field of dimension {field.ndim}.")

    if periodic:
        return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2.0 * spacing)
    else:
        return np.gradient(field, spacing, axis=axis)

class GradientTensorOperator:
    """Computes the full velocity gradient tensor field using finite differences.

    This operator calculates A_ij = ∂v_i / ∂x_j, resulting in a 3x3 tensor field
    at every grid point, required for strain/vorticity analysis in EXP3.
    """

    def __init__(self, metadata: GridMetadata) -> None:
        """Initializes the operator with grid-specific spacing and topology.

        Args:
            metadata: Metadata containing grid dimensions, spacing, and periodicity.
        """
        self.metadata = metadata

    def compute_gradient_tensor(self, velocity_field: np.ndarray) -> np.ndarray:
        """Calculates the 3x3 gradient tensor for a vector velocity field.

        Args:
            velocity_field: Array of shape (3, Nx, Ny, Nz) representing (u, v, w).

        Returns:
            np.ndarray: Tensor field of shape (3, 3, Nx, Ny, Nz) where 
                index [i, j, ...] corresponds to ∂v_i / ∂x_j.
        """
        if velocity_field.shape[0] != 3:
            raise ValueError(f"Expected 3 components in velocity field, got {velocity_field.shape[0]}")
            
        nx, ny, nz = self.metadata.dimensions
        if velocity_field.shape[1:] != (nx, ny, nz):
            raise ValueError(f"Velocity field dimensions {velocity_field.shape[1:]} do not match metadata {self.metadata.dimensions}")

        gradient = np.empty((3, 3, nx, ny, nz), dtype=velocity_field.dtype)
        spacings = self.metadata.spacing
        periodic = self.metadata.is_periodic
        
        for i in range(3):
            for j in range(3):
                gradient[i, j] = compute_central_difference(
                    velocity_field[i],
                    axis=j,
                    spacing=spacings[j],
                    periodic=periodic
                )
        return gradient

class DecomposedGradient(NamedTuple):
    """Container for the symmetric and antisymmetric parts of the velocity gradient.

    Attributes:
        strain_rate_tensor: Symmetric part S = 0.5 * (∇v + (∇v)^T).
        vorticity_tensor: Antisymmetric part Ω = 0.5 * (∇v - (∇v)^T).
    """
    strain_rate_tensor: np.ndarray
    vorticity_tensor: np.ndarray

def decompose_strain_vorticity(gradient_tensor: np.ndarray) -> DecomposedGradient:
    """Decomposes the velocity gradient tensor into strain-rate and vorticity tensors.

    This decomposition is Step 2 of EXP3 and is essential for calculating 
    the Q-criterion (Eq 1: Q = 0.5 * (||Ω||^2 - ||S||^2)).

    Args:
        gradient_tensor: Array of shape (3, 3, Nx, Ny, Nz).

    Returns:
        DecomposedGradient: A named tuple containing the S and Ω tensor fields.
    """
    # Swap first two axes for transpose: (3, 3, Nx, Ny, Nz) -> (3, 3, Nx, Ny, Nz) with axes 0 and 1 swapped
    grad_T = np.transpose(gradient_tensor, (1, 0, 2, 3, 4))
    
    strain_rate = 0.5 * (gradient_tensor + grad_T)
    vorticity_tensor = 0.5 * (gradient_tensor - grad_T)
    
    return DecomposedGradient(strain_rate_tensor=strain_rate, vorticity_tensor=vorticity_tensor)

class VorticityOperator:
    """Applies the curl operator to calculate the vorticity vector field.

    The vorticity field ω = ∇ × v represents the local rotation of the fluid.
    This operator uses finite differences to compute the curl components 
    on a structured grid with periodic boundaries.
    """

    def __init__(self, metadata: GridMetadata) -> None:
        """Initializes the curl operator with grid spacing and topology.

        Args:
            metadata: Metadata containing dimensions, spacing, and periodicity.
        """
        self.metadata = metadata

    def compute_vorticity(self, velocity_field: np.ndarray) -> np.ndarray:
        """Computes the curl (vorticity) of the provided velocity field.

        Args:
            velocity_field: Vector field of shape (3, Nx, Ny, Nz).

        Returns:
            np.ndarray: The resulting vorticity vector field of shape (3, Nx, Ny, Nz).
                Component index 0 is (∂w/∂y - ∂v/∂z), etc.

        Raises:
            ValueError: If the input field does not match the metadata dimensions.
        """
        if velocity_field.shape[0] != 3:
            raise ValueError(f"Expected 3 components in velocity field, got {velocity_field.shape[0]}")
        
        nx, ny, nz = self.metadata.dimensions
        if velocity_field.shape[1:] != (nx, ny, nz):
            raise ValueError(f"Velocity field dimensions {velocity_field.shape[1:]} do not match metadata {self.metadata.dimensions}")

        u = velocity_field[0]
        v = velocity_field[1]
        w = velocity_field[2]
        
        dx, dy, dz = self.metadata.spacing
        periodic = self.metadata.is_periodic
        
        dw_dy = compute_central_difference(w, axis=1, spacing=dy, periodic=periodic)
        dv_dz = compute_central_difference(v, axis=2, spacing=dz, periodic=periodic)
        
        du_dz = compute_central_difference(u, axis=2, spacing=dz, periodic=periodic)
        dw_dx = compute_central_difference(w, axis=0, spacing=dx, periodic=periodic)
        
        dv_dx = compute_central_difference(v, axis=0, spacing=dx, periodic=periodic)
        du_dy = compute_central_difference(u, axis=1, spacing=dy, periodic=periodic)
        
        vorticity = np.stack([
            dw_dy - dv_dz,
            du_dz - dw_dx,
            dv_dx - du_dy
        ])
        
        return vorticity
