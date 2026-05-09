import numpy as np


def apply_velocity_scale(velocity_field: np.ndarray, scale_factor: float) -> np.ndarray:
    """Applies a global scale factor to a 3D velocity vector field.

    This utility is used to normalize the velocity magnitudes derived from 
    DNS snapshots (e.g., Athena++ outputs) to ensure they are consistent 
    with the L=1 non-dimensional domain tracking requirements in EXP1.

    Args:
        velocity_field: A 4D numpy array of shape (NX, NY, NZ, 3) representing 
            the solenoidal velocity vectors on the structured grid.
        scale_factor: The multiplier applied to every component of the field.

    Returns:
        A new numpy array containing the scaled velocity vector field.

    Raises:
        ValueError: If the input velocity_field does not have the expected 
            4-dimensional structure (3 spatial dimensions + 1 vector dimension).
    """
    if velocity_field.ndim != 4:
        raise ValueError(
            f"Expected a 4D velocity field, but received a {velocity_field.ndim}D array. "
            f"Input must have shape (NX, NY, NZ, 3)."
        )

    return velocity_field * scale_factor
