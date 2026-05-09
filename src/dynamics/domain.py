import numpy as np
from typing import Tuple

class DomainBox:
    """
    Defines the spatial constraints and box dimensions for the dynamic simulation.

    This class encapsulates the physical boundaries and scale (L) of the simulation 
    volume. It is used to parameterize the periodic modulo logic required during 
    Lagrangian tracer integration (EXP1) to maintain particles within a 
    consistent coordinate system.

    Args:
        extent: The characteristic length L of the cubic domain (default is 1.0 
            as specified in EXP1).
        origin: The minimum coordinates (x, y, z) of the domain box corner.
    """

    def __init__(self, extent: float = 1.0, origin: Tuple[float, float, float] = (-0.5, -0.5, -0.5)) -> None:
        if extent <= 0:
            raise ValueError("Domain extent must be positive.")
        self._extent = float(extent)
        self._origin = np.array(origin, dtype=float)
        if self._origin.shape != (3,):
            raise ValueError("Origin must have exactly 3 components.")

    @property
    def box_size(self) -> float:
        """Returns the side length L used for modulo operations."""
        return self._extent

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the lower and upper coordinate boundaries as arrays."""
        return self._origin, self._origin + self._extent


def wrap_to_periodic_domain(positions: np.ndarray, domain: DomainBox) -> np.ndarray:
    """
    Enforces periodic boundary conditions by applying modulo logic based on the domain box size.

    Implements the core 'boundary wrap' requirement for EXP1 (Procedure Step 4), 
    ensuring that tracer positions stay within the valid L=1 cubic domain [origin, origin + L) 
    at each integration sub-step.

    Args:
        positions: Array of particle positions of shape (N, 3).
        domain: The DomainBox configuration holding the size and origin constraints.

    Returns:
        The wrapped coordinates where each component is mapped into the primary 
        domain via (pos - origin) % box_size + origin.

    Raises:
        ValueError: If the domain configuration is invalid or positions array shape is incorrect.
    """
    if not isinstance(domain, DomainBox):
        raise ValueError("The 'domain' argument must be an instance of DomainBox.")
    
    # Ensure positions is a numpy array
    pos_arr = np.asanyarray(positions)
    
    # Check shape - we expect the last dimension to be 3 (x, y, z)
    if pos_arr.size == 0:
        return pos_arr
    
    if pos_arr.shape[-1] != 3:
        raise ValueError(
            f"Positions array must have a last dimension of size 3, but got shape {pos_arr.shape}"
        )
    
    # Perform periodic wrapping
    # formula: (pos - origin) % box_size + origin
    # Numpy's % operator handles negative numbers correctly for periodic BCs (e.g., -0.1 % 1.0 = 0.9)
    origin, _ = domain.bounds
    box_size = domain.box_size
    
    return ((pos_arr - origin) % box_size) + origin
