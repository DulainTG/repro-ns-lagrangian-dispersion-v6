from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class GridMetadata:
    """Physical and computational metadata for the solenoidal turbulence grid.

    Attributes:
        dimensions: Number of grid points in (x, y, z).
        origin: Physical coordinates of the (0,0,0) grid point.
        spacing: Physical distance between adjacent grid points (dx, dy, dz).
        is_periodic: Whether the domain uses periodic boundary conditions.
        extent: The characteristic length L of the cubic domain (default 1.0).
    """
    dimensions: Tuple[int, int, int]
    origin: Tuple[float, float, float]
    spacing: Tuple[float, float, float]
    is_periodic: bool = True
    extent: float = 1.0