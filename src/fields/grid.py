from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple
import numpy as np

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


class FieldComponentLayout(Enum):
    """Defines the memory layout of vector field components in a structured grid.

    The layout determines if multiple components (e.g., vx, vy, vz) are stored 
    consecutively for each point (interleaved) or in separate component-major blocks.
    """
    INTERLEAVED = auto()
    FLAT = auto()
    SCALAR = auto()


class StructuredField:
    """Encapsulates a numerical field stored on a structured point grid.

    This class manages the lifecycle and access patterns of the underlying numpy 
    arrays extracted from VTK snapshots (e.g., density or velocity fields).

    Args:
        data: The underlying numerical array.
        layout: FieldComponentLayout used by the data.
    """

    def __init__(self, data: np.ndarray, layout: FieldComponentLayout) -> None:
        self._data = data
        self._layout = layout

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the grid dimensions including component depth."""
        return self._data.shape

    def get_component(self, component_idx: int) -> np.ndarray:
        """Extract a specific scalar component block from the field.

        Args:
            component_idx: Index of the component (0 for X, 1 for Y, etc.).

        Returns:
            A 3D numpy array representing the scalar field of that component.

        Raises:
            ValueError: If the component index is invalid for the current layout.
        """
        if self._layout == FieldComponentLayout.SCALAR:
            if component_idx != 0:
                raise ValueError(
                    f"Scalar field only has component 0, requested index {component_idx}"
                )
            return self._data

        if self._layout == FieldComponentLayout.FLAT:
            # Expected shape: (C, NX, NY, NZ)
            if component_idx < 0 or component_idx >= self._data.shape[0]:
                raise ValueError(
                    f"Invalid component index {component_idx} for FLAT layout with shape {self._data.shape}"
                )
            return self._data[component_idx]

        if self._layout == FieldComponentLayout.INTERLEAVED:
            # Expected shape: (NX, NY, NZ, C)
            if component_idx < 0 or component_idx >= self._data.shape[-1]:
                raise ValueError(
                    f"Invalid component index {component_idx} for INTERLEAVED layout with shape {self._data.shape}"
                )
            return self._data[..., component_idx]

        raise ValueError(f"Unsupported or unknown layout: {self._layout}")
class CoordinateMapper:
    """Translates between physical world coordinates and discrete grid indices.

    Essential for EXP1 and EXP3 to perform trilinear interpolation of velocity 
    and Q-criterion fields at arbitrary tracer positions.

    Args:
        metadata: Metadata describing the grid origin, spacing, and dimensions.
    """

    def __init__(self, metadata: 'GridMetadata') -> None:
        self._metadata = metadata
        self._origin = np.array(metadata.origin)
        self._spacing = np.array(metadata.spacing)

    def to_index(self, world_coords: np.ndarray) -> np.ndarray:
        """Convert physical coordinates to continuous grid indices.

        Args:
            world_coords: Array of (x, y, z) positions, shape (..., 3).

        Returns:
            Floating-point indices (i, j, k) for interpolation, shape (..., 3).
        """
        return (world_coords - self._origin) / self._spacing

    def to_world(self, grid_indices: np.ndarray) -> np.ndarray:
        """Convert grid indices back to physical world coordinates.

        Args:
            grid_indices: Floating-point or integer indices (i, j, k).

        Returns:
            Physical (x, y, z) coordinates.
        """
        return (grid_indices * self._spacing) + self._origin


class PeriodicTopology:
    """Enforces periodic boundary conditions on a cubic domain.

    Implements the L=1 periodic cubic domain logic required for EXP1 Lagrangian 
    tracer integration, ensuring tracers stay within the physical bounds [-0.5, 0.5].

    Args:
        extent: The characteristic length L of the cubic domain (default 1.0).
        origin: The physical center or starting point of the domain.
    """

    def __init__(self, extent: float = 1.0, origin: float = -0.5) -> None:
        self._extent = extent
        self._origin = origin

    def wrap_positions(self, positions: np.ndarray) -> np.ndarray:
        """Apply modulo L logic to wrap positions back into the periodic domain.

        Args:
            positions: Array of tracer coordinates, shape (..., 3).

        Returns:
            Wrapped coordinates within the range [origin, origin + extent).
        """
        return ((positions - self._origin) % self._extent) + self._origin

    def calculate_displacement(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Calculate the shortest Periodic distance vector between two points.

        Args:
            start: Starting position vector.
            end: Ending position vector.

        Returns:
            The displacement vector accounting for periodic boundary crossings.
        """
        diff = end - start
        return (diff + self._extent / 2) % self._extent - self._extent / 2
