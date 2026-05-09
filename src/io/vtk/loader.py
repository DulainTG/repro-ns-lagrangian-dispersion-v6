import numpy as np
import re
from pathlib import Path
from typing import Sequence, NamedTuple

class VtkPathResolver:
    """Resolves and maps indices to VTK snapshot file paths for DS1.

    Specific to the Solenoidal Turbulence DNS Velocity Snapshots naming convention: 
    'Turb.hydro_w.{index}.vtk'.
    """

    def __init__(self, raw_data_dir: Path) -> None:
        """Initialize with the directory containing snapshots.

        Args:
            raw_data_dir: Path to the directory containing binary VTK files.
        """
        self.raw_data_dir = Path(raw_data_dir)

    def get_path(self, snapshot_index: int) -> Path:
        """Get the absolute path for a specific snapshot index.

        Args:
            snapshot_index: The numerical index of the snapshot (e.g., 18903).

        Returns:
            Path: The resolved file path.

        Raises:
            FileNotFoundError: If the mapped path does not exist.
        """
        path = self.raw_data_dir / f"Turb.hydro_w.{snapshot_index}.vtk"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {path}")
        return path

    def list_available_indices(self) -> Sequence[int]:
        """Lists all valid snapshot indices found in the data directory.

        Returns:
            Sequence[int]: Sorted list of available indices.
        """
        indices = []
        pattern = re.compile(r"Turb\.hydro_w\.(\d+)\.vtk")
        if not self.raw_data_dir.exists():
             return []
        for file_path in self.raw_data_dir.glob("Turb.hydro_w.*.vtk"):
            match = pattern.match(file_path.name)
            if match:
                indices.append(int(match.group(1)))
        return sorted(indices)

class VtkSnapshotFields(NamedTuple):
    """Container for the primary fields extracted from a VTK snapshot."""
    velocity: np.ndarray
    density: np.ndarray
