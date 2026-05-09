import numpy as np
import re
from pathlib import Path
from typing import Sequence, NamedTuple, Protocol, runtime_checkable

from src.fields.grid import GridMetadata
from src.io.vtk.parser import AthenaHeaderParser, AthenaBinaryDataParser, VtkGridExtractor

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


@runtime_checkable
class VtkSnapshotLoader(Protocol):
    """Protocol defining the interface for loading fluid fields from VTK snapshots."""

    def load_metadata(self, path: Path) -> GridMetadata:
        """Extract physical and computational grid metadata.

        Args:
            path: Path to the VTK file.

        Returns:
            GridMetadata: Structured metadata including dimensions and spacing.
        """
        ...

    def load_time(self, path: Path) -> float:
        """Extract the physical simulation time from the snapshot.

        Args:
            path: Path to the VTK file.

        Returns:
            float: The simulation time.
        """
        ...

    def load_fields(self, path: Path) -> VtkSnapshotFields:
        """Load the primary velocity and density fields.

        Args:
            path: Path to the VTK file.

        Returns:
            VtkSnapshotFields: The vector velocity and scalar density fields.
        """
        ...


class AthenaVtkLoader(VtkSnapshotLoader):
    """Concrete loader for Athena++ VTK snapshots (DS1).

    Orchestrates header parsing and binary data block extraction to provide
    high-level access to turbulent velocity and density fields.
    """

    def __init__(
        self, 
        header_parser: AthenaHeaderParser, 
        data_parser: AthenaBinaryDataParser,
        grid_extractor: VtkGridExtractor
    ) -> None:
        """Initialize with specialized Athena++ parsers.

        Args:
            header_parser: Parser for VTK headers and Athena-specific comments.
            data_parser: Parser for extracting binary numeric field blocks.
            grid_extractor: Parser for grid geometry.
        """
        self.header_parser = header_parser
        self.data_parser = data_parser
        self.grid_extractor = grid_extractor

    def load_metadata(self, path: Path) -> GridMetadata:
        """Extract metadata from Athena VTK snapshot.

        Args:
            path: Path to the VTK file.

        Returns:
            GridMetadata: Extracted grid information.
        """
        with open(path, "rb") as f:
            # Parse header metadata and validate Athena format
            _ = self.header_parser.parse_header_metadata(f)
            return self.grid_extractor.extract_grid(f)

    def load_time(self, path: Path) -> float:
        """Extract simulation time from Athena VTK snapshot.

        Args:
            path: Path to the VTK file.

        Returns:
            float: Extracted simulation time.
        """
        with open(path, "rb") as f:
            metadata = self.header_parser.parse_header_metadata(f)
            return float(metadata.get("time", 0.0))

    def load_fields(self, path: Path) -> VtkSnapshotFields:
        """Load velocity and density fields from an Athena VTK file.

        Args:
            path: Path to the VTK file.

        Returns:
            VtkSnapshotFields: Parsed field data (velocity and density).
        """
        grid = self.load_metadata(path)
        with open(path, "rb") as f:
            velocity = self.data_parser.read_velocity_field(f, grid)
            density = self.data_parser.read_density_field(f, grid)
        return VtkSnapshotFields(velocity=velocity, density=density)


