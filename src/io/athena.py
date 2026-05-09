from typing import NamedTuple, Sequence, Iterator, Tuple
from pathlib import Path
import numpy as np

from src.fields.grid import GridMetadata
from src.io.vtk.loader import VtkPathResolver, VtkSnapshotLoader, VtkSnapshotFields

class SnapshotData(NamedTuple):
    """Container for a single loaded snapshot's data and context.
    
    Attributes:
        index: The file index of the snapshot (e.g., 18903).
        physical_time: The simulation time associated with the snapshot.
        metadata: Physical and computational grid metadata.
        fields: Record containing the grid field data (velocity, density).
    """
    index: int
    physical_time: float
    metadata: GridMetadata
    fields: VtkSnapshotFields


class AthenaSnapshotSequence:
    """Manages the ingestion of a temporal sequence of Athena simulation snapshots.
    
    Allows lazy iteration over the collection of VTK files, enabling memory-efficient 
    processing of the 100 snapshots required for Lagrangian tracer tracking in EXP1.
    """

    def __init__(self, resolver: VtkPathResolver, loader: VtkSnapshotLoader, indices: Sequence[int]) -> None:
        """
        Args:
            resolver: Component to resolve snapshot indices to file system paths.
            loader: Specialized loader for Athena VTK files.
            indices: The sequence of snapshot indices to ingest (e.g., from 18903 to 19893).
        """
        self.resolver = resolver
        self.loader = loader
        self.indices = sorted(list(indices))

    def __iter__(self) -> Iterator[SnapshotData]:
        """Provides an iterator over the metadata and fields for every snapshot in the sequence.
        
        Returns:
            Iterator[SnapshotData]: A stream of snapshots for temporal analysis.

        Raises:
            FileNotFoundError: If a required snapshot file is missing from the data directory.
            IOError: If a snapshot file cannot be read or is corrupted.
        """
        for index in self.indices:
            try:
                path = self.resolver.get_path(index)
                
                # Load metadata, time, and fields lazily
                metadata = self.loader.load_metadata(path)
                physical_time = self.loader.load_time(path)
                fields = self.loader.load_fields(path)
                
                yield SnapshotData(
                    index=index,
                    physical_time=physical_time,
                    metadata=metadata,
                    fields=fields
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Snapshot file for index {index} not found.") from e
            except Exception as e:
                # Catching general Exceptions during file processing and re-raising as IOError
                raise IOError(f"Error processing snapshot at index {index}: {str(e)}") from e
class VelocitySnapshotStream:
    """A specialized stream for DNS velocity snapshots as required by Lagrangian tracking experiments.
    
    Filters a snapshot sequence to provide (time, velocity) pairs specifically 
    tailored for dispersion calculations (EXP1) and anisotropy analysis (EXP2).
    """

    def __init__(self, sequence: AthenaSnapshotSequence) -> None:
        """
        Args:
            sequence: The underlying snapshot sequence instance.
        """
        self.sequence = sequence

    def __iter__(self) -> Iterator[Tuple[float, np.ndarray]]:
        """Iterates through the sequence specifically providing velocity fields.
        
        Yields:
            Tuple[float, np.ndarray]: A pair containing (physical_time, velocity_field), 
                where velocity_field is an array of shape (NX, NY, NZ, 3).
        """
        for snapshot in self.sequence:
            # Transpose from (3, NX, NY, NZ) to (NX, NY, NZ, 3) as per docstring
            yield snapshot.physical_time, snapshot.fields.velocity.transpose(1, 2, 3, 0)
def create_ds1_sequence(raw_data_dir: Path, loader: VtkSnapshotLoader) -> AthenaSnapshotSequence:
    """Factory to create a sequence for the primary 'Solenoidal Turbulence DNS Velocity Snapshots' dataset (DS1).
    
    Configures the sequence with the exact index range (18903-19893) and step (10) 
    specified for the paper's primary 100-snapshot reproduction duration.

    Args:
        raw_data_dir: Path to the directory containing Turb.hydro_w.*.vtk files.
        loader: An instantiated Athena VTK loader.

    Returns:
        AthenaSnapshotSequence: A pre-configured sequence for the main experiments.
    """
    resolver = VtkPathResolver(raw_data_dir)
    indices = range(18903, 19893 + 1, 10)
    return AthenaSnapshotSequence(resolver, loader, indices)
