from typing import NamedTuple, Sequence, Iterator
from pathlib import Path

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
