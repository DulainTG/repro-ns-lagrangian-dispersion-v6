import numpy as np
from typing import Dict, Optional, Mapping
from src.analysis.trajectories import TrajectoryEnsemble

class PositionCoordinateBuffer:
    """
    Pre-allocated buffer for storing Lagrangian trajectory coordinates (x, y, z).
    
    This storage handles the 3D position history for all tracers, enabling 
    Mean-Square Displacement (MSD) calculation (Eq 3) and anisotropy analysis (Eq 4, 5).
    """

    def __init__(self, num_tracers: int, num_snapshots: int) -> None:
        """
        Initialize the coordinate buffer.
        
        Args:
            num_tracers: Number of tracers to track (e.g., 8,000 for EXP1).
            num_snapshots: Number of time indices to record (e.g., 100 snapshots).
        """
        self.num_tracers = num_tracers
        self.num_snapshots = num_snapshots
        # Pre-allocate buffer for (x, y, z) coordinates. Uses float64 for precision in MSD.
        self._data = np.zeros((num_tracers, num_snapshots, 3), dtype=np.float64)

    def record_step(self, snapshot_idx: int, positions: np.ndarray) -> None:
        """
        Append current positions to the buffer at the specified snapshot index.
        
        Args:
            snapshot_idx: Temporal index in the history sequence (0 to N-1).
            positions: Float array of shape (num_tracers, 3).
        
        Raises:
            IndexError: If snapshot_idx is out of pre-allocated bounds.
            ValueError: If positions shape does not match (num_tracers, 3).
        """
        if not (0 <= snapshot_idx < self.num_snapshots):
            raise IndexError(f"Snapshot index {snapshot_idx} out of range [0, {self.num_snapshots-1}]")
            
        if positions.shape != (self.num_tracers, 3):
            raise ValueError(f"Expected positions shape {(self.num_tracers, 3)}, got {positions.shape}")
            
        self._data[:, snapshot_idx, :] = positions

    def get_data(self) -> np.ndarray:
        """
        Retrieve the full position history tensor.
        
        Returns:
            An array of shape (num_tracers, num_snapshots, 3).
        """
        return self._data


class StateHistoryRecords:
    """
    Registry for time-series flow properties sampled along tracer trajectories.
    
    Stores scalar or vector fields (e.g., Q-criterion, filtered velocity) 
    sampled at snapshot timestamps, supporting EXP3 autocorrelation analysis.
    """

    def __init__(self, num_tracers: int, num_snapshots: int) -> None:
        """
        Initialize the records container.
        
        Args:
            num_tracers: Number of tracers being tracked.
            num_snapshots: Capacity for temporal sequence data.
        """
        self.num_tracers = num_tracers
        self.num_snapshots = num_snapshots
        self._records: Dict[str, np.ndarray] = {}

    def record_sample(self, property_name: str, snapshot_idx: int, values: np.ndarray) -> None:
        """
        Store sampled property values for the current time step.
        
        Args:
            property_name: The unique identifier (e.g., 'q_criterion', 'v_ls_unit').
            snapshot_idx: Current temporal index in the history.
            values: Array of shape (num_tracers, ...) containing sampled data.
        """
        if not (0 <= snapshot_idx < self.num_snapshots):
            raise IndexError(f"Snapshot index {snapshot_idx} out of range [0, {self.num_snapshots-1}]")
            
        if values.shape[0] != self.num_tracers:
            raise ValueError(f"Expected values.shape[0] to be {self.num_tracers}, got {values.shape[0]}")

        if property_name not in self._records:
            # Dynamic pre-allocation on first sample
            trailing_shape = values.shape[1:]
            buffer_shape = (self.num_tracers, self.num_snapshots) + trailing_shape
            self._records[property_name] = np.zeros(buffer_shape, dtype=values.dtype)
        
        # Verify shape consistency with pre-allocated buffer
        expected_shape = (self.num_tracers,) + self._records[property_name].shape[2:]
        if values.shape != expected_shape:
            raise ValueError(f"Shape mismatch for '{property_name}': expected {expected_shape}, got {values.shape}")

        self._records[property_name][:, snapshot_idx, ...] = values

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Extract all recorded properties as a dictionary of tensors.
        
        Returns:
            Mapping of property names to arrays of shape (num_tracers, num_snapshots, ...).
        """
        return self._records
class TracerStateBuffer:
    """
    Central manager for current particle positions and their associated history.
    
    Maintains the 'living' state of tracers during integration sub-steps while
    interfacing with historical coordinate and state buffers to build a TrajectoryEnsemble.
    """

    def __init__(self, initial_positions: np.ndarray, coord_buffer: PositionCoordinateBuffer, state_records: StateHistoryRecords) -> None:
        """
        Set up the tracking buffer with initial coordinates and history containers.
        
        Args:
            initial_positions: Starting coordinates (num_tracers, 3).
            coord_buffer: Permanent storage for trajectory positions.
            state_records: Permanent storage for sampled flow properties.
        """
        self._positions = np.array(initial_positions, copy=True, dtype=np.float64)
        self._coord_buffer = coord_buffer
        self._state_records = state_records

    @property
    def current_positions(self) -> np.ndarray:
        """The instantaneous world-coordinates of all tracers used for interpolation."""
        # Enforce periodic boundary conditions on L=1 cubic domain
        return self._positions % 1.0

    def update_positions(self, new_positions: np.ndarray) -> None:
        """Update the active coordinates after an integration step or boundary wrap."""
        self._positions = np.array(new_positions, copy=True, dtype=np.float64)

    def commit_to_history(self, snapshot_idx: int, samples: Optional[Mapping[str, np.ndarray]] = None) -> None:
        """
        Persist current snapshot state and properties to long-term storage.
        
        Args:
            snapshot_idx: The current temporal index to record.
            samples: Optional mapping of field values (Q, V_LS) sampled at current positions.
        """
        self._coord_buffer.record_step(snapshot_idx, self._positions)
        if samples is not None:
            for property_name, values in samples.items():
                self._state_records.record_sample(property_name, snapshot_idx, values)

    def build_ensemble(self, times: np.ndarray) -> TrajectoryEnsemble:
        """
        Construct a finalized trajectory dataset for analysis.
        
        Args:
            times: Array of physical times or indices for each recorded snapshot.
            
        Returns:
            A TrajectoryEnsemble suitable for EXP1/2/3 calculations.
        """
        # Ensure times is a numpy array
        times_arr = np.array(times, copy=True)
        return TrajectoryEnsemble(
            times=times_arr,
            positions=self._coord_buffer.get_data().copy(),
            sampled_properties=self._state_records.to_dict().copy()
        )

class HistoryStateRecorder:
    """
    Service responsible for periodic snapshot capture from the tracking system.
    
    Synchronizes snapshot-level tracking logic with the underlying buffers,
    ensuring integration sub-steps are correctly mapped to history records.
    """

    def __init__(self, tracking_buffer: TracerStateBuffer) -> None:
        """
        Initialize the recorder.
        
        Args:
            tracking_buffer: The target state buffer involved in the simulation.
        """
        self.tracking_buffer = tracking_buffer

    def capture(self, snapshot_idx: int, sampled_properties: Optional[Mapping[str, np.ndarray]] = None) -> None:
        """
        Execute a record operation for the current integration state.
        
        Args:
            snapshot_idx: The index of the VTK snapshot/time-point being recorded.
            sampled_properties: Fields like Q-criterion or V_LS sampled *at* tracer positions.
        """
        self.tracking_buffer.commit_to_history(snapshot_idx, samples=sampled_properties)
