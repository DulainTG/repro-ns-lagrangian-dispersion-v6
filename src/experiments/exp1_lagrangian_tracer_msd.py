import numpy as np
import os
from pathlib import Path
from typing import List

from src.experiments import BaseExperiment
from src.experiments.outputs import ReproductionRunSettings, MsdAlphaResult, ReproductionCsvSerializer
from src.io.vtk.loader import AthenaVtkLoader, VtkPathResolver
from src.io.vtk.parser import AthenaHeaderParser, AthenaBinaryDataParser, VtkGridExtractor
from src.dynamics.domain import DomainBox
from src.dynamics.initialization import ParticleSeeder
from src.dynamics.tracking import PositionCoordinateBuffer, StateHistoryRecords, TracerStateBuffer
from src.dynamics.solver import RK4SolverIterationRoutine
from src.dynamics.interpolation import EightPointTrilinearKernel
from src.analysis.statistics import MsdCalculator, AlphaExponentEstimator


class MsdRegimeExperiment(BaseExperiment):
    """Implementation of EXP1: Lagrangian Tracer Integration and MSD Regimes.

    Reproduces the tracer trajectory reconstruction and temporal transport 
    regime identification (ballistic, superdiffusive, diffusive) as defined 
    by the Mean-Square Displacement (MSD) and the scaling exponent alpha.

    This experiment addresses Claim C4 by processing Solenoidal Turbulence 
    DNS snapshots (DS1).

    Required Outputs:
        Columns: lag_time, msd_mean, msd_std, alpha_exponent.
    """

    def __init__(self, config: ReproductionRunSettings):
        """Initialize EXP1 with specified number_of_tracers and RK4 settings.

        Args:
            config: Standardized settings for the reproduction run.
        """
        super().__init__(config)
        self.config: ReproductionRunSettings = config
        self.raw_data_dir = Path(os.environ.get('RAW_DATA_DIR', '/raw_data'))

        # Internal components
        self.path_resolver = VtkPathResolver(self.raw_data_dir)
        self.loader = AthenaVtkLoader(
            header_parser=AthenaHeaderParser(),
            data_parser=AthenaBinaryDataParser(),
            grid_extractor=VtkGridExtractor()
        )

        self.domain = DomainBox(extent=self.config.domain_extent)
        self.seeder = ParticleSeeder(self.domain)

        self.msd_calculator = MsdCalculator()
        self.alpha_estimator = AlphaExponentEstimator()
        self.serializer = ReproductionCsvSerializer()

        # Placeholders for data that will be prepared
        self.snapshot_indices: List[int] = []
        self.initial_positions: np.ndarray = np.empty((0, 3))
        self.grid_metadata = None

    def prepare(self) -> None:
        """Load snapshots from DS1 and initialize 8,000 tracers.

        Ensures the domain L=1 cubic and periodic boundary conditions are set.
        """
        self.snapshot_indices = self.path_resolver.list_available_indices()
        if not self.snapshot_indices:
            raise RuntimeError(f"No snapshots found in data directory: {self.raw_data_dir}")

        # Initialize 8,000 tracers uniformly in the domain
        # Using the number of tracers from config (expected to be 8,000)
        self.initial_positions = self.seeder.generate_initial_positions(self.config.number_of_tracers)

        # Load metadata from the first snapshot to initialize grid-dependent components
        first_path = self.path_resolver.get_path(self.snapshot_indices[0])
        self.grid_metadata = self.loader.load_metadata(first_path)

    def run(self) -> MsdAlphaResult:
        """Execute the core numerical procedure for EXP1.

        1. Perform RK4 integration with 10 sub-steps per snapshot.
        2. Store tracer positions x(t) at every snapshot.
        3. Calculate ensemble MSD(t) (Eq 3).
        4. Compute local scaling exponent alpha(t).

        Returns:
            Aggregated MSD and alpha statistics.
        """
        if self.grid_metadata is None:
            raise RuntimeError("Experiment not prepared. Call prepare() first.")

        # Setup integration components
        kernel = EightPointTrilinearKernel(self.grid_metadata)
        routine = RK4SolverIterationRoutine(kernel, self.domain)

        num_tracers = self.config.number_of_tracers
        num_snapshots = len(self.snapshot_indices)
        K = self.config.sub_steps_per_snapshot

        # Buffers for recording positions at snapshots
        coord_buffer = PositionCoordinateBuffer(num_tracers, num_snapshots)
        state_records = StateHistoryRecords(num_tracers, num_snapshots)
        buffer = TracerStateBuffer(self.initial_positions, coord_buffer, state_records)

        times = []
        
        # Load the first snapshot
        path_start = self.path_resolver.get_path(self.snapshot_indices[0])
        t_start = self.loader.load_time(path_start)
        fields_start = self.loader.load_fields(path_start)
        times.append(t_start)

        # Integration loop across snapshot intervals
        for i in range(num_snapshots - 1):
            # Load the next snapshot
            path_end = self.path_resolver.get_path(self.snapshot_indices[i+1])
            t_end = self.loader.load_time(path_end)
            fields_end = self.loader.load_fields(path_end)
            
            dt_interval = t_end - t_start
            times.append(t_end)

            # Record position at the start of the interval (snapshot i)
            buffer.commit_to_history(i)

            # Perform K integration sub-steps
            dt = dt_interval / K
            current_positions = buffer.current_positions
            for k in range(K):
                alpha_t = k / K
                current_positions = routine.step(
                    positions=current_positions,
                    v_start=fields_start.velocity,
                    v_end=fields_end.velocity,
                    alpha_t=alpha_t,
                    dt=dt,
                    dt_interval=dt_interval
                )
            
            # Update the buffer with positions at the end of the interval
            buffer.update_positions(current_positions)
            
            # Move to the next interval: reuse the end snapshot as the start of next
            t_start = t_end
            fields_start = fields_end

        # Record position at the final snapshot (snapshot num_snapshots - 1)
        buffer.commit_to_history(num_snapshots - 1)

        # Analysis
        ensemble = buffer.build_ensemble(np.array(times))
        msd_mean, msd_std = self.msd_calculator.compute_temporal_msd(ensemble)
        lag_times = ensemble.times - ensemble.times[0]
        alpha = self.alpha_estimator.calculate_local_alpha_slope(lag_times, msd_mean)

        return MsdAlphaResult(
            lag_times=lag_times,
            msd_mean=msd_mean,
            msd_std=msd_std,
            alpha_exponent=alpha
        )

    def save_artifacts(self, results: MsdAlphaResult) -> None:
        """Save the ensemble MSD and alpha values to the required CSV artifact.

        Args:
            results: The calculated result container.
        """
        output_path = Path("exp1_msd_alpha_results.csv")
        df = results.to_dataframe()
        self.serializer.save_table(df, output_path)
