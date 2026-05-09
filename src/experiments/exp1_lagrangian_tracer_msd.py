import numpy as np
import os
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from src.experiments import BaseExperiment
from src.experiments.outputs import ReproductionRunSettings, MsdAlphaResult, ReproductionCsvSerializer
from src.io.vtk.loader import AthenaVtkLoader, VtkPathResolver
from src.io.vtk.parser import AthenaHeaderParser, AthenaBinaryDataParser, VtkGridExtractor
from src.dynamics.domain import DomainBox
from src.dynamics.initialization import ParticleSeeder
from src.dynamics.tracking import PositionCoordinateBuffer, StateHistoryRecords, TracerStateBuffer
from src.dynamics.solver import RK4SolverIterationRoutine
from src.dynamics.interpolation import EightPointTrilinearKernel
from src.analysis.statistics import MsdCalculator, AlphaExponentEstimator, TransportRegime, identify_diffusion_regime


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

        if self.config.snapshot_limit > 0:
            self.snapshot_indices = self.snapshot_indices[:self.config.snapshot_limit]

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
        # Bridge shape mismatch: interpolation expects (NX, NY, NZ, 3)
        v_start = fields_start.velocity
        if v_start.shape[0] == 3 and v_start.ndim == 4:
            v_start = v_start.transpose(1, 2, 3, 0)
        times.append(t_start)

        # Integration loop across snapshot intervals
        for i in range(num_snapshots - 1):
            # Load the next snapshot
            path_end = self.path_resolver.get_path(self.snapshot_indices[i+1])
            t_end = self.loader.load_time(path_end)
            fields_end = self.loader.load_fields(path_end)
            # Bridge shape mismatch: interpolation expects (NX, NY, NZ, 3)
            v_end = fields_end.velocity
            if v_end.shape[0] == 3 and v_end.ndim == 4:
                v_end = v_end.transpose(1, 2, 3, 0)
            
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
                    v_start=v_start,
                    v_end=v_end,
                    alpha_t=alpha_t,
                    dt=dt,
                    dt_interval=dt_interval
                )
            
            # Update the buffer with positions at the end of the interval
            buffer.update_positions(current_positions)
            
            # Move to the next interval: reuse the end snapshot as the start of next
            t_start = t_end
            v_start = v_end

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
@dataclass(frozen=True)
class RegimeValidationReport:
    """Verification metrics for MSD transport regimes."""
    detected_regimes: List[TransportRegime]
    transition_times: List[float]
    is_c4_satisfied: bool
    ballistic_agreement_error: float
    diffusive_agreement_error: float


class RegimeTransitionValidator:
    """Verifies Claim C4: MSD transitions through ballistic and diffusive regimes.

    Validates that the scaling exponent alpha transitions from approx 2.0 to 1.0
    during the temporal evolution of the Lagrangian tracers.
    """

    def validate_msd_regimes(self, msd_results: MsdAlphaResult) -> RegimeValidationReport:
        """Perform diagnostic verification of transport regimes.

        Args:
            msd_results: Calculated MSD and alpha values from EXP1.

        Returns:
            Validation report comparing results against C4 claim thresholds.
        """
        lag_times = msd_results.lag_times
        alpha = msd_results.alpha_exponent

        boundaries = self.identify_regime_boundaries(lag_times, alpha)

        detected_regimes = [b[2] for b in boundaries]
        # Transition times correspond to the start of each new regime (after the first one)
        transition_times = [b[0] for b in boundaries[1:]]

        # C4 satisfaction Requirement: Transitions through ballistic (alpha=2) and diffusive (alpha=1)
        has_ballistic = TransportRegime.BALLISTIC in detected_regimes
        has_diffusive = TransportRegime.DIFFUSIVE in detected_regimes
        is_c4_satisfied = has_ballistic and has_diffusive

        # Calculate average deviation from ideal scaling in each detected regime
        ballistic_errors = []
        diffusive_errors = []

        for a in alpha:
            regime = identify_diffusion_regime(a)
            if regime == TransportRegime.BALLISTIC:
                ballistic_errors.append(abs(a - 2.0))
            elif regime == TransportRegime.DIFFUSIVE:
                diffusive_errors.append(abs(a - 1.0))

        ballistic_agreement_error = float(np.mean(ballistic_errors)) if ballistic_errors else float('nan')
        diffusive_agreement_error = float(np.mean(diffusive_errors)) if diffusive_errors else float('nan')

        return RegimeValidationReport(
            detected_regimes=detected_regimes,
            transition_times=transition_times,
            is_c4_satisfied=is_c4_satisfied,
            ballistic_agreement_error=ballistic_agreement_error,
            diffusive_agreement_error=diffusive_agreement_error
        )

    def identify_regime_boundaries(self, lag_times: np.ndarray, alpha: np.ndarray) -> List[Tuple[float, float, TransportRegime]]:
        """Identify continuous time intervals for each transport regime.

        Args:
            lag_times: Time indices for the trajectory snapshots.
            alpha: Local scaling exponent values.

        Returns:
            List of (start_time, end_time, regime) tuples.
        """
        if len(lag_times) == 0:
            return []

        regimes = [identify_diffusion_regime(a) for a in alpha]
        
        boundaries = []
        if not regimes:
            return boundaries

        current_regime = regimes[0]
        current_start_time = lag_times[0]
        
        for i in range(1, len(regimes)):
            if regimes[i] != current_regime:
                # Append previous regime info: (start_t, end_t, regime)
                boundaries.append((current_start_time, lag_times[i-1], current_regime))
                # Reset for new regime
                current_regime = regimes[i]
                current_start_time = lag_times[i]
        
        # Append the final regime
        boundaries.append((current_start_time, lag_times[-1], current_regime))
        
        return boundaries
