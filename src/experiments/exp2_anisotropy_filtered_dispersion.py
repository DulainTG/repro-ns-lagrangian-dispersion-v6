import numpy as np
import os
from pathlib import Path
from typing import List
from dataclasses import dataclass

from src.experiments import BaseExperiment
from src.experiments.outputs import ReproductionRunSettings, ReproductionCsvSerializer
from src.analysis.dispersion import AnisotropyMetrics, AnisotropyAnalyzer
from src.io.vtk.loader import AthenaVtkLoader, VtkPathResolver
from src.io.vtk.parser import AthenaHeaderParser, AthenaBinaryDataParser, VtkGridExtractor
from src.dynamics.domain import DomainBox
from src.dynamics.initialization import ParticleSeeder
from src.dynamics.tracking import PositionCoordinateBuffer, StateHistoryRecords, TracerStateBuffer
from src.dynamics.solver import RK4SolverIterationRoutine
from src.dynamics.interpolation import EightPointTrilinearKernel
from src.fields.spectral.transforms import SpectralTransformer, WavenumberIndexMap
from src.fields.spectral.filters import apply_sharp_band_filter


class FilteredAnisotropyExperiment(BaseExperiment):
    """Implementation of EXP2: Anisotropy of Filtered Dispersion.

    Goal: Reconstruct the dynamicanisotropy ratio lambda(t) by decomposing 
    displacements relative to a filtered large-scale velocity field (n=1 to 3).

    Required Outputs (per content contract):
        - csv_table: Columns [lag_time, msd_parallel, msd_perp, lambda_ratio]

    Args:
        config: Configuration containing spectral_filter_range (1, 3), 
            analysis_time_threshold (0.5), and number_of_tracers (8000).
    """

    def __init__(self, config: ReproductionRunSettings):
        """Initialize the experiment with typed settings."""
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
        self.analyzer = AnisotropyAnalyzer()
        self.serializer = ReproductionCsvSerializer()

        # Placeholders for data that will be prepared
        self.snapshot_indices: List[int] = []
        self.initial_positions: np.ndarray = np.empty((0, 3))
        self.grid_metadata = None

    def prepare(self) -> None:
        """Load DNS velocity snapshots and prepare the V_LS filtered field via 3D FFT.

        Raises:
            FileNotFoundError: If the snapshots from DS1 are missing.
            ValueError: If the filter range is outside valid wavenumber bounds.
        """
        self.snapshot_indices = self.path_resolver.list_available_indices()
        if not self.snapshot_indices:
            raise FileNotFoundError(f"No snapshots found in data directory: {self.raw_data_dir}")

        # Apply snapshot limit for smoke mode or customized runs
        if self.config.snapshot_limit > 0:
            self.snapshot_indices = self.snapshot_indices[:self.config.snapshot_limit]

        self.initial_positions = self.seeder.generate_initial_positions(self.config.number_of_tracers)

        first_path = self.path_resolver.get_path(self.snapshot_indices[0])
        self.grid_metadata = self.loader.load_metadata(first_path)

    def run(self) -> AnisotropyMetrics:
        """Execute displacement projections and calculate the anisotropy ratio lambda(t).

        Steps:
            1. Interpolate V_LS at tracer positions.
            2. Project delta_x(t) onto V_LS_unit and transverse directions.
            3. Calculate ensemble-averaged parallel and perpendicular MSD.

        Returns:
            AnisotropyMetrics: Structured container with lag times and MSD components.
        """
        if self.grid_metadata is None:
            raise RuntimeError("Experiment not prepared. Call prepare() first.")

        # Setup integration and filtering components
        kernel = EightPointTrilinearKernel(self.grid_metadata)
        routine = RK4SolverIterationRoutine(kernel, self.domain)
        transformer = SpectralTransformer(self.grid_metadata)
        index_map = WavenumberIndexMap(self.grid_metadata)

        num_tracers = self.config.number_of_tracers
        num_snapshots = len(self.snapshot_indices)
        K = self.config.sub_steps_per_snapshot

        # Buffers for recording positions at snapshots
        coord_buffer = PositionCoordinateBuffer(num_tracers, num_snapshots)
        state_records = StateHistoryRecords(num_tracers, num_snapshots)
        buffer = TracerStateBuffer(self.initial_positions, coord_buffer, state_records)

        times = []
        
        # Load the first snapshot and calculate V_LS at t=0
        path_start = self.path_resolver.get_path(self.snapshot_indices[0])
        t_start = self.loader.load_time(path_start)
        fields_start = self.loader.load_fields(path_start)
        times.append(t_start)

        # Calculate V_LS at t=0 for projection
        # V_LS is defined by wavenumber range n in [1, 3]
        v_ls_start = self._calculate_v_ls(fields_start.velocity, transformer, index_map)
        
        # v_ls_start has shape (NX, NY, NZ, 3).
        v_ls_at_p0 = kernel.interpolate(v_ls_start, self.initial_positions)
        v_ls_mag = np.linalg.norm(v_ls_at_p0, axis=1, keepdims=True)
        v_ls_unit_p0 = v_ls_at_p0 / np.where(v_ls_mag > 0, v_ls_mag, 1.0)

        # Bridge shape mismatch: interpolation expects (NX, NY, NZ, 3)
        v_start_interp = fields_start.velocity
        if v_start_interp.ndim == 4 and v_start_interp.shape[0] == 3:
            v_start_interp = v_start_interp.transpose(1, 2, 3, 0)

        # Integration loop across snapshot intervals
        for i in range(num_snapshots - 1):
            # Load the next snapshot
            path_end = self.path_resolver.get_path(self.snapshot_indices[i+1])
            t_end = self.loader.load_time(path_end)
            fields_end = self.loader.load_fields(path_end)
            
            v_end_interp = fields_end.velocity
            if v_end_interp.ndim == 4 and v_end_interp.shape[0] == 3:
                v_end_interp = v_end_interp.transpose(1, 2, 3, 0)
            
            dt_interval = t_end - t_start
            times.append(t_end)

            # Record position (and other properties if needed)
            # For EXP2, we need initial V_LS unit vector
            buffer.commit_to_history(i, samples={'v_ls_unit': v_ls_unit_p0})

            # Perform K integration sub-steps
            dt = dt_interval / K
            current_positions = buffer.current_positions
            for k in range(K):
                alpha_t = k / K
                current_positions = routine.step(
                    positions=current_positions,
                    v_start=v_start_interp,
                    v_end=v_end_interp,
                    alpha_t=alpha_t,
                    dt=dt,
                    dt_interval=dt_interval
                )
            
            buffer.update_positions(current_positions)
            
            t_start = t_end
            v_start_interp = v_end_interp

        # Record position at the final snapshot
        buffer.commit_to_history(num_snapshots - 1, samples={'v_ls_unit': v_ls_unit_p0})

        ensemble = buffer.build_ensemble(np.array(times))
        return self.analyzer.calculate_time_series(ensemble)

    def _calculate_v_ls(self, velocity: np.ndarray, transformer: SpectralTransformer, index_map: WavenumberIndexMap) -> np.ndarray:
        """Helper to calculate filtered V_LS field from velocity components.
        
        Correctly handles both (3, NX, NY, NZ) and (NX, NY, NZ, 3) input shapes.
        """
        is_channels_first = (velocity.ndim == 4 and velocity.shape[0] == 3)
        
        v_ls_components = []
        for d in range(3):
            # Extract component d: should have shape (NX, NY, NZ)
            v_comp = velocity[d, ...] if is_channels_first else velocity[..., d]
            
            coeffs = transformer.forward_fft_3d(v_comp)
            filtered_coeffs = apply_sharp_band_filter(coeffs, index_map, self.config.filter_range)
            v_ls_comp = transformer.reconstruct_physical_field(filtered_coeffs)
            v_ls_components.append(v_ls_comp)
            
        # return (NX, NY, NZ, 3) for the rest of the pipeline
        return np.stack(v_ls_components, axis=-1)

    def save_artifacts(self, results: AnisotropyMetrics) -> None:
        """Save the parallel/perpendicular MSD and ratio to a CSV table.

        Args:
            results: The metrics calculated during the run.
        """
        output_path = Path("exp2_anisotropy_results.csv")
        df = results.to_dataframe()
        self.serializer.save_table(df, output_path)


@dataclass(frozen=True)
class AnisotropyValidationReport:
    """Validation contract for Claim C1: Transverse-Dominant Anisotropic Dispersion.

    Attributes:
        asymptotic_lambda: The mean anisotropy ratio lambda(t) for t > 0.5.
        is_transverse_dominant: True if asymptotic_lambda < 1.0.
        is_c1_satisfied: True if the dispersion is systematically transverse-dominant.
        max_parallel_deviation: Maximum deviation from the mean for the parallel component.
        threshold_time: The time t after which the ratio is averaged (default 0.5).
    """
    asymptotic_lambda: float
    is_transverse_dominant: bool
    is_c1_satisfied: bool
    max_parallel_deviation: float
    threshold_time: float = 0.5


class TransverseDominanceValidator:
    """Verification engine for Claim C1 regarding anisotropic transport.

    Claim C1 states: Dispersion perpendicular to the instantaneous local 
    large-scale velocity field systematically exceeds parallel dispersion (lambda < 1).
    """

    def validate_anisotropy_ratio(self, metrics: AnisotropyMetrics, threshold_time: float = 0.5) -> AnisotropyValidationReport:
        """Validate the calculated anisotropy against the Transverse-Dominant claim.

        Args:
            metrics: The time-series metrics from EXP2.
            threshold_time: The lag time index after which the average lambda is computed.

        Returns:
            AnisotropyValidationReport: Detailed status of claim verification.
        """
        analyzer = AnisotropyAnalyzer()
        asymptotic_lambda = analyzer.calculate_asymptotic_ratio(metrics, start_time=threshold_time)
        is_transverse_dominant = analyzer.check_transverse_dominance(asymptotic_lambda)
        
        # Calculate max parallel deviation in the asymptotic regime
        mask = metrics.lag_times > threshold_time
        if np.any(mask):
            parallel_regime = metrics.msd_parallel[mask]
            mean_parallel = np.mean(parallel_regime)
            max_parallel_deviation = float(np.max(np.abs(parallel_regime - mean_parallel)))
        else:
            max_parallel_deviation = 0.0

        is_c1_satisfied = is_transverse_dominant

        return AnisotropyValidationReport(
            asymptotic_lambda=asymptotic_lambda,
            is_transverse_dominant=is_transverse_dominant,
            is_c1_satisfied=is_c1_satisfied,
            max_parallel_deviation=max_parallel_deviation,
            threshold_time=threshold_time
        )

    def check_claim_c1(self, report: AnisotropyValidationReport) -> bool:
        """Final binary verification of Claim C1 based on the validation report.

        Args:
            report: The pre-calculated validation report.

        Returns:
            bool: True if Claim C1 is strictly satisfied (lambda < 1).
        """
        return report.is_c1_satisfied
