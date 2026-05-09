import numpy as np
import os
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from src.experiments import BaseExperiment
from src.experiments.outputs import (
    ReproductionRunSettings, 
    TimescaleSummary, 
    ReproductionCsvSerializer,
    ReportFormatter
)
from src.io.vtk.loader import AthenaVtkLoader, VtkPathResolver
from src.io.vtk.parser import AthenaHeaderParser, AthenaBinaryDataParser, VtkGridExtractor
from src.dynamics.domain import DomainBox
from src.dynamics.initialization import ParticleSeeder
from src.dynamics.tracking import PositionCoordinateBuffer, StateHistoryRecords, TracerStateBuffer
from src.dynamics.solver import RK4SolverIterationRoutine
from src.dynamics.interpolation import EightPointTrilinearKernel
from src.fields.operators import GradientTensorOperator, decompose_strain_vorticity
from src.fields.diagnostics import VortexDiagnostic
from src.analysis.time_series import compute_q_signal_autocorrelation
from src.analysis.vorticity import VorticityTimescaleDeriver


@dataclass(frozen=True)
class VortexAutocorrelationResult:
    """Structured results for the Vortex Residence and Q-Autocorrelation experiment.

    Coordinates with the required experiment outputs for EXP3.

    Attributes:
        lag_times: Array of time lags for the autocorrelation function.
        autocorrelation_q_normalized: Ensemble-averaged normalized Q-criterion autocorrelation.
        summary: Calculated timescale metrics (tau_Q and tau_Q/Te).
    """
    lag_times: np.ndarray
    autocorrelation_q_normalized: np.ndarray
    summary: TimescaleSummary


class VortexTrappingExperiment(BaseExperiment):
    """Implementation of Experiment 3: Vortex Residence and Q-Autocorrelation.

    Characterizes vortex trapping duration by calculating the Q-criterion along 
    trajectories and its Lagrangian autocorrelation to identify the timescale tau_Q.

    Experiment ID: EXP3.
    """

    def __init__(self, config: ReproductionRunSettings):
        """Initialize the experiment with specific reproduction settings.

        Args:
            config: Standardized reproduction run settings.
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
        self.vortex_diagnostic = VortexDiagnostic()
        self.serializer = ReproductionCsvSerializer()

        # Placeholders for data that will be prepared
        self.snapshot_indices: List[int] = []
        self.initial_positions: np.ndarray = np.empty((0, 3))
        self.grid_metadata = None

    def prepare(self) -> None:
        """Load snapshots and compute the Q-criterion scalar field (Eq 1)."""
        self.snapshot_indices = self.path_resolver.list_available_indices()
        if not self.snapshot_indices:
            raise RuntimeError(f"No snapshots found in data directory: {self.raw_data_dir}")

        self.initial_positions = self.seeder.generate_initial_positions(self.config.number_of_tracers)

        # Load metadata from the first snapshot
        if self.snapshot_indices:
            first_path = self.path_resolver.get_path(self.snapshot_indices[0])
            self.grid_metadata = self.loader.load_metadata(first_path)

    def run(self) -> VortexAutocorrelationResult:
        """Execute the Lagrangian sampling, autocorrelation calculation, and ensemble averaging.

        Returns:
            VortexAutocorrelationResult: The normalized autocorrelation curve and summary metrics.
        """
        if self.grid_metadata is None:
            raise RuntimeError("Experiment not prepared. Call prepare() first.")

        # Setup integration and diagnostic components
        kernel = EightPointTrilinearKernel(self.grid_metadata)
        routine = RK4SolverIterationRoutine(kernel, self.domain)
        grad_op = GradientTensorOperator(self.grid_metadata)

        num_tracers = self.config.number_of_tracers
        num_snapshots = len(self.snapshot_indices)
        K = self.config.sub_steps_per_snapshot

        # Buffers for recording positions and Q-criterion at snapshots
        coord_buffer = PositionCoordinateBuffer(num_tracers, num_snapshots)
        state_records = StateHistoryRecords(num_tracers, num_snapshots)
        buffer = TracerStateBuffer(self.initial_positions, coord_buffer, state_records)

        times = []

        # Load the first snapshot
        path_start = self.path_resolver.get_path(self.snapshot_indices[0])
        t_start = self.loader.load_time(path_start)
        fields_start = self.loader.load_fields(path_start)

        # Integration loop across snapshot intervals
        for i in range(num_snapshots - 1):
            # Load the next snapshot
            path_end = self.path_resolver.get_path(self.snapshot_indices[i + 1])
            t_end = self.loader.load_time(path_end)
            fields_end = self.loader.load_fields(path_end)

            dt_interval = t_end - t_start
            times.append(t_start)

            # --- Sample Q at current snapshot ---
            grad_tensor = grad_op.compute_gradient_tensor(fields_start.velocity)
            decomp = decompose_strain_vorticity(grad_tensor)
            q_field = self.vortex_diagnostic.generate_q_criterion_field(decomp)
            
            # Sample Q at current positions
            q_sampled = kernel.interpolate(q_field, buffer.current_positions)
            buffer.commit_to_history(i, samples={'q_criterion': q_sampled})

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

            # Move to the next interval
            t_start = t_end
            fields_start = fields_end

        # Record at the final snapshot
        times.append(t_start)
        grad_tensor = grad_op.compute_gradient_tensor(fields_start.velocity)
        decomp = decompose_strain_vorticity(grad_tensor)
        q_field = self.vortex_diagnostic.generate_q_criterion_field(decomp)
        q_sampled = kernel.interpolate(q_field, buffer.current_positions)
        buffer.commit_to_history(num_snapshots - 1, samples={'q_criterion': q_sampled})

        # --- Analysis ---
        ensemble = buffer.build_ensemble(np.array(times))
        autocorr = compute_q_signal_autocorrelation(ensemble)
        lag_times = ensemble.times - ensemble.times[0]

        validator = VortexTrappingValidator()
        tau_q = validator.estimate_residence_time(autocorr, lag_times)
        
        # Reference large-eddy turnover time Te. Default to 1.0
        te_reference = 1.0 
        summary = TimescaleSummary(
            tau_q=tau_q,
            tau_q_over_te=tau_q / te_reference,
            vortex_threshold=0.0
        )

        return VortexAutocorrelationResult(
            lag_times=lag_times,
            autocorrelation_q_normalized=autocorr,
            summary=summary
        )

    def save_artifacts(self, results: VortexAutocorrelationResult) -> None:
        """Save the Lagrangian autocorrelation CSV and the timescale report.

        Args:
            results: The data to be persisted as defined in the content contracts.
        """
        # Save CSV identifying the autocorrelation curve
        df = pd.DataFrame({
            'lag_time': results.lag_times,
            'autocorrelation_q_normalized': results.autocorrelation_q_normalized
        })
        output_path_csv = Path("exp3_vortex_autocorr.csv")
        df.to_csv(output_path_csv, index=False, float_format='%.6e')

        # Save Text Report for Claim C2 verification
        formatter = ReportFormatter()
        report_text = formatter.format_timescale_report(results.summary)
        output_path_txt = Path("exp3_timescale_report.txt")
        with open(output_path_txt, "w") as f:
            f.write(report_text)


@dataclass(frozen=True)
class TrappingValidationReport:
    """Comparison of measured vortex trapping statistics against Paper Claim C2.

    Attributes:
        tau_q: The measured 1/e decay timescale.
        tau_q_over_te: The measured ratio to large-eddy turnover time (Te).
        is_c2_satisfied: Boolean indicating if the ratio is close to the target 0.07.
        relative_deviation: Relative error between measured and targeted (0.07) ratio.
    """
    tau_q: float
    tau_q_over_te: float
    is_c2_satisfied: bool
    relative_deviation: float


class VortexTrappingValidator:
    """Validates vortex residence statistics against established turbulence claims.

    Specifically handles Claim C2 regarding the transience of vortex trapping events.
    """

    def validate_trapping_timescale(self, summary: TimescaleSummary, target_ratio: float=0.07) -> TrappingValidationReport:
        """Validates the measured trapping timescale against the expected paper ratio.

        Args:
            summary: The calculated timescale metrics from EXP3.
            target_ratio: The claim value (tau_Q / Te approx 0.07).

        Returns:
            TrappingValidationReport: Quantitative comparison and pass/fail status.
        """
        tau_q = summary.tau_q
        ratio = summary.tau_q_over_te
        
        # C2 check: tolerance of 50% around the claimed value 0.07 (0.035 - 0.105)
        is_c2_satisfied = 0.035 <= ratio <= 0.105
        
        relative_deviation = abs(ratio - target_ratio) / target_ratio
        
        return TrappingValidationReport(
            tau_q=tau_q,
            tau_q_over_te=ratio,
            is_c2_satisfied=bool(is_c2_satisfied),
            relative_deviation=float(relative_deviation)
        )

    def check_timescale_bounds(self, tau_q: float, min_tau: float, max_tau: float) -> bool:
        """Verify if the estimated tau_Q falls within physically plausible bounds for solenoidal turbulence.

        Args:
            tau_q: Estimated residence timescale.
            min_tau: Lower bound for validity.
            max_tau: Upper bound for validity.

        Returns:
            bool: True if within bounds.
        """
        return min_tau <= tau_q <= max_tau

    def estimate_residence_time(self, autocorrelation: np.ndarray, lag_times: np.ndarray) -> float:
        """Estimate the residence timescale using the 1/e crossing method.

        Args:
            autocorrelation: Normalized autocorrelation values.
            lag_times: Corresponding lag time steps.

        Returns:
            float: The interpolated time where autocorrelation equals 1/e.
        """
        deriver = VorticityTimescaleDeriver()
        return deriver.derive_tau_q(lag_times, autocorrelation)
