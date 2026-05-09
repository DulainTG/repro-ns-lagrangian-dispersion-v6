from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReproductionRunSettings:
    """Parameters defining the exact configuration used for a reproduction run.

    Matches requirements for EXP1, EXP2, and EXP3 integration and analysis settings.

    Attributes:
        number_of_tracers: Total tracers in the ensemble (normally 8,000).
        integration_scheme: Numerical method used (e.g., 'RK4').
        sub_steps_per_snapshot: Internal steps between snapshots (normally 10).
        interpolation: Spatial interpolation method (e.g., 'trilinear').
        filter_range: Spectral wavenumbers for LS decomposition (EXP2, normally [1, 3]).
        analysis_time_threshold: Start time for averaging asymptotic ratios (EXP2, t > 0.5).
    """
    number_of_tracers: int
    integration_scheme: Literal['RK4', 'Euler']
    sub_steps_per_snapshot: int
    interpolation: Literal['trilinear', 'nearest']
    filter_range: tuple[int, int]
    analysis_time_threshold: float
    domain_extent: float = 1.0


@dataclass(frozen=True)
class MsdAlphaResult:
    """Structured container for Mean-Square Displacement and its scaling exponent.

    Satisfies the EXP1 content contract for CSV output.

    Attributes:
        lag_times: Array of time intervals t.
        msd_mean: Ensemble-averaged MSD values at each lag.
        msd_std: Standard deviation of MSD across the ensemble.
        alpha_exponent: Local scaling exponent d(log MSD)/d(log t).
    """
    lag_times: np.ndarray
    msd_mean: np.ndarray
    msd_std: np.ndarray
    alpha_exponent: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert analysis results into a standard tabular format for serialization.

        Returns:
            DataFrame with columns: lag_time, msd_mean, msd_std, alpha_exponent.
        """
        return pd.DataFrame({
            'lag_time': self.lag_times,
            'msd_mean': self.msd_mean,
            'msd_std': self.msd_std,
            'alpha_exponent': self.alpha_exponent
        })


@dataclass(frozen=True)
class TimescaleSummary:
    """Summary metrics for vortex residence time analysis (EXP3).

    Attributes:
        tau_q: The time where Lagrangian Q-autocorrelation drops to 1/e.
        tau_q_over_te: The timescale normalized by the large-eddy turnover time.
        vortex_threshold: The Q-criterion value used to define trapping (usually > 0).
    """
    tau_q: float
    tau_q_over_te: float
    vortex_threshold: float = 0.0


class ReportFormatter:
    """Converts experiment results into formatted text summaries for CLI or file export."""

    def format_timescale_report(self, summary: TimescaleSummary) -> str:
        """Generate a human-readable text report for EXP3 characteristic timescales.

        Args:
            summary: The calculated trapping timescale metrics.

        Returns:
            A string containing fields: tau_Q, tau_Q_over_Te.
        """
        lines = [
            "--------------------------------------------------",
            "EXP3: VORTEX TRAPPING TIMESCALE ANALYSIS",
            "--------------------------------------------------",
            f"Vortex Criterion (Q_threshold):  {summary.vortex_threshold:.4e}",
            f"Lagrangian Auto-correlation Time (tau_Q): {summary.tau_q:.6f}",
            f"Normalized Timescale (tau_Q / Te):        {summary.tau_q_over_te:.6f}",
            "--------------------------------------------------"
        ]
        return "\n".join(lines)


class ReproductionCsvSerializer:
    """Handles persistent storage of experimental results in CSV format.

    Ensures filenames and column schemas follow the paper's reproduction requirements.
    """

    def save_table(self, data: pd.DataFrame, output_path: Path) -> None:
        """Serialize a dataframe to CSV with consistent precision and encoding.

        Args:
            data: The tabular results (e.g., MSD or Anisotropy results).
            output_path: Destination file system path.

        Raises:
            IOError: If the directory is not writable.
            ValueError: If the dataframe schema does not match expected output contracts.
        """
        if not isinstance(output_path, Path):
            output_path = Path(output_path)

        # Validation of column contracts for EXP1 and EXP2
        required_exp1 = {'lag_time', 'msd_mean', 'msd_std', 'alpha_exponent'}
        required_exp2 = {'lag_time', 'msd_parallel', 'msd_perp', 'lambda_ratio'}

        actual_cols = set(data.columns)
        is_exp1 = required_exp1.issubset(actual_cols)
        is_exp2 = required_exp2.issubset(actual_cols)

        if not (is_exp1 or is_exp2):
            raise ValueError(
                f"Dataframe columns {data.columns.tolist()} do not match EXP1 or EXP2 contracts."
            )

        try:
            # Ensure the parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize to CSV with fixed precision (6 decimal places in scientific notation)
            # as is common in scientific data sharing for this field.
            data.to_csv(output_path, index=False, float_format='%.6e', encoding='utf-8')
        except (PermissionError, OSError) as e:
            raise IOError(f"Could not write to output path {output_path}: {e}") from e
