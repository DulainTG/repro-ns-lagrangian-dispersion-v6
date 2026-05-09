from dataclasses import dataclass
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
