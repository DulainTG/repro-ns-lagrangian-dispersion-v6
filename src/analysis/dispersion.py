import numpy as np
from dataclasses import dataclass
from .trajectories import TrajectoryEnsemble, calculate_trajectory_displacements

@dataclass(frozen=True)
class AnisotropyMetrics:
    """Data contract for filtered dispersion anisotropy results (EXP2).

    Attributes:
        lag_times: Array of time lags corresponding to snapshots.
        msd_parallel: Mean-square displacement projected onto the local V_LS direction.
        msd_perp: Mean-square displacement perpendicular to the local V_LS direction.
        lambda_ratio: The ratio of parallel to perpendicular MSD (msd_parallel / msd_perp).
    """
    lag_times: np.ndarray
    msd_parallel: np.ndarray
    msd_perp: np.ndarray
    lambda_ratio: np.ndarray

    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert anisotropy results into a standard tabular format for serialization.

        Returns:
            DataFrame with columns: lag_time, msd_parallel, msd_perp, lambda_ratio.
        """
        import pandas as pd
        return pd.DataFrame({
            'lag_time': self.lag_times,
            'msd_parallel': self.msd_parallel,
            'msd_perp': self.msd_perp,
            'lambda_ratio': self.lambda_ratio
        })



class AnisotropyAnalyzer:
    """Calculates dispersion anisotropy by decomposing Lagrangian displacements.

    This analyzer supports EXP2 (Anisotropy of Filtered Dispersion) by comparing 
    particle transport relative to the instantaneous local large-scale velocity field (V_LS).
    """

    def calculate_time_series(self, ensemble: TrajectoryEnsemble) -> AnisotropyMetrics:
        """Calculates the time-evolving parallel/perpendicular MSDs and lambda ratio.

        Uses Eq 4 (MSD_parallel) and Eq 5 (MSD_perp) from the paper specification.

        Args:
            ensemble: Trajectory data containing positions and 'v_ls_unit' samples.

        Returns:
            AnisotropyMetrics containing the time series for EXP2 output.

        Raises:
            KeyError: If 'v_ls_unit' is missing from the ensemble properties.
        """
        if 'v_ls_unit' not in ensemble.sampled_properties:
            raise KeyError("Property 'v_ls_unit' is missing from the ensemble properties.")

        # Calculate unwrapped displacements Delta_x(t) = x(t) - x(0)
        # Assuming L=1 periodic domain as specified for these experiments.
        displacements = calculate_trajectory_displacements(ensemble, extent=1.0, unwrap=True)

        # direction of the large-scale velocity at the initial position
        v_ls_0 = ensemble.sampled_properties['v_ls_unit'][:, 0, :]

        # Project displacements onto the local initial V_LS direction
        # disp_parallel shape: (N, T)
        disp_parallel = np.einsum('nti,ni->nt', displacements, v_ls_0)

        # MSD_parallel(t) = < (Delta x . n)^2 > (Eq 4)
        msd_parallel = np.mean(disp_parallel**2, axis=0)

        # Total squared displacement
        total_disp_sq = np.sum(displacements**2, axis=2)

        # Squared perpendicular displacement
        # |Delta x_perp|^2 = |Delta x|^2 - (Delta x . n)^2
        perp_disp_sq = total_disp_sq - disp_parallel**2

        # MSD_perp(t) = 1/2 < |Delta x_perp|^2 > (Eq 5)
        msd_perp = 0.5 * np.mean(perp_disp_sq, axis=0)

        # Calculate lambda(t) = MSD_parallel / MSD_perp
        # Handle division by zero at t=0
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_ratio = msd_parallel / msd_perp
            lambda_ratio = np.nan_to_num(lambda_ratio, nan=1.0, posinf=1.0, neginf=1.0)

        return AnisotropyMetrics(
            lag_times=ensemble.times - ensemble.times[0],
            msd_parallel=msd_parallel,
            msd_perp=msd_perp,
            lambda_ratio=lambda_ratio
        )

    def calculate_asymptotic_ratio(self, metrics: AnisotropyMetrics, start_time: float = 0.5) -> float:
        """Computes the average anisotropy ratio for the asymptotic regime.

        Corresponds to step 7 of the EXP2 procedure (average lambda for t > 0.5).

        Args:
            metrics: The calculated anisotropy time series.
            start_time: The physical time threshold for the average (default 0.5).

        Returns:
            The mean lambda_ratio for times greater than the threshold.
        """
        mask = metrics.lag_times > start_time
        if not np.any(mask):
            return float(metrics.lambda_ratio[-1])
        return float(np.mean(metrics.lambda_ratio[mask]))

    def check_transverse_dominance(self, asymptotic_ratio: float) -> bool:
        """Evaluates if the dispersion is transverse-dominant (lambda < 1).

        Directly tests Claim C1: dispersion perpendicular to V_LS exceeds parallel dispersion.

        Args:
            asymptotic_ratio: The mean ratio calculated from the asymptotic regime.

        Returns:
            True if lambda < 1, indicating transverse dominance.
        """
        return asymptotic_ratio < 1.0
