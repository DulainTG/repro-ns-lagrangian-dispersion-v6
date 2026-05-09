import numpy as np
from enum import Enum
from typing import Tuple
from src.analysis.trajectories import TrajectoryEnsemble, calculate_trajectory_displacements

class TransportRegime(Enum):
    """Classification of Lagrangian transport regimes based on the alpha exponent."""
    SUBDIFFUSIVE = 'subdiffusive'
    DIFFUSIVE = 'diffusive'
    SUPERDIFFUSIVE = 'superdiffusive'
    BALLISTIC = 'ballistic'

class MsdCalculator:
    """Calculates Mean-Square Displacement (MSD) statistics for Lagrangian ensembles.

    This class implements the ensemble averaging logic for tracer displacements
    as defined in Eq 3: MSD(t) = <|x(t) - x(0)|^2>.
    """

    def compute_temporal_msd(self, ensemble: TrajectoryEnsemble) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates ensemble mean and standard deviation of squared displacement.

        Args:
            ensemble: The trajectory data containing tracer positions over time.

        Returns:
            A tuple of (msd_mean, msd_std), both as 1D arrays of length T.

        Raises:
            ValueError: If the ensemble contains no position data or has inconsistent dimensions.
        """
        if ensemble.positions is None or ensemble.positions.size == 0:
            raise ValueError("Ensemble contains no position data.")
        
        # Calculate unwrapped displacements Δx(t) = x(t) - x(0)
        # Assuming periodic extent L=1.0 as default for these experiments
        displacements = calculate_trajectory_displacements(ensemble, extent=1.0, unwrap=True)
        # shape: (num_tracers, num_steps, 3)
        
        # Squared displacement |Δx(t)|^2 = Δx^2 + Δy^2 + Δz^2
        sq_dist = np.sum(displacements**2, axis=-1)
        # shape: (num_tracers, num_steps)
        
        msm = np.mean(sq_dist, axis=0)  # msd_mean
        mss = np.std(sq_dist, axis=0)   # msd_std
        
        return msm, mss

    def compute_windowed_msd(self, ensemble: TrajectoryEnsemble, window_size: int) -> np.ndarray:
        """Calculates MSD using a sliding temporal window for improved statistics.

        Args:
            ensemble: The trajectory data.
            window_size: The number of snapshots to include in the rolling average window.

        Returns:
            1D array of MSD values calculated over the specified temporal window.

        Raises:
            ValueError: If window_size exceeds the total duration of the ensemble.
        """
        if window_size > ensemble.num_steps:
             raise ValueError(f"window_size {window_size} exceeds total ensemble duration {ensemble.num_steps}")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")

        # Reconstruct unwrapped displacements from zero for time-averaging
        # Since x_unw(t+tau) - x_unw(t) = (x_unw(t+tau)-x_unw(0)) - (x_unw(t)-x_unw(0))
        d_from_zero = calculate_trajectory_displacements(ensemble, extent=1.0, unwrap=True)
        
        msd_windowed = np.zeros(window_size)
        
        for tau in range(window_size):
            if tau == 0:
                msd_windowed[tau] = 0.0
                continue
            
            # Displacement vectors: x(t + tau) - x(t)
            # diffs.shape = (num_tracers, total_steps - tau, 3)
            diffs = d_from_zero[:, tau:, :] - d_from_zero[:, :-tau, :]
            sq_dist = np.sum(diffs**2, axis=-1)
            # Ensemble and time average for this lag tau
            msd_windowed[tau] = np.mean(sq_dist)
            
        return msd_windowed

class AlphaExponentEstimator:
    """Estimates the local scaling exponent alpha from MSD time series data.

    Implements the derivative alpha(t) = d(log MSD) / d(log t) to identify 
    transport regimes (EXP1).
    """

    def calculate_local_alpha_slope(self, lag_times: np.ndarray, msd_values: np.ndarray) -> np.ndarray:
        """Computes the local slope of the MSD in log-log space.

        Args:
            lag_times: Array of time intervals (dt).
            msd_values: Array of corresponding Mean-Square Displacement values.

        Returns:
            1D array of alpha exponent values for each time step.

        Raises:
            ValueError: If lag_times contains non-positive values or input lengths mismatch.
        """
        if len(lag_times) != len(msd_values):
             raise ValueError("Input arrays lag_times and msd_values must have the same length.")
        
        # Validate lag_times and msd_values for log-log calculation
        # Identify points where we can compute log(t) and log(MSD)
        valid_indices = (lag_times > 0) & (msd_values > 0)
        
        alpha = np.zeros_like(msd_values, dtype=float)
        
        if np.sum(valid_indices) < 2:
            return alpha
            
        log_t = np.log(lag_times[valid_indices])
        log_msd = np.log(msd_values[valid_indices])
        
        # Numerical gradient in log-log space gives local scaling exponent alpha(t)
        # alpha = d(log MSD) / d(log t)
        alpha[valid_indices] = np.gradient(log_msd, log_t)
        
        return alpha

    def estimate_asymptotic_exponent(self, lag_times: np.ndarray, msd_values: np.ndarray, tail_fraction: float=0.2) -> float:
        """Estimates the final scaling exponent at the end of the simulation.

        Args:
            lag_times: Time indices.
            msd_values: MSD series.
            tail_fraction: The fraction of the total duration used for the fit.

        Returns:
            A single scalar representing the late-time scaling alpha.
        """
        num_points = len(lag_times)
        if num_points < 2:
            return 0.0
        
        num_tail_points = max(2, int(num_points * tail_fraction))
        
        tail_t = lag_times[-num_tail_points:]
        tail_msd = msd_values[-num_tail_points:]
        
        # Filter for valid values before taking logs
        valid = (tail_t > 0) & (tail_msd > 0)
        if np.sum(valid) < 2:
            return 0.0
            
        log_t = np.log(tail_t[valid])
        log_msd = np.log(tail_msd[valid])
        
        # Linear regression to find the slope which represents the scaling exponent
        slope, _ = np.polyfit(log_t, log_msd, 1)
        return float(slope)

def identify_diffusion_regime(alpha: float) -> TransportRegime:
    """Maps a scaling exponent to a specific transport regime label.

    Uses thresholds defined in EXP1: Ballistic (alpha=2), Diffusive (alpha=1).

    Args:
        alpha: The local scaling exponent value.

    Returns:
        The identified TransportRegime enum member.

    Raises:
        ValueError: If alpha is negative (physically inconsistent with MSD).
    """
    if alpha < 0:
        raise ValueError(f"Scaling exponent alpha must be non-negative, got {alpha}")
    
    # Classification thresholds for Lagrangian transport regimes
    if alpha < 0.8:
        return TransportRegime.SUBDIFFUSIVE
    elif alpha <= 1.2:
        # Diffusive regime around alpha=1
        return TransportRegime.DIFFUSIVE
    elif alpha < 1.8:
        # Superdiffusive regime between diffusion and ballistic motion
        return TransportRegime.SUPERDIFFUSIVE
    else:
        # Ballistic regime around alpha=2
        return TransportRegime.BALLISTIC
