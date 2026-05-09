import numpy as np
from src.analysis.trajectories import TrajectoryEnsemble

def calculate_lag_phase_average(ensemble_series: np.ndarray) -> np.ndarray:
    """Calculate the ensemble mean of a property as a function of lag time indices.

    Args:
        ensemble_series: A 2D array of shape (num_tracers, num_steps) containing 
            the sampled property values for each tracer.

    Returns:
        A 1D array of shape (num_steps,) representing the ensemble average at each lag phase.

    Raises:
        ValueError: If the input array is not 2-dimensional.
    """
    if ensemble_series.ndim != 2:
        raise ValueError(f"Input ensemble_series must be 2-dimensional (received {ensemble_series.ndim}D).")
    
    return np.mean(ensemble_series, axis=0)


def calculate_normalized_decay(signal: np.ndarray) -> np.ndarray:
    """Normalize a correlation signal by its initial value to represent a decay function.

    Args:
        signal: Real-valued array representing a correlation or autocorrelation trace.

    Returns:
        The signal normalized such that the first element is 1.0 (assuming maximum at lag 0).

    Raises:
        ZeroDivisionError: If the first element (lag 0) is zero.
    """
    if signal.size == 0:
        return signal
    
    if signal[0] == 0:
        raise ZeroDivisionError("The first element of the signal is zero; cannot normalize a decay starting from zero.")
    
    return signal / signal[0]


def integrate_normalized_correlation(lag_times: np.ndarray, correlation_values: np.ndarray) -> float:
    """Integrate the normalized correlation function over lag time to estimate integral scales.

    Args:
        lag_times: 1D array of physical lag times corresponding to the correlation values.
        correlation_values: 1D array of normalized correlation values (starting at 1.0).

    Returns:
        The integral of the correlation function, often used as a characteristic timescale.

    Raises:
        ValueError: If input arrays have mismatched dimensions.
    """
    if lag_times.shape != correlation_values.shape:
        raise ValueError(
            f"Shape mismatch: lag_times {lag_times.shape} and correlation_values {correlation_values.shape} "
            "must have the same dimensions."
        )

    # Use the trapezoidal rule to integrate. 
    # np.trapezoid is the modern name for np.trapz in NumPy 2.x
    return float(np.trapezoid(correlation_values, x=lag_times))
def compute_q_signal_autocorrelation(ensemble: TrajectoryEnsemble) -> np.ndarray:
    """Compute the ensemble-averaged Lagrangian autocorrelation of the Q-criterion for EXP3.

    This function satisfies the EXP3 requirement for artifact 'autocorrelation_q_normalized'. 
    It processes the 'q_criterion' property sampled along trajectories as specified 
    in Equation 1 and the XP3 procedure.

    Args:
        ensemble: A TrajectoryEnsemble containing 'q_criterion' in its sampled_properties.

    Returns:
        A 1D array of normalized autocorrelation values indexed by lag time.

    Raises:
        KeyError: If 'q_criterion' is missing from the ensemble properties.
        ValueError: If the ensemble has insufficient steps for correlation calculation.
    """
    if 'q_criterion' not in ensemble.sampled_properties:
        raise KeyError("Property 'q_criterion' was not found in the trajectory ensemble sampled_properties.")

    q_series = ensemble.sampled_properties['q_criterion']

    if ensemble.num_steps < 1:
        raise ValueError("The trajectory ensemble contains no time steps, which is insufficient for autocorrelation.")

    # Lagrangian autocorrelation R_Q(tau) = <Q(0) * Q(tau)> / <Q(0)^2>
    # Calculate the product Q_i(0) * Q_i(t) for all tracers i and lag times t
    q_initial = q_series[:, 0:1]  # shape (N, 1) to enable broadcasting over (N, T)
    lag_products = q_series * q_initial

    # Ensemble average over all tracers to get the correlation signal
    correlation_signal = calculate_lag_phase_average(lag_products)

    # Normalize by the initial value (lag 0) to yield the decay function starting at 1.0
    return calculate_normalized_decay(correlation_signal)
def find_one_over_e_threshold_crossing(times: np.ndarray, signal: np.ndarray) -> float:
    """Search for the first time index where a signal drops below 1/e (~0.3678).

    Specifically used in EXP3 to determine the vortex trapping timescale tau_Q 
    from the Lagrangian Q-autocorrelation function.

    Args:
        times: 1D array of time coordinates.
        signal: 1D array of signal values (expected to decay from 1.0 to 0.0).

    Returns:
        The interpolated time coordinate where the signal crosses 1/e.

    Raises:
        ValueError: If the signal never crosses the 1/e threshold.
    """
    if times.shape != signal.shape:
        raise ValueError(f"Input mismatch: times {times.shape} and signal {signal.shape} must be the same size.")

    # Target threshold is 1/e for the characteristic decay timescale tau_Q
    threshold = 1.0 / np.exp(1.0)

    # Identify indices where the signal is below the 1/e threshold
    below_indices = np.where(signal < threshold)[0]

    if below_indices.size == 0:
        raise ValueError("The signal does not drop below the 1/e threshold within the provided range.")

    # Use the first occurrence where the signal drops below threshold
    idx = below_indices[0]

    if idx == 0:
        # Signal is already below threshold at the beginning
        return float(times[0])

    # Interpolate between the points [idx-1, idx] to find the precise crossing time
    t1, t2 = times[idx-1], times[idx]
    s1, s2 = signal[idx-1], signal[idx]

    # Linear interpolation: t_crossing = t1 + (t2 - t1) * (threshold - s1) / (s2 - s1)
    # Since s1 >= threshold and s2 < threshold, s2 - s1 is always non-zero.
    t_crossing = t1 + (t2 - t1) * (threshold - s1) / (s2 - s1)

    return float(t_crossing)
