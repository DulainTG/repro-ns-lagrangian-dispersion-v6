import numpy as np

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
