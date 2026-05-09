import numpy as np
from typing import NamedTuple, Dict, List
from .trajectories import TrajectoryEnsemble


class ResidenceTimeStats(NamedTuple):
    """Statistics for tracer residence times within vortex structures.
    
    Attributes:
        mean_duration: Average time a tracer stays in a Q > 0 region.
        median_duration: Median time a tracer stays in a Q > 0 region.
        all_durations: Raw array of all individual trapping event durations.
    """
    mean_duration: float
    median_duration: float
    all_durations: np.ndarray
class VortexResidenceAnalyzer:
    """Analyzes the duration tracers spend trapped in vortex regions.
    
    This diagnostic supports EXP3 (Vortex Residence and Q-Autocorrelation) and 
    Claim C2 regarding the transience of vortex trapping events. It identifies
    contiguous temporal segments where the Q-criterion (Q > 0) is satisfied
    along a trajectory.
    """

    def calculate_residence_durations(self, ensemble: TrajectoryEnsemble, q_threshold: float=0.0) -> ResidenceTimeStats:
        """Derives residence times for the full ensemble of tracers.
        
        Args:
            ensemble: TrajectoryEnsemble containing 'q_criterion' in sampled_properties.
            q_threshold: The threshold value to define a vortex region (default 0.0).

        Returns:
            ResidenceTimeStats containing the distribution and averages of trapping durations.

        Raises:
            KeyError: If 'q_criterion' is missing from the ensemble sampled_properties.
        """
        if 'q_criterion' not in ensemble.sampled_properties:
            raise KeyError("Property 'q_criterion' is missing from ensemble sampled_properties.")

        q_criterion = ensemble.sampled_properties['q_criterion']
        times = ensemble.times
        
        all_durations = []
        
        for i in range(ensemble.num_tracers):
            is_vortex = q_criterion[i] > q_threshold
            
            if not np.any(is_vortex):
                continue
                
            # Find contiguous segments of True
            padded = np.zeros(len(is_vortex) + 2, dtype=bool)
            padded[1:-1] = is_vortex
            diff = np.diff(padded.astype(int))
            
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1
            
            for s, e in zip(starts, ends):
                # Duration is times[e] - times[s]
                duration = times[e] - times[s]
                all_durations.append(duration)
        
        if not all_durations:
            return ResidenceTimeStats(
                mean_duration=0.0,
                median_duration=0.0,
                all_durations=np.array([], dtype=float)
            )
            
        durations_array = np.array(all_durations)
        return ResidenceTimeStats(
            mean_duration=float(np.mean(durations_array)),
            median_duration=float(np.median(durations_array)),
            all_durations=durations_array
        )
class VorticityTimescaleDeriver:
    """Derives characteristic vorticity timescales from Lagrangian signals.
    
    Specifically calculates the tau_Q timescale corresponding to the 1/e 
    decay of the Q-criterion Lagrangian autocorrelation, as required by 
    the EXP3 text report artifact.
    """

    def derive_tau_q(self, lag_times: np.ndarray, autocorrelation: np.ndarray) -> float:
        """Calculates the 1/e decay timescale (tau_Q) from a correlation curve.
        
        Args:
            lag_times: Array of time offsets corresponding to the correlation values.
            autocorrelation: Normalized Lagrangian autocorrelation values.

        Returns:
            The time value tau_Q where the autocorrelation first drops to 1/e.

        Raises:
            ValueError: If the signal does not cross the 1/e threshold.
        """
        threshold = 1.0 / np.e
        
        # Find the first index where autocorrelation is below threshold
        indices = np.where(autocorrelation <= threshold)[0]
        if len(indices) == 0:
            raise ValueError("The signal does not cross the 1/e threshold.")
        
        idx = indices[0]
        if idx == 0:
            # If it's already below threshold at lag 0 (unlikely for autocorrelation)
            return float(lag_times[0])
            
        # Linear interpolation between idx-1 and idx
        t1, t2 = lag_times[idx-1], lag_times[idx]
        c1, c2 = autocorrelation[idx-1], autocorrelation[idx]
        
        # c(t) = c1 + (c2 - c1) / (t2 - t1) * (t - t1)
        # solve for c(t) = threshold
        tau_q = t1 + (threshold - c1) * (t2 - t1) / (c2 - c1)
        
        return float(tau_q)

    def calculate_normalized_metrics(self, tau_q: float, large_eddy_turnover_time: float) -> Dict[str, float]:
        """Computes summary metrics for the EXP3 text report.
        
        Args:
            tau_q: The calculated trapping timescale.
            large_eddy_turnover_time: The reference Te (large-eddy turnover time).

        Returns:
            Dictionary containing standard fields: 'tau_Q' and 'tau_Q_over_Te'.
            Matches the EXP3 Content Contract for text_report.
        """
        return {
            'tau_Q': tau_q,
            'tau_Q_over_Te': tau_q / large_eddy_turnover_time
        }
