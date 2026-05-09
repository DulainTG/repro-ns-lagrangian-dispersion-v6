import numpy as np
from src.analysis.trajectories import TrajectoryEnsemble
from src.analysis.statistics import MsdCalculator, AlphaExponentEstimator

def test_methods(fraction=0.5, n_runs=1000):
    num_tracers = 200
    num_steps = 100
    times = np.linspace(0, 1.0, num_steps)
    dt = times[1] - times[0]
    
    ms_poly = []
    ms_median_local = []
    
    for _ in range(n_runs):
        D = 0.1
        steps = np.random.normal(0, np.sqrt(2 * D * dt), (num_tracers, num_steps, 3))
        steps[:, 0, :] = 0
        positions = np.cumsum(steps, axis=1)
        ensemble = TrajectoryEnsemble(times=times, positions=positions)
        msd_mean, _ = MsdCalculator().compute_temporal_msd(ensemble)
        
        # Method 1: Polyfit
        alpha_poly = AlphaExponentEstimator().estimate_asymptotic_exponent(times, msd_mean, tail_fraction=fraction)
        ms_poly.append(alpha_poly)
        
        # Method 2: Median of local slopes
        est = AlphaExponentEstimator()
        local_alphas = est.calculate_local_alpha_slope(times, msd_mean)
        num_tail_points = max(2, int(len(times) * fraction))
        alpha_median = np.median(local_alphas[-num_tail_points:])
        ms_median_local.append(alpha_median)
    
    for name, alphas in [("Polyfit", ms_poly), ("MedianLocal", ms_median_local)]:
        alphas = np.array(alphas)
        failures = np.sum(np.abs(alphas - 1.0) > 0.2)
        print(f"Method: {name}, Failures: {failures}/{n_runs}, Mean: {np.mean(alphas):.3f}, Std: {np.std(alphas):.3f}")

if __name__ == "__main__":
    test_methods(0.3)
