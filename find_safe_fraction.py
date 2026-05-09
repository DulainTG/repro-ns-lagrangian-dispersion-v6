import numpy as np
from src.analysis.trajectories import TrajectoryEnsemble
from src.analysis.statistics import MsdCalculator, AlphaExponentEstimator

def test_fraction(fraction, n_runs=1000):
    num_tracers = 200
    num_steps = 100
    times = np.linspace(0, 1.0, num_steps)
    dt = times[1] - times[0]
    
    failures = 0
    alphas = []
    for _ in range(n_runs):
        D = 0.1
        steps = np.random.normal(0, np.sqrt(2 * D * dt), (num_tracers, num_steps, 3))
        steps[:, 0, :] = 0
        positions = np.cumsum(steps, axis=1)
        ensemble = TrajectoryEnsemble(times=times, positions=positions)
        msd_mean, _ = MsdCalculator().compute_temporal_msd(ensemble)
        alpha = AlphaExponentEstimator().estimate_asymptotic_exponent(times, msd_mean, tail_fraction=fraction)
        alphas.append(alpha)
        if not np.allclose(alpha, 1.0, atol=0.2):
            failures += 1
    
    print(f"Fraction: {fraction}, Failures: {failures}/{n_runs}, Mean: {np.mean(alphas):.3f}, Std: {np.std(alphas):.3f}, Max: {np.max(alphas):.3f}")

if __name__ == "__main__":
    for f in [0.5, 0.6, 0.7, 0.8]:
        test_fraction(f)
