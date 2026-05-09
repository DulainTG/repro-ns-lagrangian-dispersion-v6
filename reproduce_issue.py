import numpy as np
from src.analysis.trajectories import TrajectoryEnsemble
from src.analysis.statistics import MsdCalculator, AlphaExponentEstimator, identify_diffusion_regime, TransportRegime

def test_repro(tail_fraction):
    num_tracers = 200
    num_steps = 100
    times = np.linspace(0, 1.0, num_steps)
    dt = times[1] - times[0]

    alphas = []
    for _ in range(100):
        # x(t) = random walk
        D = 0.1
        steps = np.random.normal(0, np.sqrt(2 * D * dt), (num_tracers, num_steps, 3))
        steps[:, 0, :] = 0
        positions_diffusive = np.cumsum(steps, axis=1)

        ensemble_diffusive = TrajectoryEnsemble(times=times, positions=positions_diffusive)
        msd_calc = MsdCalculator()
        msd_mean_diff, _ = msd_calc.compute_temporal_msd(ensemble_diffusive)

        alpha_est = AlphaExponentEstimator()
        alpha_asymp_diff = alpha_est.estimate_asymptotic_exponent(times, msd_mean_diff, tail_fraction=tail_fraction)
        alphas.append(alpha_asymp_diff)

    print(f"Tail fraction: {tail_fraction}")
    print(f"Mean alpha: {np.mean(alphas)}")
    print(f"Std alpha: {np.std(alphas)}")
    print(f"Max alpha: {np.max(alphas)}")
    print(f"Min alpha: {np.min(alphas)}")
    print("-" * 20)

if __name__ == "__main__":
    for f in [0.2, 0.5, 0.8]:
        test_repro(f)
