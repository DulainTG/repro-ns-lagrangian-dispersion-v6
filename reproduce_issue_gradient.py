import numpy as np
from src.analysis.trajectories import TrajectoryEnsemble
from src.analysis.statistics import MsdCalculator, AlphaExponentEstimator

def test_repro():
    num_tracers = 200
    num_steps = 100
    times = np.linspace(0, 1.0, num_steps)
    dt = times[1] - times[0]

    alphas_polyfit = []
    alphas_gradient_mean = []
    for _ in range(100):
        D = 0.1
        steps = np.random.normal(0, np.sqrt(2 * D * dt), (num_tracers, num_steps, 3))
        steps[:, 0, :] = 0
        positions_diffusive = np.cumsum(steps, axis=1)

        ensemble_diffusive = TrajectoryEnsemble(times=times, positions=positions_diffusive)
        msd_calc = MsdCalculator()
        msd_mean_diff, _ = msd_calc.compute_temporal_msd(ensemble_diffusive)

        alpha_est = AlphaExponentEstimator()
        
        # Current method: polyfit on tail (0.2)
        alpha_asymp = alpha_est.estimate_asymptotic_exponent(times, msd_mean_diff, tail_fraction=0.2)
        alphas_polyfit.append(alpha_asymp)
        
        # New method trial: mean of local alpha slopes on same tail
        local_alphas = alpha_est.calculate_local_alpha_slope(times, msd_mean_diff)
        num_tail_points = max(2, int(len(times) * 0.2))
        alphas_gradient_mean.append(np.mean(local_alphas[-num_tail_points:]))

    print(f"Polyfit tail 0.2 - Mean: {np.mean(alphas_polyfit)}, Std: {np.std(alphas_polyfit)}, Max: {np.max(alphas_polyfit)}")
    print(f"Gradient mean tail 0.2 - Mean: {np.mean(alphas_gradient_mean)}, Std: {np.std(alphas_gradient_mean)}, Max: {np.max(alphas_gradient_mean)}")

if __name__ == "__main__":
    test_repro()
