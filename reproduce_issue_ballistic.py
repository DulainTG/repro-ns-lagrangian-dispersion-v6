import numpy as np
from src.analysis.trajectories import TrajectoryEnsemble
from src.analysis.statistics import MsdCalculator, AlphaExponentEstimator, identify_diffusion_regime, TransportRegime

def test_ballistic(tail_fraction):
    num_tracers = 200
    num_steps = 100
    times = np.linspace(0, 1.0, num_steps)
    v0 = np.random.uniform(-1.0, 1.0, (num_tracers, 3))
    positions_ballistic = v0[:, np.newaxis, :] * times[np.newaxis, :, np.newaxis]

    ensemble_ballistic = TrajectoryEnsemble(times=times, positions=positions_ballistic)
    msd_calc = MsdCalculator()
    msd_mean, msd_std = msd_calc.compute_temporal_msd(ensemble_ballistic)

    alpha_est = AlphaExponentEstimator()
    alpha_asymp = alpha_est.estimate_asymptotic_exponent(times, msd_mean, tail_fraction=tail_fraction)
    
    print(f"Tail fraction: {tail_fraction}, Alpha: {alpha_asymp}")

if __name__ == "__main__":
    for f in [0.2, 0.5, 0.8]:
        test_ballistic(f)
