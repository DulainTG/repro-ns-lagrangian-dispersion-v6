import pytest
import numpy as np
from src.analysis.trajectories import TrajectoryEnsemble, calculate_trajectory_displacements, estimate_lagrangian_velocities, PropertyHistoryBuffer
from src.analysis.statistics import MsdCalculator, AlphaExponentEstimator, TransportRegime, identify_diffusion_regime
from src.analysis.dispersion import AnisotropyAnalyzer, AnisotropyMetrics
from src.analysis.vorticity import VortexResidenceAnalyzer, VorticityTimescaleDeriver, ResidenceTimeStats
from src.analysis.time_series import compute_q_signal_autocorrelation, find_one_over_e_threshold_crossing, integrate_normalized_correlation
from src.analysis.vectors import VectorProjector

def test_msd_and_transport_regime_integration():
    """Test integration between TrajectoryEnsemble, MsdCalculator, and AlphaExponentEstimator."""
    # Use more tracers to reduce stochastic noise in alpha estimation
    num_tracers = 200 
    num_steps = 100
    times = np.linspace(0, 1.0, num_steps)
    dt = times[1] - times[0]
    
    # 1. Test Ballistic Motion (alpha = 2)
    # x(t) = v * t
    # Keep velocities small so jumps are < 0.5 for extent=1.0 at dt=0.01
    # max v * dt < 0.5 -> v < 50
    v0 = np.random.uniform(-1.0, 1.0, (num_tracers, 3))
    positions_ballistic = v0[:, np.newaxis, :] * times[np.newaxis, :, np.newaxis]
    
    ensemble_ballistic = TrajectoryEnsemble(times=times, positions=positions_ballistic)
    msd_calc = MsdCalculator()
    msd_mean, msd_std = msd_calc.compute_temporal_msd(ensemble_ballistic)
    
    alpha_est = AlphaExponentEstimator()
    alpha_asymp = alpha_est.estimate_asymptotic_exponent(times, msd_mean)
    
    assert np.allclose(alpha_asymp, 2.0, atol=0.1)
    assert identify_diffusion_regime(alpha_asymp) == TransportRegime.BALLISTIC

    # 2. Test Diffusive Motion (alpha = 1)
    # x(t) = random walk
    D = 0.1
    steps = np.random.normal(0, np.sqrt(2 * D * dt), (num_tracers, num_steps, 3))
    steps[:, 0, :] = 0 
    positions_diffusive = np.cumsum(steps, axis=1)
    
    ensemble_diffusive = TrajectoryEnsemble(times=times, positions=positions_diffusive)
    msd_mean_diff, _ = msd_calc.compute_temporal_msd(ensemble_diffusive)
    
    alpha_asymp_diff = alpha_est.estimate_asymptotic_exponent(times, msd_mean_diff)
    
    # Expect alpha close to 1
    assert np.allclose(alpha_asymp_diff, 1.0, atol=0.2)
    assert identify_diffusion_regime(alpha_asymp_diff) == TransportRegime.DIFFUSIVE

def test_anisotropy_analysis_integration():
    """Test integration between TrajectoryEnsemble and AnisotropyAnalyzer."""
    num_tracers = 20
    num_steps = 50
    times = np.linspace(0, 1, num_steps)
    
    # Create tracers moving primarily perpendicular to a reference direction
    v_ls_unit = np.zeros((num_tracers, num_steps, 3))
    v_ls_unit[..., 0] = 1.0
    
    t_grid = times[np.newaxis, :, np.newaxis]
    positions = np.zeros((num_tracers, num_steps, 3))
    positions[..., 0] = 0.1 * t_grid[..., 0]
    positions[..., 1] = 1.0 * t_grid[..., 0]
    positions[..., 2] = 1.0 * t_grid[..., 0]
    
    ensemble = TrajectoryEnsemble(
        times=times, 
        positions=positions, 
        sampled_properties={'v_ls_unit': v_ls_unit}
    )
    
    analyzer = AnisotropyAnalyzer()
    metrics = analyzer.calculate_time_series(ensemble)
    
    assert isinstance(metrics, AnisotropyMetrics)
    assert np.allclose(metrics.lambda_ratio[1:], 0.01)
    
    asymp_ratio = analyzer.calculate_asymptotic_ratio(metrics, start_time=0.5)
    assert np.isclose(asymp_ratio, 0.01)

def test_vortex_analysis_and_time_series_integration():
    """Test integration between TrajectoryEnsemble, time_series functions, and vorticity analyzers."""
    num_tracers = 50
    num_steps = 200
    times = np.linspace(0, 10, num_steps)
    
    # Create Q-criterion signals that decay
    tau_true = 2.0
    q_criterion = np.exp(-times[np.newaxis, :] / tau_true) 
    # Add minor noise
    q_criterion += 0.001 * np.random.randn(*q_criterion.shape)
    
    ensemble = TrajectoryEnsemble(
        times=times,
        positions=np.zeros((num_tracers, num_steps, 3)),
        sampled_properties={'q_criterion': q_criterion}
    )
    
    # 1. Test autocorrelation computation
    autocorr = compute_q_signal_autocorrelation(ensemble)
    assert autocorr.shape == (num_steps,)
    
    # 2. Test tau_Q derivation
    deriver = VorticityTimescaleDeriver()
    tau_q = deriver.derive_tau_q(times, autocorr)
    assert np.isclose(tau_q, tau_true, rtol=0.2)
    
    # 3. Test vortex residence durations
    q_binary = np.zeros((num_tracers, num_steps))
    trapping_steps = 40
    q_binary[:, 10:10+trapping_steps] = 1.0
    
    ensemble_binary = TrajectoryEnsemble(
        times=times,
        positions=np.zeros((num_tracers, num_steps, 3)),
        sampled_properties={'q_criterion': q_binary}
    )
    
    res_analyzer = VortexResidenceAnalyzer()
    res_stats = res_analyzer.calculate_residence_durations(ensemble_binary, q_threshold=0.5)
    
    expected_duration = times[10+trapping_steps-1] - times[10]
    assert np.allclose(res_stats.all_durations, expected_duration)

def test_trajectory_utilities_and_integration():
    """Test utilities in trajectories.py and integration of time series metrics."""
    num_tracers = 5
    num_steps = 10
    times = np.linspace(0, 1.0, num_steps)
    
    v_true = np.array([1, 2, 3])
    # Build positions with shape (num_tracers, num_steps, 3)
    positions = v_true[np.newaxis, np.newaxis, :] * times[np.newaxis, :, np.newaxis]
    positions = np.repeat(positions, num_tracers, axis=0) # Shape (5, 10, 3)
    
    ensemble = TrajectoryEnsemble(
        times=times,
        positions=positions,
        sampled_properties={'scalar': np.random.rand(num_tracers, num_steps)}
    )
    
    # 1. Test Lagrangian Velocities
    v_est = estimate_lagrangian_velocities(ensemble)
    assert v_est.shape == (num_tracers, num_steps, 3)
    for i in range(num_tracers):
        assert np.allclose(v_est[i], v_true[np.newaxis, :])
        
    # 2. Test PropertyHistoryBuffer
    buffer = PropertyHistoryBuffer(ensemble)
    series = buffer.get_property_series('scalar')
    assert np.array_equal(series, ensemble.sampled_properties['scalar'])
    
    # 3. Test Integration of Correlation Signal
    lag_times = np.linspace(0, 1, 100)
    correlation = 1 - lag_times
    integral = integrate_normalized_correlation(lag_times, correlation)
    assert np.isclose(integral, 0.5)

def test_vector_decomposition_integration():
    """Test VectorProjector integration."""
    projector = VectorProjector()
    vectors = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 0]])
    directions = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    decomp = projector.decompose_vector_field(vectors, directions)
    assert np.allclose(decomp.parallel, [[1, 0, 0], [0, 0, 0], [1, 0, 0]])
    assert np.allclose(decomp.perpendicular, [[0, 0, 0], [0, 1, 0], [0, 1, 0]])

def test_error_handling_integration():
    """Test error handling across analysis modules."""
    ensemble = TrajectoryEnsemble(times=np.array([0, 1]), positions=np.zeros((1, 2, 3)))
    msd_calc = MsdCalculator()
    with pytest.raises(ValueError, match="window_size must be positive"):
        msd_calc.compute_windowed_msd(ensemble, 0)
        
    ans_analyzer = AnisotropyAnalyzer()
    with pytest.raises(KeyError, match="v_ls_unit"):
        ans_analyzer.calculate_time_series(ensemble)
        
    vort_deriver = VorticityTimescaleDeriver()
    # Signal that doesn't decay enough
    with pytest.raises(ValueError, match="signal does not cross the 1/e threshold"):
        vort_deriver.derive_tau_q(np.array([0, 1]), np.array([1.0, 0.9]))
