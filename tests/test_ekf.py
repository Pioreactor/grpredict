# test_ekf.py

# tests/test_culture_growth_ekf.py

import numpy as np
import pytest

# Replace `your_module` with the actual module/path where CultureGrowthEKF is defined.
from grpredict import CultureGrowthEKF
from grpredict import build_filter_from_observation_summary
from grpredict import estimate_normalization_factor_from_warmup_observations
from grpredict import normalize_observation_by_factor
from grpredict import normalize_observations_by_factor
from grpredict import summarize_warmup_observations
from tests.simulation_utils import plausible_growth_rate_profiles
from tests.simulation_utils import simulate_profiled_od_observations


def make_ekf(dt: float) -> CultureGrowthEKF:
    """
    Helper to construct a CultureGrowthEKF with:
      - 2×2 initial covariance = identity
      - small process-noise covariance (to keep it nearly deterministic)
      - small observation-noise covariance (one sensor)
      - a large outlier threshold (to disable outlier handling)
    """
    initial_state = np.array([1.0, 0.0])  # OD = 1.0, r = 0.0 (we’ll overwrite r later)
    initial_covariance = np.eye(2)
    # Very small process noise on both OD and r (positive definite)
    process_noise_covariance = np.array([[1e-6, 0.0],
                                         [0.0, 1e-8]])
    # Very small observation noise (one sensor)
    observation_noise_covariance = np.array([[1e-6]])
    outlier_std_threshold = 1e6  # effectively disable outlier checks

    ekf = CultureGrowthEKF(
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        process_noise_covariance=process_noise_covariance,
        observation_noise_covariance=observation_noise_covariance,
        outlier_std_threshold=outlier_std_threshold,
    )
    return ekf


def test_exponential_growth_fixed_rate():
    """
    Simulate perfect exponential growth:
      OD_true[t] = OD_true[t-1] * exp(r_true * dt)
    with r_true = 0.2. Observations are noise-free.
    After a few updates, the EKF’s estimated r should approach 0.2.
    """
    dt = 1.0
    r_true = 0.2
    steps = 20

    # Build EKF and override its initial state to match true OD/r
    ekf = make_ekf(dt)
    ekf.state_ = np.array([1.0, r_true])  # start EKF exactly on the true trajectory
    ekf.covariance_ = np.eye(2)

    # Track estimates
    estimated_rates = []

    OD_true = 1.0
    for _ in range(steps):
        # advance the "true" OD
        OD_true = OD_true * np.exp(r_true * dt)
        obs = [OD_true]  # single-sensor observation

        # run one EKF update
        state_est, cov_est = ekf.update(obs, dt)
        estimated_rates.append(state_est[1])

    # After several iterations without noise, EKF should lock in r ≈ r_true
    # Check the last few estimated rates are within 1e-3 of true value
    for est_r in estimated_rates[-5:]:
        assert pytest.approx(r_true, rel=1e-3) == est_r


def test_flat_growth_zero_rate():
    """
    Simulate a flat OD (r_true = 0). Observations are exactly constant.
    The EKF’s estimated r should converge to 0.0 (within small tolerance).
    """
    dt = 1.0
    r_true = 0.0
    steps = 30

    ekf = make_ekf(dt)
    # Override initial state so the EKF “knows” OD=1.0 but starts with some nonzero rate
    ekf.state_ = np.array([1.0, 0.5])  # initial guessed r=0.5 (incorrect)
    ekf.covariance_ = np.eye(2)

    OD_true = 1.0
    estimated_rates = []

    for _ in range(steps):
        # OD stays constant
        obs = [OD_true]
        state_est, cov_est = ekf.update(obs, dt)
        estimated_rates.append(state_est[1])

    # After a handful of corrections, the EKF’s r should approach 0.0
    for est_r in estimated_rates[-5:]:
        assert abs(est_r) < 1e-3


def test_linearly_increasing_growth_rate():
    """
    Simulate a scenario where the “true” growth rate increases by dr each step:
      r_true[t] = r_true[t-1] + dr * dt
      OD_true[t] = OD_true[t-1] * exp(r_true[t-1] * dt)
    We check that the EKF’s estimated r tracks the upward trend.
    """
    dt = 1.0
    dr = 0.01       # growth-rate increment per step
    steps = 50

    ekf = make_ekf(dt)
    # Start EKF with OD=1.0 and r=0.0
    ekf.state_ = np.array([1.0, 0.0])
    ekf.covariance_ = np.eye(2)

    OD_true = 1.0
    r_true = 0.0
    estimated_rates = []
    true_rates = []

    for _ in range(steps):
        # Save the current “true” rate before increment
        true_rates.append(r_true)

        # Advance OD using the previous rate
        OD_true = OD_true * np.exp(r_true * dt)

        # Simulate one-step “linear” increase in r
        r_true = r_true + dr * dt

        # EKF sees the perfect observation (no noise)
        obs = [OD_true]
        state_est, cov_est = ekf.update(obs, dt)

        estimated_rates.append(state_est[1])

    # Now compare the last part of estimated_rates to true_rates
    # They will not be exactly equal (EKF lags a little), so allow some tolerance.
    # In particular, once the filter has “spun up,” the slope should be ≈ dr
    #
    # We assert that over the final 5 points, (est[i+1] – est[i]) ≈ dr
    diffs = np.diff(estimated_rates[-6:])  # last 6 → 5 differences
    for delta in diffs:
        assert pytest.approx(dr, rel=0.1) == delta  # within 10% of dr

    # Also check absolute value near the true rate at the last timestep
    assert pytest.approx(true_rates[-1], rel=0.1) == estimated_rates[-1]


def test_warmup_normalization_recenters_startup_window_near_one() -> None:
    raw_observations = np.array([2.45, 2.48, 2.43, 2.50, 2.47, 2.60, 2.71], dtype=float)
    warmup_observations = raw_observations[:5]

    normalization_factor = estimate_normalization_factor_from_warmup_observations(
        warmup_observations,
    )
    normalized = normalize_observations_by_factor(raw_observations, normalization_factor)

    assert normalization_factor > 0.0
    assert pytest.approx(1.0, rel=0.02) == float(np.median(normalized[:5]))


def test_summarize_warmup_observations_handles_gain_scaled_au_data() -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    latent_signal = np.array([1.00, 1.00, 1.01, 1.02, 1.02, 1.03, 1.05, 1.07], dtype=float)
    raw_observations = 0.42 + 2.7 * latent_signal
    scaled_observations = 3.0 * raw_observations
    warmup_observations = raw_observations[:5]
    scaled_warmup_observations = scaled_observations[:5]

    summary = summarize_warmup_observations(warmup_observations, dt_hours)
    summary_scaled = summarize_warmup_observations(
        scaled_warmup_observations,
        dt_hours,
    )

    assert pytest.approx(3.0, rel=1e-3) == (
        summary_scaled["normalization_factor"] / summary["normalization_factor"]
    )
    np.testing.assert_allclose(
        normalize_observations_by_factor(raw_observations, summary["normalization_factor"]),
        normalize_observations_by_factor(scaled_observations, summary_scaled["normalization_factor"]),
        rtol=1e-6,
        atol=1e-6,
    )

    ekf = build_filter_from_observation_summary(summary)
    normalized_observations = normalize_observations_by_factor(
        raw_observations,
        summary["normalization_factor"],
    )
    state, covariance = ekf.update([float(normalized_observations[5])], dt_hours)
    assert np.all(np.isfinite(state))
    assert np.all(np.isfinite(covariance))


def test_summarize_warmup_observations_returns_warmup_only_normalized_data() -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    raw_observations = np.array([2.45, 2.48, 2.43, 2.50, 2.47, 2.60, 2.71], dtype=float)
    warmup_observations = raw_observations[:5]

    summary = summarize_warmup_observations(warmup_observations, dt_hours)

    normalized_warmup = np.asarray(summary["normalized_warmup_observations"], dtype=float)
    assert normalized_warmup.shape == warmup_observations.shape
    assert pytest.approx(1.0, rel=0.02) == float(np.median(normalized_warmup))


def test_normalize_observation_by_factor_matches_batch_helper() -> None:
    raw_observations = np.array([2.45, 2.48, 2.43, 2.50, 2.47], dtype=float)
    normalization_factor = 2.47

    normalized_batch = normalize_observations_by_factor(raw_observations, normalization_factor)
    normalized_streaming = np.asarray(
        [
            normalize_observation_by_factor(float(observation), normalization_factor)
            for observation in raw_observations
        ],
        dtype=float,
    )

    np.testing.assert_allclose(normalized_batch, normalized_streaming, rtol=1e-12, atol=1e-12)


def test_growth_rate_is_clipped_at_initialization() -> None:
    ekf = CultureGrowthEKF(
        initial_state=np.array([1.0, 5.0]),
        initial_covariance=np.eye(2),
        process_noise_covariance=np.diag([1e-6, 1e-8]),
        observation_noise_covariance=np.array([[1e-6]]),
        outlier_std_threshold=5.0,
    )

    assert ekf.state_[1] == 3.0


def test_growth_rate_is_clipped_when_syncing_and_updating() -> None:
    dt = 1.0
    ekf = make_ekf(dt)
    ekf.state_ = np.array([1.0, 10.0])
    ekf.covariance_ = np.eye(2)

    state, _ = ekf.update([1.0], dt)

    assert -1.0 <= float(state[1]) <= 3.0


def test_growth_rate_process_noise_variance_does_not_affect_race_filter_outputs() -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    growth_rates = plausible_growth_rate_profiles(6.0, dt_hours)["constant_growth"]
    simulated = simulate_profiled_od_observations(
        growth_rates,
        profile_name="noisy_colored",
        dt_hours=dt_hours,
        seed=123,
    )

    low_process_noise_ekf = CultureGrowthEKF(
        initial_state=np.array([1.0, 0.0], dtype=float),
        initial_covariance=np.diag([0.10**2, 0.15**2]),
        process_noise_covariance=np.diag([1e-5, 1e-12]),
        observation_noise_covariance=np.array([[0.003**2]], dtype=float),
        outlier_std_threshold=5.0,
    )
    high_process_noise_ekf = CultureGrowthEKF(
        initial_state=np.array([1.0, 0.0], dtype=float),
        initial_covariance=np.diag([0.10**2, 0.15**2]),
        process_noise_covariance=np.diag([1e-5, 1e2]),
        observation_noise_covariance=np.array([[0.003**2]], dtype=float),
        outlier_std_threshold=5.0,
    )

    low_states: list[np.ndarray] = []
    high_states: list[np.ndarray] = []
    low_covariances: list[np.ndarray] = []
    high_covariances: list[np.ndarray] = []

    for observation in simulated["observed_od"][1:]:
        low_state, low_covariance = low_process_noise_ekf.update([float(observation)], dt_hours)
        high_state, high_covariance = high_process_noise_ekf.update([float(observation)], dt_hours)
        low_states.append(np.asarray(low_state, dtype=float).copy())
        high_states.append(np.asarray(high_state, dtype=float).copy())
        low_covariances.append(np.asarray(low_covariance, dtype=float).copy())
        high_covariances.append(np.asarray(high_covariance, dtype=float).copy())

    np.testing.assert_allclose(
        np.asarray(low_states, dtype=float),
        np.asarray(high_states, dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(low_covariances, dtype=float),
        np.asarray(high_covariances, dtype=float),
        rtol=0.0,
        atol=0.0,
    )
