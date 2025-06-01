# test_ekf.py

# tests/test_culture_growth_ekf.py

import numpy as np
import pytest

# Replace `your_module` with the actual module/path where CultureGrowthEKF is defined.
from grpredict import CultureGrowthEKF


def make_ekf(dt: float) -> CultureGrowthEKF:
    """
    Helper to construct a CultureGrowthEKF with:
      - 2×2 initial covariance = identity
      - small process-noise covariance (to keep it nearly deterministic)
      - small observation-noise covariance (one sensor)
      - a single “identity” angle (so h(x) = [OD])
      - a large outlier threshold (to disable outlier handling)
    """
    initial_state = np.array([1.0, 0.0])  # OD = 1.0, r = 0.0 (we’ll overwrite r later)
    initial_covariance = np.eye(2)
    # Very small process noise on both OD and r (positive definite)
    process_noise_covariance = np.array([[1e-6, 0.0],
                                         [0.0, 1e-8]])
    # Very small observation noise (one sensor)
    observation_noise_covariance = np.array([[1e-6]])
    angles = ["0"]  # “0” means use identity (h(x) = OD)
    outlier_std_threshold = 1e6  # effectively disable outlier checks

    ekf = CultureGrowthEKF(
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        process_noise_covariance=process_noise_covariance,
        observation_noise_covariance=observation_noise_covariance,
        angles=angles,
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
