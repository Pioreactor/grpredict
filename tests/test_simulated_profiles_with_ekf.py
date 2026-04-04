from __future__ import annotations

import math

import numpy as np
import pytest

from grpredict import CultureGrowthEKF
from tests.simulation_utils import NOISE_PROFILE_PARAMETERS
from tests.simulation_utils import plausible_growth_rate_profiles
from tests.simulation_utils import simulate_profiled_od_observations


def make_single_sensor_ekf(profile_name: str) -> CultureGrowthEKF:
    parameters = NOISE_PROFILE_PARAMETERS[profile_name]
    initial_state = np.array([1.0, 0.0], dtype=float)
    initial_covariance = np.diag([0.10**2, 0.15**2])
    process_noise_covariance = np.diag([1e-5, 1e-6 * (1.0 + 0.5 * parameters["rho"])])
    observation_noise_covariance = np.array([[0.003**2]], dtype=float)
    return CultureGrowthEKF(
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        process_noise_covariance=process_noise_covariance,
        observation_noise_covariance=observation_noise_covariance,
        angles=["0"],
        outlier_std_threshold=5.0,
    )


def run_ekf_over_observations(
    observed_od: np.ndarray,
    dt_hours: float,
    profile_name: str,
) -> np.ndarray:
    ekf = make_single_sensor_ekf(profile_name)
    estimated_rates: list[float] = []
    for observation in observed_od[1:]:
        state, _ = ekf.update([float(observation)], dt_hours)
        estimated_rates.append(float(state[1]))
    return np.asarray(estimated_rates, dtype=float)


def root_mean_square_error(estimated: np.ndarray, actual: np.ndarray) -> float:
    return math.sqrt(float(np.mean((estimated - actual) ** 2)))


@pytest.mark.parametrize(
    "growth_profile_name",
    ["diauxic_shift", "late_restart", "constant_growth", "crash_recovery", "lag_log_stationary", "nutrient_pulse" ],
)
@pytest.mark.parametrize("profile_name", ["nominal_colored", "nominal_near_iid", "noisy_colored"])
def test_ekf_tracks_positive_growth_on_simulated_profiles(
    growth_profile_name: str,
    profile_name: str,
) -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    growth_rates = plausible_growth_rate_profiles(12.0, dt_hours)[growth_profile_name]
    simulated = simulate_profiled_od_observations(
        growth_rates,
        profile_name=profile_name,
        dt_hours=dt_hours,
        seed=123,
    )

    estimated_rates = run_ekf_over_observations(simulated["observed_od"], dt_hours, profile_name)
    true_tail_mean = float(np.mean(growth_rates[-20:]))
    estimated_tail_mean = float(np.mean(estimated_rates[-20:]))

    assert np.all(np.isfinite(estimated_rates))
    assert estimated_tail_mean > 0.0
    assert estimated_tail_mean > 0.75 * true_tail_mean




@pytest.mark.parametrize("profile_name", ["nominal_colored", "nominal_near_iid", "noisy_colored"])
def test_ekf_outputs_remain_finite_for_crash_recovery_profile(profile_name: str) -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    growth_rates = plausible_growth_rate_profiles(12.0, dt_hours)["crash_recovery"]
    simulated = simulate_profiled_od_observations(
        growth_rates,
        profile_name=profile_name,
        dt_hours=dt_hours,
        seed=222,
    )
    estimated_rates = run_ekf_over_observations(simulated["observed_od"], dt_hours, profile_name)

    assert np.all(np.isfinite(estimated_rates))
    assert np.max(np.abs(estimated_rates)) < 1.0


@pytest.mark.parametrize("profile_name", ["nominal_colored", "nominal_near_iid"])
def test_ekf_detects_negative_growth_during_crash_recovery(profile_name: str) -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    growth_rates = plausible_growth_rate_profiles(12.0, dt_hours)["crash_recovery"]
    simulated = simulate_profiled_od_observations(
        growth_rates,
        profile_name=profile_name,
        dt_hours=dt_hours,
        seed=456,
    )
    estimated_rates = run_ekf_over_observations(simulated["observed_od"], dt_hours, profile_name)

    assert np.min(estimated_rates) < -0.005


@pytest.mark.parametrize("profile_name", ["nominal_colored", "nominal_near_iid", "noisy_colored"])
def test_ekf_reports_near_zero_growth_for_zero_growth_profile_across_noise_families(profile_name: str) -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    growth_rates = plausible_growth_rate_profiles(12.0, dt_hours)["zero_growth"]
    simulated = simulate_profiled_od_observations(
        growth_rates,
        profile_name=profile_name,
        dt_hours=dt_hours,
        seed=321,
    )
    estimated_rates = run_ekf_over_observations(simulated["observed_od"], dt_hours, profile_name)

    assert np.all(np.isfinite(estimated_rates))
    assert abs(float(np.median(estimated_rates[-30:]))) < 0.03


@pytest.mark.parametrize("profile_name", ["nominal_colored", "nominal_near_iid", "noisy_colored"])
def test_ekf_reports_near_constant_growth_for_constant_growth_profile_across_noise_families(profile_name: str) -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    growth_rates = plausible_growth_rate_profiles(6.0, dt_hours)["constant_growth"]
    simulated = simulate_profiled_od_observations(
        growth_rates,
        profile_name=profile_name,
        dt_hours=dt_hours,
        seed=321,
    )
    from matplotlib import pyplot as plt
    estimated_rates = run_ekf_over_observations(simulated["observed_od"], dt_hours, profile_name)

    assert np.all(np.isfinite(estimated_rates))
    assert abs(0.25 - float(np.median(estimated_rates[-30:]))) < 0.05


def test_ekf_mean_per_scenario_rmse_stays_below_target_across_explicit_profiles() -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    total_hours_by_profile = {
        "diauxic_shift": 12.0,
        "crash_recovery": 12.0,
        "late_restart": 12.0,
        "narrow_peak": 12.0,
        "zero_growth": 12.0,
        "constant_growth": 6.0,
        "lag_log_stationary": 12.0,
        "nutrient_pulse": 12.0,
        "byproduct_inhibition": 12.0,
        "washout_recovery": 12.0,
    }
    noise_families = [
        "nominal_colored",
        "nominal_near_iid",
        "noisy_colored",
    ]

    per_scenario_rmses: list[float] = []
    for growth_profile_name, total_hours in total_hours_by_profile.items():
        growth_rates = plausible_growth_rate_profiles(total_hours, dt_hours)[growth_profile_name]
        for noise_family in noise_families:
            simulated = simulate_profiled_od_observations(
                growth_rates,
                profile_name=noise_family,
                dt_hours=dt_hours,
                seed=321,
            )
            estimated_rates = run_ekf_over_observations(
                simulated["observed_od"],
                dt_hours,
                noise_family,
            )
            per_scenario_rmses.append(root_mean_square_error(estimated_rates, growth_rates))

    mean_per_scenario_rmse = float(np.mean(per_scenario_rmses))
    assert mean_per_scenario_rmse < 0.045
