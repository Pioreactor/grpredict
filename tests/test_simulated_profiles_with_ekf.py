from __future__ import annotations

import math

import numpy as np
import pytest

from grpredict import CultureGrowthEKF
from tests.simulation_utils import NOISE_PROFILE_PARAMETERS
from tests.simulation_utils import plausible_growth_rate_profiles
from tests.simulation_utils import simulate_noisy_observations_from_latent_od
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


def run_ekf_over_observations_with_dilution_flags(
    observed_od: np.ndarray,
    dt_hours: float,
    profile_name: str,
    recent_dilution_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ekf = make_single_sensor_ekf(profile_name)
    estimated_od: list[float] = []
    estimated_rates: list[float] = []

    for observation, recent_dilution in zip(
        observed_od[1:],
        recent_dilution_flags,
        strict=True,
    ):
        state, _ = ekf.update(
            [float(observation)],
            dt_hours,
            recent_dilution=bool(recent_dilution),
        )
        estimated_od.append(float(state[0]))
        estimated_rates.append(float(state[1]))

    return np.asarray(estimated_od, dtype=float), np.asarray(estimated_rates, dtype=float)


def simulate_constant_growth_with_regular_dosing(
    total_hours: float,
    dt_hours: float,
    profile_name: str,
    *,
    seed: int,
    initial_od: float = 1.0,
    start_dosing_after_hours: float = 3.0,
    dosing_interval_hours: float = 2.0,
    dosing_drop_fraction: float = 0.18,
    dosing_drop_fraction_noise: float = 0.015,
) -> dict[str, np.ndarray]:
    growth_rates = plausible_growth_rate_profiles(total_hours, dt_hours)["constant_growth"]
    generator = np.random.default_rng(seed)

    latent_od = np.empty(growth_rates.size + 1, dtype=float)
    latent_od[0] = initial_od
    recent_dilution_flags = np.zeros(growth_rates.size, dtype=bool)

    start_dosing_step = int(round(start_dosing_after_hours / dt_hours))
    dosing_interval_steps = int(round(dosing_interval_hours / dt_hours))
    dosing_steps = np.arange(
        start_dosing_step,
        growth_rates.size + 1,
        dosing_interval_steps,
        dtype=int,
    )

    dosing_drop_fractions: list[float] = []
    for step_index, rate in enumerate(growth_rates, start=1):
        latent_od[step_index] = latent_od[step_index - 1] * math.exp(float(rate) * dt_hours)

        if step_index in dosing_steps:
            drop_fraction = dosing_drop_fraction + generator.normal(
                loc=0.0,
                scale=dosing_drop_fraction_noise,
            )
            drop_fraction = float(np.clip(drop_fraction, 0.08, 0.35))
            latent_od[step_index] *= 1.0 - drop_fraction
            recent_dilution_flags[step_index - 1] = True
            dosing_drop_fractions.append(drop_fraction)

    observed_od = simulate_noisy_observations_from_latent_od(
        latent_od,
        profile_name=profile_name,
        rng=generator,
    )
    dosing_indices = np.flatnonzero(recent_dilution_flags)

    return {
        "growth_rates": growth_rates,
        "latent_od": latent_od,
        "observed_od": observed_od,
        "recent_dilution_flags": recent_dilution_flags,
        "dosing_indices": dosing_indices,
        "dosing_drop_fractions": np.asarray(dosing_drop_fractions, dtype=float),
    }


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
    estimated_rates = run_ekf_over_observations(simulated["observed_od"], dt_hours, profile_name)

    assert np.all(np.isfinite(estimated_rates))
    assert abs(0.25 - float(np.median(estimated_rates[-30:]))) < 0.05


@pytest.mark.parametrize("profile_name", ["nominal_colored", "nominal_near_iid", "noisy_colored"])
@pytest.mark.parametrize("seed", [7, 35, 10])
def test_ekf_handles_regular_dosing_without_losing_constant_growth_estimate(
    profile_name: str,
    seed: int,
) -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    simulated = simulate_constant_growth_with_regular_dosing(
        total_hours=12.0,
        dt_hours=dt_hours,
        profile_name=profile_name,
        seed=seed,
    )

    estimated_od, estimated_rates = run_ekf_over_observations_with_dilution_flags(
        simulated["observed_od"],
        dt_hours,
        profile_name,
        simulated["recent_dilution_flags"],
    )
    from matplotlib import pyplot as plt
    #print(simulated["observed_od"][:100])
    #print(estimated_od[:100])
    plt.plot(simulated["observed_od"])
    plt.plot(estimated_rates)
    plt.plot(estimated_od)
    plt.show()

    assert np.all(np.isfinite(estimated_od))
    assert np.all(np.isfinite(estimated_rates))
    assert simulated["dosing_indices"].size >= 3

    for dosing_index in simulated["dosing_indices"]:
        assert estimated_od[dosing_index] < 0.97 * estimated_od[dosing_index - 1]

        pre_window = estimated_rates[max(dosing_index - 12, 0) : dosing_index]
        post_window = estimated_rates[dosing_index + 3 : dosing_index + 15]

        assert pre_window.size >= 6
        assert post_window.size >= 6
        assert abs(float(np.median(post_window)) - float(np.median(pre_window))) < 0.05

    non_dosing_rates = estimated_rates[~simulated["recent_dilution_flags"]]
    assert abs(float(np.median(non_dosing_rates[-24:])) - 0.25) < 0.06


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
