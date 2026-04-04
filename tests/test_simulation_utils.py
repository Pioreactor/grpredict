from __future__ import annotations

import numpy as np
import pytest

from tests.simulation_utils import NOISE_PROFILE_PARAMETERS
from tests.simulation_utils import plausible_growth_rate_profiles
from tests.simulation_utils import simulate_latent_od_from_growth_rates
from tests.simulation_utils import simulate_profiled_od_observations




@pytest.mark.parametrize("profile_name", ["nominal_colored", "nominal_near_iid", "noisy_colored"])
def test_simulate_profiled_od_observations_returns_expected_shapes(profile_name: str) -> None:
    growth_rates = plausible_growth_rate_profiles(12.0, 5.0 / 60.0)["diauxic_shift"]
    simulated = simulate_profiled_od_observations(
        growth_rates,
        profile_name=profile_name,
        dt_hours=5.0 / 60.0,
        seed=7,
    )
    assert simulated["growth_rates"].shape == (144,)
    assert simulated["latent_od"].shape == (145,)
    assert simulated["observed_od"].shape == (145,)
    assert simulated["time_hours"].shape == (145,)


def test_noise_profiles_produce_distinct_observation_series() -> None:
    growth_rates = plausible_growth_rate_profiles(12.0, 5.0 / 60.0)["diauxic_shift"]
    nominal = simulate_profiled_od_observations(
        growth_rates,
        profile_name="nominal_colored",
        dt_hours=5.0 / 60.0,
        seed=11,
    )
    iid = simulate_profiled_od_observations(
        growth_rates,
        profile_name="nominal_near_iid",
        dt_hours=5.0 / 60.0,
        seed=11,
    )
    noisy = simulate_profiled_od_observations(
        growth_rates,
        profile_name="noisy_colored",
        dt_hours=5.0 / 60.0,
        seed=11,
    )

    nominal_error = nominal["observed_od"] - nominal["latent_od"]
    iid_error = iid["observed_od"] - iid["latent_od"]
    noisy_error = noisy["observed_od"] - noisy["latent_od"]

    assert np.std(noisy_error) > np.std(nominal_error)
    assert np.std(noisy_error) > np.std(iid_error)
    assert NOISE_PROFILE_PARAMETERS["nominal_near_iid"]["rho"] < NOISE_PROFILE_PARAMETERS["nominal_colored"]["rho"]


def test_zero_growth_latent_od_stays_constant() -> None:
    dt_hours = 5.0 / 60.0
    growth_rates = plausible_growth_rate_profiles(12.0, dt_hours)["zero_growth"]
    latent_od = simulate_latent_od_from_growth_rates(growth_rates, dt_hours, initial_od=1.23)

    assert np.allclose(latent_od, 1.23)


@pytest.mark.parametrize(
    "profile_name",
    ["diauxic_shift", "late_restart", "narrow_peak", "zero_growth", "constant_growth"],
)
def test_nonnegative_growth_profiles_produce_monotone_latent_od(profile_name: str) -> None:
    dt_hours = 5.0 / 60.0
    profiles = plausible_growth_rate_profiles(12.0, dt_hours)
    latent_od = simulate_latent_od_from_growth_rates(profiles[profile_name], dt_hours, initial_od=1.0)
    assert np.all(np.diff(latent_od) >= -1e-12), profile_name


@pytest.mark.parametrize("profile_name", ["nominal_colored", "nominal_near_iid", "noisy_colored"])
def test_same_seed_produces_identical_simulated_observations(profile_name: str) -> None:
    dt_hours = 5.0 / 60.0
    growth_rates = plausible_growth_rate_profiles(12.0, dt_hours)["late_restart"]
    first = simulate_profiled_od_observations(
        growth_rates,
        profile_name=profile_name,
        dt_hours=dt_hours,
        seed=99,
    )
    second = simulate_profiled_od_observations(
        growth_rates,
        profile_name=profile_name,
        dt_hours=dt_hours,
        seed=99,
    )

    assert np.array_equal(first["observed_od"], second["observed_od"])
    assert np.array_equal(first["latent_od"], second["latent_od"])


@pytest.mark.parametrize("profile_name", ["nominal_colored", "nominal_near_iid", "noisy_colored"])
def test_different_seeds_change_observed_noise_realization(profile_name: str) -> None:
    dt_hours = 5.0 / 60.0
    growth_rates = plausible_growth_rate_profiles(12.0, dt_hours)["late_restart"]
    first = simulate_profiled_od_observations(
        growth_rates,
        profile_name=profile_name,
        dt_hours=dt_hours,
        seed=99,
    )
    second = simulate_profiled_od_observations(
        growth_rates,
        profile_name=profile_name,
        dt_hours=dt_hours,
        seed=100,
    )

    assert not np.array_equal(first["observed_od"], second["observed_od"])
    assert np.array_equal(first["latent_od"], second["latent_od"])


def test_noise_families_preserve_expected_autocorrelation_ordering() -> None:
    dt_hours = 5.0 / 60.0
    growth_rates = plausible_growth_rate_profiles(12.0, dt_hours)["zero_growth"]

    def average_lag1(profile_name: str) -> float:
        lag1_values: list[float] = []
        for seed in range(10):
            simulated = simulate_profiled_od_observations(
                growth_rates,
                profile_name=profile_name,
                dt_hours=dt_hours,
                seed=seed,
            )
            error = simulated["observed_od"] - simulated["latent_od"]
            lag1_values.append(float(np.corrcoef(error[:-1], error[1:])[0, 1]))
        return float(np.mean(lag1_values))

    iid_lag1 = average_lag1("nominal_near_iid")
    nominal_lag1 = average_lag1("nominal_colored")
    noisy_lag1 = average_lag1("noisy_colored")

    assert abs(iid_lag1 - NOISE_PROFILE_PARAMETERS["nominal_near_iid"]["rho"]) < 0.10
    assert iid_lag1 < nominal_lag1 < noisy_lag1
    assert nominal_lag1 > 0.35
    assert noisy_lag1 > 0.50
