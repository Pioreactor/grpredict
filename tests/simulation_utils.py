"""Helpers for generating synthetic OD datasets with realistic noise profiles.

1. Build a plausible growth-rate trajectory in units of 1/hour.
2. Integrate that growth curve forward to obtain a latent OD trajectory.
3. Convert latent OD into observed OD by adding a profile-specific noise process.

The observation model combines:
- an additive floor, which matters most at low OD,
- a multiplicative component, which scales with OD magnitude,
- optional AR(1) temporal correlation in the residuals,
- occasional rare shock events to produce heavier-than-Gaussian tails.

Given growth-rate curve `growth_rates[t]`, initial OD `initial_od`, and cadence
`dt_hours`, simulate:

latent_od[t] = latent_od[t-1] * exp(growth_rates[t] * dt_hours)
error_state[t] = rho * error_state[t-1] + sqrt(1 - rho^2) * innovation_t
innovation_t ~ Normal(0, sigma(latent_od[t]))
shock_t = 0 most of the time, with rare jumps of `shock_scale * sigma(latent_od[t])`
observed_od[t] = latent_od[t] + error_state[t] + shock_t
sigma(x) = sqrt(sigma_floor^2 + (cv * x)^2)

This helper does not currently simulate a separate per-unit calibration factor
`c_unit`; if needed later, that would multiply `latent_od` before observation
noise is added.


These helpers live under `tests/` because they are intended to support estimator
tests and regression cases, not to become part of the public package API.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

# `sigma_floor`: additive OD noise floor, independent of signal magnitude.
# `cv`: multiplicative noise level; observation std grows roughly like `cv * OD`.
# `rho`: AR(1) residual persistence; larger values create more colored noise.
# `shock_probability`: per-observation chance of a rare outlier event.
# `shock_scale`: outlier magnitude in units of the local observation sigma.
NOISE_PROFILE_PARAMETERS: dict[str, dict[str, float]] = {
    "nominal_colored": {
        "sigma_floor": 9.85e-04,
        "cv": 0.0054,
        "rho": 0.59,
        "shock_probability": 0.0073,
        "shock_scale": 5.0,
    },
    "nominal_near_iid": {
        "sigma_floor": 0.0,
        "cv": 0.0056,
        "rho": 0.02,
        "shock_probability": 0.0030,
        "shock_scale": 4.0,
    },
    "noisy_colored": {
        "sigma_floor": 7.82e-03,
        "cv": 0.0114,
        "rho": 0.83,
        "shock_probability": 0.0150,
        "shock_scale": 6.0,
    },
}


def interpolate_profile(
    total_hours: float,
    dt_hours: float,
    knot_hours: list[float],
    knot_rates: list[float],
) -> np.ndarray:
    if len(knot_hours) != len(knot_rates):
        raise ValueError("knot_hours and knot_rates must have the same length")
    step_times = np.arange(1, int(round(total_hours / dt_hours)) + 1, dtype=float) * dt_hours
    return np.interp(step_times, np.asarray(knot_hours, dtype=float), np.asarray(knot_rates, dtype=float))


def diauxic_shift_profile(total_hours: float, dt_hours: float) -> np.ndarray:
    return interpolate_profile(
        total_hours,
        dt_hours,
        knot_hours=[0.0, 1.2, 2.8, 4.6, 5.6, 7.0, 9.0, total_hours],
        knot_rates=[0.01, 0.01, 0.23, 0.23, 0.03, 0.16, 0.12, 0.05],
    )




def crash_recovery_profile(total_hours: float, dt_hours: float) -> np.ndarray:
    return interpolate_profile(
        total_hours,
        dt_hours,
        knot_hours=[0.0, 1.2, 3.8, 5.0, 5.1, 6.8, 8.8, total_hours],
        knot_rates=[0.02, 0.02, 0.25, 0.22, -0.03, 0.10, 0.07, 0.02],
    )


def late_restart_profile(total_hours: float, dt_hours: float) -> np.ndarray:
    return interpolate_profile(
        total_hours,
        dt_hours,
        knot_hours=[0.0, 2.0, 4.8, 6.2, 7.2, 9.0, total_hours],
        knot_rates=[0.005, 0.005, 0.18, 0.015, 0.015, 0.14, 0.05],
    )


def lag_log_stationary_profile(total_hours: float, dt_hours: float) -> np.ndarray:
    return interpolate_profile(
        total_hours,
        dt_hours,
        knot_hours=[0.0, 1.8, 3.2, 5.2, 7.8, 9.8, total_hours],
        knot_rates=[0.004, 0.004, 0.11, 0.24, 0.17, 0.06, 0.015],
    )


def nutrient_pulse_profile(total_hours: float, dt_hours: float) -> np.ndarray:
    return interpolate_profile(
        total_hours,
        dt_hours,
        knot_hours=[0.0, 1.5, 3.8, 5.4, 6.0, 7.5, 9.5, total_hours],
        knot_rates=[0.02, 0.02, 0.15, 0.13, 0.23, 0.18, 0.09, 0.03],
    )


def byproduct_inhibition_profile(total_hours: float, dt_hours: float) -> np.ndarray:
    return interpolate_profile(
        total_hours,
        dt_hours,
        knot_hours=[0.0, 1.2, 3.5, 5.2, 7.0, 8.4, 10.0, total_hours],
        knot_rates=[0.01, 0.01, 0.22, 0.21, 0.12, 0.05, 0.01, -0.01],
    )


def washout_recovery_profile(total_hours: float, dt_hours: float) -> np.ndarray:
    return interpolate_profile(
        total_hours,
        dt_hours,
        knot_hours=[0.0, 1.6, 3.6, 5.0, 5.8, 7.4, 9.2, total_hours],
        knot_rates=[0.015, 0.015, 0.17, 0.16, -0.025, 0.045, 0.08, 0.03],
    )


def narrow_peak_profile(total_hours: float, dt_hours: float) -> np.ndarray:
    step_times = np.arange(1, int(round(total_hours / dt_hours)) + 1, dtype=float) * dt_hours
    baseline = np.full_like(step_times, 0.02)
    peak = 0.24 * np.exp(-0.5 * ((step_times - 5.0) / 0.7) ** 2)
    shoulder = 0.08 * np.exp(-0.5 * ((step_times - 8.2) / 1.1) ** 2)
    return baseline + peak + shoulder


def zero_growth_profile(total_hours: float, dt_hours: float) -> np.ndarray:
    return np.zeros(int(round(total_hours / dt_hours)), dtype=float)


def constant_growth_profile(total_hours: float, dt_hours: float) -> np.ndarray:
    return 0.25 * np.ones(int(round(total_hours / dt_hours)), dtype=float)


def plausible_growth_rate_profiles(total_hours: float, dt_hours: float) -> dict[str, np.ndarray]:
    return {
        "diauxic_shift": diauxic_shift_profile(total_hours, dt_hours),
        "crash_recovery": crash_recovery_profile(total_hours, dt_hours),
        "late_restart": late_restart_profile(total_hours, dt_hours),
        "lag_log_stationary": lag_log_stationary_profile(total_hours, dt_hours),
        "nutrient_pulse": nutrient_pulse_profile(total_hours, dt_hours),
        "byproduct_inhibition": byproduct_inhibition_profile(total_hours, dt_hours),
        "washout_recovery": washout_recovery_profile(total_hours, dt_hours),
        "narrow_peak": narrow_peak_profile(total_hours, dt_hours),
        "zero_growth": zero_growth_profile(total_hours, dt_hours),
        "constant_growth": constant_growth_profile(total_hours, dt_hours),
    }


def simulate_latent_od_from_growth_rates(
    growth_rates: Iterable[float],
    dt_hours: float,
    initial_od: float = 1.0,
) -> np.ndarray:
    rates = np.asarray(list(growth_rates), dtype=float)
    latent_od = np.empty(rates.size + 1, dtype=float)
    latent_od[0] = initial_od
    for index, rate in enumerate(rates, start=1):
        latent_od[index] = latent_od[index - 1] * math.exp(float(rate) * dt_hours)
    return latent_od


def _sigma_for_observation(od: float, profile_name: str) -> float:
    parameters = NOISE_PROFILE_PARAMETERS[profile_name]
    return math.sqrt(parameters["sigma_floor"] ** 2 + (parameters["cv"] * od) ** 2)


def simulate_noisy_observations_from_latent_od(
    latent_od: Iterable[float],
    profile_name: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if profile_name not in NOISE_PROFILE_PARAMETERS:
        raise ValueError(f"Unknown profile: {profile_name}")

    latent = np.asarray(list(latent_od), dtype=float)
    observations = np.empty(latent.size, dtype=float)
    parameters = NOISE_PROFILE_PARAMETERS[profile_name]
    generator = rng or np.random.default_rng()

    error_state = 0.0
    for index, od in enumerate(latent):
        sigma = _sigma_for_observation(float(od), profile_name)
        innovation = generator.normal(loc=0.0, scale=sigma)
        error_state = parameters["rho"] * error_state + math.sqrt(
            max(1.0 - parameters["rho"] ** 2, 0.0)
        ) * innovation
        shock = 0.0
        if generator.uniform() < parameters["shock_probability"]:
            shock = generator.choice([-1.0, 1.0]) * parameters["shock_scale"] * sigma
        observations[index] = max(float(od) + error_state + shock, 0.0)
    return observations


def simulate_profiled_od_observations(
    growth_rates: Iterable[float],
    profile_name: str,
    dt_hours: float,
    *,
    initial_od: float = 1.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rates = np.asarray(list(growth_rates), dtype=float)
    latent_od = simulate_latent_od_from_growth_rates(rates, dt_hours, initial_od=initial_od)
    observed_od = simulate_noisy_observations_from_latent_od(
        latent_od,
        profile_name=profile_name,
        rng=rng,
    )
    time_hours = np.arange(latent_od.size, dtype=float) * dt_hours
    return {
        "time_hours": time_hours,
        "growth_rates": rates,
        "latent_od": latent_od,
        "observed_od": observed_od,
    }
