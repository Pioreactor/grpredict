from __future__ import annotations

import numpy as np
import pytest

from grpredict import build_filter_from_observation_summary
from grpredict import CultureGrowthEKF
from grpredict import _is_positive_definite
from grpredict import normalize_observation_by_factor
from grpredict import normalize_observations_by_factor
from grpredict import summarize_warmup_observations


def run_filter_over_initialized_observations(
    warmup_observations: np.ndarray,
    raw_observations: np.ndarray,
    dt_hours: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    summary = summarize_warmup_observations(
        warmup_observations,
        dt_hours,
    )
    ekf = build_filter_from_observation_summary(summary)
    assert isinstance(ekf, CultureGrowthEKF)

    estimated_signal: list[float] = []
    estimated_growth_rate: list[float] = []

    for raw_observation in raw_observations[1:]:
        normalized_observation = normalize_observation_by_factor(
            float(raw_observation),
            float(summary["normalization_factor"]),
        )
        state, _ = ekf.update([normalized_observation], dt_hours)
        estimated_signal.append(float(np.exp(state[0])))
        estimated_growth_rate.append(float(state[1]))

    return (
        summary,
        np.asarray(estimated_signal, dtype=float),
        np.asarray(estimated_growth_rate, dtype=float),
    )


def test_end_to_end_warmup_initialization_tracks_a_simple_au_growth_trace() -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    raw_observations = np.array(
        [
            2.43,
            2.44,
            2.45,
            2.46,
            2.45,
            2.47,
            2.49,
            2.52,
            2.55,
            2.59,
            2.63,
            2.68,
            2.73,
            2.79,
            2.86,
        ],
        dtype=float,
    )
    warmup_observations = raw_observations[:5]

    summary, estimated_signal, estimated_growth_rate = run_filter_over_initialized_observations(
        warmup_observations,
        raw_observations,
        dt_hours,
    )

    normalized_observations = np.asarray(
        normalize_observations_by_factor(raw_observations, summary["normalization_factor"]),
        dtype=float,
    )
    assert pytest.approx(1.0, rel=0.02) == float(np.median(normalized_observations[:5]))
    assert np.all(np.isfinite(estimated_signal))
    assert np.all(np.isfinite(estimated_growth_rate))
    assert estimated_signal[-1] > estimated_signal[0]
    assert estimated_growth_rate[-1] > 0.0


def test_end_to_end_gain_scaled_au_trace_produces_same_growth_rate_path() -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    raw_observations = np.array(
        [
            2.43,
            2.44,
            2.45,
            2.46,
            2.45,
            2.47,
            2.49,
            2.52,
            2.55,
            2.59,
            2.63,
            2.68,
            2.73,
            2.79,
            2.86,
        ],
        dtype=float,
    )
    scaled_observations = 3.7 * raw_observations
    warmup_observations = raw_observations[:5]
    scaled_warmup_observations = scaled_observations[:5]

    base_summary, base_signal, base_growth_rate = run_filter_over_initialized_observations(
        warmup_observations,
        raw_observations,
        dt_hours,
    )
    scaled_summary, scaled_signal, scaled_growth_rate = run_filter_over_initialized_observations(
        scaled_warmup_observations,
        scaled_observations,
        dt_hours,
    )

    assert pytest.approx(3.7, rel=1e-3) == (
        float(scaled_summary["normalization_factor"])
        / float(base_summary["normalization_factor"])
    )
    np.testing.assert_allclose(base_signal, scaled_signal, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(base_growth_rate, scaled_growth_rate, rtol=1e-6, atol=1e-6)


def test_end_to_end_noisy_offset_au_trace_yields_finite_matrices_and_positive_growth() -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    raw_observations = np.array(
        [
            1.84,
            1.86,
            1.85,
            1.87,
            1.86,
            1.87,
            1.90,
            1.89,
            1.94,
            1.98,
            2.01,
            2.05,
            2.04,
            2.11,
            2.15,
            2.18,
        ],
        dtype=float,
    )
    warmup_observations = raw_observations[:6]

    summary, estimated_signal, estimated_growth_rate = run_filter_over_initialized_observations(
        warmup_observations,
        raw_observations,
        dt_hours,
    )

    initial_covariance = np.asarray(summary["initial_covariance"], dtype=float)
    process_noise_covariance = np.asarray(summary["process_noise_covariance"], dtype=float)
    observation_noise_covariance = np.asarray(
        summary["observation_noise_covariance"],
        dtype=float,
    )

    assert _is_positive_definite(initial_covariance)
    assert _is_positive_definite(process_noise_covariance)
    assert _is_positive_definite(observation_noise_covariance)
    assert np.all(np.isfinite(estimated_signal))
    assert np.all(np.isfinite(estimated_growth_rate))
    assert float(np.median(estimated_growth_rate[-5:])) > 0.0
