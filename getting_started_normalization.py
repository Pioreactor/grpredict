#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from grpredict import build_filter_from_observation_summary
from grpredict import summarize_warmup_observations
from grpredict import normalize_observations_by_factor


def stream_observations(dt_hours: float) -> np.ndarray:
    truth = 0.5
    rng = np.random.default_rng(11)
    while True:
        truth = truth * np.exp(0.1 * dt_hours)
        yield np.array(
            [
                truth + 0.04 * rng.normal(),
                3.4 * truth + 0.09 * rng.normal(),
            ],
            dtype=float,
        )


def main() -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    warmup_length = 5
    stream_steps = 12

    observation_stream = stream_observations(dt_hours)

    warmup_observations = np.asarray(
        [next(observation_stream) for _ in range(warmup_length)],
        dtype=float,
    )

    summary = summarize_warmup_observations(
        warmup_observations,
        dt_hours,
    )
    ekf = build_filter_from_observation_summary(summary, outlier_std_threshold=5.0)
    normalized_warmup = np.asarray(summary["normalized_warmup_observations"], dtype=float)
    normalization_factors = np.asarray(summary["normalization_factors"], dtype=float)

    print("Warmup length:", warmup_length)
    print("Warmup observations (AU):\n", np.round(warmup_observations, 6))
    print("Normalization factors:", np.round(normalization_factors, 6))
    print(
        "Initial hidden state [log_od, growth_rate, growth_rate_drift]:",
        np.round(summary["initial_state"], 6),
    )
    print("Initial covariance:\n", np.round(summary["initial_covariance"], 6))
    print("Process noise covariance:\n", np.round(summary["process_noise_covariance"], 6))
    print("Observation noise covariance:\n", np.round(summary["observation_noise_covariance"], 6))
    print()
    print("Normalized warmup observations:\n", np.round(normalized_warmup, 6))
    print()
    print(
        "stream_step  raw_sensor_1  raw_sensor_2  normalized_1  normalized_2  estimated_od  estimated_growth_rate"
    )

    for step_index, raw_observation in enumerate(observation_stream, start=warmup_length):
        if step_index >= warmup_length + stream_steps:
            break
        normalized_observation = np.asarray(
            normalize_observations_by_factor(
                raw_observation,
                normalization_factors,
            ),
            dtype=float,
        )
        state, _ = ekf.update(normalized_observation, dt_hours)
        print(
            f"{step_index:>4}  "
            f"{float(raw_observation[0]):>12.6f}  "
            f"{float(raw_observation[1]):>12.6f}  "
            f"{float(normalized_observation[0]):>12.6f}  "
            f"{float(normalized_observation[1]):>12.6f}  "
            f"{float(np.exp(state[0])):>12.6f}  "
            f"{float(state[1]):>21.6f}"
        )


if __name__ == "__main__":
    main()
