#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from grpredict import build_filter_from_observation_summary
from grpredict import normalize_observation_by_factor
from grpredict import summarize_warmup_observations


def stream_observations(dt_hours: float) -> float:
    truth = 0.5
    while True:
        truth = truth * np.exp(0.1 * dt_hours)
        yield truth + 0.1 * np.random.randn()


def main() -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    warmup_length = 5

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

    print("Warmup length:", warmup_length)
    print("Warmup observations (AU):", np.round(warmup_observations, 6))
    print("Normalization factor:", round(float(summary["normalization_factor"]), 6))
    print("Initial state:", np.round(summary["initial_state"], 6))
    print("Initial covariance:\n", np.round(summary["initial_covariance"], 6))
    print("Process noise covariance:\n", np.round(summary["process_noise_covariance"], 6))
    print("Observation noise covariance:\n", np.round(summary["observation_noise_covariance"], 6))
    print()
    print("Normalized warmup observations:", np.round(normalized_warmup, 6))
    print()
    print("stream_step  raw_au  normalized_signal  estimated_signal  estimated_growth_rate")

    for step_index, raw_observation in enumerate(observation_stream, start=warmup_length):
        normalized_observation = normalize_observation_by_factor(
            raw_observation,
            float(summary["normalization_factor"]),
        )
        state, _ = ekf.update([normalized_observation], dt_hours)
        print(
            f"{step_index:>4}  "
            f"{raw_observation:>6.3f}  "
            f"{normalized_observation:>17.6f}  "
            f"{float(state[0]):>16.6f}  "
            f"{float(state[1]):>21.6f}"
        )


if __name__ == "__main__":
    main()
