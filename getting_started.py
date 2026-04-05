#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from grpredict import CultureGrowthEKF


def stream_observations(dt_hours: float) -> float:
    truth = 0.5
    rng = np.random.default_rng(7)
    while True:
        truth = truth * np.exp(0.1 * dt_hours)
        yield float(truth + 0.05 * rng.normal())


def main() -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    observation_stream = stream_observations(dt_hours)

    first_observation = next(observation_stream)

    ekf = CultureGrowthEKF(
        initial_state=np.array([first_observation, 0.0], dtype=float),
        initial_covariance=np.diag([0.10**2, 0.15**2]),
        process_noise_covariance=np.diag([1e-5, 1e-6]),
        observation_noise_covariance=np.array([[0.05**2]], dtype=float),
        outlier_std_threshold=5.0,
    )

    print("First raw observation (AU):", round(float(first_observation), 6))
    print("Initial state:", np.round(ekf.state_, 6))
    print("Initial covariance:\n", np.round(ekf.covariance_, 6))
    print("Process noise covariance:\n", np.round(ekf.process_noise_covariance, 6))
    print("Observation noise covariance:\n", np.round(ekf.observation_noise_covariance, 6))
    print()
    print("stream_step  raw_au  estimated_signal  estimated_growth_rate")

    for step_index, raw_observation in enumerate(observation_stream):
        state, _ = ekf.update([raw_observation], dt_hours)
        print(
            f"{step_index:>4}  "
            f"{raw_observation:>6.3f}  "
            f"{float(state[0]):>16.6f}  "
            f"{float(state[1]):>21.6f}"
        )


if __name__ == "__main__":
    main()
