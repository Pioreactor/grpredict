#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from grpredict import CultureGrowthEKF


def stream_observations(dt_hours: float) -> np.ndarray:
    truth = 0.5
    rng = np.random.default_rng(7)
    while True:
        truth = truth * np.exp(0.1 * dt_hours)
        yield truth + 0.05 * rng.normal(),


def main() -> None:
    dt_hours = 5.0 / 60.0 / 60.0
    stream_steps = 12
    observation_stream = stream_observations(dt_hours)

    first_observation = next(observation_stream)
    initial_log_od = float(np.log(max(np.median(first_observation), 1e-9)))

    ekf = CultureGrowthEKF(
        initial_state=np.array([initial_log_od, 0.0, 0.0], dtype=float),
        initial_covariance=np.diag([0.10**2, 0.15**2, 0.15**2]),
        process_noise_covariance=np.diag([1e-8, 6e-8, 6e-6]),
        observation_noise_covariance=np.diag([0.05**2]),
        outlier_std_threshold=5.0,
    )

    print("First raw observations (AU):", np.round(first_observation, 6))
    print("Initial hidden state [log_od, growth_rate, growth_rate_drift]:", np.round(ekf.state_, 6))
    print("Initial covariance:\n", np.round(ekf.covariance_, 6))
    print("Process noise covariance:\n", np.round(ekf.process_noise_covariance, 6))
    print("Observation noise covariance:\n", np.round(ekf.observation_noise_covariance, 6))
    print()
    print("stream_step  raw_sensor_1  estimated_od  estimated_growth_rate")

    for step_index, raw_observation in enumerate(observation_stream, start=1):
        if step_index > stream_steps:
            break
        state, _ = ekf.update(raw_observation, dt_hours)
        print(
            f"{step_index:>4}  "
            f"{float(raw_observation[0]):>12.6f}  "
            f"{float(np.exp(state[0])):>12.6f}  "
            f"{float(state[1]):>21.6f}"
        )


if __name__ == "__main__":
    main()
