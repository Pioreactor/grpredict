# -*- coding: utf-8 -*-
from __future__ import annotations
from math import log
from math import sqrt
from typing import Any
from typing import TypeAlias

FloatVectorLike: TypeAlias = Any
FloatMatrixLike: TypeAlias = Any

MINIMUM_OBSERVATION_VALUE = 1e-9

def _as_positive_1d_array(values: FloatVectorLike) -> FloatVectorLike:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional sequence of observations")
    if array.size == 0:
        raise ValueError("Expected at least one observation")
    return np.maximum(array, MINIMUM_OBSERVATION_VALUE)


def _is_positive_definite(A: FloatMatrixLike) -> bool:
    import numpy as np

    A = np.asarray(A, dtype=float)
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def _robust_std(values: FloatVectorLike) -> float:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return 1.4826 * mad


def estimate_normalization_factor_from_warmup_observations(
    observations: FloatVectorLike,
) -> float:
    import numpy as np

    positive_observations = _as_positive_1d_array(observations)
    if positive_observations.size < 2:
        raise ValueError("Need at least two warmup observations")
    log_reference = float(np.median(np.log(positive_observations)))
    return float(np.exp(log_reference))


def normalize_observation_by_factor(
    observation: float,
    normalization_factor: float,
) -> float:
    return max(float(observation), MINIMUM_OBSERVATION_VALUE) / float(normalization_factor)


def normalize_observations_by_factor(
    observations: FloatVectorLike,
    normalization_factor: float,
) -> FloatVectorLike:
    import numpy as np

    if normalization_factor <= 0:
        raise ValueError("normalization_factor must be positive")
    positive_observations = _as_positive_1d_array(observations)
    return np.maximum(
        positive_observations / float(normalization_factor),
        MINIMUM_OBSERVATION_VALUE,
    )


def estimate_observation_noise_covariance_from_warmup_observations(
    normalized_observations: FloatVectorLike,
    dt_hours: float,
) -> FloatMatrixLike:
    import numpy as np

    positive_observations = _as_positive_1d_array(normalized_observations)
    if positive_observations.size < 2:
        raise ValueError("Need at least two warmup observations")

    time_hours = np.arange(positive_observations.size, dtype=float) * float(dt_hours)
    log_warmup = np.log(positive_observations)
    design = np.column_stack([np.ones(positive_observations.size, dtype=float), time_hours])
    coefficients, _, _, _ = np.linalg.lstsq(design, log_warmup, rcond=None)
    fitted_log_signal = design @ coefficients
    log_residuals = log_warmup - fitted_log_signal
    log_residual_std = max(_robust_std(log_residuals), 1e-3)
    return np.array([[log_residual_std * log_residual_std]], dtype=float)


def estimate_initial_covariance_from_warmup_observations(
    normalized_observations: FloatVectorLike,
    dt_hours: float,
) -> FloatMatrixLike:
    import numpy as np

    positive_observations = _as_positive_1d_array(normalized_observations)
    if positive_observations.size < 2:
        raise ValueError("Need at least two warmup observations")

    log_warmup = np.log(positive_observations)
    sigma_log_od0 = max(2.0 * _robust_std(log_warmup), 0.05)

    if positive_observations.size < 3:
        sigma_gr0 = 0.15
    else:
        startup_growth_samples = np.diff(log_warmup) / float(dt_hours)
        sigma_gr0 = max(2.0 * _robust_std(startup_growth_samples), 0.15)

    sigma_gr_drift0 = max(sigma_gr0, 0.01)
    return np.diag(
        [
            sigma_log_od0 * sigma_log_od0,
            sigma_gr0 * sigma_gr0,
            sigma_gr_drift0 * sigma_gr_drift0,
        ]
    ).astype(float)


def make_process_noise_covariance(
    dt_hours: float,
    *,
    reference_dt_hours: float = 5.0 / 60.0 / 60.0,
) -> FloatMatrixLike:
    import numpy as np

    scale = max(float(dt_hours) / float(reference_dt_hours), 0.25)
    return np.diag([1e-8 * scale, 6e-8 * scale, 6e-6 * scale]).astype(float)


def summarize_warmup_observations(
    observations: FloatVectorLike,
    dt_hours: float,
) -> dict[str, object]:
    import numpy as np

    normalization_factor = estimate_normalization_factor_from_warmup_observations(observations)
    normalized_observations = normalize_observations_by_factor(
        observations,
        normalization_factor,
    )
    initial_state = np.array([0.0, 0.0, 0.0], dtype=float)
    initial_covariance = estimate_initial_covariance_from_warmup_observations(
        normalized_observations,
        dt_hours,
    )
    process_noise_covariance = make_process_noise_covariance(dt_hours)
    observation_noise_covariance = estimate_observation_noise_covariance_from_warmup_observations(
        normalized_observations,
        dt_hours,
    )
    return {
        "normalization_factor": normalization_factor,
        "normalized_warmup_observations": normalized_observations,
        "initial_state": initial_state,
        "initial_covariance": initial_covariance,
        "process_noise_covariance": process_noise_covariance,
        "observation_noise_covariance": observation_noise_covariance,
    }


def build_filter_from_observation_summary(
    summary: dict[str, object],
    *,
    outlier_std_threshold: float = 5.0,
    min_growth_rate: float = -1.0,
    max_growth_rate: float = 3.0,
) -> "CultureGrowthEKF":
    return CultureGrowthEKF(
        initial_state=summary["initial_state"],
        initial_covariance=summary["initial_covariance"],
        process_noise_covariance=summary["process_noise_covariance"],
        observation_noise_covariance=summary["observation_noise_covariance"],
        outlier_std_threshold=outlier_std_threshold,
        min_growth_rate=min_growth_rate,
        max_growth_rate=max_growth_rate,
    )


class CultureGrowthEKF:
    """
    Single-filter growth-rate estimator in log-OD space.

    The canonical state is `[log_od, growth_rate, growth_rate_drift]`.

    `state_[0]` is `log_od`, not `od`. That means an initial optical density of
    1.0 is represented as `log(1.0) = 0.0`, so a neutral initialization is
    `[0.0, 0.0, 0.0]`.

    Each update converts the sensor observations into an implied `log_od`
    measurement, fuses them into a single robust observation, and runs a linear
    Kalman filter step directly in that hidden space.

    Public API is the hidden state directly.
    """

    handle_outliers = True

    def __init__(
        self,
        initial_state: FloatVectorLike,
        initial_covariance: FloatMatrixLike,
        process_noise_covariance: FloatMatrixLike,
        observation_noise_covariance: FloatMatrixLike,
        outlier_std_threshold: float,
        min_growth_rate: float = -1.0,
        max_growth_rate: float = 3.0,
    ) -> None:
        """
        Parameters
        ----------
        initial_state
            Hidden-state initialization in the order
            `[log_od, growth_rate, growth_rate_drift]`.

            This is intentionally not `[od, growth_rate]`. For example, if the
            culture should start at `od=1.0`, pass `log_od=0.0`.
        initial_covariance
            3x3 covariance matrix for `[log_od, growth_rate, growth_rate_drift]`.
        process_noise_covariance
            3x3 process noise covariance in the same hidden-state basis.
        observation_noise_covariance
            Sensor covariance in observed OD space.
        """
        import numpy as np

        initial_state = np.asarray(initial_state, dtype=float)
        initial_covariance = np.asarray(initial_covariance, dtype=float)
        process_noise_covariance = np.asarray(process_noise_covariance, dtype=float)
        observation_noise_covariance = np.asarray(observation_noise_covariance, dtype=float)

        if initial_state.shape[0] != 3:
            raise ValueError(
                "initial_state must have shape (3,) and follow "
                "[log_od, growth_rate, growth_rate_drift]"
            )
        if (
            initial_state.shape[0] != initial_covariance.shape[0]
            or initial_covariance.shape[0] != initial_covariance.shape[1]
        ):
            raise ValueError(
                "initial_covariance must be square and match the size of initial_state"
            )
        if process_noise_covariance.shape != initial_covariance.shape:
            raise ValueError(
                "process_noise_covariance must have the same shape as initial_covariance"
            )
        if not _is_positive_definite(process_noise_covariance):
            raise ValueError("process_noise_covariance must be positive definite")
        if not _is_positive_definite(initial_covariance):
            raise ValueError("initial_covariance must be positive definite")
        if not _is_positive_definite(observation_noise_covariance):
            raise ValueError("observation_noise_covariance must be positive definite")
        if min_growth_rate > max_growth_rate:
            raise ValueError("min_growth_rate must be less than or equal to max_growth_rate")
        if not np.all(np.isfinite(initial_state)):
            raise ValueError("initial_state must contain only finite values")

        self.process_noise_covariance = process_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance
        self.state_ = initial_state
        self.min_growth_rate = float(min_growth_rate)
        self.max_growth_rate = float(max_growth_rate)
        self.state_[1] = self._clip_growth_rate(float(self.state_[1]))
        self.covariance_ = initial_covariance
        self.n_sensors = observation_noise_covariance.shape[0]
        self.n_states = initial_state.shape[0]
        self.outlier_std_threshold = outlier_std_threshold
        self._minimum_state_value = 1e-9
        self._base_log_observation_noise_variance = max(
            2e-5,
            3.0 * float(np.mean(np.diag(self.observation_noise_covariance)))
            / max(np.exp(float(self.state_[0])) ** 2, self._minimum_state_value),
        )
        self._log_observation_noise_variance = self._base_log_observation_noise_variance

    def _clip_growth_rate(self, growth_rate: float) -> float:
        return min(max(float(growth_rate), self.min_growth_rate), self.max_growth_rate)

    def _safe_log(self, value: float) -> float:
        return log(max(float(value), self._minimum_state_value))

    def _hidden_transition_matrix(self, dt: float) -> FloatMatrixLike:
        import numpy as np

        return np.array(
            [
                [1.0, dt, 0.5 * dt * dt],
                [0.0, 1.0, dt],
                [0.0, 0.0, 0.995],
            ],
            dtype=float,
        )

    def _process_noise_covariance_with_dilution_jump(
        self,
        recent_dilution: bool,
    ) -> FloatMatrixLike:
        import numpy as np

        process_covariance = self.process_noise_covariance.copy().astype(float)
        if recent_dilution:
            process_covariance[0, 0] += 0.05
            process_covariance[1, 1] += 1e-4
        return process_covariance

    def _od_measurement_and_variance_from_sensor(
        self, measurement: float, sensor_index: int
    ) -> tuple[float, float]:
        variance = max(
            float(self.observation_noise_covariance[sensor_index, sensor_index]),
            1e-12,
        )
        od_measurement = max(float(measurement), self._minimum_state_value)
        derivative = 1.0 / od_measurement
        return od_measurement, variance * derivative * derivative

    def _combine_log_measurements(self, observation: FloatVectorLike) -> tuple[float, float]:
        import numpy as np

        observation = np.asarray(observation, dtype=float)
        log_measurements = np.zeros(self.n_sensors, dtype=float)
        variances = np.zeros(self.n_sensors, dtype=float)
        for sensor_index, measurement in enumerate(observation):
            od_measurement, log_variance = self._od_measurement_and_variance_from_sensor(
                float(measurement), sensor_index
            )
            log_measurements[sensor_index] = self._safe_log(od_measurement)
            variances[sensor_index] = max(
                log_variance,
                self._base_log_observation_noise_variance,
            )

        weights = 1.0 / variances
        combined_variance = 1.0 / float(np.sum(weights))
        combined_measurement = float(np.sum(weights * log_measurements) * combined_variance)
        return combined_measurement, combined_variance

    def _update_observation_noise_covariance_from_log_variance(self) -> None:
        import numpy as np

        od = max(np.exp(float(self.state_[0])), self._minimum_state_value)
        updated_observation_noise = self.observation_noise_covariance.copy().astype(float)
        np.fill_diagonal(
            updated_observation_noise,
            self._log_observation_noise_variance * od * od,
        )
        self.observation_noise_covariance = updated_observation_noise

    def update(
        self,
        obs: FloatVectorLike,
        dt: float,
        recent_dilution: bool = False,
    ) -> tuple[FloatVectorLike, FloatMatrixLike]:
        import numpy as np

        observation = np.asarray(obs, dtype=float)
        if observation.shape[0] != self.n_sensors:
            raise ValueError(
                f"obs must contain one value per sensor: got {observation.shape[0]}, expected {self.n_sensors}"
            )

        combined_log_measurement, measurement_variance = self._combine_log_measurements(observation)
        transition = self._hidden_transition_matrix(dt)
        state_prediction = transition @ self.state_
        covariance_prediction = (
            transition @ self.covariance_ @ transition.T
            + self._process_noise_covariance_with_dilution_jump(recent_dilution=recent_dilution)
        )

        innovation = combined_log_measurement - float(state_prediction[0])
        residual_covariance = float(
            covariance_prediction[0, 0]
            + max(measurement_variance, self._log_observation_noise_variance)
        )
        standardized_innovation = innovation / sqrt(max(residual_covariance, 1e-12))
        currently_is_outlier = self.handle_outliers and (
            abs(standardized_innovation) > self.outlier_std_threshold
        )
        effective_measurement_variance = max(measurement_variance, self._log_observation_noise_variance)
        if currently_is_outlier:
            inflation = max(
                (abs(standardized_innovation) / max(self.outlier_std_threshold, 1e-12)) ** 2,
                1.0,
            )
            effective_measurement_variance *= inflation
            residual_covariance = float(covariance_prediction[0, 0] + effective_measurement_variance)

        kalman_gain = covariance_prediction[:, 0] / max(residual_covariance, 1e-12)
        self.state_ = state_prediction + kalman_gain * innovation
        self.covariance_ = covariance_prediction - np.outer(
            kalman_gain, covariance_prediction[0, :]
        )

        self.covariance_[0, 0] = max(float(self.covariance_[0, 0]), 1e-12)
        self.covariance_[1, 1] = max(float(self.covariance_[1, 1]), 1e-12)
        self.covariance_[2, 2] = max(float(self.covariance_[2, 2]), 1e-12)
        self.state_[1] = self._clip_growth_rate(float(self.state_[1]))

        if np.isnan(self.state_).any():
            raise ValueError("NaNs detected after calculation.")

        if (not currently_is_outlier) and (not recent_dilution):
            innovation_squared = min(innovation * innovation, 16.0 * residual_covariance)
            self._log_observation_noise_variance = max(
                self._base_log_observation_noise_variance,
                0.98 * self._log_observation_noise_variance + 0.02 * innovation_squared,
            )
            self._update_observation_noise_covariance_from_log_variance()

        return self.state_, self.covariance_
