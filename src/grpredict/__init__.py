# -*- coding: utf-8 -*-
from __future__ import annotations
from collections.abc import Sequence
from math import log
from math import sqrt
from typing import Any
from typing import Optional
from typing import TypeAlias

FloatVectorLike: TypeAlias = Any
FloatMatrixLike: TypeAlias = Any

class ExponentialMovingAverage:
    """
    Models the following:

    mean_n = (1 - alpha)·x + alpha·mean_{n-1}

    Ex: if alpha = 0, use latest value only.
    """

    def __init__(self, alpha: float) -> None:
        if alpha < 0 or alpha > 1:
            raise ValueError
        self.value: Optional[float] = None
        self.alpha = alpha

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = (1 - self.alpha) * new_value + self.alpha * self.value
        return self.value

    def get_latest(self) -> float:
        if self.value is None:
            raise ValueError("No values provided yet!")
        assert isinstance(self.value, float)
        return self.value

    def clear(self) -> None:
        self.value = None


class ExponentialMovingStd:
    """
    Models the following:

    var_n = (1 - alpha)·(x - mean_n)(x - mean_{n-1}) + alpha·var_{n-1}
    std_n = sqrt(var_n)

    Ex: if alpha = 0, use latest value only.
    """

    def __init__(
        self,
        alpha: float,
        ema_alpha: Optional[float] = None,
        initial_std_value: Optional[float] = None,
        initial_mean_value: Optional[float] = None,
    ) -> None:
        self._var_value = initial_std_value**2 if initial_std_value is not None else None
        self.value: Optional[float] = initial_std_value if initial_std_value is not None else None
        self.alpha = alpha
        self.ema = ExponentialMovingAverage(ema_alpha or self.alpha)
        if initial_mean_value is not None:
            self.ema.update(initial_mean_value)

    def update(self, new_value: float) -> Optional[float]:
        if self.ema.value is None:
            # need at least two data points for this algo
            self.ema.update(new_value)
            return None  # None

        mean_prev = self.ema.get_latest()
        self.ema.update(new_value)
        mean_curr = self.ema.get_latest()
        assert mean_prev is not None
        assert mean_curr is not None

        if self._var_value is None:
            self._var_value = (new_value - mean_curr) * (new_value - mean_prev)
        else:
            self._var_value = (1 - self.alpha) * (new_value - mean_curr) * (
                new_value - mean_prev
            ) + self.alpha * self._var_value
        self.value = sqrt(self._var_value)
        return self.value

    def get_latest(self) -> float:
        if self.value is None:
            raise ValueError("No values provided yet!")
        assert isinstance(self.value, float)
        return self.value

    def clear(self) -> None:
        self.value = None
        self._var_value = None
        self.ema.clear()


class CultureGrowthEKF:
    """
    Modified from the algorithm in
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181923#pone.0181923.s007

    The idea is that each sensor will evolve as follows, using notation from the wikipedia page

    ```
    m_{1,t}, m_{2,t} are our sensor measurements (could be more, example here is for two sensors)
    Note that we normalized the raw sensor measurements by their initial values, so initially they
    "hover" around 1.0.

    m_{1,t} = g1(OD_{t-1}) + σ0B*noise   # noise here includes temperature noise, EM noise, stirring noise
    m_{2,t} = g2(OD_{t-1}) + σ0A*noise

    OD_t = OD_{t-1} * exp(r_{t-1} * Δt) + σ1*noise
    r_t = r_{t-1} + σ2*noise         => r_t = sum(sum(σ2*noise_i)

    # g1 and g2 are generic functions. Often they are the identity functions in OD,
    # but if using a 180deg sensor then it would be the inverse function, like exp(-OD)
    # they could also be functions that model saturation.

    Let X = [OD, r]

    f([OD, r], Δt) = [OD * exp(r Δt), r]
    h([OD, r], Δt) = [g1(OD), g2(OD)]   # recall: this is a function of the number of sensor, here we are using two sensors.

    jac(f) = [
        [exp(r Δt),  OD * exp(r Δt) * Δt, ],
        [0,          1,                   ],
    ]

    jac(h) = [
        [1, 0],  # because d(identity)/dOD = 1, d(identity)/dr = 0,
        [1, 0],
        ...
    ]

    ```

    Example
    ---------

        initial_state = np.array([obs.iloc[0], 0.0])
        initial_covariance = np.eye(2)
        process_noise_covariance = np.array([[0.00001, 0], [0, 1e-13]])
        observation_noise_covariance = np.array([[0.2]])
        ekf = CultureGrowthEKF(initial_state, initial_covariance, process_noise_covariance, observation_noise_covariance)

        ekf.update(...)
        ekf.state_


    Scaling
    ---------
    1. Obs. covariance matrix is dynamic, as the sensor noise scales with OD increasing.


    Note on 180°
    -------------
    The measurement model for 180 is obs_t = exp(-(od_t - 1)), which comes from the beer lambert model:

      T = T_0 / 10**A

    T_0, in our model, is equal to the initial average signal from growth_rate_calculating,

      T_t = 10**{A_0} / 10**A_t = 10**{-(A_t - A_0)}

    Absorbance, A, is proportional to the optical density (in a certain range)

      T_t = 10**{-k(od_t - od_0)}

    10 is silly, so we use e.

      T_t = exp{-k(od_t - od_0)}

    The factor of of k just scales the deviations from the blank, and this can be incorporated into the Kalman Filter parameters

      T_t = exp{-(od_t - od_0)}

    Note the transformation that is often used with transmission:

      -log(T_t) = od_t - od_0

      Note: in our model, od_0 = 1.



    Useful Resources
    -------------------
    - https://dsp.stackexchange.com/questions/2347/how-to-understand-kalman-gain-intuitively
     > R is reflects in noise in the sensors, Q reflects how confident we are in the current state

    - https://perso.crans.org/club-krobot/doc/kalman.pdf
     > Another way of thinking about the weighting by K (Kalman Gain) is that as the measurement error covariance R approaches zero, the actual measurement, z, is “trusted” more and more,
     while the predicted measurement is trusted less and less. On the other hand, as the a priori estimate error covariance, Q, approaches zero the actual measurement, z,
     is trusted less and less, while the predicted measurement is trusted more and more
    """

    handle_outliers = True

    def __init__(
        self,
        initial_state: FloatVectorLike,
        initial_covariance: FloatMatrixLike,
        process_noise_covariance: FloatMatrixLike,
        observation_noise_covariance: FloatMatrixLike,
        angles: Sequence[str],
        outlier_std_threshold: float,
    ) -> None:
        import numpy as np

        initial_state = np.asarray(initial_state, dtype=float)
        initial_covariance = np.asarray(initial_covariance, dtype=float)
        process_noise_covariance = np.asarray(process_noise_covariance, dtype=float)
        observation_noise_covariance = np.asarray(observation_noise_covariance, dtype=float)

        assert initial_state.shape[0] == 2
        assert (
            initial_state.shape[0] == initial_covariance.shape[0] == initial_covariance.shape[1]
        ), f"Shapes are not correct,{initial_state.shape[0]=}, {initial_covariance.shape[0]=}, {initial_covariance.shape[1]=}"
        assert process_noise_covariance.shape == initial_covariance.shape
        assert self._is_positive_definite(process_noise_covariance)
        assert self._is_positive_definite(initial_covariance)
        assert self._is_positive_definite(observation_noise_covariance)

        self.process_noise_covariance = process_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance
        self.state_ = initial_state
        self.covariance_ = initial_covariance
        self.n_sensors = observation_noise_covariance.shape[0]
        self.n_states = initial_state.shape[0]
        self.angles = angles
        self.outlier_std_threshold = outlier_std_threshold
        self._sigma2_gr_baseline = self.process_noise_covariance[1, 1]
        self.ems = ExponentialMovingStd(
            0.975, 0.80, initial_std_value=np.sqrt(observation_noise_covariance[0][0])
        )
        self._minimum_state_value = 1e-9
        self._baseline_state_ = self.state_.copy()
        self._baseline_covariance_ = self.covariance_.copy()
        self._log_state_ = np.array(
            [self._safe_log(self.state_[0]), float(self.state_[1]), 0.0], dtype=float
        )
        self._log_covariance_ = self._public_covariance_to_hidden_covariance(
            self.state_, self.covariance_
        )
        self._base_log_observation_noise_variance = max(
            2e-5,
            3.0 * float(np.mean(np.diag(self.observation_noise_covariance)))
            / max(float(self.state_[0]) ** 2, self._minimum_state_value),
        )
        self._rate_blend_weight = 1.0
        self._last_public_state_ = self.state_.copy()
        self._last_public_covariance_ = self.covariance_.copy()

    def _safe_log(self, value: float) -> float:
        return log(max(float(value), self._minimum_state_value))

    def _public_covariance_to_hidden_covariance(
        self,
        public_state: FloatVectorLike,
        public_covariance: FloatMatrixLike,
    ) -> FloatMatrixLike:
        import numpy as np

        od, rate = np.asarray(public_state, dtype=float)
        covariance = np.asarray(public_covariance, dtype=float)
        od = max(float(od), self._minimum_state_value)

        hidden_covariance = np.zeros((3, 3), dtype=float)
        hidden_covariance[0, 0] = max(float(covariance[0, 0]) / (od * od), 1e-12)
        hidden_covariance[1, 1] = max(float(covariance[1, 1]), 1e-12)
        hidden_covariance[0, 1] = float(covariance[0, 1]) / od
        hidden_covariance[1, 0] = hidden_covariance[0, 1]
        hidden_covariance[2, 2] = max(hidden_covariance[1, 1], 1e-4)
        return hidden_covariance

    def _sync_hidden_state_if_public_attributes_changed(self) -> None:
        import numpy as np

        if (not np.allclose(self.state_, self._last_public_state_)) or (
            not np.allclose(self.covariance_, self._last_public_covariance_)
        ):
            self._baseline_state_ = np.asarray(self.state_, dtype=float).copy()
            self._baseline_covariance_ = np.asarray(self.covariance_, dtype=float).copy()
            self._log_state_ = np.array(
                [self._safe_log(self.state_[0]), float(self.state_[1]), 0.0], dtype=float
            )
            self._log_covariance_ = self._public_covariance_to_hidden_covariance(
                self.state_, self.covariance_
            )

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

    def _od_measurement_and_variance_from_sensor(
        self, measurement: float, sensor_index: int
    ) -> tuple[float, float]:
        variance = max(
            float(self.observation_noise_covariance[sensor_index, sensor_index]),
            1e-12,
        )
        if self.angles[sensor_index] == "180":
            transmission = max(float(measurement), self._minimum_state_value)
            od_measurement = max(1.0 - log(transmission), self._minimum_state_value)
            derivative = 1.0 / max(od_measurement * transmission, self._minimum_state_value)
            return od_measurement, variance * derivative * derivative

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

    def update(
        self,
        obs: FloatVectorLike,
        dt: float,
        recent_dilution: bool = False,
    ) -> tuple[FloatVectorLike, FloatMatrixLike]:
        import numpy as np

        self._sync_hidden_state_if_public_attributes_changed()

        observation = np.asarray(obs, dtype=float)
        assert observation.shape[0] == self.n_sensors, (observation, self.n_sensors)

        state_prediction = self.update_state_from_previous_state(self._baseline_state_, dt)
        covariance_prediction = self.update_covariance_from_old_covariance(
            self._baseline_state_,
            self._baseline_covariance_,
            dt,
            recent_dilution=recent_dilution,
        )

        residual_state = observation - self.update_observations_from_state(state_prediction)
        H = self._J_update_observations_from_state(state_prediction)

        huber_threshold = self.outlier_std_threshold * (self.ems.value or 10_000)
        currently_is_outlier = abs(residual_state[0]) > huber_threshold
        if recent_dilution:
            covariance_prediction[0, 1] = 0.0
            covariance_prediction[1, 0] = 0.0
        elif self.handle_outliers and currently_is_outlier:
            covariance_prediction[0, 0] = 2 * covariance_prediction[0, 0]

        self.ems.update(residual_state[0])

        if (not currently_is_outlier) and (not recent_dilution):
            self.observation_noise_covariance = self.update_observation_noise_cov(residual_state)

        residual_covariance = H @ covariance_prediction @ H.T + self.observation_noise_covariance
        kalman_gain_ = np.linalg.solve(residual_covariance.T, (H @ covariance_prediction.T)).T

        if self.handle_outliers and currently_is_outlier:
            kalman_gain_[0, 0] *= huber_threshold / max(abs(residual_state[0]), 1e-20)
            kalman_gain_[1:, 0] = 0

        baseline_state = state_prediction + kalman_gain_ @ residual_state
        baseline_covariance = (np.eye(self.n_states) - kalman_gain_ @ H) @ covariance_prediction

        combined_log_measurement, measurement_variance = self._combine_log_measurements(observation)
        hidden_prediction = self._hidden_transition_matrix(dt) @ self._log_state_
        hidden_covariance_prediction = (
            self._hidden_transition_matrix(dt) @ self._log_covariance_ @ self._hidden_transition_matrix(dt).T
            + np.diag([1e-8, 1e-7, 1e-5])
        )
        secondary_innovation = combined_log_measurement - float(hidden_prediction[0])
        secondary_innovation_variance = float(
            hidden_covariance_prediction[0, 0]
            + max(measurement_variance, self._base_log_observation_noise_variance)
        )
        secondary_gain = hidden_covariance_prediction[:, 0] / max(
            secondary_innovation_variance, 1e-12
        )
        self._log_state_ = hidden_prediction + secondary_gain * secondary_innovation
        self._log_covariance_ = hidden_covariance_prediction - np.outer(
            secondary_gain, hidden_covariance_prediction[0, :]
        )
        self._log_covariance_[0, 0] = max(float(self._log_covariance_[0, 0]), 1e-12)
        self._log_covariance_[1, 1] = max(float(self._log_covariance_[1, 1]), 1e-12)
        self._log_covariance_[2, 2] = max(float(self._log_covariance_[2, 2]), 1e-12)
        secondary_rate = float(self._log_state_[1])

        blended_rate = (1.0 - self._rate_blend_weight) * float(baseline_state[1]) + (
            self._rate_blend_weight * secondary_rate
        )
        self._baseline_state_ = baseline_state
        self._baseline_covariance_ = baseline_covariance
        self.state_ = baseline_state.copy()
        self.state_[1] = blended_rate
        self.covariance_ = baseline_covariance

        if np.isnan(self.state_).any():
            raise ValueError("NaNs detected after calculation.")

        nis = (residual_state[0] ** 2) / residual_covariance[0, 0]
        if nis > 9.0:
            self.process_noise_covariance[1, 1] *= 2
        else:
            factor = 0.95
            baseline = self._sigma2_gr_baseline
            self.process_noise_covariance[1, 1] = (
                1 - factor
            ) * baseline + factor * self.process_noise_covariance[1, 1]

        self._last_public_state_ = self.state_.copy()
        self._last_public_covariance_ = self.covariance_.copy()
        return self.state_, self.covariance_

    def update_observation_noise_cov(self, residual_state: FloatVectorLike) -> FloatMatrixLike:
        """
        Exponentially-weighted measurement noise covariance.
        """
        import numpy as np

        lambda_ = 0.97  # controls the “memory” of the estimator. 0.97 means the filter effectively averages the last ≈ 1 / (1-0.97) ≈ 33 timesteps.
        residual_state = np.asarray(residual_state, dtype=float)
        rrT = np.outer(residual_state, residual_state)

        observation_noise_covariance = lambda_ * self.observation_noise_covariance + (1.0 - lambda_) * rrT

        # keep a floor on the diagonal to avoid numerical issues
        diag = np.diag_indices_from(observation_noise_covariance)
        observation_noise_covariance[diag] = np.maximum(observation_noise_covariance[diag], 1e-10)

        return observation_noise_covariance

    def update_state_from_previous_state(self, state: FloatVectorLike, dt: float) -> FloatVectorLike:
        """
        Denoted "f" in literature, x_{k} = f(x_{k-1})

        state = [OD, r]

        OD_t = OD_{t-1}·exp(r_{t-1}·Δt)
        r_t  = r_{t-1}

        """
        import numpy as np

        od, rate = np.asarray(state, dtype=float)
        return np.array([od * np.exp(rate * dt), rate])

    def _J_update_observations_from_state(self, state_prediction: FloatVectorLike) -> FloatMatrixLike:
        """
        Jacobian of observations model, encoded as update_observations_from_state

        measurement model is:

        m_{1,t} = g1(OD_{t-1})
        m_{2,t} = g2(OD_{t-1})
        ...

        gi are generic functions. Often they are the identity function, but if using a 180deg sensor
        then it would be the inverse function. One day it could model saturation, too.

        jac(h) = [
            [1, 0, 0],
            [1, 0, 0],
            ...
        ]

        """
        from numpy import exp
        from numpy import zeros

        od = state_prediction[0]
        J = zeros((self.n_sensors, 2))
        for i in range(self.n_sensors):
            angle = self.angles[i]
            J[i, 0] = 1.0 if (angle != "180") else -exp(-(od - 1))
        return J

    def update_covariance_from_old_covariance(
        self,
        state: FloatVectorLike,
        covariance: FloatMatrixLike,
        dt: float,
        recent_dilution: bool,
    ) -> FloatMatrixLike:
        import numpy as np

        state = np.asarray(state, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        Q = self.process_noise_covariance.copy().astype(float)

        if recent_dilution:
            jump_var = max(1e-12, 0.25 * state[0] ** 2)  # keep strictly >0 for Cholesky safety
            Q[0, 0] += jump_var

        jacobian = self._J_update_state_from_previous_state(state, dt)
        return jacobian @ covariance @ jacobian.T + Q

    def update_observations_from_state(self, state_predictions: FloatVectorLike) -> FloatVectorLike:
        """
        "h" in the literature, z_k = h(x_k).

        Return shape is (n_sensors,)
        """
        import numpy as np

        obs = np.zeros((self.n_sensors,))
        od = state_predictions[0]

        for i in range(self.n_sensors):
            angle = self.angles[i]
            obs[i] = od if (angle != "180") else np.exp(-(od - 1))
        return obs

    def _J_update_state_from_previous_state(self, state: FloatVectorLike, dt: float) -> FloatMatrixLike:
        """
        The prediction process is (encoded in update_state_from_previous_state)

            state = [OD, r, a]

            OD_t = OD_{t-1} * exp(r_{t-1} * Δt)
            r_t = r_{t-1}

        So jacobian should look like:

        [
            [exp(r Δt),  OD * exp(r Δt) * Δt, ],
            [0,          1,                   ],
        ]


        """
        import numpy as np

        J = np.zeros((2, 2))

        od, rate = np.asarray(state, dtype=float)
        J[0, 0] = np.exp(rate * dt)
        J[1, 1] = 1

        J[0, 1] = od * np.exp(rate * dt) * dt

        return J

    @staticmethod
    def _is_positive_definite(A: FloatMatrixLike) -> bool:
        import numpy as np

        A = np.asarray(A, dtype=float)
        if np.array_equal(A, A.T):
            try:
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False
