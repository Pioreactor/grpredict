from collections.abc import Iterable, Sequence
from typing import ClassVar, TypeAlias

FloatVectorLike: TypeAlias = Iterable[float]
FloatMatrixLike: TypeAlias = Iterable[Iterable[float]]


class ExponentialMovingAverage:
    value: float | None
    alpha: float
    def __init__(self, alpha: float) -> None: ...
    def update(self, new_value: float) -> float: ...
    def get_latest(self) -> float: ...
    def clear(self) -> None: ...


class ExponentialMovingStd:
    value: float | None
    alpha: float
    ema: ExponentialMovingAverage
    def __init__(
        self,
        alpha: float,
        ema_alpha: float | None = None,
        initial_std_value: float | None = None,
        initial_mean_value: float | None = None,
    ) -> None: ...
    def update(self, new_value: float) -> float | None: ...
    def get_latest(self) -> float: ...
    def clear(self) -> None: ...


class CultureGrowthEKF:
    handle_outliers: ClassVar[bool]
    process_noise_covariance: object
    observation_noise_covariance: object
    state_: object
    covariance_: object
    n_sensors: int
    n_states: int
    angles: Sequence[str]
    outlier_std_threshold: float

    def __init__(
        self,
        initial_state: FloatVectorLike,
        initial_covariance: FloatMatrixLike,
        process_noise_covariance: FloatMatrixLike,
        observation_noise_covariance: FloatMatrixLike,
        angles: Sequence[str],
        outlier_std_threshold: float,
    ) -> None: ...
    def update(
        self, obs: FloatVectorLike, dt: float, recent_dilution: bool = False
    ) -> tuple[object, object]: ...
    def update_observation_noise_cov(self, residual_state: FloatVectorLike) -> object: ...
    def update_state_from_previous_state(self, state: FloatVectorLike, dt: float) -> object: ...
    def _J_update_observations_from_state(self, state_prediction: FloatVectorLike) -> object: ...
    def update_covariance_from_old_covariance(
        self,
        state: FloatVectorLike,
        covariance: FloatMatrixLike,
        dt: float,
        recent_dilution: bool,
    ) -> object: ...
    def update_observations_from_state(self, state_predictions: FloatVectorLike) -> object: ...
    def _J_update_state_from_previous_state(self, state: FloatVectorLike, dt: float) -> object: ...
    @staticmethod
    def _is_positive_definite(A: FloatMatrixLike) -> bool: ...
