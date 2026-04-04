from collections.abc import Iterable
from typing import ClassVar, TypeAlias

FloatVectorLike: TypeAlias = Iterable[float]
FloatMatrixLike: TypeAlias = Iterable[Iterable[float]]


class CultureGrowthEKF:
    handle_outliers: ClassVar[bool]
    process_noise_covariance: object
    observation_noise_covariance: object
    state_: object
    covariance_: object
    n_sensors: int
    n_states: int
    outlier_std_threshold: float

    def __init__(
        self,
        initial_state: FloatVectorLike,
        initial_covariance: FloatMatrixLike,
        process_noise_covariance: FloatMatrixLike,
        observation_noise_covariance: FloatMatrixLike,
        outlier_std_threshold: float,
    ) -> None: ...
    def update(
        self, obs: FloatVectorLike, dt: float, recent_dilution: bool = False
    ) -> tuple[object, object]: ...
    @staticmethod
    def _is_positive_definite(A: FloatMatrixLike) -> bool: ...
