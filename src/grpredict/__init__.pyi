from collections.abc import Iterable
from typing import ClassVar, TypeAlias

FloatVectorLike: TypeAlias = Iterable[float]
FloatMatrixLike: TypeAlias = Iterable[Iterable[float]]


def estimate_normalization_factor_from_warmup_observations(
    observations: FloatVectorLike,
    *,
    minimum_value: float = 1e-9,
) -> float: ...
def normalize_observation_by_factor(
    observation: float,
    normalization_factor: float,
    *,
    minimum_value: float = 1e-9,
) -> float: ...
def normalize_observations_by_factor(
    observations: FloatVectorLike,
    normalization_factor: float,
    *,
    minimum_value: float = 1e-9,
) -> object: ...
def estimate_observation_noise_covariance_from_warmup_observations(
    normalized_observations: FloatVectorLike,
    dt_hours: float,
    *,
    minimum_value: float = 1e-9,
) -> object: ...
def estimate_initial_covariance_from_warmup_observations(
    normalized_observations: FloatVectorLike,
    dt_hours: float,
    *,
    minimum_value: float = 1e-9,
) -> object: ...
def make_process_noise_covariance(
    dt_hours: float,
    *,
    reference_dt_hours: float = 5.0 / 60.0 / 60.0,
) -> object: ...
def summarize_warmup_observations(
    observations: FloatVectorLike,
    dt_hours: float,
    *,
    minimum_value: float = 1e-9,
) -> dict[str, object]: ...
def build_filter_from_observation_summary(
    summary: dict[str, object],
    *,
    outlier_std_threshold: float = 5.0,
    min_growth_rate: float = -1.0,
    max_growth_rate: float = 3.0,
) -> "CultureGrowthEKF": ...


class CultureGrowthEKF:
    handle_outliers: ClassVar[bool]
    process_noise_covariance: object
    observation_noise_covariance: object
    state_: object
    covariance_: object
    n_sensors: int
    n_states: int
    outlier_std_threshold: float
    min_growth_rate: float
    max_growth_rate: float

    def __init__(
        self,
        initial_state: FloatVectorLike,
        initial_covariance: FloatMatrixLike,
        process_noise_covariance: FloatMatrixLike,
        observation_noise_covariance: FloatMatrixLike,
        outlier_std_threshold: float,
        min_growth_rate: float = -1.0,
        max_growth_rate: float = 3.0,
    ) -> None: ...
    def update(
        self, obs: FloatVectorLike, dt: float, recent_dilution: bool = False
    ) -> tuple[object, object]: ...
    @staticmethod
    def _is_positive_definite(A: FloatMatrixLike) -> bool: ...
