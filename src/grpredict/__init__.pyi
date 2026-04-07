from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatVectorLike: TypeAlias = Iterable[float]
FloatMatrixLike: TypeAlias = Iterable[Iterable[float]]
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class WarmupObservationSummary:
    normalization_factors: FloatArray
    normalized_warmup_observations: FloatArray
    initial_state: FloatArray
    initial_covariance: FloatArray
    process_noise_covariance: FloatArray
    observation_noise_covariance: FloatArray


def estimate_normalization_factor_from_warmup_observations(
    observations: FloatVectorLike,
) -> FloatArray: ...
def normalize_observations_by_factor(
    observations: FloatVectorLike,
    normalization_factor: FloatVectorLike,
) -> FloatArray: ...
def estimate_observation_noise_covariance_from_warmup_observations(
    normalized_observations: FloatVectorLike,
    dt_hours: float,
) -> FloatArray: ...
def estimate_initial_covariance_from_warmup_observations(
    normalized_observations: FloatVectorLike,
    dt_hours: float,
) -> FloatArray: ...
def make_process_noise_covariance(
    dt_hours: float,
    *,
    reference_dt_hours: float = 5.0 / 60.0 / 60.0,
) -> FloatArray: ...
def summarize_warmup_observations(
    observations: FloatVectorLike,
    dt_hours: float,
) -> WarmupObservationSummary: ...
def build_filter_from_observation_summary(
    summary: WarmupObservationSummary,
    *,
    outlier_std_threshold: float = 5.0,
    min_growth_rate: float = -1.0,
    max_growth_rate: float = 3.0,
) -> "CultureGrowthEKF": ...
def _is_positive_definite(A: FloatMatrixLike) -> bool: ...


class CultureGrowthEKF:
    """State order is `[log_od, growth_rate, growth_rate_drift]`."""

    handle_outliers: ClassVar[bool]
    process_noise_covariance: FloatArray
    observation_noise_covariance: FloatArray
    state_: FloatArray
    covariance_: FloatArray
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
    ) -> tuple[FloatArray, FloatArray]: ...
