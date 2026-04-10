#!/Users/camerondavidson-pilon/code/grpredict/.venv/bin/python3.14
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig_grpredict"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.simulation_utils import plausible_growth_rate_profiles
from tests.simulation_utils import simulate_profiled_od_observations
from tests.test_simulated_profiles_with_ekf import make_single_sensor_ekf
from tests.test_simulated_profiles_with_ekf import run_ekf_over_observations

PLOTS_DIR = ROOT / "scratch" / "plots"
GROWTH_RATE_OUTPUT_PATH = PLOTS_DIR / "ekf_profile_noise_grid.png"
OD_OUTPUT_PATH = PLOTS_DIR / "ekf_profile_noise_od_grid.png"

DT_HOURS = 5.0 / 60.0 / 60.0
SEED = 321
PROFILE_LAYOUT: list[tuple[str, float, str]] = [
    ("lag_log_stationary", 12.0, "lag_log_stationary"),
    ("constant_growth", 12.0, "constant"),
    ("washout_recovery", 12.0, "washout_recovery."),
]
NOISE_FAMILIES: list[tuple[str, str]] = [
    ("nominal_near_iid", "Nominal near-iid"),
    ("nominal_colored", "Nominal colored"),
    ("noisy_colored", "Noisy colored"),
]


def build_panel_data(profile_name: str, total_hours: float, noise_family: str) -> dict[str, np.ndarray]:
    growth_rates = plausible_growth_rate_profiles(total_hours, DT_HOURS)[profile_name]
    simulated = simulate_profiled_od_observations(
        growth_rates,
        profile_name=noise_family,
        dt_hours=DT_HOURS,
        seed=SEED,
    )
    estimated_rates = run_ekf_over_observations(
        simulated["observed_od"],
        DT_HOURS,
        noise_family,
    )
    ekf = make_single_sensor_ekf(noise_family)
    estimated_od = np.empty_like(simulated["observed_od"])
    estimated_od[0] = float(np.exp(ekf.state_[0]))
    for index, observation in enumerate(simulated["observed_od"][1:], start=1):
        state, _ = ekf.update([float(observation)], DT_HOURS)
        estimated_od[index] = float(np.exp(state[0]))
    return {
        "time_hours": simulated["time_hours"],
        "rate_time_hours": simulated["time_hours"][1:],
        "true_rates": simulated["growth_rates"],
        "estimated_rates": estimated_rates,
        "latent_od": simulated["latent_od"],
        "observed_od": simulated["observed_od"],
        "estimated_od": estimated_od,
    }


def collect_panels() -> dict[tuple[int, int], dict[str, np.ndarray]]:
    panels: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for row_index, (profile_name, total_hours, _) in enumerate(PROFILE_LAYOUT):
        for col_index, (noise_family, _) in enumerate(NOISE_FAMILIES):
            panels[(row_index, col_index)] = build_panel_data(profile_name, total_hours, noise_family)
    return panels


def render_growth_rate_grid(panels: dict[tuple[int, int], dict[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=False, sharey=True, constrained_layout=False)
    max_abs_rate = 0.0
    for panel in panels.values():
        panel_max = float(
            np.max(np.abs(np.concatenate([panel["true_rates"], panel["estimated_rates"]])))
        )
        max_abs_rate = max(max_abs_rate, panel_max)
    y_limit = max(0.30, np.ceil(max_abs_rate * 20.0) / 20.0)
    for row_index, (_, _, profile_label) in enumerate(PROFILE_LAYOUT):
        for col_index, (_, noise_label) in enumerate(NOISE_FAMILIES):
            axis = axes[row_index, col_index]
            panel = panels[(row_index, col_index)]

            axis.plot(
                panel["rate_time_hours"],
                panel["true_rates"],
                color="#1f77b4",
                linewidth=2.2,
                label="True growth rate",
            )
            axis.plot(
                panel["rate_time_hours"],
                panel["estimated_rates"],
                color="#d62728",
                linewidth=1.5,
                alpha=0.9,
                label="EKF estimate",
            )
            axis.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.5)
            axis.set_ylim(-0.08 if y_limit > 0.08 else -y_limit, y_limit)
            axis.grid(alpha=0.18)

            if row_index == 0:
                axis.set_title(noise_label)
            if col_index == 0:
                axis.set_ylabel(f"{profile_label}\nGrowth rate (1/h)")
            if row_index == len(PROFILE_LAYOUT) - 1:
                axis.set_xlabel("Time (hours)")

            rmse = float(np.sqrt(np.mean((panel["estimated_rates"] - panel["true_rates"]) ** 2)))
            axis.text(
                0.02,
                0.96,
                f"RMSE {rmse:.3f}",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Simulated Growth Rate vs EKF Estimate Across Profiles and Noise Families\n"
        f"dt={DT_HOURS * 3600:.0f}s, seed={SEED}",
        fontsize=14,
        y=0.972,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.935))
    figure.savefig(GROWTH_RATE_OUTPUT_PATH, dpi=180)
    plt.close(figure)


def render_observed_od_grid(panels: dict[tuple[int, int], dict[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=False, sharey=False, constrained_layout=False)
    for row_index, (_, _, profile_label) in enumerate(PROFILE_LAYOUT):
        for col_index, (_, noise_label) in enumerate(NOISE_FAMILIES):
            axis = axes[row_index, col_index]
            panel = panels[(row_index, col_index)]

            axis.plot(
                panel["time_hours"],
                panel["latent_od"],
                color="#1f77b4",
                linewidth=2.0,
                label="Latent OD",
            )
            axis.plot(
                panel["time_hours"],
                panel["observed_od"],
                color="#2ca02c",
                linewidth=0.9,
                alpha=0.65,
                label="Observed OD",
            )
            axis.plot(
                panel["time_hours"],
                panel["estimated_od"],
                color="#d62728",
                linewidth=2.6,
                alpha=1.0,
                label="KF OD",
                zorder=4,
            )
            axis.scatter(
                panel["time_hours"],
                panel["observed_od"],
                color="#2ca02c",
                s=4,
                alpha=0.22,
                zorder=2,
            )
            axis.grid(alpha=0.18)

            if row_index == 0:
                axis.set_title(noise_label)
            if col_index == 0:
                axis.set_ylabel(f"{profile_label}\nOD")
            if row_index == len(PROFILE_LAYOUT) - 1:
                axis.set_xlabel("Time (hours)")

            residual_std = float(np.std(panel["observed_od"] - panel["latent_od"]))
            axis.text(
                0.02,
                0.96,
                f"resid sd {residual_std:.3f}",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Simulated Latent OD, Observed OD, and KF OD Across Profiles and Noise Families\n"
        f"dt={DT_HOURS * 3600:.0f}s, seed={SEED}",
        fontsize=14,
        y=0.972,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.935))
    figure.savefig(OD_OUTPUT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    panels = collect_panels()
    render_growth_rate_grid(panels)
    render_observed_od_grid(panels)
    print(GROWTH_RATE_OUTPUT_PATH)
    print(OD_OUTPUT_PATH)


if __name__ == "__main__":
    main()
