"""Figures: colorblind-safe, units on every axis, data shown with the summary.

Ground rules section 8. Functions return a Matplotlib figure and never write files or
call ``plt.show()``; saving belongs in the notebook layer.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito, which stays distinguishable under the common colour-vision deficiencies.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
SEQUENTIAL = "viridis"


def apply_style() -> None:
    """Set the rcParams used by every figure in the notebook."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "image.cmap": SEQUENTIAL,
            "figure.autolayout": True,
        }
    )


def culture_trajectories(history: pd.DataFrame, title: str = "", feed_marks: bool = True):
    """Cells, nutrients, by-products and titer against time, with feed points marked."""
    required = {"time_d", "xv", "viability_percent", "glc", "gln", "lac", "amm", "titer"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing {sorted(missing)}")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    time = history["time_d"]

    axes[0, 0].plot(time, history["xv"], marker="o", label="viable")
    if "xd" in history:
        axes[0, 0].plot(time, history["xd"], marker="s", label="dead")
    axes[0, 0].set_ylabel("cell density [1e6 cells/mL]")
    axes[0, 0].legend(frameon=False)
    twin = axes[0, 0].twinx()
    twin.plot(time, history["viability_percent"], color=OKABE_ITO[2], linestyle="--")
    twin.set_ylabel("viability [%]", color=OKABE_ITO[2])
    twin.grid(False)

    axes[0, 1].plot(time, history["glc"], marker="o", label="glucose")
    axes[0, 1].plot(time, history["gln"], marker="s", label="glutamine")
    axes[0, 1].set_ylabel("nutrient [mM]")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(time, history["lac"], marker="o", label="lactate")
    axes[1, 0].plot(time, history["amm"], marker="s", label="ammonium")
    axes[1, 0].set_ylabel("by-product [mM]")
    axes[1, 0].set_xlabel("time [d]")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(time, history["titer"], marker="o", color=OKABE_ITO[1])
    axes[1, 1].set_ylabel("titer [mg/L]")
    axes[1, 1].set_xlabel("time [d]")

    if feed_marks and "feed_volume_added_L" in history:
        fed = history[history["feed_volume_added_L"] > 0]["time_d"]
        for axis in axes.ravel():
            for moment in fed:
                axis.axvline(moment, color="0.75", linewidth=0.7, zorder=0)
    if title:
        figure.suptitle(title)
    return figure


def campaign_overlay(samples: pd.DataFrame, column: str, unit: str, group_column: str = "feed_strategy"):
    """Every batch's trajectory, coloured by group: the data, not just the mean."""
    if column not in samples.columns:
        raise ValueError(f"{column!r} is not a column")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    groups = list(dict.fromkeys(samples[group_column]))
    colours = {group: OKABE_ITO[i % len(OKABE_ITO)] for i, group in enumerate(groups)}
    # Grouping by batch as well as by group keeps each run a separate line; without it
    # the trajectories join end to end into one zigzag.
    for keys, frame in samples.groupby(["batch_id", group_column], sort=False):
        axis.plot(frame["time_d"], frame[column], color=colours[keys[1]], alpha=0.45, linewidth=1.0)
    for group in groups:
        mean = samples[samples[group_column] == group].groupby("time_d")[column].mean()
        axis.plot(mean.index, mean.to_numpy(), color=colours[group], linewidth=2.5, label=f"{group} (mean)")
    axis.set_xlabel("time [d]")
    axis.set_ylabel(f"{column} [{unit}]")
    axis.legend(frameon=False, title=group_column.replace("_", " "))
    axis.set_title(f"{len(samples['batch_id'].unique())} batches, individual runs shown")
    return figure


def parity(observed, predicted, *, label: str, unit: str, scores: dict | None = None, groups=None):
    """Predicted against observed with the 1:1 line and the baseline RMSE in the corner."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    figure, axis = plt.subplots(figsize=(5.5, 5.2))
    if groups is None:
        axis.scatter(observed, predicted, s=38, alpha=0.8, edgecolor="white")
    else:
        groups = np.asarray(groups)
        for index, group in enumerate(dict.fromkeys(groups.tolist())):
            mask = groups == group
            axis.scatter(
                observed[mask],
                predicted[mask],
                s=38,
                alpha=0.85,
                color=OKABE_ITO[index % len(OKABE_ITO)],
                edgecolor="white",
                label=str(group),
            )
        if len(set(groups.tolist())) <= 8:
            axis.legend(frameon=False, fontsize=8)
    limits = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
    span = limits[1] - limits[0]
    limits = [limits[0] - 0.05 * span, limits[1] + 0.05 * span]
    axis.plot(limits, limits, color="0.4", linestyle="--", linewidth=1, zorder=0, label="1:1")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel(f"observed {label} [{unit}]")
    axis.set_ylabel(f"predicted {label} [{unit}]")
    if scores:
        text = (
            f"RMSE {scores['rmse']:.3g}\n"
            f"mean-predictor {scores['rmse_mean_baseline']:.3g}\n"
            f"skill {scores['skill_vs_mean']:+.2f}   n = {scores['n_test']}"
        )
        axis.text(
            0.03,
            0.97,
            text,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
    return figure


def breakthrough(curves: dict[str, pd.DataFrame], breakthrough_fraction: float = 0.10):
    """Breakthrough curves with the sizing breakthrough marked."""
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for index, (label, curve) in enumerate(curves.items()):
        axis.plot(
            curve["load_g_per_L_resin"],
            100 * curve["breakthrough_fraction"],
            color=OKABE_ITO[index % len(OKABE_ITO)],
            label=label,
        )
    axis.axhline(
        100 * breakthrough_fraction,
        color="0.4",
        linestyle="--",
        linewidth=1,
        label=f"{breakthrough_fraction:.0%} breakthrough",
    )
    axis.set_xlabel("load [g product / L resin]")
    axis.set_ylabel("breakthrough C/C0 [%]")
    axis.legend(frameon=False)
    return figure


def yield_waterfall(steps: pd.DataFrame):
    """Product mass surviving each downstream step, with cumulative yield labelled."""
    required = {"step", "product_g", "cumulative_yield_fraction"}
    missing = required - set(steps.columns)
    if missing:
        raise ValueError(f"steps are missing {sorted(missing)}")
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    positions = np.arange(len(steps))
    axis.bar(positions, steps["product_g"], color=OKABE_ITO[0], alpha=0.85)
    for position, (mass, cumulative) in enumerate(
        zip(steps["product_g"], steps["cumulative_yield_fraction"], strict=True)
    ):
        axis.text(position, mass, f"{mass:.1f} g\n{cumulative:.0%}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels(steps["step"], rotation=20, ha="right")
    axis.set_ylabel("product [g]")
    axis.set_ylim(0, steps["product_g"].max() * 1.25)
    axis.set_title("Product mass through the purification train")
    return figure


def sensitivity_bars(indices: pd.DataFrame, title: str = "Sobol sensitivity"):
    """First-order and total Sobol indices with their confidence intervals."""
    required = {"parameter", "S1", "ST"}
    missing = required - set(indices.columns)
    if missing:
        raise ValueError(f"indices are missing {sorted(missing)}")
    ordered = indices.sort_values("ST")
    positions = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(7.5, 0.45 * len(ordered) + 2.0))
    axis.barh(
        positions + 0.2,
        ordered["ST"],
        height=0.38,
        color=OKABE_ITO[0],
        label="total (ST)",
        xerr=ordered.get("ST_conf"),
        error_kw={"elinewidth": 1},
    )
    axis.barh(
        positions - 0.2,
        ordered["S1"],
        height=0.38,
        color=OKABE_ITO[4],
        label="first order (S1)",
        xerr=ordered.get("S1_conf"),
        error_kw={"elinewidth": 1},
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(ordered["parameter"])
    axis.set_xlabel("Sobol index [-]  (error bars: 95% bootstrap CI)")
    axis.set_title(title)
    axis.legend(frameon=False)
    return figure


def spectra_plot(wavenumbers, spectra, *, sample_count: int = 12, title: str = "Simulated Raman spectra"):
    """A sample of spectra, coloured by index so drift is visible."""
    spectra = np.atleast_2d(np.asarray(spectra, dtype=float))
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    step = max(1, len(spectra) // sample_count)
    shown = spectra[::step]
    colours = plt.get_cmap(SEQUENTIAL)(np.linspace(0, 1, len(shown)))
    for row, colour in zip(shown, colours, strict=True):
        axis.plot(wavenumbers, row, color=colour, linewidth=0.9)
    axis.set_xlabel("wavenumber [cm$^{-1}$]")
    axis.set_ylabel("intensity [arb. units]")
    axis.set_title(f"{title} ({len(shown)} of {len(spectra)} shown)")
    return figure


def save(figure, path, *, formats=("svg", "png")) -> list:
    """Save a figure as vector plus a raster preview (ground rules section 8)."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        target = path.with_suffix(f".{suffix}")
        figure.savefig(target, bbox_inches="tight")
        written.append(target)
    return written
