"""Downstream purification: depth filtration, Protein A capture, polishing, UF/DF.

Units are carried in the names: concentrations ``g/L``, volumes ``L``, column volumes
``CV``, times ``min``, flow ``cm/h`` or ``CV/h``, areas ``cm2``, lengths ``cm``.

The capture step uses the lumped kinetic model (LKM) in the form used for breakthrough
prediction by Narayanan et al. (2021), doi:10.1016/j.chroma.2021.462248: a column of
equal-sized tanks in series, Langmuir equilibrium, a lumped mass-transfer coefficient.
Tanks-in-series is how the axial dispersion is represented, so the tank count is a
model parameter (``Pe/2``-like), not a numerical refinement knob: the notebook's
verification section shows the result converging in time step at a fixed tank count,
and treats the tank count as a fitted/physical parameter instead.

Polishing (CEX/AEX) is modelled at the level a process-development screen needs: a
yield and a log-reduction per impurity class, from a step's operating point. That is a
correlation, not a mechanism, and it is labelled as such wherever it is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Impurity classes tracked end to end. Values are concentrations in g/L except HCP and
# DNA, which are reported per gram of product as the assays report them.
IMPURITY_UNITS = {
    "hcp_ng_per_mg": "ng/mg",
    "dna_pg_per_mg": "pg/mg",
    "aggregate_percent": "%",
    "leached_protein_a_ng_per_mg": "ng/mg",
}


@dataclass(frozen=True)
class Column:
    """Packed-bed geometry and packing properties."""

    diameter_cm: float
    bed_height_cm: float
    porosity_total: float = 0.85  # interstitial + intraparticle, typical for Protein A
    porosity_interstitial: float = 0.35

    def __post_init__(self) -> None:
        if min(self.diameter_cm, self.bed_height_cm) <= 0:
            raise ValueError("column dimensions must be > 0")
        if not 0 < self.porosity_interstitial < self.porosity_total < 1:
            raise ValueError("require 0 < interstitial < total porosity < 1")

    @property
    def area_cm2(self) -> float:
        return np.pi * (self.diameter_cm / 2.0) ** 2

    @property
    def volume_L(self) -> float:
        """One column volume (CV), in litres."""
        return self.area_cm2 * self.bed_height_cm / 1000.0

    def residence_time_min(self, velocity_cm_per_h: float) -> float:
        """Bed residence time ``L/u`` in minutes (the usual way a capture step is set)."""
        if velocity_cm_per_h <= 0:
            raise ValueError("velocity must be > 0 cm/h")
        return 60.0 * self.bed_height_cm / velocity_cm_per_h


@dataclass(frozen=True)
class LangmuirResin:
    """Equilibrium and mass transfer for one resin/product pair.

    ``q_max_g_per_L_resin`` is capacity on the solid phase, ``k_eq_L_per_g`` the
    Langmuir constant, ``k_mass_transfer_per_min`` the lumped film+pore coefficient.
    """

    q_max_g_per_L_resin: float
    k_eq_L_per_g: float
    k_mass_transfer_per_min: float
    name: str = "Protein A"

    def __post_init__(self) -> None:
        if min(self.q_max_g_per_L_resin, self.k_eq_L_per_g, self.k_mass_transfer_per_min) <= 0:
            raise ValueError("resin parameters must be > 0")

    def equilibrium_q(self, concentration_g_per_L):
        """Solid-phase load in equilibrium with a liquid concentration (g/L resin)."""
        c = np.maximum(np.asarray(concentration_g_per_L, dtype=float), 0.0)
        return self.q_max_g_per_L_resin * self.k_eq_L_per_g * c / (1.0 + self.k_eq_L_per_g * c)


def breakthrough_curve(
    column: Column,
    resin: LangmuirResin,
    feed_g_per_L: float,
    velocity_cm_per_h: float,
    *,
    load_cv: float = 60.0,
    n_tanks: int = 30,
    points: int = 400,
) -> pd.DataFrame:
    """Simulate loading and return the breakthrough curve.

    Returns a frame indexed by load with ``load_cv``, ``load_g_per_L_resin``,
    ``c_out_g_per_L``, ``breakthrough_fraction`` and the bed-average ``q_g_per_L_resin``.

    Raises
    ------
    ValueError
        If an input is non-positive, or the solver fails.
    """
    if feed_g_per_L <= 0 or load_cv <= 0:
        raise ValueError("feed concentration and load must be > 0")
    if n_tanks < 2:
        raise ValueError("n_tanks must be >= 2 to resolve a front")

    epsilon = column.porosity_total
    # Mobile-phase residence time per tank, in minutes.
    tau_min = column.residence_time_min(velocity_cm_per_h) * epsilon / n_tanks
    duration_min = load_cv * column.residence_time_min(velocity_cm_per_h) * epsilon

    def rhs(_t, y):
        c = np.maximum(y[:n_tanks], 0.0)
        q = np.maximum(y[n_tanks:], 0.0)
        uptake = resin.k_mass_transfer_per_min * (resin.equilibrium_q(c) - q)  # g/(L resin.min)
        inflow = np.concatenate(([feed_g_per_L], c[:-1]))
        dc = (inflow - c) / tau_min - (1.0 - epsilon) / epsilon * uptake
        return np.concatenate([dc, uptake])

    times_min = np.linspace(0.0, duration_min, points)
    solution = solve_ivp(
        rhs,
        (0.0, duration_min),
        np.zeros(2 * n_tanks),
        t_eval=times_min,
        method="LSODA",  # stiff while the front is sharp
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise ValueError(f"breakthrough integration failed: {solution.message}")

    c_out = np.maximum(solution.y[n_tanks - 1], 0.0)
    q_mean = np.maximum(solution.y[n_tanks:], 0.0).mean(axis=0)
    cv = times_min / (column.residence_time_min(velocity_cm_per_h) * epsilon)
    return pd.DataFrame(
        {
            "time_min": times_min,
            "load_cv": cv,
            "load_g_per_L_resin": cv * epsilon * feed_g_per_L,
            "c_out_g_per_L": c_out,
            "breakthrough_fraction": c_out / feed_g_per_L,
            "q_g_per_L_resin": q_mean,
        }
    )


def dynamic_binding_capacity(curve: pd.DataFrame, breakthrough_fraction: float = 0.10) -> float:
    """DBC at a breakthrough fraction (g product per L resin), by interpolation.

    ``DBC10`` is the figure a capture step is sized on.

    Raises
    ------
    ValueError
        If the curve never reaches that breakthrough: the load was too short to size on.
    """
    if not 0 < breakthrough_fraction < 1:
        raise ValueError("breakthrough_fraction must be in (0, 1)")
    fraction = curve["breakthrough_fraction"].to_numpy()
    if fraction.max() < breakthrough_fraction:
        raise ValueError(
            f"curve reaches only {fraction.max():.3f} breakthrough; load further than "
            f"{curve['load_cv'].iloc[-1]:.0f} CV to size on {breakthrough_fraction:.0%}"
        )
    index = int(np.argmax(fraction >= breakthrough_fraction))
    if index == 0:
        return float(curve["load_g_per_L_resin"].iloc[0])
    x = fraction[index - 1 : index + 1]
    y = curve["load_g_per_L_resin"].to_numpy()[index - 1 : index + 1]
    return float(np.interp(breakthrough_fraction, x, y))


def capture_step(
    column: Column,
    resin: LangmuirResin,
    feed_g_per_L: float,
    feed_volume_L: float,
    velocity_cm_per_h: float,
    *,
    load_fraction_of_dbc: float = 0.80,
    elution_recovery: float = 0.95,
    hcp_log_reduction: float = 3.0,
    dna_log_reduction: float = 3.5,
    aggregate_removal_fraction: float = 0.30,
    leached_protein_a_ng_per_mg: float = 8.0,
    cycles_max: int = 12,
) -> dict[str, float]:
    """Size and run a bind-elute capture step on one batch.

    The column is loaded to ``load_fraction_of_dbc`` of its 10% DBC per cycle and the
    batch is processed in as many cycles as that needs. Step yield is the product of the
    load yield (what did not break through) and ``elution_recovery``.

    Raises
    ------
    ValueError
        If the batch needs more than ``cycles_max`` cycles on this column.
    """
    if not 0 < load_fraction_of_dbc <= 1:
        raise ValueError("load_fraction_of_dbc must be in (0, 1]")
    if not 0 < elution_recovery <= 1:
        raise ValueError("elution_recovery must be in (0, 1]")

    curve = breakthrough_curve(column, resin, feed_g_per_L, velocity_cm_per_h, load_cv=120.0)
    dbc10 = dynamic_binding_capacity(curve, 0.10)
    load_g_per_cycle = load_fraction_of_dbc * dbc10 * column.volume_L
    product_g = feed_g_per_L * feed_volume_L
    cycles = int(np.ceil(product_g / load_g_per_cycle))
    if cycles > cycles_max:
        raise ValueError(
            f"{cycles} cycles needed (max {cycles_max}): the column is too small for "
            f"{product_g:.1f} g at DBC10 = {dbc10:.1f} g/L resin"
        )

    # Product lost in the flow-through up to the chosen load.
    target_load = load_fraction_of_dbc * dbc10
    mask = curve["load_g_per_L_resin"] <= target_load
    loaded = curve[mask]
    lost_fraction = float(np.trapezoid(loaded["breakthrough_fraction"], loaded["load_g_per_L_resin"]) / target_load)
    load_yield = 1.0 - lost_fraction
    yield_fraction = load_yield * elution_recovery

    return {
        "dbc10_g_per_L_resin": dbc10,
        "residence_time_min": column.residence_time_min(velocity_cm_per_h),
        "cycles": cycles,
        "load_g_per_L_resin": target_load,
        "load_yield_fraction": load_yield,
        "yield_fraction": yield_fraction,
        "product_out_g": product_g * yield_fraction,
        "hcp_log_reduction": hcp_log_reduction,
        "dna_log_reduction": dna_log_reduction,
        "aggregate_removal_fraction": aggregate_removal_fraction,
        "leached_protein_a_ng_per_mg": leached_protein_a_ng_per_mg,
        "buffer_volume_L": cycles * column.volume_L * 12.0,  # equilibration, wash, elute, strip, CIP
    }


def polishing_step(
    name: str,
    product_in_g: float,
    impurities: dict[str, float],
    *,
    yield_fraction: float,
    hcp_log_reduction: float = 1.5,
    dna_log_reduction: float = 2.0,
    aggregate_removal_fraction: float = 0.70,
    protein_a_log_reduction: float = 1.5,
) -> tuple[float, dict[str, float]]:
    """Apply a polishing step's yield and clearance. Returns (product out, impurities out).

    Clearance of a per-mg impurity has to account for product loss as well: losing
    product concentrates whatever impurity remains, which is why the ratio is scaled by
    the yield.

    Raises
    ------
    ValueError
        For a yield outside (0, 1] or a negative impurity.
    """
    if not 0 < yield_fraction <= 1:
        raise ValueError(f"{name}: yield must be in (0, 1]")
    if any(v < 0 for v in impurities.values()):
        raise ValueError(f"{name}: negative impurity levels {impurities}")

    out = dict(impurities)
    for key, log_reduction in (
        ("hcp_ng_per_mg", hcp_log_reduction),
        ("dna_pg_per_mg", dna_log_reduction),
        ("leached_protein_a_ng_per_mg", protein_a_log_reduction),
    ):
        if key in out:
            out[key] = out[key] * 10.0**-log_reduction / yield_fraction
    if "aggregate_percent" in out:
        remaining = out["aggregate_percent"] * (1.0 - aggregate_removal_fraction)
        out["aggregate_percent"] = remaining / yield_fraction
    return product_in_g * yield_fraction, out


def ufdf_step(
    product_in_g: float,
    volume_in_L: float,
    *,
    target_concentration_g_per_L: float,
    diavolumes: float = 7.0,
    yield_fraction: float = 0.95,
    membrane_flux_LMH: float = 40.0,
    area_m2: float = 0.5,
) -> dict[str, float]:
    """Ultrafiltration/diafiltration to a target concentration.

    Buffer exchange follows the ideal diafiltration law ``exp(-N)`` for a fully
    permeable salt, which is the standard first estimate.

    Raises
    ------
    ValueError
        If an input is non-positive or the yield is outside (0, 1].
    """
    if target_concentration_g_per_L <= 0 or volume_in_L <= 0 or diavolumes < 0:
        raise ValueError("concentration and volume must be > 0, diavolumes >= 0")
    if not 0 < yield_fraction <= 1:
        raise ValueError("yield must be in (0, 1]")
    product_out_g = product_in_g * yield_fraction
    final_volume_L = product_out_g / target_concentration_g_per_L
    permeate_L = volume_in_L - final_volume_L + diavolumes * final_volume_L
    return {
        "product_out_g": product_out_g,
        "final_volume_L": final_volume_L,
        "concentration_factor": volume_in_L / final_volume_L,
        "buffer_exchange_fraction": 1.0 - np.exp(-diavolumes),
        "process_time_h": permeate_L / (membrane_flux_LMH * area_m2),
        "buffer_volume_L": diavolumes * final_volume_L,
    }


def train_summary(steps: list[dict]) -> pd.DataFrame:
    """One row per step: yield, cumulative yield, product mass, impurity levels."""
    if not steps:
        raise ValueError("no steps")
    frame = pd.DataFrame(steps)
    required = {"step", "yield_fraction", "product_g"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"steps are missing {sorted(missing)}")
    frame["cumulative_yield_fraction"] = frame["yield_fraction"].cumprod()
    return frame
