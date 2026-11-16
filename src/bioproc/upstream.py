"""Fed-batch mammalian cell culture: an unstructured Monod-type mechanistic model.

States and units (units are carried in the names): time ``h``, viable and dead cells
``xv``/``xd`` in ``1e6 cells/mL``, ``glc``/``gln``/``lac``/``amm`` in ``mM``, ``titer``
in ``mg/L``, ``volume`` in ``L``. Specific rates are per ``1e9 cells`` per hour, and
``1e6 cells/mL == 1e9 cells/L``, so ``q * xv`` is mM/h directly.

The structure is the one reviewed by Kyriakopoulos et al. (2017),
doi:10.1002/biot.201700229, and used as the mechanistic half of the hybrid models in
Narayanan et al. (2019), doi:10.1002/bit.27097: Monod growth limited by glucose and
glutamine, inhibited by lactate and ammonium; death rising with ammonium and with
starvation; Luedeking-Piret product formation; and the switch to lactate consumption
once glucose is low (Gagnon et al. 2011, doi:10.1002/bit.23072).

Out of scope: amino-acid pools, pH and dissolved-oxygen dynamics, glycosylation. A
temperature shift enters as a step change in growth and productivity. The default
parameters give CHO-like trajectories; they are illustrative, not fitted to a cell line.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

STATES = ("xv", "xd", "glc", "gln", "lac", "amm", "titer", "volume", "glc_consumed")
# Cumulative totals, not concentrations: they start at zero, are never diluted by a
# feed, and do not have to be supplied in an initial state.
CUMULATIVE_STATES = ("glc_consumed",)
STATE_UNITS = {
    "xv": "1e6 cells/mL",
    "xd": "1e6 cells/mL",
    "glc": "mM",
    "gln": "mM",
    "lac": "mM",
    "amm": "mM",
    "titer": "mg/L",
    "volume": "L",
    "glc_consumed": "mmol",
}
# What an analyser and a cell counter report, mapped to the model state.
SAMPLE_COLUMNS = {
    "vcd_1e6_per_mL": "xv",
    "glucose_mM": "glc",
    "glutamine_mM": "gln",
    "lactate_mM": "lac",
    "ammonium_mM": "amm",
    "titer_mg_per_L": "titer",
}
# Switches maintenance consumption off as a nutrient runs out, so a concentration
# cannot be driven negative between solver steps (mM).
MAINTENANCE_SATURATION_mM = 0.1


@dataclass(frozen=True)
class CHOParameters:
    """Kinetic parameters. Defaults are illustrative CHO-like values, not a fit.

    The defaults were set so the specific rates land in the ranges reported for CHO
    fed-batch culture, because a model whose rates are an order of magnitude out
    produces a culture that starves on day 3 whatever else is right:

    ===================== ================== =====================================
    Quantity              Default here       Reported range
    ===================== ================== =====================================
    doubling time         ~21 h              20-30 h
    qGlc                  0.34 pmol/cell/d   0.2-0.5
    qGln                  0.08 pmol/cell/d   0.05-0.15
    qLac (overflow)       0.42 pmol/cell/d   0.2-0.6
    qp                    22 pg/cell/d       10-30 for a production clone
    ===================== ================== =====================================

    The notebook prints these rates so they can be checked against the ranges above
    before any result is read.
    """

    mu_max_per_h: float = 0.045
    ks_glc_mM: float = 1.0
    ks_gln_mM: float = 0.3
    ki_lac_mM: float = 80.0
    ki_amm_mM: float = 12.0
    kd_min_per_h: float = 0.0005
    kd_amm_max_per_h: float = 0.005
    kd_amm_mM: float = 8.0
    kd_starve_per_h: float = 0.030
    k_starve_mM: float = 0.2
    # Dead cells lyse rather than accumulating for ever. Without this term viability
    # falls monotonically from inoculation and a 14-day culture ends below 30%, which
    # no real process does. 0.02/h is a ~35 h half-life for a dead cell.
    k_lysis_per_h: float = 0.020
    y_x_glc_1e9cells_per_mmol: float = 2.8
    m_glc_mmol_per_1e9cells_h: float = 0.002
    y_lac_glc_mol_per_mol: float = 1.3
    q_lac_uptake_mmol_per_1e9cells_h: float = 0.008
    k_switch_glc_mM: float = 2.0
    k_lac_uptake_mM: float = 5.0
    y_x_gln_1e9cells_per_mmol: float = 12.0
    m_gln_mmol_per_1e9cells_h: float = 0.0004
    k_gln_degradation_per_h: float = 0.0025
    y_amm_gln_mol_per_mol: float = 0.85
    qp_alpha_mg_per_1e9cells: float = 0.0
    qp_beta_mg_per_1e9cells_h: float = 0.90
    shift_mu_factor: float = 0.60
    shift_qp_factor: float = 1.35

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {value}")
        if self.y_x_glc_1e9cells_per_mmol <= 0 or self.y_x_gln_1e9cells_per_mmol <= 0:
            raise ValueError("biomass yields must be > 0")


@dataclass(frozen=True)
class Feed:
    """A bolus addition: a volume carrying nutrient concentrations (mM)."""

    volume_L: float
    glc_mM: float = 0.0
    gln_mM: float = 0.0
    label: str = "feed"

    def __post_init__(self) -> None:
        if min(self.volume_L, self.glc_mM, self.gln_mM) < 0:
            raise ValueError(f"negative feed: {self}")


def monod(concentration, half_saturation):
    """``c / (c + k)``, clipped at zero. Vectorised."""
    c = np.maximum(np.asarray(concentration, dtype=float), 0.0)
    return c / (c + half_saturation)


def specific_rates(p: CHOParameters, glc, gln, lac, amm, shifted=0.0) -> dict[str, np.ndarray]:
    """Specific rates at a state; vectorised over arrays.

    Returns growth ``mu`` and death ``kd`` (1/h); consumption ``q_glc``, ``q_gln`` and
    net lactate and ammonium ``q_lac``, ``q_amm`` (mmol per 1e9 cells per h, positive
    meaning production for ``q_lac``/``q_amm``); and ``q_p`` (mg per 1e9 cells per h).
    ``shifted`` is 0 before the temperature shift and 1 after it.
    """
    glc, gln, lac, amm = (np.maximum(np.asarray(x, dtype=float), 0.0) for x in (glc, gln, lac, amm))
    shifted = np.asarray(shifted, dtype=float)
    mu_scale = 1.0 + shifted * (p.shift_mu_factor - 1.0)
    qp_scale = 1.0 + shifted * (p.shift_qp_factor - 1.0)

    inhibition = (p.ki_lac_mM / (p.ki_lac_mM + lac)) * (p.ki_amm_mM / (p.ki_amm_mM + amm))
    mu = p.mu_max_per_h * mu_scale * monod(glc, p.ks_glc_mM) * monod(gln, p.ks_gln_mM) * inhibition
    kd = (
        p.kd_min_per_h
        + p.kd_amm_max_per_h * monod(amm, p.kd_amm_mM)
        + p.kd_starve_per_h * (1.0 - monod(glc, p.k_starve_mM))
    )

    # Growth-associated plus maintenance; maintenance fades out as the nutrient does.
    q_glc = mu / p.y_x_glc_1e9cells_per_mmol + p.m_glc_mmol_per_1e9cells_h * monod(glc, MAINTENANCE_SATURATION_mM)
    q_gln = mu / p.y_x_gln_1e9cells_per_mmol + p.m_gln_mmol_per_1e9cells_h * monod(gln, MAINTENANCE_SATURATION_mM)

    # Glycolytic overflow while glucose is plentiful; net uptake once it is scarce and
    # lactate has accumulated (the metabolic shift).
    overflow = p.y_lac_glc_mol_per_mol * q_glc * monod(glc, p.k_switch_glc_mM)
    uptake = p.q_lac_uptake_mmol_per_1e9cells_h * monod(lac, p.k_lac_uptake_mM) * (1.0 - monod(glc, p.k_switch_glc_mM))
    q_lac = overflow - uptake
    q_amm = p.y_amm_gln_mol_per_mol * q_gln
    q_p = qp_scale * (p.qp_alpha_mg_per_1e9cells * mu + p.qp_beta_mg_per_1e9cells_h)
    return {"mu": mu, "kd": kd, "q_glc": q_glc, "q_gln": q_gln, "q_lac": q_lac, "q_amm": q_amm, "q_p": q_p}


# Ranges reported for CHO fed-batch culture, used to sanity-check a parameter set
# before it is trusted. Units: pmol per cell per day, except where noted.
PLAUSIBLE_RATES = {
    "doubling_time_h": (18.0, 35.0),
    "q_glc_pmol_per_cell_day": (0.15, 0.60),
    "q_gln_pmol_per_cell_day": (0.03, 0.20),
    "q_lac_pmol_per_cell_day": (0.10, 0.80),
    "qp_pg_per_cell_day": (5.0, 40.0),
}


def rate_check(p: CHOParameters, glc_mM: float = 25.0, gln_mM: float = 4.0) -> pd.DataFrame:
    """Specific rates in the units the literature reports, next to plausible ranges.

    A parameter set whose rates are an order of magnitude out produces a culture that
    starves in three days however carefully everything else is done, so this is worth
    looking at before any result (ground rules section 1: a number is not a result
    until it can be interpreted).

    Returns one row per quantity with its value, the reported range and whether it
    falls inside. ``in_range`` being False is a warning about the parameters, not an
    error: an unusual clone is allowed, it just has to be a deliberate choice.
    """
    rates = specific_rates(p, glc_mM, gln_mM, 0.5, 0.5)
    mu = float(rates["mu"])
    # q is mmol per 1e9 cells per h; x24 h/day gives mmol per 1e9 cells per day, which
    # is numerically pmol per cell per day.
    values = {
        "doubling_time_h": np.log(2) / mu if mu > 0 else np.inf,
        "q_glc_pmol_per_cell_day": float(rates["q_glc"]) * 24.0,
        "q_gln_pmol_per_cell_day": float(rates["q_gln"]) * 24.0,
        "q_lac_pmol_per_cell_day": float(rates["q_lac"]) * 24.0,
        "qp_pg_per_cell_day": float(rates["q_p"]) * 24.0,
    }
    return pd.DataFrame(
        [
            {
                "quantity": name,
                "value": value,
                "reported_low": PLAUSIBLE_RATES[name][0],
                "reported_high": PLAUSIBLE_RATES[name][1],
                "in_range": bool(PLAUSIBLE_RATES[name][0] <= value <= PLAUSIBLE_RATES[name][1]),
            }
            for name, value in values.items()
        ]
    )


def derivatives(t: float, y: np.ndarray, p: CHOParameters, shift_time_h: float) -> np.ndarray:
    """Right-hand side of the ODE system, in the order of :data:`STATES`."""
    xv, xd, glc, gln, lac, amm, titer, volume, _consumed = y
    xv, xd = max(xv, 0.0), max(xd, 0.0)
    rates = specific_rates(p, glc, gln, lac, amm, shifted=float(t >= shift_time_h))
    mu, kd = float(rates["mu"]), float(rates["kd"])
    return np.array(
        [
            (mu - kd) * xv,
            kd * xv - p.k_lysis_per_h * xd,
            -float(rates["q_glc"]) * xv,
            -float(rates["q_gln"]) * xv - p.k_gln_degradation_per_h * max(gln, 0.0),
            float(rates["q_lac"]) * xv,
            float(rates["q_amm"]) * xv + p.k_gln_degradation_per_h * max(gln, 0.0),
            float(rates["q_p"]) * xv,
            0.0,  # volume changes only at a bolus feed
            # Cumulative glucose consumed, in mmol: q is mmol per 1e9 cells per h and
            # xv is 1e9 cells/L, so q*xv*V is mmol/h. Integrating it in the solver,
            # rather than by quadrature on sample points afterwards, makes the mass
            # balance a test of the feeding bookkeeping instead of a test of trapezoid
            # accuracy.
            float(rates["q_glc"]) * xv * max(volume, 0.0),
        ]
    )


Policy = Callable[[float, dict[str, float], float], list[Feed]]


def simulate(
    initial: dict[str, float],
    parameters: CHOParameters,
    *,
    duration_h: float = 14 * 24,
    sample_interval_h: float = 24.0,
    feed_policy: Policy | None = None,
    shift_time_h: float = np.inf,
    max_step_h: float = 1.0,
) -> pd.DataFrame:
    """Integrate the culture, applying ``feed_policy`` at each sample point.

    ``feed_policy(t_h, state, sample_interval_h) -> list[Feed]`` sees the state just
    before the feed, which is what an operator sees: the sample is drawn, the analyser
    reports, and the bolus is added on that reading.

    Returns one row per sample point with the states, the specific rates, the feed
    volume added at that point, and cumulative ``feed_volume_L``.

    Raises
    ------
    ValueError
        If a state is missing or negative, the horizon or interval is not positive, or
        the solver fails.
    """
    missing = [s for s in STATES if s not in initial and s not in CUMULATIVE_STATES]
    if missing:
        raise ValueError(f"initial state is missing {missing}")
    if initial["volume"] <= 0:
        raise ValueError("initial volume must be > 0 L")
    negative = {k: v for k, v in initial.items() if k in STATES and v < 0}
    if negative:
        raise ValueError(f"negative initial states: {negative}")
    if duration_h <= 0 or sample_interval_h <= 0:
        raise ValueError("duration_h and sample_interval_h must be > 0")

    y = np.array([float(initial.get(s, 0.0)) for s in STATES])
    times = np.arange(0.0, duration_h + 1e-9, sample_interval_h)
    rows, cumulative_feed_L = [], 0.0

    for index, t in enumerate(times):
        state = dict(zip(STATES, y, strict=True))
        added_L = 0.0
        if feed_policy is not None and 0 < index < len(times) - 1:
            for feed in feed_policy(float(t), dict(state), sample_interval_h):
                if feed.volume_L <= 0:
                    continue
                total_L = state["volume"] + feed.volume_L
                # Dilution: cells and product are diluted, nutrients also gain the bolus.
                for key in ("xv", "xd", "titer", "lac", "amm"):
                    state[key] *= state["volume"] / total_L
                state["glc"] = (state["glc"] * state["volume"] + feed.glc_mM * feed.volume_L) / total_L
                state["gln"] = (state["gln"] * state["volume"] + feed.gln_mM * feed.volume_L) / total_L
                state["volume"] = total_L
                added_L += feed.volume_L
                cumulative_feed_L += feed.volume_L
            y = np.array([state[s] for s in STATES])

        rates = specific_rates(parameters, state["glc"], state["gln"], state["lac"], state["amm"], t >= shift_time_h)
        rows.append(
            {
                "time_h": float(t),
                "time_d": float(t) / 24.0,
                **{k: float(v) for k, v in state.items()},
                "viability_percent": 100.0 * state["xv"] / max(state["xv"] + state["xd"], 1e-12),
                "feed_volume_added_L": added_L,
                "feed_volume_L": cumulative_feed_L,
                "temperature_shifted": bool(t >= shift_time_h),
                **{k: float(v) for k, v in rates.items()},
            }
        )

        if index < len(times) - 1:
            solution = solve_ivp(
                derivatives,
                (float(t), float(times[index + 1])),
                y,
                args=(parameters, shift_time_h),
                method="LSODA",  # the rates stiffen sharply at the metabolic switch
                max_step=max_step_h,
                rtol=1e-7,
                atol=1e-9,
            )
            if not solution.success:
                raise ValueError(f"integration failed at t = {t:.1f} h: {solution.message}")
            y = np.maximum(solution.y[:, -1], 0.0)

    return pd.DataFrame(rows)


def glucose_setpoint_policy(target_glc_mM: float, feed_glc_mM: float, feed_gln_mM: float = 0.0) -> Policy:
    """Feed on the measured glucose: top it back up to ``target_glc_mM``.

    This is the Gagnon-style feedback used in section 5.9 of Khuat et al. (2023),
    doi:10.48550/arXiv.2310.09991 - the bolus is sized from the reading just taken.

    Raises
    ------
    ValueError
        If the feed is not more concentrated than the target.
    """
    if feed_glc_mM <= target_glc_mM:
        raise ValueError(f"feed glucose {feed_glc_mM} mM must exceed the target {target_glc_mM} mM")

    def policy(_t_h: float, state: dict[str, float], _interval_h: float) -> list[Feed]:
        deficit_mM = target_glc_mM - state["glc"]
        if deficit_mM <= 0:
            return []
        volume_L = deficit_mM * state["volume"] / (feed_glc_mM - target_glc_mM)
        return [Feed(volume_L=volume_L, glc_mM=feed_glc_mM, gln_mM=feed_gln_mM, label="glucose setpoint")]

    return policy


def fixed_bolus_policy(fraction_of_volume: float, feed_glc_mM: float, feed_gln_mM: float = 0.0) -> Policy:
    """Feed a fixed fraction of the current volume at every sample point."""
    if not 0 <= fraction_of_volume < 1:
        raise ValueError("fraction_of_volume must be in [0, 1)")

    def policy(_t_h: float, state: dict[str, float], _interval_h: float) -> list[Feed]:
        return [
            Feed(
                volume_L=fraction_of_volume * state["volume"],
                glc_mM=feed_glc_mM,
                gln_mM=feed_gln_mM,
                label="fixed bolus",
            )
        ]

    return policy


def harvest_metrics(history: pd.DataFrame, viability_floor_percent: float = 70.0) -> dict[str, float]:
    """Key performance indicators at harvest.

    Harvest is the first sample at or below ``viability_floor_percent``, else the last.
    Returns titer (mg/L), harvested mass (mg), peak and final VCD, integral viable cell
    density (IVCD, ``1e6 cell.day/mL``) and the overall cell-specific productivity
    ``qp`` (pg per cell per day) that is the usual way to compare clones.
    """
    if history.empty:
        raise ValueError("empty history")
    below = history.index[history["viability_percent"] <= viability_floor_percent]
    end = int(below[0]) if len(below) else int(history.index[-1])
    cut = history.loc[:end]
    ivcd = float(np.trapezoid(cut["xv"], cut["time_d"]))
    # mg/L / (1e6 cell.day/mL) = mg per 1e9 cell.day = pg per cell per day.
    return {
        "harvest_time_d": float(cut["time_d"].iloc[-1]),
        "titer_mg_per_L": float(cut["titer"].iloc[-1]),
        "harvest_mass_mg": float(cut["titer"].iloc[-1] * cut["volume"].iloc[-1]),
        "peak_vcd_1e6_per_mL": float(cut["xv"].max()),
        "final_viability_percent": float(cut["viability_percent"].iloc[-1]),
        "ivcd_1e6_cell_day_per_mL": ivcd,
        "qp_pg_per_cell_day": float(cut["titer"].iloc[-1] / ivcd) if ivcd > 0 else float("nan"),
        "final_lactate_mM": float(cut["lac"].iloc[-1]),
        "final_ammonium_mM": float(cut["amm"].iloc[-1]),
    }
