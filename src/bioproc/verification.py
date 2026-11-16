"""Verification: does the code solve the equations correctly?

Separate from validation (are these the right equations?), which needs independent
experimental data and lives in :mod:`bioproc.validation`. Ground rules section 3.

Checks here:

- analytic solutions, where one exists (batch growth with no limitation; a Langmuir
  isotherm at equilibrium; ideal diafiltration);
- conservation of mass over a simulated culture and over a purification train;
- time-step convergence with the observed order of the integrator;
- a tanks-in-series check: the capture model's dispersion is set by the tank count, so
  the count is shown to be a model parameter rather than a refinement knob.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Check:
    """One verification result. ``passed`` is the assertion; ``note`` says why."""

    name: str
    passed: bool
    observed: float
    expected: float
    tolerance: float
    note: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def relative_error(self) -> float:
        scale = abs(self.expected) if self.expected != 0 else 1.0
        return abs(self.observed - self.expected) / scale


def close(name: str, observed: float, expected: float, tolerance: float, note: str = "") -> Check:
    """Build a Check from a relative tolerance.

    ``tolerance`` is relative unless ``expected`` is zero, when it is absolute.

    >>> close("demo", 1.001, 1.0, 0.01).passed
    True
    """
    scale = abs(expected) if expected != 0 else 1.0
    return Check(name, abs(observed - expected) / scale <= tolerance, observed, expected, tolerance, note)


def exponential_growth(mu_per_h: float, x0: float, hours: float) -> float:
    """Analytic unlimited growth ``x0 * exp(mu * t)``; the limiting case of the model."""
    if mu_per_h < 0 or x0 <= 0 or hours < 0:
        raise ValueError("require mu >= 0, x0 > 0, t >= 0")
    return float(x0 * np.exp(mu_per_h * hours))


def verify_growth_against_analytic(
    simulate_unlimited, mu_per_h: float, x0: float, hours: float, tolerance: float = 2e-3
) -> Check:
    """Compare the integrator against exponential growth with nutrients in excess.

    ``simulate_unlimited(mu_per_h, x0, hours) -> float`` runs the model with limitation,
    inhibition and death switched off, where the answer is known exactly.
    """
    observed = float(simulate_unlimited(mu_per_h, x0, hours))
    expected = exponential_growth(mu_per_h, x0, hours)
    return close(
        "Batch growth matches x0*exp(mu*t) with nutrients in excess",
        observed,
        expected,
        tolerance,
        "Analytic solution of the model's own limiting case",
    )


def verify_langmuir_equilibrium(resin, concentration_g_per_L: float, tolerance: float = 1e-10) -> Check:
    """At equilibrium the isotherm must reproduce its closed form."""
    c = float(concentration_g_per_L)
    expected = resin.q_max_g_per_L_resin * resin.k_eq_L_per_g * c / (1.0 + resin.k_eq_L_per_g * c)
    return close("Langmuir isotherm closed form", float(resin.equilibrium_q(c)), expected, tolerance)


def verify_glucose_balance(history: pd.DataFrame, feed_glc_mM: float, tolerance: float = 1e-6) -> Check:
    """Glucose in = glucose out + glucose consumed, over the whole culture.

    What is in the vessel at the end must equal what started there, plus what was fed,
    minus what the cells consumed. Consumption comes from ``glc_consumed``, which the
    solver integrates as a state: that makes this a test of the feeding and dilution
    bookkeeping, which is where errors hide, rather than a test of how accurately a
    trapezoid rule reproduces a rate sampled once a day.
    """
    required = {"glc", "volume", "glc_consumed", "feed_volume_added_L"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing {sorted(missing)}")
    inventory_mmol = history["glc"] * history["volume"]
    fed_mmol = float((history["feed_volume_added_L"] * feed_glc_mM).sum())
    consumed_mmol = float(history["glc_consumed"].iloc[-1])
    change_mmol = float(inventory_mmol.iloc[-1] - inventory_mmol.iloc[0])
    residual = change_mmol - (fed_mmol - consumed_mmol)
    scale = max(abs(fed_mmol), abs(consumed_mmol), 1e-12)
    return Check(
        "Glucose mass balance over the culture",
        abs(residual) / scale <= tolerance,
        float(abs(residual) / scale),
        0.0,
        tolerance,
        f"fed {fed_mmol:.3f} mmol, consumed {consumed_mmol:.3f} mmol (integrated as a state)",
    )


def verify_product_balance(steps: pd.DataFrame, tolerance: float = 1e-9) -> Check:
    """Each downstream step's output must equal its input times its yield."""
    required = {"step", "product_g", "yield_fraction"}
    missing = required - set(steps.columns)
    if missing:
        raise ValueError(f"steps are missing {sorted(missing)}")
    expected = steps["product_g"].iloc[0] * steps["yield_fraction"].iloc[1:].cumprod()
    observed = steps["product_g"].iloc[1:]
    worst = float((observed - expected).abs().max() / max(abs(steps["product_g"].iloc[0]), 1e-12))
    return Check(
        "Product mass balance across the purification train",
        worst <= tolerance,
        worst,
        0.0,
        tolerance,
        "Each step: out = in * yield",
    )


def verify_diafiltration(buffer_exchange_fraction: float, diavolumes: float, tolerance: float = 1e-9) -> Check:
    """Ideal diafiltration: the retained salt fraction is ``exp(-N)``."""
    return close(
        "Diafiltration buffer exchange = 1 - exp(-N)",
        float(buffer_exchange_fraction),
        float(1.0 - np.exp(-diavolumes)),
        tolerance,
        "Fully permeable solute, perfectly mixed retentate",
    )


def observed_order(step_sizes, errors) -> float:
    """Slope of log(error) against log(step size): the observed order of convergence.

    >>> round(observed_order([0.4, 0.2, 0.1], [0.16, 0.04, 0.01]), 3)
    2.0

    Raises
    ------
    ValueError
        If fewer than two points, or an error is zero (the fit is undefined).
    """
    step_sizes = np.asarray(step_sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if step_sizes.size < 2:
        raise ValueError("need at least two refinement levels")
    if np.any(errors <= 0):
        raise ValueError("an error is zero or negative; cannot fit a log-log slope")
    slope, _intercept = np.polyfit(np.log(step_sizes), np.log(errors), 1)
    return float(slope)


def convergence_study(run, step_sizes, reference=None) -> tuple[pd.DataFrame, float]:
    """Run at several step sizes and report the error and the observed order.

    ``run(step) -> float`` returns the quantity of interest. Without a ``reference``,
    the finest step is taken as the reference and dropped from the fit, which measures
    self-convergence rather than accuracy.

    Returns the table and the observed order.
    """
    step_sizes = sorted(map(float, step_sizes), reverse=True)
    if len(step_sizes) < 3 and reference is None:
        raise ValueError("self-convergence needs at least 3 step sizes")
    values = [float(run(step)) for step in step_sizes]
    if reference is None:
        reference, fit_steps, fit_values = values[-1], step_sizes[:-1], values[:-1]
    else:
        reference, fit_steps, fit_values = float(reference), step_sizes, values
    errors = [abs(value - reference) for value in fit_values]
    table = pd.DataFrame({"step": step_sizes, "value": values})
    table["abs_error"] = [abs(value - reference) for value in values]
    table["relative_error"] = table["abs_error"] / max(abs(reference), 1e-30)
    return table, observed_order(fit_steps, errors)


def summarise(checks: list[Check]) -> pd.DataFrame:
    """Checks as a table, failures first."""
    if not checks:
        raise ValueError("no checks to summarise")
    frame = pd.DataFrame(
        [
            {
                "check": c.name,
                "status": c.status,
                "observed": c.observed,
                "expected": c.expected,
                "relative_error": c.relative_error,
                "tolerance": c.tolerance,
                "note": c.note,
            }
            for c in checks
        ]
    )
    return frame.sort_values("status", ascending=True, kind="stable").reset_index(drop=True)


def raise_on_failure(checks: list[Check]) -> None:
    """Raise if any check failed, naming every failure.

    Raises
    ------
    AssertionError
        Listing each failed check with its observed and expected values.
    """
    failed = [c for c in checks if not c.passed]
    if failed:
        lines = [
            f"  {c.name}: observed {c.observed:.6g}, expected {c.expected:.6g} (tol {c.tolerance:g})" for c in failed
        ]
        raise AssertionError(f"{len(failed)} verification check(s) failed:\n" + "\n".join(lines))
