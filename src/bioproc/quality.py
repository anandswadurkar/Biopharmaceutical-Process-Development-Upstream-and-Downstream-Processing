"""QbD bookkeeping: CQAs, acceptance, control charts and capability.

A CQA is a property with a limit (ICH Q8/Q11, as summarised by Eon-Duval et al. 2012,
doi:10.1002/btpr.1548). This module decides whether a batch meets its criteria, builds
the control charts that say whether the process is stable, and computes capability.

A batch that misses a criterion is a *process* finding, not a code failure: it is
reported as ``OUT OF SPEC`` in the release table, never raised as an error. Only a
broken calculation raises (ground rules section 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Criterion:
    """One acceptance criterion. ``lower``/``upper`` are inclusive; ``None`` is one-sided."""

    name: str
    unit: str
    lower: float | None = None
    upper: float | None = None
    kind: str = "CQA"  # CQA, KPI or PROCESS

    def __post_init__(self) -> None:
        if self.lower is None and self.upper is None:
            raise ValueError(f"{self.name}: a criterion needs at least one limit")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"{self.name}: lower {self.lower} > upper {self.upper}")

    def passes(self, value: float) -> bool:
        """Whether a value meets the criterion.

        >>> Criterion("aggregate", "%", upper=2.0).passes(1.5)
        True
        """
        if not np.isfinite(value):
            raise ValueError(f"{self.name}: value is not finite ({value})")
        if self.lower is not None and value < self.lower:
            return False
        return not (self.upper is not None and value > self.upper)

    def margin(self, value: float) -> float:
        """Distance to the nearest limit; negative when out of spec."""
        distances = []
        if self.lower is not None:
            distances.append(value - self.lower)
        if self.upper is not None:
            distances.append(self.upper - value)
        return float(min(distances))

    def describe(self) -> str:
        if self.lower is not None and self.upper is not None:
            return f"{self.lower:g} to {self.upper:g} {self.unit}"
        if self.lower is not None:
            return f">= {self.lower:g} {self.unit}"
        return f"<= {self.upper:g} {self.unit}"


def criteria_from_config(acceptance: dict) -> list[Criterion]:
    """Build criteria from the config's ``[acceptance]`` table.

    Each entry is ``name = {unit = "...", lower = x, upper = y, kind = "CQA"}``.

    Raises
    ------
    ValueError
        If an entry is not a table or has no limit.
    """
    criteria = []
    for name, entry in acceptance.items():
        if not isinstance(entry, dict):
            raise ValueError(f"[acceptance.{name}] must be a table with unit and limits, got {entry!r}")
        criteria.append(
            Criterion(
                name=name,
                unit=str(entry.get("unit", "")),
                lower=entry.get("lower"),
                upper=entry.get("upper"),
                kind=str(entry.get("kind", "CQA")),
            )
        )
    return criteria


def release_table(values: dict[str, float], criteria: list[Criterion]) -> pd.DataFrame:
    """Compare a batch's results with its criteria.

    A criterion with no measured value is reported as ``NOT MEASURED`` rather than
    silently dropped: an attribute nobody measured is not an attribute that passed.
    """
    rows = []
    for criterion in criteria:
        if criterion.name not in values:
            rows.append(
                {
                    "attribute": criterion.name,
                    "kind": criterion.kind,
                    "value": np.nan,
                    "unit": criterion.unit,
                    "criterion": criterion.describe(),
                    "margin": np.nan,
                    "status": "NOT MEASURED",
                }
            )
            continue
        value = float(values[criterion.name])
        passed = criterion.passes(value)
        rows.append(
            {
                "attribute": criterion.name,
                "kind": criterion.kind,
                "value": value,
                "unit": criterion.unit,
                "criterion": criterion.describe(),
                "margin": criterion.margin(value),
                "status": "PASS" if passed else "OUT OF SPEC",
            }
        )
    return pd.DataFrame(rows)


def control_chart(values, *, labels=None, sigma: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """Individuals chart with limits from the moving range (an I-MR chart).

    Limits come from the average moving range (``d2 = 1.128`` for n = 2), not from the
    standard deviation of all the points: that is what makes the chart sensitive to a
    shift rather than absorbing it into the limits.

    Returns the per-point table and the limits.

    Raises
    ------
    ValueError
        With fewer than 3 points, where limits are not meaningful.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        raise ValueError("a control chart needs at least 3 batches")
    if not np.isfinite(values).all():
        raise ValueError("non-finite values")
    moving_range = np.abs(np.diff(values))
    centre = float(values.mean())
    spread = float(moving_range.mean()) / 1.128
    limits = {
        "centre": centre,
        "sigma_estimate": spread,
        "upper": centre + sigma * spread,
        "lower": centre - sigma * spread,
        "mr_mean": float(moving_range.mean()),
        "mr_upper": 3.267 * float(moving_range.mean()),
    }

    table = pd.DataFrame(
        {"batch": list(labels) if labels is not None else np.arange(1, values.size + 1), "value": values}
    )
    table["moving_range"] = np.concatenate([[np.nan], moving_range])
    table["out_of_control"] = (values > limits["upper"]) | (values < limits["lower"])
    # Western Electric rule 2: nine points in a row on one side of the centre line.
    side = np.sign(values - centre)
    run_flag = np.zeros(values.size, dtype=bool)
    for index in range(8, values.size):
        window = side[index - 8 : index + 1]
        if np.all(window == window[0]) and window[0] != 0:
            run_flag[index] = True
    table["run_rule"] = run_flag
    return table, limits


def capability(values, criterion: Criterion) -> dict[str, float]:
    """Process capability ``Cp``/``Cpk`` against a criterion.

    ``Cpk`` below 1.0 means the process produces out-of-spec batches at its current
    spread and centring; 1.33 is the usual target. One-sided criteria give ``Cp = nan``
    and a one-sided ``Cpk``.

    Raises
    ------
    ValueError
        With fewer than 3 values, or zero spread (capability is undefined).
    """
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        raise ValueError("capability needs at least 3 batches")
    sd = float(np.std(values, ddof=1))
    if sd == 0:
        raise ValueError("zero spread: capability is undefined")
    mean = float(values.mean())
    cpk_parts = []
    if criterion.upper is not None:
        cpk_parts.append((criterion.upper - mean) / (3 * sd))
    if criterion.lower is not None:
        cpk_parts.append((mean - criterion.lower) / (3 * sd))
    cp = (
        (criterion.upper - criterion.lower) / (6 * sd)
        if criterion.lower is not None and criterion.upper is not None
        else float("nan")
    )
    return {
        "n": int(values.size),
        "mean": mean,
        "sd": sd,
        "cp": float(cp),
        "cpk": float(min(cpk_parts)),
        "out_of_spec_batches": int(sum(not criterion.passes(v) for v in values)),
    }


def golden_batch(
    samples: pd.DataFrame,
    column: str,
    *,
    batch_column: str = "batch_id",
    time_column: str = "time_d",
    bands: tuple[float, float] = (10.0, 90.0),
) -> pd.DataFrame:
    """Median trajectory with a percentile band across batches: the 'golden batch'.

    Used to judge whether a new run is tracking the historical process, which is what
    multivariate batch monitoring does (Kirdar et al. 2007, doi:10.1021/bp060377u).

    Raises
    ------
    ValueError
        If a column is missing or fewer than 3 batches are present.
    """
    for name in (column, batch_column, time_column):
        if name not in samples.columns:
            raise ValueError(f"{name!r} is not a column")
    if samples[batch_column].nunique() < 3:
        raise ValueError("a reference band needs at least 3 batches")
    grouped = samples.groupby(time_column)[column]
    return pd.DataFrame(
        {
            "median": grouped.median(),
            f"p{int(bands[0]):02d}": grouped.quantile(bands[0] / 100),
            f"p{int(bands[1]):02d}": grouped.quantile(bands[1] / 100),
            "n_batches": grouped.count(),
        }
    ).reset_index()


def comparability(group_a, group_b, *, name: str = "comparison", alpha: float = 0.05) -> dict:
    """Compare two groups of batches and state the smallest difference detectable.

    "No significant difference" is meaningless without the detection limit, so the
    result always carries ``min_detectable_difference`` from the smaller group
    (ground rules section 1).

    Raises
    ------
    ValueError
        If either group has fewer than 2 replicates - one run against one run is not a
        comparison.
    """
    from scipy import stats

    from bioproc.evaluation import detection_limit

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError(f"{name}: need at least 2 replicates on both sides, got {a.size} and {b.size}")
    result = stats.ttest_ind(a, b, equal_var=False)
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    smaller = a if a.size <= b.size else b
    limit = detection_limit(smaller)
    difference = float(a.mean() - b.mean())
    return {
        "name": name,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "n_a": int(a.size),
        "n_b": int(b.size),
        "difference": difference,
        "cohens_d": float(difference / pooled_sd) if pooled_sd > 0 else float("nan"),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
        "min_detectable_difference": limit["min_detectable_difference"],
        "verdict": (
            f"difference {difference:+.3g} (p = {result.pvalue:.3f}); this design could detect "
            f"{limit['min_detectable_difference']:.3g} at 80% power"
        ),
    }
