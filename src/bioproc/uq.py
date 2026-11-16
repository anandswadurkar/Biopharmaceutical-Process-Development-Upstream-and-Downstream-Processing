"""Uncertainty and sensitivity: which inputs move the answer, and by how much.

Ground rules section 3. Two tools:

- :func:`sobol_indices` - variance-based global sensitivity (SALib), for ranking the
  kinetic parameters and operating conditions that drive a KPI;
- :func:`monte_carlo` - propagate input distributions to an interval on the output, so
  results are reported as intervals rather than single trajectories.

Both take a plain ``evaluate(sample) -> float`` so they work on the mechanistic model,
the hybrid model or a purification train without knowing which.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def _problem(bounds: dict[str, tuple[float, float]]) -> dict:
    if not bounds:
        raise ValueError("no parameters given")
    for name, (low, high) in bounds.items():
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"{name}: bounds must be finite with low < high, got ({low}, {high})")
    return {"num_vars": len(bounds), "names": list(bounds), "bounds": [list(b) for b in bounds.values()]}


def sobol_indices(
    evaluate: Callable[[dict[str, float]], float],
    bounds: dict[str, tuple[float, float]],
    *,
    n_samples: int = 256,
    seed: int = 0,
    progress_every: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """First-order and total Sobol indices for ``evaluate`` over ``bounds``.

    ``n_samples`` is the Saltelli base; the model is called ``n_samples * (2D + 2)``
    times, so a 6-parameter study at 256 is 3584 runs. Keep the model cheap, or lower
    it and read the confidence intervals, which widen accordingly.

    Returns the indices (with 95% bootstrap confidence intervals) and a record of the
    settings, including the evaluation count actually performed.

    Raises
    ------
    ValueError
        For bad bounds, or if any evaluation returns a non-finite value.
    """
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    problem = _problem(bounds)
    if n_samples < 8 or (n_samples & (n_samples - 1)) != 0:
        raise ValueError(f"n_samples should be a power of 2 and >= 8 for Saltelli sampling, got {n_samples}")
    design = sobol_sample.sample(problem, n_samples, calc_second_order=False, seed=seed)

    outputs = np.empty(len(design))
    for index, row in enumerate(design):
        value = float(evaluate(dict(zip(problem["names"], row, strict=True))))
        if not np.isfinite(value):
            raise ValueError(f"evaluation {index} returned {value} for {dict(zip(problem['names'], row, strict=True))}")
        outputs[index] = value
        if progress_every and (index + 1) % progress_every == 0:
            print(f"  {index + 1}/{len(design)} evaluations")

    analysis = sobol_analyze.analyze(problem, outputs, calc_second_order=False, seed=seed)
    indices = pd.DataFrame(
        {
            "parameter": problem["names"],
            "S1": analysis["S1"],
            "S1_conf": analysis["S1_conf"],
            "ST": analysis["ST"],
            "ST_conf": analysis["ST_conf"],
        }
    ).sort_values("ST", ascending=False, ignore_index=True)
    record = {
        "n_base_samples": n_samples,
        "n_evaluations": int(len(design)),
        "seed": seed,
        "output_mean": float(outputs.mean()),
        "output_sd": float(outputs.std(ddof=1)),
        "bounds": {k: list(v) for k, v in bounds.items()},
    }
    return indices, record


def monte_carlo(
    evaluate: Callable[[dict[str, float]], float],
    distributions: dict[str, tuple[str, float, float]],
    *,
    n_samples: int = 200,
    seed: int = 0,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> tuple[pd.DataFrame, dict]:
    """Propagate input distributions to an interval on the output.

    ``distributions`` maps a name to ``(kind, a, b)``: ``("normal", mean, sd)``,
    ``("lognormal", median, cv)`` (CV as a fraction) or ``("uniform", low, high)``.
    Lognormal is the right default for a positive quantity such as a rate constant or a
    raw-material potency - a normal with a wide CV puts mass below zero.

    Returns the sample table (inputs plus ``output``) and a summary with the quantiles.

    Raises
    ------
    ValueError
        For an unknown distribution kind or invalid parameters.
    """
    generator = np.random.default_rng(seed)
    draws: dict[str, np.ndarray] = {}
    for name, (kind, a, b) in distributions.items():
        if kind == "normal":
            if b < 0:
                raise ValueError(f"{name}: sd must be >= 0")
            draws[name] = generator.normal(a, b, n_samples)
        elif kind == "lognormal":
            if a <= 0 or b < 0:
                raise ValueError(f"{name}: lognormal needs median > 0 and cv >= 0")
            sigma = np.sqrt(np.log(1.0 + b**2))
            draws[name] = generator.lognormal(np.log(a), sigma, n_samples)
        elif kind == "uniform":
            if a >= b:
                raise ValueError(f"{name}: uniform needs low < high")
            draws[name] = generator.uniform(a, b, n_samples)
        else:
            raise ValueError(f"{name}: unknown distribution {kind!r}; use normal, lognormal or uniform")

    table = pd.DataFrame(draws)
    outputs = []
    for _, row in table.iterrows():
        value = float(evaluate(row.to_dict()))
        outputs.append(value)
    table["output"] = outputs
    finite = table["output"].to_numpy()
    if not np.isfinite(finite).all():
        raise ValueError(f"{int((~np.isfinite(finite)).sum())} evaluations returned non-finite values")

    summary = {
        "n_samples": n_samples,
        "seed": seed,
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)),
        **{f"q{int(q * 100):02d}": float(np.quantile(finite, q)) for q in quantiles},
        "distributions": {k: list(v) for k, v in distributions.items()},
    }
    return table, summary


def interval_string(summary: dict, unit: str = "", quantiles=(5, 95)) -> str:
    """Format a Monte Carlo summary as ``median [q05, q95] unit``.

    >>> interval_string({"q50": 2.0, "q05": 1.5, "q95": 2.6}, "g/L")
    '2 [1.5, 2.6] g/L'
    """
    low, high = (f"q{q:02d}" for q in quantiles)
    missing = [key for key in ("q50", low, high) if key not in summary]
    if missing:
        raise ValueError(f"summary is missing {missing}")
    suffix = f" {unit}" if unit else ""
    return f"{summary['q50']:.3g} [{summary[low]:.3g}, {summary[high]:.3g}]{suffix}"
