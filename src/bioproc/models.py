"""Data-driven, hybrid and soft-sensor models, each with the baseline it must beat.

Three families, in the order the review of Khuat et al. (2023),
doi:10.48550/arXiv.2310.09991, sets them out:

- **MVDA / ML** on batch descriptors: PLS, random forest, Gaussian process. These
  predict a batch KPI (titer, peak VCD) from process parameters.
- **Soft sensors** on spectra: a PLS calibration from a Raman-like spectrum to a
  concentration, the workhorse of sections 5.7 and 6 of that review.
- **Hybrid**: the mechanistic model carries the mass balances and a data-driven part
  supplies a rate it cannot predict. Here the ML part corrects the specific
  productivity, which is where clone-to-clone differences actually live.

Every fit here goes through :mod:`bioproc.evaluation`, so a score always arrives next
to its baseline and is always computed on batches the model never saw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bioproc import evaluation

MODEL_CHOICES = ("pls", "ridge", "random_forest", "gaussian_process")


def build_model(name: str, *, seed: int = 0, n_components: int = 2, n_estimators: int = 300):
    """Return an unfitted scikit-learn estimator with scaling built in.

    Raises
    ------
    ValueError
        For an unknown name.
    """
    if name not in MODEL_CHOICES:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODEL_CHOICES)}")
    if name == "pls":
        # PLS scales internally; a second scaler would be harmless but misleading.
        return PLSRegression(n_components=n_components, scale=True)
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=2)
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)),
        ]
    )


def _flatten(predictions) -> np.ndarray:
    """PLS returns a column vector; everything else returns 1-D."""
    return np.asarray(predictions, dtype=float).ravel()


@dataclass
class FittedModel:
    """A fitted estimator with its out-of-fold scores and the split that produced them."""

    name: str
    estimator: object
    features: list[str]
    target: str
    scores: dict[str, float]
    split: str
    predictions: np.ndarray
    observed: np.ndarray
    groups: np.ndarray

    def beats_baseline(self) -> bool:
        """True when the model beats the mean predictor out of fold."""
        return self.scores.get("skill_vs_mean", float("nan")) > 0

    def summary_line(self) -> str:
        verdict = "beats" if self.beats_baseline() else "does NOT beat"
        return (
            f"{self.name}: RMSE {self.scores['rmse']:.3g} vs mean-predictor "
            f"{self.scores['rmse_mean_baseline']:.3g} ({self.split}) - {verdict} the baseline"
        )


def fit_batch_model(
    table: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    model: str = "pls",
    group_column: str = "batch_id",
    seed: int = 0,
    n_components: int = 2,
) -> FittedModel:
    """Fit a KPI model with batch-aware cross-validation and baseline comparison.

    Raises
    ------
    ValueError
        For missing columns, non-finite values, or fewer samples than features (where
        every model would interpolate the training set perfectly and mean nothing).
    """
    missing = [c for c in [*features, target, group_column] if c not in table.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    frame = table[[*features, target, group_column]].dropna()
    if len(frame) <= len(features):
        raise ValueError(
            f"{len(frame)} samples for {len(features)} features: the fit is underdetermined "
            "(the Low-N problem). Reduce features or add batches."
        )
    x = frame[features].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    groups = frame[group_column].to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite values in the features or target")

    estimator = build_model(model, seed=seed, n_components=min(n_components, x.shape[1]))
    predictions, split = evaluation.cross_validated_predictions(estimator, x, y, groups)
    predictions = _flatten(predictions)
    scores = evaluation.regression_scores(y, predictions, y_train=y)
    from sklearn.base import clone

    return FittedModel(model, clone(estimator).fit(x, y), list(features), target, scores, split, predictions, y, groups)


def simulate_spectra(
    concentrations: pd.DataFrame,
    *,
    seed: int,
    n_wavenumbers: int = 300,
    noise_level: float = 0.01,
    fluorescence_drift: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate Raman-like spectra from known concentrations (a positive control).

    Each analyte gets a Gaussian band at a fixed position; the spectrum is their sum
    plus a broad fluorescence baseline and detector noise. That baseline is exactly the
    interference that makes real Raman calibration hard (Gautam et al. 2015,
    doi:10.1140/epjti/s40485-015-0018-6), and it is why the notebook compares raw
    spectra against baseline-corrected ones rather than assuming preprocessing helps.

    Returns ``(spectra, wavenumbers, truth)``.
    """
    if concentrations.empty:
        raise ValueError("no concentrations given")
    generator = np.random.default_rng(seed)
    wavenumbers = np.linspace(400.0, 1800.0, n_wavenumbers)
    analytes = list(concentrations.columns)
    centres = np.linspace(600.0, 1600.0, len(analytes))
    widths = generator.uniform(25.0, 60.0, size=len(analytes))
    strengths = generator.uniform(0.5, 1.5, size=len(analytes))

    values = concentrations.to_numpy(dtype=float)
    scale = np.where(np.ptp(values, axis=0) > 0, np.ptp(values, axis=0), 1.0)
    scaled = (values - values.min(axis=0)) / scale

    bands = np.stack([np.exp(-0.5 * ((wavenumbers - c) / w) ** 2) for c, w in zip(centres, widths, strict=True)])
    spectra = scaled @ (strengths[:, None] * bands)
    # Broad, batch-varying fluorescence background plus noise.
    slope = generator.uniform(0.0, fluorescence_drift, size=len(values))[:, None]
    baseline = slope * np.exp((wavenumbers - wavenumbers[0]) / (wavenumbers[-1] - wavenumbers[0]))[None, :]
    spectra = spectra + baseline + generator.normal(0.0, noise_level, size=spectra.shape)

    truth = {
        "analytes": analytes,
        "band_centres_cm-1": dict(zip(analytes, centres.round(1), strict=True)),
        "band_widths_cm-1": dict(zip(analytes, widths.round(1), strict=True)),
        "noise_level": noise_level,
        "fluorescence_drift": fluorescence_drift,
        "warning": "SYNTHETIC spectra. They show the calibration pipeline recovers a known answer.",
    }
    return spectra, wavenumbers, truth


def baseline_correct(spectra: np.ndarray, *, degree: int = 4, iterations: int = 20) -> np.ndarray:
    """Remove a smooth fluorescence baseline by iterative polynomial fitting (ModPoly).

    Fit a low-order polynomial, clip the spectrum down to that fit wherever it lies
    above it, and refit. The peaks are progressively excluded and the polynomial
    settles onto the background, which is what distinguishes this from a plain
    polynomial fit (that one is dragged upwards by the peaks themselves).

    A rolling-minimum envelope was tried first and made the calibration five times
    worse: the envelope tracks noise troughs, so subtracting it injects noise into
    every channel. This is why the notebook measures the effect of preprocessing
    instead of assuming it helps.

    Raises
    ------
    ValueError
        For a degree that the number of channels cannot support.
    """
    spectra = np.atleast_2d(np.asarray(spectra, dtype=float))
    if degree < 1 or degree >= spectra.shape[1]:
        raise ValueError(f"degree must be in [1, {spectra.shape[1] - 1}], got {degree}")
    x = np.linspace(-1.0, 1.0, spectra.shape[1])  # scaled, so the fit stays conditioned
    corrected = np.empty_like(spectra)
    for index, row in enumerate(spectra):
        working = row.copy()
        baseline = np.zeros_like(row)
        for _ in range(iterations):
            baseline = np.polyval(np.polyfit(x, working, degree), x)
            previous, working = working, np.minimum(working, baseline)
            if np.allclose(previous, working, rtol=1e-6, atol=1e-9):
                break
        corrected[index] = row - baseline
    return corrected


def fit_soft_sensor(
    spectra: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    n_components: int = 5,
    seed: int = 0,
) -> FittedModel:
    """Calibrate a PLS soft sensor from spectra to one concentration.

    Cross-validation is grouped by batch: a random split over time points from the same
    run leaks adjacent, nearly identical spectra into the test set.

    Raises
    ------
    ValueError
        If shapes disagree, or there are fewer samples than components.
    """
    spectra = np.atleast_2d(np.asarray(spectra, dtype=float))
    target = np.asarray(target, dtype=float).ravel()
    groups = np.asarray(groups)
    if not len(spectra) == len(target) == len(groups):
        raise ValueError("spectra, target and groups must have the same length")
    if len(spectra) <= n_components:
        raise ValueError(f"{len(spectra)} samples cannot support {n_components} PLS components")

    estimator = PLSRegression(n_components=n_components, scale=True)
    predictions, split = evaluation.cross_validated_predictions(estimator, spectra, target, groups)
    predictions = _flatten(predictions)
    scores = evaluation.regression_scores(target, predictions, y_train=target)
    scores["n_components"] = n_components
    from sklearn.base import clone

    return FittedModel(
        f"PLS soft sensor ({n_components} components)",
        clone(estimator).fit(spectra, target),
        [f"wavenumber_{i}" for i in range(spectra.shape[1])],
        "concentration",
        scores,
        split,
        predictions,
        target,
        groups,
    )


def choose_components(spectra, target, groups, candidates=(1, 2, 3, 5, 8, 12)) -> pd.DataFrame:
    """Cross-validated RMSE against the number of PLS components.

    Chosen on out-of-fold error, never on the fit: this is the threshold-tuning trap in
    ground rules section 1, and the notebook shows the whole scan rather than one cut.
    """
    rows = []
    for k in candidates:
        if k >= len(spectra):
            continue
        try:
            fitted = fit_soft_sensor(spectra, target, groups, n_components=int(k))
        except ValueError:
            continue
        rows.append(
            {
                "n_components": int(k),
                "rmse": fitted.scores["rmse"],
                "rmse_mean_baseline": fitted.scores["rmse_mean_baseline"],
                "skill_vs_mean": fitted.scores["skill_vs_mean"],
            }
        )
    if not rows:
        raise ValueError("no component count could be evaluated on this data")
    return pd.DataFrame(rows)


def hybrid_productivity_correction(
    batch_table: pd.DataFrame,
    features: list[str],
    *,
    seed: int = 0,
    group_column: str = "batch_id",
) -> FittedModel:
    """Learn a multiplier on the mechanistic specific productivity from batch features.

    This is the serial hybrid structure of Narayanan et al. (2019),
    doi:10.1002/bit.27097: mass balances stay mechanistic, the data-driven part supplies
    a rate the mechanism does not predict. The target is the ratio of the observed
    cell-specific productivity to the nominal one, so a model that learns nothing
    predicts 1.0 and the mechanistic model is recovered unchanged.

    Raises
    ------
    ValueError
        If the nominal productivity column is missing or non-positive.
    """
    required = {"qp_pg_per_cell_day", "qp_nominal_pg_per_cell_day"}
    missing = required - set(batch_table.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    if (batch_table["qp_nominal_pg_per_cell_day"] <= 0).any():
        raise ValueError("nominal productivity must be > 0 to form a ratio")
    frame = batch_table.copy()
    frame["qp_ratio"] = frame["qp_pg_per_cell_day"] / frame["qp_nominal_pg_per_cell_day"]
    return fit_batch_model(frame, features, "qp_ratio", model="random_forest", seed=seed, group_column=group_column)
