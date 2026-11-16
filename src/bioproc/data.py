"""Where the data comes from: a synthetic campaign, a user file, or an open dataset.

Three sources, and the notebook always states which one it used.

1. **Synthetic campaign** (:func:`simulate_campaign`). Batches are generated from the
   mechanistic model with a stated ground truth: a known clone effect, a known
   feed-strategy effect, known measurement noise. It is a positive control - it shows
   that the analysis recovers an answer that is known. It is not evidence about any
   real process, and every figure drawn from it says so.
2. **Your own export** (:func:`read_batch_table`). A long or wide CSV/Excel of batch
   runs. Column names are mapped explicitly; nothing is guessed from magnitude.
3. **An open dataset** (:func:`fetch_open_dataset`). Downloaded through ``pooch`` with
   a checksum and the access date recorded. The one registered here is the
   anonymised, 0-1 normalised CHO upstream set published with Gangadharan et al.
   (2021), doi:10.1016/j.compchemeng.2021.107226 - 106 cultures, 5 L to 500 L.

Because that dataset is normalised, it carries no units. Section 2 of the notebook
refuses to attach units to it, and the models trained on it report normalised errors
only (ground rules section 2: read units from the source, never infer them).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from bioproc import upstream
from bioproc.ui import data_directory

# Registered open datasets: url, sha256 (None until verified on first download),
# and what the file actually contains.
OPEN_DATASETS = {
    "gangadharan2021_cho": {
        "url": "https://ars.els-cdn.com/content/image/1-s2.0-S0098135421000041-mmc3.xlsx",
        "sha256": None,
        "citation": "Gangadharan et al. (2021), Comput Chem Eng 146:107226, doi:10.1016/j.compchemeng.2021.107226",
        "licence": "Elsevier supplementary material; check the publisher's terms before redistributing",
        "normalised": True,
        "note": (
            "106 CHO fed-batch cultures, 5 L to 500 L, up to 17 days, 25 parameters. "
            "Anonymised and min-max normalised to 0-1 by the authors, so it carries no "
            "units and no absolute concentrations."
        ),
    }
}

CLONE_EFFECTS = {
    # Multipliers on the default parameters. The ground truth the synthetic analysis
    # must recover: clone B grows slower but is more productive per cell.
    "A": {"mu_max_per_h": 1.00, "qp_beta_mg_per_1e9cells_h": 1.00},
    "B": {"mu_max_per_h": 0.88, "qp_beta_mg_per_1e9cells_h": 1.45},
    "C": {"mu_max_per_h": 1.06, "qp_beta_mg_per_1e9cells_h": 0.80},
}


def simulate_campaign(
    n_batches: int,
    *,
    seed: int,
    clones=("A", "B"),
    feed_strategies=("glucose_setpoint", "fixed_bolus"),
    scales_L=(5.0, 50.0, 500.0),
    duration_h: float = 14 * 24,
    sample_interval_h: float = 24.0,
    shift_time_h: float = 72.0,
    lot_variation_cv: float = 0.08,
    measurement_noise: dict[str, float] | None = None,
    parameters: upstream.CHOParameters | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate a campaign of fed-batch runs with a known ground truth.

    Batch-to-batch variation is a lognormal multiplier (``lot_variation_cv``) on the
    growth and productivity parameters, standing in for lot-to-lot raw-material
    variation. Measurement noise is multiplicative and is applied to the samples only,
    never to the states the model integrated.

    Returns ``(samples, batch_summary, truth)``: one row per sample point, one row per
    batch, and the dictionary of what was actually imposed.

    Raises
    ------
    ValueError
        For a non-positive batch count or an unknown clone.
    """
    if n_batches < 1:
        raise ValueError("n_batches must be >= 1")
    unknown = set(clones) - set(CLONE_EFFECTS)
    if unknown:
        raise ValueError(f"unknown clones {sorted(unknown)}; known: {sorted(CLONE_EFFECTS)}")

    noise = {
        "vcd_1e6_per_mL": 0.07,
        "glucose_mM": 0.04,
        "glutamine_mM": 0.06,
        "lactate_mM": 0.05,
        "ammonium_mM": 0.08,
        "titer_mg_per_L": 0.06,
    }
    noise.update(measurement_noise or {})
    base = parameters or upstream.CHOParameters()
    generator = np.random.default_rng(seed)

    sample_rows, batch_rows = [], []
    for index in range(n_batches):
        clone = str(clones[index % len(clones)])
        strategy = str(feed_strategies[index % len(feed_strategies)])
        scale_L = float(scales_L[index % len(scales_L)])
        # Lognormal with the requested CV, median 1: multiplies, never goes negative.
        sigma = np.sqrt(np.log(1.0 + lot_variation_cv**2))
        lot = {
            name: float(generator.lognormal(-0.5 * sigma**2, sigma))
            for name in ("mu_max_per_h", "qp_beta_mg_per_1e9cells_h", "kd_amm_max_per_h")
        }
        effects = CLONE_EFFECTS[clone]
        parameters_batch = upstream.CHOParameters(
            **{
                **{k: v for k, v in base.__dict__.items()},
                "mu_max_per_h": base.mu_max_per_h * effects["mu_max_per_h"] * lot["mu_max_per_h"],
                "qp_beta_mg_per_1e9cells_h": (
                    base.qp_beta_mg_per_1e9cells_h
                    * effects["qp_beta_mg_per_1e9cells_h"]
                    * lot["qp_beta_mg_per_1e9cells_h"]
                ),
                "kd_amm_max_per_h": base.kd_amm_max_per_h * lot["kd_amm_max_per_h"],
            }
        )

        seed_density = float(generator.normal(0.4, 0.03))
        initial = {
            "xv": max(seed_density, 0.1),
            "xd": 0.02,
            "glc": 25.0,
            "gln": 4.0,
            "lac": 0.5,
            "amm": 0.5,
            "titer": 0.0,
            "volume": scale_L,
        }
        if strategy == "glucose_setpoint":
            policy = upstream.glucose_setpoint_policy(target_glc_mM=12.0, feed_glc_mM=250.0, feed_gln_mM=30.0)
        elif strategy == "fixed_bolus":
            policy = upstream.fixed_bolus_policy(fraction_of_volume=0.035, feed_glc_mM=250.0, feed_gln_mM=30.0)
        else:
            raise ValueError(f"unknown feed strategy {strategy!r}")

        history = upstream.simulate(
            initial,
            parameters_batch,
            duration_h=duration_h,
            sample_interval_h=sample_interval_h,
            feed_policy=policy,
            shift_time_h=shift_time_h,
        )
        batch_id = f"B{index + 1:03d}"
        history.insert(0, "batch_id", batch_id)
        history["clone"] = clone
        history["feed_strategy"] = strategy
        history["scale_L"] = scale_L

        # Samples: what the analyser reports, with multiplicative noise.
        for column, state in upstream.SAMPLE_COLUMNS.items():
            factor = generator.normal(1.0, noise.get(column, 0.0), size=len(history))
            history[column] = np.maximum(history[state] * factor, 0.0)

        metrics = upstream.harvest_metrics(history)
        sample_rows.append(history)
        batch_rows.append(
            {
                "batch_id": batch_id,
                "clone": clone,
                "feed_strategy": strategy,
                "scale_L": scale_L,
                "seed_density_1e6_per_mL": initial["xv"],
                "mu_max_per_h": parameters_batch.mu_max_per_h,
                "qp_beta_mg_per_1e9cells_h": parameters_batch.qp_beta_mg_per_1e9cells_h,
                **metrics,
            }
        )

    truth = {
        "generator": "bioproc.data.simulate_campaign",
        "seed": seed,
        "clone_effects": {clone: CLONE_EFFECTS[clone] for clone in clones},
        "lot_variation_cv": lot_variation_cv,
        "measurement_noise_cv": noise,
        "shift_time_h": shift_time_h,
        "base_parameters": dict(base.__dict__),
        "warning": "SYNTHETIC. A positive control for the analysis, not evidence about any real process.",
    }
    return pd.concat(sample_rows, ignore_index=True), pd.DataFrame(batch_rows), truth


# ---------------------------------------------------------------------------------
REQUIRED_SAMPLE_COLUMNS = ("batch_id", "time_h")


def read_batch_table(
    path: str | Path,
    column_map: dict[str, str],
    *,
    sheet: str | int = 0,
    units_declared: bool = False,
) -> pd.DataFrame:
    """Read a batch export and rename its columns explicitly.

    ``column_map`` maps *your* column names to the canonical names
    (``batch_id``, ``time_h``, ``vcd_1e6_per_mL``, ``glucose_mM``, ...). Nothing is
    guessed: a column that is not mapped is kept under its original name and is not
    treated as a known quantity.

    ``units_declared`` must be set by the caller once they have confirmed the file's
    units against its own documentation; it is recorded on the frame's ``attrs``.

    Raises
    ------
    FileNotFoundError, ValueError
        Missing file, unsupported suffix, missing required column, or a mapping that
        points at a column the file does not have.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name=sheet)
    else:
        raise ValueError(f"unsupported file type {path.suffix!r}; use .csv or .xlsx")

    absent = [source for source in column_map if source not in frame.columns]
    if absent:
        raise ValueError(f"{path.name}: mapped columns not in the file: {absent}. Found: {list(frame.columns)[:20]}")
    frame = frame.rename(columns=column_map)
    missing = [c for c in REQUIRED_SAMPLE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: still missing required columns {missing} after mapping")

    frame.attrs.update(
        source=str(path),
        source_sha256=_sha256(path),
        read_on=dt.date.today().isoformat(),
        units_declared=bool(units_declared),
        column_map=dict(column_map),
    )
    return frame.sort_values(list(REQUIRED_SAMPLE_COLUMNS)).reset_index(drop=True)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_open_dataset(name: str, *, known_hash: str | None = None) -> tuple[Path, dict]:
    """Download a registered open dataset through ``pooch`` into the data directory.

    Returns the cached path and the registry entry (citation, licence, caveats). The
    first download prints the file's hash so it can be pinned in :data:`OPEN_DATASETS`.

    Raises
    ------
    KeyError
        For an unregistered name.
    RuntimeError
        If the download fails; the registry entry's url is given so it can be fetched
        by hand (ground rules section 7: external endpoints are unreliable).
    """
    if name not in OPEN_DATASETS:
        raise KeyError(f"unknown dataset {name!r}; registered: {sorted(OPEN_DATASETS)}")
    entry = OPEN_DATASETS[name]
    import pooch

    target = data_directory() / "open-datasets"
    target.mkdir(parents=True, exist_ok=True)
    try:
        path = Path(
            pooch.retrieve(
                url=entry["url"],
                known_hash=known_hash or entry["sha256"],
                fname=f"{name}{Path(entry['url']).suffix}",
                path=target,
            )
        )
    except Exception as error:  # noqa: BLE001 - any network/hash failure is reported the same way
        raise RuntimeError(
            f"could not fetch {name}: {error}. Download {entry['url']} by hand into {target} and pass the path instead."
        ) from error
    record = dict(entry)
    record.update(path=str(path), sha256_observed=_sha256(path), accessed=dt.date.today().isoformat())
    return path, record


def missing_value_report(frame: pd.DataFrame, group_column: str = "batch_id") -> pd.DataFrame:
    """Per-column missingness overall and in the worst batch.

    Bioprocess data is missing-at-random by column and day (Mante et al. 2019,
    doi:10.1007/s00449-018-02059-5), so a column that looks 5% missing overall can be
    absent from a whole batch. That is the number that decides whether imputation or
    dropping is honest.
    """
    if group_column not in frame.columns:
        raise ValueError(f"{group_column!r} is not a column")
    overall = frame.isna().mean()
    per_batch = frame.groupby(group_column).apply(lambda g: g.isna().mean(), include_groups=False)
    report = pd.DataFrame(
        {
            "missing_fraction": overall,
            "worst_batch_missing_fraction": per_batch.max(),
            "batches_fully_missing": (per_batch == 1.0).sum(),
        }
    ).drop(index=[group_column], errors="ignore")
    return report.sort_values("missing_fraction", ascending=False)
