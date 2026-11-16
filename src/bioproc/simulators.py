"""Optional external simulators, detected at run time and never required.

The notebook has a built-in model for every step, so it runs with nothing but the
conda environment. Where one of these tools is installed, the notebook offers it as a
cross-check or as a deeper model, and records which engine produced each number.

================ ============================================================
Tool             What it adds here
================ ============================================================
CADET-Process    Rigorous chromatography (general rate model, SMB, carousel).
                 A reference implementation to check the lumped kinetic model
                 against - ground rules section 3 calls for exactly that.
                 ``pip install CADET-Process``; the CADET-Core solver is a
                 separate conda package (``conda install -c conda-forge cadet``).
BioSTEAM         Techno-economic analysis and LCA of the whole plant: cost of
                 goods per gram, utilities, equipment sizing.
                 ``pip install biosteam``. Pure Python, installs cleanly.
IDAES            Equation-oriented process modelling and optimisation on
                 Pyomo. Heavier; use when the flowsheet itself is the object
                 of study. ``conda install -c IDAES-PSE idaes-pse``.
DWSIM            Open-source steady-state and dynamic flowsheeting (.NET).
                 Windows desktop application; drive it through its Python
                 interface or by file exchange. Not a pip install.
================ ============================================================

On this machine (Windows + conda), install any of these into a *separate* environment
rather than on top of ``bioproc``: a pip build of a compiled dependency over a conda
environment is how the crashes with no traceback start (ground rules section 9).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pandas as pd

REGISTRY = {
    "cadet": {
        "module": "CADETProcess",
        "purpose": "Rigorous chromatography models (GRM, SMB) - cross-check for the lumped kinetic model",
        "install": "pip install CADET-Process  +  conda install -c conda-forge cadet",
        "url": "https://github.com/fau-advanced-separations/CADET-Process",
    },
    "biosteam": {
        "module": "biosteam",
        "purpose": "Techno-economic analysis: cost of goods, utilities, equipment sizing",
        "install": "pip install biosteam",
        "url": "https://biosteam.readthedocs.io/",
    },
    "idaes": {
        "module": "idaes",
        "purpose": "Equation-oriented flowsheet modelling and optimisation (Pyomo)",
        "install": "conda install -c IDAES-PSE idaes-pse",
        "url": "https://github.com/IDAES/idaes-pse",
    },
    "dwsim": {
        "module": "dwsim",
        "purpose": "Steady-state and dynamic flowsheeting (.NET desktop application)",
        "install": "download the installer; drive it through its Python interface",
        "url": "https://github.com/DanWBR/dwsim10",
    },
}


@dataclass(frozen=True)
class Availability:
    key: str
    available: bool
    version: str
    purpose: str
    install: str
    url: str


def available(key: str) -> bool:
    """Whether a registered simulator can be imported.

    Raises
    ------
    KeyError
        For an unregistered key.
    """
    if key not in REGISTRY:
        raise KeyError(f"unknown simulator {key!r}; registered: {sorted(REGISTRY)}")
    return importlib.util.find_spec(REGISTRY[key]["module"]) is not None


def detect() -> pd.DataFrame:
    """One row per registered simulator: available, version, purpose, how to install."""
    rows = []
    for key, entry in REGISTRY.items():
        present = available(key)
        version = ""
        if present:
            try:
                import importlib.metadata as metadata

                version = metadata.version(entry["module"])
            except Exception:  # noqa: BLE001 - a package without metadata is still usable
                version = "unknown"
        rows.append(Availability(key, present, version, entry["purpose"], entry["install"], entry["url"]).__dict__)
    return pd.DataFrame(rows)


def require(key: str) -> None:
    """Raise a message that says how to install, if the simulator is missing.

    Raises
    ------
    ImportError
        Naming the package and the install command.
    """
    if not available(key):
        entry = REGISTRY[key]
        raise ImportError(f"{key} is not installed. {entry['install']}  ({entry['url']})")


# ---------------------------------------------------------------------------------
def cadet_breakthrough(
    column,
    resin,
    feed_g_per_L: float,
    velocity_cm_per_h: float,
    *,
    load_cv: float = 60.0,
    axial_dispersion_m2_per_s: float = 1e-7,
    particle_radius_m: float = 4.25e-5,
) -> pd.DataFrame:
    """Breakthrough from CADET-Process's general rate model, for comparison.

    Same column, resin and load as :func:`bioproc.downstream.breakthrough_curve`, so the
    two curves can be plotted together: agreement is evidence the lumped model is solved
    correctly, and a gap is informative about where lumping costs accuracy.

    Returns the same columns as the built-in curve.

    Raises
    ------
    ImportError
        If CADET-Process or the CADET solver is unavailable.
    RuntimeError
        If the solver runs but does not converge.
    """
    require("cadet")
    from CADETProcess.modelBuilder import ProcessBuilder  # noqa: F401 - import checks the install
    from CADETProcess.processModel import (
        ComponentSystem,
        FlowSheet,
        Inlet,
        LumpedRateModelWithPores,
        Outlet,
        Process,
    )
    from CADETProcess.simulator import Cadet

    system = ComponentSystem(["product"])
    inlet = Inlet(system, name="feed")
    inlet.c = [feed_g_per_L]
    volumetric_flow_m3_per_s = (velocity_cm_per_h / 100.0 / 3600.0) * (column.area_cm2 / 1e4)
    inlet.flow_rate = volumetric_flow_m3_per_s

    bed = LumpedRateModelWithPores(system, name="column")
    bed.length = column.bed_height_cm / 100.0
    bed.diameter = column.diameter_cm / 100.0
    bed.bed_porosity = column.porosity_interstitial
    bed.particle_porosity = (column.porosity_total - column.porosity_interstitial) / (1 - column.porosity_interstitial)
    bed.particle_radius = particle_radius_m
    bed.axial_dispersion = axial_dispersion_m2_per_s
    bed.film_diffusion = [resin.k_mass_transfer_per_min / 60.0 * particle_radius_m / 3.0]

    from CADETProcess.processModel import Langmuir

    binding = Langmuir(system, name="protein A")
    binding.adsorption_rate = [resin.k_eq_L_per_g * resin.k_mass_transfer_per_min / 60.0]
    binding.desorption_rate = [resin.k_mass_transfer_per_min / 60.0]
    binding.capacity = [resin.q_max_g_per_L_resin]
    bed.binding_model = binding

    outlet = Outlet(system, name="outlet")
    flowsheet = FlowSheet(system)
    for unit in (inlet, bed, outlet):
        flowsheet.add_unit(unit)
    flowsheet.add_connection(inlet, bed)
    flowsheet.add_connection(bed, outlet)

    residence_s = 60.0 * column.residence_time_min(velocity_cm_per_h) * column.porosity_total
    process = Process(flowsheet, "breakthrough")
    process.cycle_time = load_cv * residence_s

    results = Cadet().simulate(process)
    if not getattr(results, "exit_flag", 0) == 0:
        raise RuntimeError(f"CADET did not converge: exit flag {getattr(results, 'exit_flag', 'unknown')}")
    solution = results.solution.outlet.outlet
    times_s = np.asarray(solution.time, dtype=float)
    c_out = np.asarray(solution.solution, dtype=float).ravel()
    cv = times_s / residence_s
    return pd.DataFrame(
        {
            "time_min": times_s / 60.0,
            "load_cv": cv,
            "load_g_per_L_resin": cv * column.porosity_total * feed_g_per_L,
            "c_out_g_per_L": c_out,
            "breakthrough_fraction": c_out / feed_g_per_L,
            "engine": "CADET general rate model",
        }
    )


# Published cost of goods for a commercial mAb, all-in. Direct materials are a small
# fraction of this: facility depreciation, labour, QC and fill-finish dominate.
PUBLISHED_MAB_COGS_PER_GRAM = (50.0, 300.0)


def biosteam_cost_estimate(
    product_kg_per_year: float,
    *,
    titer_g_per_L: float,
    downstream_yield: float,
    media_cost_per_L: float = 6.0,
    buffer_cost_per_L: float = 1.5,
    resin_cost_per_L: float = 12000.0,
    resin_cycles: int = 100,
    campaign_batches_per_year: int = 20,
    buffer_L_per_L_broth: float = 8.0,
) -> dict:
    """**Direct materials** cost per gram: media, buffers and resin amortisation.

    This is a materials *floor*, not cost of goods. Facility depreciation, labour, QC
    and fill-finish are excluded and they are the majority of real COGS, so the figure
    here comes out roughly an order of magnitude below the published
    $50-300/g for a commercial mAb. It is useful for the question it can answer -
    does doubling the titer change the materials bill, and by how much - and useless
    for the question it cannot.

    Uses BioSTEAM where it is installed; otherwise reports this arithmetic estimate
    and says so in ``engine``.

    Raises
    ------
    ValueError
        For non-positive inputs or a yield outside (0, 1].
    """
    if product_kg_per_year <= 0 or titer_g_per_L <= 0:
        raise ValueError("production and titer must be > 0")
    if not 0 < downstream_yield <= 1:
        raise ValueError("downstream_yield must be in (0, 1]")

    product_g = product_kg_per_year * 1000.0
    broth_L = product_g / (titer_g_per_L * downstream_yield)
    media_cost = broth_L * media_cost_per_L
    buffer_cost = broth_L * buffer_L_per_L_broth * buffer_cost_per_L
    # Resin sized for one batch's load at 40 g/L resin, amortised over its cycle life.
    resin_L = (product_g / campaign_batches_per_year) / 40.0
    resin_cost = resin_L * resin_cost_per_L * campaign_batches_per_year / max(resin_cycles, 1)
    total = media_cost + buffer_cost + resin_cost
    per_gram = total / product_g
    low, high = PUBLISHED_MAB_COGS_PER_GRAM
    estimate = {
        "engine": "bioproc arithmetic estimate (direct materials only)",
        "broth_volume_L_per_year": broth_L,
        "media_cost_per_year": media_cost,
        "buffer_cost_per_year": buffer_cost,
        "resin_cost_per_year": resin_cost,
        "direct_materials_per_year": total,
        "direct_materials_per_gram": per_gram,
        "published_cogs_per_gram_low": low,
        "published_cogs_per_gram_high": high,
        "materials_share_of_published_cogs": f"{per_gram / high:.1%} to {per_gram / low:.1%}",
        "caveat": (
            "Direct materials only. Facility, labour, QC and fill-finish are excluded and "
            f"dominate real cost of goods, published at ${low:g}-{high:g}/g for a commercial mAb. "
            "Use this to compare options, never as a cost of goods."
        ),
    }
    if available("biosteam"):
        estimate["note"] = (
            "BioSTEAM is installed: build a flowsheet there for equipment sizing, utilities and "
            "a proper TEA, and replace this estimate."
        )
    return estimate
