"""One configuration file per studied system (ground rules section 4).

The notebook's code is identical for every system. Everything specific to a process -
cell line, feed strategy, resin, column, acceptance criteria, and the default answers
to the notebook's questions - lives in ``configs/systems/<name>.toml``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from bioproc import downstream, upstream

REPO = Path(__file__).resolve().parents[2]
SYSTEMS_DIR = REPO / "configs" / "systems"
REQUIRED_SECTIONS = ("system", "upstream", "downstream", "acceptance")


def available_systems() -> list[str]:
    """Names of the configs on disk."""
    return sorted(path.stem for path in SYSTEMS_DIR.glob("*.toml"))


def load_system(name_or_path: str | Path) -> dict:
    """Read a system config by name (``configs/systems/<name>.toml``) or by path.

    Raises
    ------
    FileNotFoundError, ValueError
        If the file is absent or a required section is missing.
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = SYSTEMS_DIR / f"{name_or_path}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; available: {available_systems()}")
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"{path.name}: missing sections {missing}")
    config["_path"] = str(path)
    return config


def cho_parameters(config: dict, overrides: dict | None = None) -> upstream.CHOParameters:
    """Build kinetic parameters from ``[upstream.kinetics]`` plus any overrides.

    Raises
    ------
    ValueError
        If a key is not a parameter of the model (a typo must not pass silently).
    """
    values = dict(config.get("upstream", {}).get("kinetics", {}))
    values.update(overrides or {})
    known = set(upstream.CHOParameters().__dict__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown kinetic parameters {sorted(unknown)}; known: {sorted(known)}")
    return upstream.CHOParameters(**values)


def initial_state(config: dict, volume_L: float | None = None) -> dict[str, float]:
    """Initial culture state from ``[upstream.initial]``.

    Raises
    ------
    ValueError
        If a state is missing from the config.
    """
    state = dict(config["upstream"]["initial"])
    if volume_L is not None:
        state["volume"] = float(volume_L)
    # Cumulative states (glucose consumed so far) start at zero and are not something
    # a config declares, so they are not required here.
    required = [key for key in upstream.STATES if key not in upstream.CUMULATIVE_STATES]
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"[upstream.initial] is missing {missing}")
    return {key: float(state[key]) for key in required}


def feed_policy(config: dict, strategy: str, target_glc_mM: float | None = None):
    """Build the feed policy named by ``strategy`` from ``[upstream.feed]``.

    Raises
    ------
    ValueError
        For an unknown strategy.
    """
    feed = config["upstream"]["feed"]
    if strategy == "glucose_setpoint":
        return upstream.glucose_setpoint_policy(
            target_glc_mM=float(target_glc_mM if target_glc_mM is not None else feed["target_glc_mM"]),
            feed_glc_mM=float(feed["glc_mM"]),
            feed_gln_mM=float(feed.get("gln_mM", 0.0)),
        )
    if strategy == "fixed_bolus":
        return upstream.fixed_bolus_policy(
            fraction_of_volume=float(feed["bolus_fraction_of_volume"]),
            feed_glc_mM=float(feed["glc_mM"]),
            feed_gln_mM=float(feed.get("gln_mM", 0.0)),
        )
    if strategy == "none":
        return None
    raise ValueError(f"unknown feed strategy {strategy!r}; use glucose_setpoint, fixed_bolus or none")


def capture_column(
    config: dict, diameter_cm: float | None = None, bed_height_cm: float | None = None
) -> downstream.Column:
    """Column geometry from ``[downstream.capture.column]``."""
    section = config["downstream"]["capture"]["column"]
    return downstream.Column(
        diameter_cm=float(diameter_cm if diameter_cm is not None else section["diameter_cm"]),
        bed_height_cm=float(bed_height_cm if bed_height_cm is not None else section["bed_height_cm"]),
        porosity_total=float(section.get("porosity_total", 0.85)),
        porosity_interstitial=float(section.get("porosity_interstitial", 0.35)),
    )


def capture_resin(config: dict) -> downstream.LangmuirResin:
    """Resin parameters from ``[downstream.capture.resin]``."""
    section = config["downstream"]["capture"]["resin"]
    return downstream.LangmuirResin(
        q_max_g_per_L_resin=float(section["q_max_g_per_L_resin"]),
        k_eq_L_per_g=float(section["k_eq_L_per_g"]),
        k_mass_transfer_per_min=float(section["k_mass_transfer_per_min"]),
        name=str(section.get("name", "Protein A")),
    )


def answers(config: dict) -> dict:
    """Default answers to the notebook's questions, from ``[answers]``."""
    return dict(config.get("answers", {}))


def acceptance_criteria(config: dict) -> dict:
    """Release criteria from ``[acceptance]``: what a batch has to meet."""
    return dict(config["acceptance"])
