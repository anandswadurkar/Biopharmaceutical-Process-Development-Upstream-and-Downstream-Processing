"""Configs, campaign generation and the open-data loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioproc import config as cfg
from bioproc import data, quality, upstream


class TestConfigs:
    def test_every_shipped_config_loads(self):
        names = cfg.available_systems()
        assert names, "no configs found"
        for name in names:
            config = cfg.load_system(name)
            assert config["system"]["name"]
            assert config["system"]["seed"]

    def test_every_config_builds_a_runnable_system(self):
        """A config has to produce parameters, a state and a policy that actually run."""
        for name in cfg.available_systems():
            config = cfg.load_system(name)
            state = cfg.initial_state(config)
            parameters = cfg.cho_parameters(config)
            policy = cfg.feed_policy(config, "glucose_setpoint")
            history = upstream.simulate(state, parameters, duration_h=48, sample_interval_h=24, feed_policy=policy)
            assert len(history) == 3, name
            assert history["titer"].iloc[-1] > 0, f"{name} produced no product"

    def test_initial_state_does_not_require_cumulative_states(self):
        """Cumulative totals start at zero; a config must not have to declare them."""
        config = cfg.load_system("demo_mab")
        state = cfg.initial_state(config)
        for cumulative in upstream.CUMULATIVE_STATES:
            assert cumulative not in state
        assert upstream.simulate(state, cfg.cho_parameters(config), duration_h=24, sample_interval_h=24) is not None

    def test_volume_override(self):
        config = cfg.load_system("demo_mab")
        assert cfg.initial_state(config, volume_L=7.5)["volume"] == 7.5

    def test_unknown_system(self):
        with pytest.raises(FileNotFoundError, match="available"):
            cfg.load_system("no_such_system")

    def test_missing_section(self, tmp_path):
        path = tmp_path / "broken.toml"
        path.write_text('[system]\nname = "x"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="missing sections"):
            cfg.load_system(path)

    def test_typo_in_a_kinetic_parameter_is_rejected(self):
        """A mistyped parameter must not be silently ignored."""
        config = cfg.load_system("demo_mab")
        with pytest.raises(ValueError, match="unknown kinetic parameters"):
            cfg.cho_parameters(config, overrides={"mu_max_per_hour": 0.05})

    def test_unknown_feed_strategy(self):
        config = cfg.load_system("demo_mab")
        with pytest.raises(ValueError, match="unknown feed strategy"):
            cfg.feed_policy(config, "osmotic")

    def test_acceptance_criteria_are_well_formed(self):
        for name in cfg.available_systems():
            config = cfg.load_system(name)
            criteria = quality.criteria_from_config(cfg.acceptance_criteria(config))
            assert criteria, name
            for criterion in criteria:
                assert criterion.unit, f"{name}: {criterion.name} has no unit"

    def test_answers_cover_every_question_the_notebook_asks(self):
        """Headless execution must never hit a question with no default.

        The keys are read out of the generator source, so adding a question without a
        default in both configs fails here rather than halfway through a CI run.
        """
        import re
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(cfg.REPO) / "tools"))
        import make_notebook

        asked = set()
        for system in make_notebook.SYSTEMS:
            for kind, source in make_notebook.build_cells(system):
                if kind != "code":
                    continue
                asked |= set(re.findall(r'ask\.ask(?:_path)?\(\s*"([a-z0-9_]+)"', source))

        # Questions that legitimately have no config default: they take one inline, or
        # are only reached on a branch the shipped configs do not take.
        inline_defaults = {"strategy_replicates", "input_cv", "batch_file", "units_declared"}
        inline_defaults |= {key for key in asked if key.startswith("map_")}

        for name in cfg.available_systems():
            answers = cfg.answers(cfg.load_system(name))
            missing = sorted(asked - set(answers) - inline_defaults)
            assert not missing, f"{name}.toml [answers] is missing defaults for {missing}"


class TestCampaign:
    def test_shape_and_grouping(self):
        samples, batches, truth = data.simulate_campaign(4, seed=1, duration_h=120, sample_interval_h=24)
        assert batches.shape[0] == 4
        assert samples["batch_id"].nunique() == 4
        assert set(upstream.SAMPLE_COLUMNS) <= set(samples.columns)
        assert "SYNTHETIC" in truth["warning"]

    def test_reproducible_from_the_seed(self):
        first, _, _ = data.simulate_campaign(3, seed=7, duration_h=72)
        second, _, _ = data.simulate_campaign(3, seed=7, duration_h=72)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_differ(self):
        first, _, _ = data.simulate_campaign(3, seed=1, duration_h=72)
        second, _, _ = data.simulate_campaign(3, seed=2, duration_h=72)
        assert not first["titer"].equals(second["titer"])

    def test_the_known_clone_effect_is_recoverable(self):
        """The ground truth: clone B is more productive per cell than clone A.

        This is what makes the synthetic data a positive control rather than noise.
        """
        _, batches, _ = data.simulate_campaign(16, seed=3, clones=("A", "B"), duration_h=14 * 24)
        by_clone = batches.groupby("clone")["qp_pg_per_cell_day"].mean()
        assert by_clone["B"] > by_clone["A"], f"imposed effect not recovered: {by_clone.to_dict()}"

    def test_measurement_noise_perturbs_samples_not_states(self):
        samples, _, _ = data.simulate_campaign(2, seed=5, duration_h=72, measurement_noise={"glucose_mM": 0.2})
        assert not np.allclose(samples["glucose_mM"], samples["glc"])
        assert (samples["glucose_mM"] >= 0).all(), "a noisy measurement must not go negative"

    def test_noise_free_samples_match_the_states(self):
        samples, _, _ = data.simulate_campaign(
            2, seed=5, duration_h=72, measurement_noise=dict.fromkeys(upstream.SAMPLE_COLUMNS, 0.0)
        )
        np.testing.assert_allclose(samples["glucose_mM"], samples["glc"], rtol=1e-12)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="n_batches must be >= 1"):
            data.simulate_campaign(0, seed=1)
        with pytest.raises(ValueError, match="unknown clones"):
            data.simulate_campaign(2, seed=1, clones=("Z",))
        with pytest.raises(ValueError, match="unknown feed strategy"):
            data.simulate_campaign(2, seed=1, feed_strategies=("osmotic",))


class TestReadBatchTable:
    def test_round_trip(self, tmp_path):
        frame = pd.DataFrame({"Run": ["a", "a", "b"], "Hours": [0, 24, 0], "VCD": [0.4, 2.0, 0.5]})
        path = tmp_path / "export.csv"
        frame.to_csv(path, index=False)
        result = data.read_batch_table(
            path, {"Run": "batch_id", "Hours": "time_h", "VCD": "vcd_1e6_per_mL"}, units_declared=True
        )
        assert list(result.columns) == ["batch_id", "time_h", "vcd_1e6_per_mL"]
        assert result.attrs["units_declared"] is True
        assert len(result.attrs["source_sha256"]) == 64

    def test_units_default_to_undeclared(self, tmp_path):
        path = tmp_path / "export.csv"
        pd.DataFrame({"Run": ["a"], "Hours": [0]}).to_csv(path, index=False)
        result = data.read_batch_table(path, {"Run": "batch_id", "Hours": "time_h"})
        assert result.attrs["units_declared"] is False, "units must never be assumed"

    def test_mapping_a_column_that_is_not_there(self, tmp_path):
        path = tmp_path / "export.csv"
        pd.DataFrame({"Run": ["a"], "Hours": [0]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="mapped columns not in the file"):
            data.read_batch_table(path, {"Batch": "batch_id"})

    def test_required_columns_enforced(self, tmp_path):
        path = tmp_path / "export.csv"
        pd.DataFrame({"VCD": [1.0]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            data.read_batch_table(path, {"VCD": "vcd_1e6_per_mL"})

    def test_unsupported_suffix(self, tmp_path):
        path = tmp_path / "export.parquet"
        path.write_bytes(b"x")
        with pytest.raises(ValueError, match="unsupported file type"):
            data.read_batch_table(path, {})

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            data.read_batch_table(tmp_path / "absent.csv", {})


class TestMissingValues:
    def test_reports_a_fully_missing_batch(self):
        frame = pd.DataFrame(
            {
                "batch_id": ["a", "a", "b", "b"],
                "glucose_mM": [5.0, 6.0, np.nan, np.nan],
            }
        )
        report = data.missing_value_report(frame)
        assert report.loc["glucose_mM", "missing_fraction"] == 0.5
        assert report.loc["glucose_mM", "worst_batch_missing_fraction"] == 1.0
        assert report.loc["glucose_mM", "batches_fully_missing"] == 1

    def test_unknown_group_column(self):
        with pytest.raises(ValueError, match="is not a column"):
            data.missing_value_report(pd.DataFrame({"x": [1]}), group_column="batch_id")


class TestOpenDataset:
    def test_unregistered_name(self):
        with pytest.raises(KeyError, match="unknown dataset"):
            data.fetch_open_dataset("not_a_dataset")

    def test_registry_entries_are_documented(self):
        for name, entry in data.OPEN_DATASETS.items():
            assert entry["citation"], name
            assert entry["licence"], name
            assert entry["url"].startswith("http"), name


class TestNotebookPreflight:
    """The first cell must diagnose a wrong kernel, which is the usual first failure."""

    @staticmethod
    def _preflight_source() -> str:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(cfg.REPO) / "tools"))
        import make_notebook

        cells = make_notebook.build_cells(make_notebook.SYSTEMS[0])
        code_cells = [source for kind, source in cells if kind == "code"]
        return code_cells[0]

    def test_preflight_runs_before_any_other_import(self):
        source = self._preflight_source()
        assert "import bioproc" in source
        assert "Preflight" in source, "the first code cell is no longer the kernel check"

    def test_preflight_names_the_kernel_and_the_recovery(self):
        source = self._preflight_source()
        for expected in [
            "Python (bioproc)",
            "Select Another Kernel",
            "Change Kernel",
            "conda env create",
            "install_kernel.py",
        ]:
            assert expected in source, f"the kernel check does not mention {expected!r}"

    def test_preflight_explains_why_it_does_not_self_install(self):
        """Silently installing into the wrong env would change the numeric stack."""
        source = self._preflight_source()
        assert "pinned" in source or "not done for you" in source

    def test_preflight_message_fires_when_the_package_is_absent(self, tmp_path, monkeypatch):
        """Execute the cell with bioproc hidden and check the message it produces."""
        import builtins

        source = self._preflight_source()
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "bioproc" or name.startswith("bioproc."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(ModuleNotFoundError) as caught:
            exec(compile(source, "<preflight>", "exec"), {"__name__": "__preflight__"})
        message = str(caught.value)
        assert "Python (bioproc)" in message
        assert "does not have bioproc" in message
