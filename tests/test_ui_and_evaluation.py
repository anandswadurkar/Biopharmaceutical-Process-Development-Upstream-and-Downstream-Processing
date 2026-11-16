"""The interaction layer and the scoring layer: the two things the notebook leans on."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bioproc import evaluation, quality, ui


class TestParsing:
    def test_numbers_and_ranges(self):
        assert ui.parse_value("2.5", float) == 2.5
        assert ui.parse_value("7", int) == 7
        with pytest.raises(ValueError, match="whole number"):
            ui.parse_value("7.5", int)
        with pytest.raises(ValueError, match="below the minimum"):
            ui.parse_value("1", float, minimum=5)
        with pytest.raises(ValueError, match="above the maximum"):
            ui.parse_value("9", float, maximum=5)
        with pytest.raises(ValueError, match="expected a number"):
            ui.parse_value("abc", float)

    def test_choices_by_name_and_number(self):
        assert ui.parse_value("rf", str, choices=["pls", "rf"]) == "rf"
        assert ui.parse_value("1", str, choices=["pls", "rf"]) == "pls"
        with pytest.raises(ValueError, match="is not one of"):
            ui.parse_value("xgboost", str, choices=["pls", "rf"])

    def test_numeric_choice_prefers_the_literal_option(self):
        """When the options are themselves numbers, '64' means 64, not 'the 64th'."""
        assert ui.parse_value("64", str, choices=["16", "32", "64"]) == "64"

    def test_multiple(self):
        assert ui.parse_value("a, c", str, choices=["a", "b", "c"], multiple=True) == ["a", "c"]
        assert ui.parse_value(["a", "b"], str, choices=["a", "b"], multiple=True) == ["a", "b"]
        with pytest.raises(ValueError, match="duplicate"):
            ui.parse_value("a, a", str, choices=["a", "b"], multiple=True)
        with pytest.raises(ValueError, match="at least one"):
            ui.parse_value("", str, choices=["a"], multiple=True)

    def test_booleans(self):
        for word in ("y", "YES", "true", "1", "on"):
            assert ui.parse_value(word, bool) is True
        for word in ("n", "No", "false", "0", "off"):
            assert ui.parse_value(word, bool) is False
        with pytest.raises(ValueError, match="expected yes/no"):
            ui.parse_value("maybe", bool)

    def test_empty_string_is_rejected(self):
        with pytest.raises(ValueError, match="empty answer"):
            ui.parse_value("   ", str)


class TestSession:
    def test_keyboard_answer_wins(self):
        session = ui.Session({"n": 5}, interactive=True, input_function=lambda _: "9")
        assert session.ask("n", "How many?", int) == 9
        assert session.answers[-1].source == "keyboard"

    def test_enter_takes_the_default(self):
        session = ui.Session({"n": 5}, interactive=True, input_function=lambda _: "")
        assert session.ask("n", "How many?", int) == 5
        assert session.answers[-1].source == "config default"

    def test_no_keyboard_falls_back_silently(self):
        """nbclient raises StdinNotImplementedError; the session must cope, not crash."""

        def no_stdin(_prompt):
            raise NotImplementedError("raw_input was called, but this frontend does not support input requests")

        session = ui.Session({"n": 5}, input_function=no_stdin)
        assert session.ask("n", "How many?", int) == 5
        assert session.interactive is False
        assert session.notes, "the fallback should be recorded for the reader"

    def test_invalid_then_valid_is_re_asked(self):
        answers = iter(["abc", "-1", "4"])
        session = ui.Session({}, interactive=True, input_function=lambda _: next(answers))
        assert session.ask("n", "How many?", int, minimum=0) == 4

    def test_persistent_invalid_input_raises(self):
        session = ui.Session({}, interactive=True, input_function=lambda _: "nonsense")
        with pytest.raises(ValueError, match="invalid answers in a row"):
            session.ask("n", "How many?", int)

    def test_no_default_and_no_keyboard_raises(self):
        session = ui.Session({}, interactive=False)
        with pytest.raises(KeyError, match="no default in the config"):
            session.ask("n", "How many?", int)

    def test_invalid_default_raises_rather_than_passing(self):
        session = ui.Session({"n": "not a number"}, interactive=False)
        with pytest.raises(ValueError, match="expected a number"):
            session.ask("n", "How many?", int)

    def test_replay_from_a_file(self, tmp_path):
        path = tmp_path / "answers.json"
        path.write_text(json.dumps({"n": 11}), encoding="utf-8")
        session = ui.Session(
            {"n": 5},
            answers_file=path,
            interactive=True,
            input_function=lambda _: pytest.fail("must not prompt when replaying"),
        )
        assert session.ask("n", "How many?", int) == 11
        assert session.answers[-1].source == "answers file"

    def test_round_trip_save_and_replay(self, tmp_path):
        first = ui.Session({"a": 1, "b": "pls"}, interactive=False)
        first.ask("a", "a?", int)
        first.ask("b", "b?", str, choices=["pls", "rf"])
        path = first.save(tmp_path / "out.json")
        second = ui.Session({}, answers_file=path, interactive=False)
        assert second.ask("a", "a?", int) == 1
        assert second.ask("b", "b?", str, choices=["pls", "rf"]) == "pls"

    def test_re_answering_replaces_rather_than_duplicates(self):
        session = ui.Session({"n": 5}, interactive=False)
        session.ask("n", "How many?", int)
        session.ask("n", "How many?", int)
        assert len([a for a in session.answers if a.key == "n"]) == 1

    def test_ask_path_rejects_the_wrong_suffix(self, tmp_path):
        bad = tmp_path / "data.docx"
        bad.write_text("x", encoding="utf-8")
        session = ui.Session({"f": str(bad)}, interactive=False)
        with pytest.raises(ValueError, match="expected one of"):
            session.ask_path("f", "File?", suffixes=[".csv"])

    def test_ask_path_rejects_a_missing_file(self, tmp_path):
        session = ui.Session({"f": str(tmp_path / "absent.csv")}, interactive=False)
        with pytest.raises(FileNotFoundError):
            session.ask_path("f", "File?", suffixes=[".csv"])


class TestScoring:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        scores = evaluation.regression_scores(y, y)
        assert scores["rmse"] == 0.0
        assert scores["skill_vs_mean"] == 1.0

    def test_mean_predictor_has_zero_skill(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        scores = evaluation.regression_scores(y, np.full_like(y, y.mean()))
        assert scores["skill_vs_mean"] == pytest.approx(0.0, abs=1e-12)

    def test_worse_than_the_mean_is_negative_skill(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        scores = evaluation.regression_scores(y, np.array([10.0, -5.0, 8.0, 0.0]))
        assert scores["skill_vs_mean"] < 0

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluation.rmse([1, 2, 3], [1, 2])

    def test_nrmse_undefined_for_a_flat_target(self):
        with pytest.raises(ValueError, match="zero range"):
            evaluation.normalised_rmse_percent([2.0, 2.0], [1.0, 3.0])

    def test_majority_baseline_has_zero_mcc(self):
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        scores = evaluation.classification_scores(y, np.zeros_like(y))
        assert scores["accuracy"] == 0.9, "accuracy flatters the majority class"
        assert scores["mcc"] == 0.0, "MCC does not"
        assert scores["balanced_accuracy"] == 0.5

    def test_shuffled_baseline_is_near_zero(self):
        generator = np.random.default_rng(0)
        y = generator.integers(0, 2, 60)
        scores = evaluation.classification_scores(y, y, seed=1)
        assert scores["mcc"] == pytest.approx(1.0)
        assert abs(scores["mcc_shuffled_mean"]) < 0.15

    def test_persistence_baseline(self):
        y = np.array([1.0, 1.0, 1.0])
        assert evaluation.persistence_baseline(y)["rmse_persistence"] == 0.0
        with pytest.raises(ValueError, match="at least 2 points"):
            evaluation.persistence_baseline([1.0])

    def test_grouped_split_needs_two_batches(self):
        with pytest.raises(ValueError, match="at least 2 batches"):
            evaluation.grouped_splits(["a"] * 10)

    def test_split_choice_follows_the_batch_count(self):
        _, name = evaluation.grouped_splits(["a", "b", "c"])
        assert name == "leave-one-batch-out"
        _, name = evaluation.grouped_splits([f"b{i}" for i in range(12)])
        assert "fold" in name

    def test_out_of_fold_predictions_cover_every_sample(self):
        from sklearn.linear_model import LinearRegression

        generator = np.random.default_rng(0)
        x = generator.normal(size=(30, 2))
        y = x[:, 0] * 2 + 1
        groups = np.repeat(np.arange(6), 5)
        predictions, _ = evaluation.cross_validated_predictions(LinearRegression(), x, y, groups)
        assert np.isfinite(predictions).all()
        assert evaluation.rmse(y, predictions) < 1e-6

    def test_detection_limit_needs_replicates(self):
        with pytest.raises(ValueError, match="at least 2 replicates"):
            evaluation.detection_limit([1.0])

    def test_detection_limit_shrinks_with_n(self):
        generator = np.random.default_rng(0)
        few = evaluation.detection_limit(generator.normal(10, 1, 3))
        many = evaluation.detection_limit(generator.normal(10, 1, 50))
        assert many["min_detectable_difference"] < few["min_detectable_difference"]


class TestQuality:
    def test_criterion_limits(self):
        criterion = quality.Criterion("aggregate", "%", upper=2.0)
        assert criterion.passes(1.9)
        assert not criterion.passes(2.1)
        assert criterion.margin(1.5) == pytest.approx(0.5)
        assert criterion.margin(2.5) < 0

    def test_criterion_needs_a_limit(self):
        with pytest.raises(ValueError, match="needs at least one limit"):
            quality.Criterion("x", "-")

    def test_inverted_limits(self):
        with pytest.raises(ValueError, match="lower .* > upper"):
            quality.Criterion("x", "-", lower=5, upper=1)

    def test_unmeasured_attribute_is_not_a_pass(self):
        criteria = [quality.Criterion("hcp", "ng/mg", upper=100)]
        table = quality.release_table({}, criteria)
        assert table["status"].iloc[0] == "NOT MEASURED"

    def test_control_chart_flags_a_shift(self):
        values = [100.0] * 10 + [130.0]
        chart, limits = quality.control_chart(values)
        assert chart["out_of_control"].iloc[-1]
        assert limits["upper"] > limits["centre"] > limits["lower"]

    def test_control_chart_needs_three_points(self):
        with pytest.raises(ValueError, match="at least 3 batches"):
            quality.control_chart([1.0, 2.0])

    def test_capability_falls_as_spread_grows(self):
        criterion = quality.Criterion("titer", "mg/L", lower=900, upper=1100)
        generator = np.random.default_rng(0)
        tight = quality.capability(generator.normal(1000, 10, 40), criterion)
        loose = quality.capability(generator.normal(1000, 60, 40), criterion)
        assert tight["cpk"] > loose["cpk"]

    def test_comparability_needs_replicates_on_both_sides(self):
        with pytest.raises(ValueError, match="at least 2 replicates"):
            quality.comparability([1.0], [1.0, 2.0])

    def test_comparability_reports_a_detection_limit(self):
        generator = np.random.default_rng(0)
        result = quality.comparability(generator.normal(10, 1, 5), generator.normal(10, 1, 5))
        assert result["min_detectable_difference"] > 0
        assert "could detect" in result["verdict"]


class TestCostEstimate:
    """The cost figure is a materials floor and must present itself as one."""

    def test_reports_its_share_of_published_cogs(self):
        from bioproc import simulators

        estimate = simulators.biosteam_cost_estimate(50.0, titer_g_per_L=4.0, downstream_yield=0.7)
        assert "materials only" in estimate["engine"].lower()
        assert "cost of goods" in estimate["caveat"].lower()
        low, high = simulators.PUBLISHED_MAB_COGS_PER_GRAM
        assert estimate["direct_materials_per_gram"] < low, (
            "a materials-only figure that reaches published COGS is being read as something it is not"
        )

    def test_higher_titer_lowers_the_materials_bill(self):
        from bioproc import simulators

        poor = simulators.biosteam_cost_estimate(50.0, titer_g_per_L=1.0, downstream_yield=0.7)
        good = simulators.biosteam_cost_estimate(50.0, titer_g_per_L=5.0, downstream_yield=0.7)
        assert good["direct_materials_per_gram"] < poor["direct_materials_per_gram"]
        assert good["broth_volume_L_per_year"] < poor["broth_volume_L_per_year"]

    def test_invalid_inputs(self):
        import pytest as _pytest

        from bioproc import simulators

        with _pytest.raises(ValueError, match="must be > 0"):
            simulators.biosteam_cost_estimate(0.0, titer_g_per_L=4.0, downstream_yield=0.7)
        with _pytest.raises(ValueError, match="downstream_yield"):
            simulators.biosteam_cost_estimate(50.0, titer_g_per_L=4.0, downstream_yield=1.5)


class TestBaselineCorrection:
    def test_removes_a_smooth_background_without_eating_peaks(self):
        """ModPoly must leave a narrow peak standing while removing a broad baseline."""
        from bioproc import models

        x = np.linspace(0, 1, 300)
        peak = 1.0 * np.exp(-0.5 * ((x - 0.5) / 0.01) ** 2)
        background = 3.0 * np.exp(2.0 * x)
        corrected = models.baseline_correct((peak + background)[None, :])[0]
        assert corrected.max() > 0.7, "the peak was flattened by the correction"
        off_peak = np.abs(x - 0.5) > 0.1
        assert np.abs(corrected[off_peak]).max() < 0.2, "the background was not removed"

    def test_rejects_an_impossible_degree(self):
        import pytest as _pytest

        from bioproc import models

        with _pytest.raises(ValueError, match="degree must be"):
            models.baseline_correct(np.zeros((2, 5)), degree=9)


class TestShortPath:
    """Committed notebooks publish whatever they print, so paths are shortened."""

    def test_repo_paths_become_relative(self):
        from pathlib import Path

        repo = Path(ui.__file__).resolve().parents[2]
        assert ui.short_path(repo / "configs" / "systems" / "demo_mab.toml") == "configs/systems/demo_mab.toml"

    def test_home_paths_become_tilde(self):
        from pathlib import Path

        assert ui.short_path(Path.home() / "bioproc-data" / "demo") == "~/bioproc-data/demo"

    def test_foreign_paths_are_left_alone(self, tmp_path):
        """A path outside both roots is still shown, so provenance is not lost."""
        result = ui.short_path(tmp_path / "elsewhere.csv")
        assert "elsewhere.csv" in result

    def test_no_home_directory_survives_shortening(self):
        from pathlib import Path

        for candidate in [Path.home() / "x", Path(ui.__file__).resolve().parents[2] / "y"]:
            assert str(Path.home()) not in ui.short_path(candidate)
