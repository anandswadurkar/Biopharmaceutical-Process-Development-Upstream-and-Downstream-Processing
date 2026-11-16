"""Upstream model: known answers, conservation, invalid inputs."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bioproc import upstream, verification

BASE_STATE = {"xv": 0.5, "xd": 0.02, "glc": 25.0, "gln": 4.0, "lac": 0.5, "amm": 0.5, "titer": 0.0, "volume": 2.0}


def unlimited_parameters(mu_per_h: float) -> upstream.CHOParameters:
    """Growth only: no death, no product, no maintenance, no limitation."""
    return upstream.CHOParameters(
        mu_max_per_h=mu_per_h,
        kd_min_per_h=0.0,
        kd_amm_max_per_h=0.0,
        kd_starve_per_h=0.0,
        qp_beta_mg_per_1e9cells_h=0.0,
        m_glc_mmol_per_1e9cells_h=0.0,
        m_gln_mmol_per_1e9cells_h=0.0,
        k_gln_degradation_per_h=0.0,
        y_lac_glc_mol_per_mol=0.0,
        y_amm_gln_mol_per_mol=0.0,
        ks_glc_mM=1e-9,
        ks_gln_mM=1e-9,
        ki_lac_mM=1e12,
        ki_amm_mM=1e12,
    )


class TestKnownAnswers:
    def test_exponential_growth(self):
        """With nutrients in excess and death off, the model is x0*exp(mu*t)."""
        state = {**BASE_STATE, "glc": 1e6, "gln": 1e6}
        history = upstream.simulate(state, unlimited_parameters(0.03), duration_h=48, sample_interval_h=48)
        expected = verification.exponential_growth(0.03, BASE_STATE["xv"], 48.0)
        # 1e-4 relative: LSODA at rtol 1e-7 over 48 h, with the sample-point restart.
        np.testing.assert_allclose(history["xv"].iloc[-1], expected, rtol=1e-4)

    def test_no_growth_means_constant_cells(self):
        history = upstream.simulate(
            {**BASE_STATE, "glc": 1e6, "gln": 1e6}, unlimited_parameters(0.0), duration_h=72, sample_interval_h=24
        )
        np.testing.assert_allclose(history["xv"], BASE_STATE["xv"], rtol=1e-9)

    def test_monod_limits(self):
        """Monod is 0 at zero, 1/2 at k, and approaches 1."""
        assert upstream.monod(0.0, 1.0) == 0.0
        np.testing.assert_allclose(upstream.monod(1.0, 1.0), 0.5)
        assert upstream.monod(1e9, 1.0) > 0.999

    def test_temperature_shift_trades_growth_for_productivity(self):
        parameters = upstream.CHOParameters()
        before = upstream.specific_rates(parameters, 20.0, 4.0, 10.0, 2.0, shifted=0.0)
        after = upstream.specific_rates(parameters, 20.0, 4.0, 10.0, 2.0, shifted=1.0)
        assert after["mu"] < before["mu"]
        assert after["q_p"] > before["q_p"]

    def test_lactate_switches_from_production_to_consumption(self):
        """High glucose: overflow. Low glucose with lactate present: net uptake."""
        parameters = upstream.CHOParameters()
        high = upstream.specific_rates(parameters, 25.0, 4.0, 20.0, 1.0)
        low = upstream.specific_rates(parameters, 0.2, 4.0, 20.0, 1.0)
        assert float(high["q_lac"]) > 0, "glucose-rich culture should produce lactate"
        assert float(low["q_lac"]) < 0, "glucose-poor culture with lactate should consume it"


class TestPlausibility:
    """The defaults have to produce a culture a bioprocess engineer would recognise.

    The first version of these parameters consumed glucose ten times faster than any
    reported CHO line, so glutamine ran out on day 3, the culture peaked at 2.3e6
    cells/mL and the demo process missed its own titer specification by a factor of
    ten. Every number below is a guard against that returning silently.
    """

    def test_specific_rates_are_in_reported_ranges(self):
        table = upstream.rate_check(upstream.CHOParameters())
        out_of_range = table[~table["in_range"]]
        assert out_of_range.empty, f"rates outside reported ranges:\n{out_of_range}"

    def test_fourteen_day_fed_batch_looks_like_one(self):
        policy = upstream.glucose_setpoint_policy(target_glc_mM=12.0, feed_glc_mM=250.0, feed_gln_mM=30.0)
        history = upstream.simulate(
            {**BASE_STATE, "volume": 500.0},
            upstream.CHOParameters(),
            duration_h=14 * 24,
            sample_interval_h=24,
            feed_policy=policy,
            shift_time_h=72,
        )
        peak = history["xv"].max()
        titer = history["titer"].iloc[-1]
        viability = history["viability_percent"].iloc[-1]
        assert 8.0 <= peak <= 30.0, f"peak VCD {peak:.1f}e6/mL is outside 8-30 for a fed-batch"
        assert 1500 <= titer <= 8000, f"day-14 titer {titer:.0f} mg/L is outside 1.5-8 g/L"
        assert viability >= 50.0, f"day-14 viability {viability:.0f}% is too low for a harvest"
        assert history["lac"].max() <= 60.0, "lactate exceeds what is seen in practice"
        assert history["amm"].max() <= 12.0, "ammonium exceeds what is seen in practice"

    def test_glutamine_is_not_the_first_thing_to_run_out(self):
        """Glucose should be the controlled nutrient; glutamine must outlast growth."""
        policy = upstream.glucose_setpoint_policy(target_glc_mM=12.0, feed_glc_mM=250.0, feed_gln_mM=30.0)
        history = upstream.simulate(
            {**BASE_STATE, "volume": 500.0},
            upstream.CHOParameters(),
            duration_h=14 * 24,
            sample_interval_h=24,
            feed_policy=policy,
        )
        peak_day = float(history.loc[history["xv"].idxmax(), "time_d"])
        assert peak_day >= 5.0, f"growth stopped on day {peak_day:.0f}; a nutrient is limiting far too early"


class TestConservation:
    def test_glucose_balance_unfed(self):
        """Consumption is a solver state, so the balance closes to solver tolerance."""
        history = upstream.simulate(BASE_STATE, upstream.CHOParameters(), duration_h=96, sample_interval_h=6)
        check = verification.verify_glucose_balance(history, feed_glc_mM=0.0, tolerance=1e-6)
        assert check.passed, f"{check.note}; residual {check.observed:.2e}"

    def test_glucose_balance_with_feeding(self):
        """The same, with many bolus feeds: this is what catches a dilution error.

        The culture starts at the setpoint so that every interval triggers a feed;
        starting above it would spend the first days unfed and test less.
        """
        policy = upstream.glucose_setpoint_policy(target_glc_mM=12.0, feed_glc_mM=250.0)
        history = upstream.simulate(
            {**BASE_STATE, "glc": 12.0},
            upstream.CHOParameters(),
            duration_h=240,
            sample_interval_h=12,
            feed_policy=policy,
        )
        assert (history["feed_volume_added_L"] > 0).sum() >= 10, "the test needs feeds to exercise dilution"
        check = verification.verify_glucose_balance(history, feed_glc_mM=250.0, tolerance=1e-6)
        assert check.passed, f"{check.note}; residual {check.observed:.2e}"

    def test_a_dilution_error_would_be_caught(self):
        """The balance check must fail on corrupted bookkeeping, or it proves nothing."""
        policy = upstream.fixed_bolus_policy(0.05, feed_glc_mM=250.0)
        history = upstream.simulate(
            BASE_STATE, upstream.CHOParameters(), duration_h=120, sample_interval_h=12, feed_policy=policy
        )
        corrupted = history.copy()
        corrupted.loc[corrupted.index[-1], "glc"] *= 1.05
        assert not verification.verify_glucose_balance(corrupted, feed_glc_mM=250.0).passed

    def test_dead_cells_accumulate_without_lysis(self):
        """With lysis switched off, xd is monotone: death is the only term."""
        parameters = upstream.CHOParameters(k_lysis_per_h=0.0)
        history = upstream.simulate(BASE_STATE, parameters, duration_h=336, sample_interval_h=24)
        assert (history["xd"].diff().dropna() >= -1e-9).all()

    def test_lysis_clears_dead_cells(self):
        """With lysis on, dead cells are cleared, so viability ends higher."""
        with_lysis = upstream.simulate(
            BASE_STATE, upstream.CHOParameters(k_lysis_per_h=0.006), duration_h=336, sample_interval_h=24
        )
        without = upstream.simulate(
            BASE_STATE, upstream.CHOParameters(k_lysis_per_h=0.0), duration_h=336, sample_interval_h=24
        )
        assert with_lysis["xd"].iloc[-1] < without["xd"].iloc[-1]
        assert with_lysis["viability_percent"].iloc[-1] > without["viability_percent"].iloc[-1]

    def test_lysis_cannot_drive_dead_cells_negative(self):
        """A large lysis rate must not produce a negative population."""
        history = upstream.simulate(
            {**BASE_STATE, "xd": 1.0}, upstream.CHOParameters(k_lysis_per_h=0.5), duration_h=240, sample_interval_h=12
        )
        assert history["xd"].min() >= -1e-9

    def test_states_never_go_negative(self):
        """A culture run to exhaustion must not produce negative concentrations."""
        history = upstream.simulate(
            {**BASE_STATE, "glc": 5.0, "gln": 0.5}, upstream.CHOParameters(), duration_h=336, sample_interval_h=12
        )
        for state in upstream.STATES:
            assert history[state].min() >= -1e-9, f"{state} went negative: {history[state].min()}"

    def test_feeding_dilutes_titer(self):
        """A bolus must dilute the product, not concentrate it."""
        policy = upstream.fixed_bolus_policy(0.1, feed_glc_mM=250.0)
        history = upstream.simulate(
            {**BASE_STATE, "titer": 1000.0},
            unlimited_parameters(0.0),
            duration_h=48,
            sample_interval_h=24,
            feed_policy=policy,
        )
        assert history["titer"].iloc[1] < 1000.0
        # Mass is conserved by dilution: titer * volume unchanged by the bolus itself.
        np.testing.assert_allclose(
            history["titer"].iloc[1] * history["volume"].iloc[1], 1000.0 * BASE_STATE["volume"], rtol=1e-9
        )


class TestFeedPolicies:
    def test_setpoint_policy_restores_the_target(self):
        policy = upstream.glucose_setpoint_policy(target_glc_mM=12.0, feed_glc_mM=250.0)
        state = {"glc": 4.0, "volume": 10.0}
        feed = policy(24.0, state, 24.0)[0]
        mixed = (state["glc"] * state["volume"] + 250.0 * feed.volume_L) / (state["volume"] + feed.volume_L)
        np.testing.assert_allclose(mixed, 12.0, rtol=1e-12)

    def test_setpoint_policy_does_nothing_above_target(self):
        policy = upstream.glucose_setpoint_policy(target_glc_mM=12.0, feed_glc_mM=250.0)
        assert policy(24.0, {"glc": 20.0, "volume": 10.0}, 24.0) == []

    def test_feed_weaker_than_target_is_rejected(self):
        with pytest.raises(ValueError, match="must exceed the target"):
            upstream.glucose_setpoint_policy(target_glc_mM=12.0, feed_glc_mM=10.0)


class TestInvalidInput:
    def test_negative_parameter(self):
        with pytest.raises(ValueError, match="must be finite and >= 0"):
            upstream.CHOParameters(mu_max_per_h=-0.01)

    def test_zero_yield(self):
        with pytest.raises(ValueError, match="yields must be > 0"):
            upstream.CHOParameters(y_x_glc_1e9cells_per_mmol=0.0)

    def test_missing_state(self):
        with pytest.raises(ValueError, match="missing"):
            upstream.simulate({"xv": 1.0}, upstream.CHOParameters())

    def test_negative_state(self):
        with pytest.raises(ValueError, match="negative initial states"):
            upstream.simulate({**BASE_STATE, "glc": -1.0}, upstream.CHOParameters())

    def test_zero_volume(self):
        with pytest.raises(ValueError, match="volume must be > 0"):
            upstream.simulate({**BASE_STATE, "volume": 0.0}, upstream.CHOParameters())

    def test_non_positive_duration(self):
        with pytest.raises(ValueError, match="must be > 0"):
            upstream.simulate(BASE_STATE, upstream.CHOParameters(), duration_h=0)

    def test_negative_feed(self):
        with pytest.raises(ValueError, match="negative feed"):
            upstream.Feed(volume_L=-1.0)

    def test_bad_bolus_fraction(self):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
            upstream.fixed_bolus_policy(1.5, feed_glc_mM=250.0)

    def test_harvest_metrics_on_empty(self):
        import pandas as pd

        with pytest.raises(ValueError, match="empty history"):
            upstream.harvest_metrics(pd.DataFrame())


class TestHarvestMetrics:
    def test_qp_is_titer_over_ivcd(self):
        """qp = titer / IVCD, both taken at the harvest point, not at the last sample.

        Harvest can fall before the end of the run when viability drops through the
        floor, which is exactly what this guards: comparing against the full history
        would silently use a different cut.
        """
        history = upstream.simulate(BASE_STATE, upstream.CHOParameters(), duration_h=240, sample_interval_h=24)
        metrics = upstream.harvest_metrics(history)
        cut = history[history["time_d"] <= metrics["harvest_time_d"] + 1e-9]
        ivcd = np.trapezoid(cut["xv"], cut["time_d"])
        np.testing.assert_allclose(metrics["ivcd_1e6_cell_day_per_mL"], ivcd, rtol=1e-12)
        np.testing.assert_allclose(metrics["qp_pg_per_cell_day"], cut["titer"].iloc[-1] / ivcd, rtol=1e-12)

    def test_harvest_stops_at_the_viability_floor(self):
        history = upstream.simulate(
            {**BASE_STATE, "glc": 6.0, "gln": 0.6}, upstream.CHOParameters(), duration_h=504, sample_interval_h=24
        )
        metrics = upstream.harvest_metrics(history, viability_floor_percent=70.0)
        assert (
            metrics["final_viability_percent"] <= 70.0 + 1e-9 or metrics["harvest_time_d"] == history["time_d"].iloc[-1]
        )


@settings(deadline=None, max_examples=25)
@given(
    mu=st.floats(min_value=0.0, max_value=0.08),
    hours=st.floats(min_value=1.0, max_value=200.0),
    x0=st.floats(min_value=0.05, max_value=5.0),
)
def test_growth_matches_analytic_for_any_rate(mu, hours, x0):
    """Invariant: unlimited growth is exponential for every rate and horizon."""
    history = upstream.simulate(
        {**BASE_STATE, "xv": x0, "glc": 1e6, "gln": 1e6},
        unlimited_parameters(mu),
        duration_h=hours,
        sample_interval_h=hours,
    )
    np.testing.assert_allclose(history["xv"].iloc[-1], x0 * np.exp(mu * hours), rtol=1e-4)
