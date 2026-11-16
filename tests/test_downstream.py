"""Downstream: known answers, conservation, invalid inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioproc import downstream, verification

COLUMN = downstream.Column(diameter_cm=10.0, bed_height_cm=20.0)
RESIN = downstream.LangmuirResin(q_max_g_per_L_resin=50.0, k_eq_L_per_g=8.0, k_mass_transfer_per_min=0.5)


class TestColumn:
    def test_volume_matches_geometry(self):
        """A 10 cm x 20 cm bed is pi*25*20 cm3 = 1.571 L."""
        np.testing.assert_allclose(COLUMN.volume_L, np.pi * 25.0 * 20.0 / 1000.0, rtol=1e-12)

    def test_residence_time(self):
        """20 cm at 300 cm/h is 4 minutes."""
        np.testing.assert_allclose(COLUMN.residence_time_min(300.0), 4.0, rtol=1e-12)

    def test_invalid_geometry(self):
        with pytest.raises(ValueError, match="dimensions must be > 0"):
            downstream.Column(diameter_cm=0.0, bed_height_cm=20.0)

    def test_invalid_porosity_ordering(self):
        with pytest.raises(ValueError, match="interstitial < total"):
            downstream.Column(diameter_cm=10.0, bed_height_cm=20.0, porosity_total=0.3, porosity_interstitial=0.5)

    def test_zero_velocity(self):
        with pytest.raises(ValueError, match="velocity must be > 0"):
            COLUMN.residence_time_min(0.0)


class TestIsotherm:
    def test_closed_form(self):
        check = verification.verify_langmuir_equilibrium(RESIN, 2.0)
        assert check.passed

    def test_saturates_at_qmax(self):
        assert RESIN.equilibrium_q(1e9) == pytest.approx(RESIN.q_max_g_per_L_resin, rel=1e-6)

    def test_zero_concentration_binds_nothing(self):
        assert RESIN.equilibrium_q(0.0) == 0.0

    def test_monotone_in_concentration(self):
        values = RESIN.equilibrium_q(np.linspace(0, 10, 50))
        assert np.all(np.diff(values) > 0)

    def test_invalid_resin(self):
        with pytest.raises(ValueError, match="must be > 0"):
            downstream.LangmuirResin(q_max_g_per_L_resin=-1.0, k_eq_L_per_g=8.0, k_mass_transfer_per_min=0.5)


class TestBreakthrough:
    @pytest.mark.slow
    def test_curve_rises_from_zero_towards_the_feed(self):
        curve = downstream.breakthrough_curve(COLUMN, RESIN, 1.0, 300.0, load_cv=150)
        assert curve["breakthrough_fraction"].iloc[0] < 1e-6
        assert curve["breakthrough_fraction"].iloc[-1] > 0.5
        assert (curve["breakthrough_fraction"].diff().dropna() >= -1e-6).all(), "breakthrough must be monotone"

    @pytest.mark.slow
    def test_capacity_approaches_the_isotherm_at_long_residence_time(self):
        """Slow loading approaches equilibrium, so DBC approaches the static capacity."""
        slow = downstream.breakthrough_curve(COLUMN, RESIN, 1.0, 60.0, load_cv=200)
        fast = downstream.breakthrough_curve(COLUMN, RESIN, 1.0, 600.0, load_cv=200)
        dbc_slow = downstream.dynamic_binding_capacity(slow, 0.10)
        dbc_fast = downstream.dynamic_binding_capacity(fast, 0.10)
        assert dbc_slow > dbc_fast, "a longer residence time must give more usable capacity"
        assert dbc_slow < RESIN.equilibrium_q(1.0) * 1.05, "DBC cannot exceed the equilibrium capacity"

    @pytest.mark.slow
    def test_dbc_rises_with_breakthrough_fraction(self):
        curve = downstream.breakthrough_curve(COLUMN, RESIN, 1.0, 300.0, load_cv=200)
        assert downstream.dynamic_binding_capacity(curve, 0.01) < downstream.dynamic_binding_capacity(curve, 0.10)

    def test_short_load_cannot_be_sized(self):
        curve = downstream.breakthrough_curve(COLUMN, RESIN, 1.0, 300.0, load_cv=2, points=50)
        with pytest.raises(ValueError, match="load further"):
            downstream.dynamic_binding_capacity(curve, 0.10)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="must be > 0"):
            downstream.breakthrough_curve(COLUMN, RESIN, 0.0, 300.0)
        with pytest.raises(ValueError, match="n_tanks must be >= 2"):
            downstream.breakthrough_curve(COLUMN, RESIN, 1.0, 300.0, n_tanks=1)
        curve = downstream.breakthrough_curve(COLUMN, RESIN, 1.0, 300.0, load_cv=20, points=50)
        with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
            downstream.dynamic_binding_capacity(curve, 1.5)


class TestCaptureStep:
    @pytest.mark.slow
    def test_yield_is_load_times_elution(self):
        result = downstream.capture_step(COLUMN, RESIN, 1.0, 20.0, 300.0, elution_recovery=0.9)
        np.testing.assert_allclose(result["yield_fraction"], result["load_yield_fraction"] * 0.9, rtol=1e-12)
        assert 0 < result["yield_fraction"] <= 1

    @pytest.mark.slow
    def test_more_product_needs_more_cycles(self):
        small = downstream.capture_step(COLUMN, RESIN, 1.0, 20.0, 300.0)
        large = downstream.capture_step(COLUMN, RESIN, 1.0, 60.0, 300.0)
        assert large["cycles"] > small["cycles"]

    @pytest.mark.slow
    def test_oversized_batch_is_refused(self):
        with pytest.raises(ValueError, match="cycles needed"):
            downstream.capture_step(COLUMN, RESIN, 5.0, 500.0, 300.0, cycles_max=3)

    def test_invalid_fractions(self):
        with pytest.raises(ValueError, match="load_fraction_of_dbc"):
            downstream.capture_step(COLUMN, RESIN, 1.0, 10.0, 300.0, load_fraction_of_dbc=1.5)


class TestPolishing:
    def test_clearance_accounts_for_product_loss(self):
        """Losing product concentrates the impurity that remains."""
        _, out = downstream.polishing_step(
            "cex", 100.0, {"hcp_ng_per_mg": 1000.0}, yield_fraction=0.5, hcp_log_reduction=1.0
        )
        # 1 log removal, but half the product survives: 1000/10/0.5 = 200.
        np.testing.assert_allclose(out["hcp_ng_per_mg"], 200.0, rtol=1e-12)

    def test_product_out_is_in_times_yield(self):
        product, _ = downstream.polishing_step("cex", 80.0, {}, yield_fraction=0.9)
        np.testing.assert_allclose(product, 72.0, rtol=1e-12)

    def test_zero_log_reduction_only_concentrates(self):
        _, out = downstream.polishing_step(
            "x", 10.0, {"hcp_ng_per_mg": 100.0}, yield_fraction=0.8, hcp_log_reduction=0.0
        )
        np.testing.assert_allclose(out["hcp_ng_per_mg"], 125.0, rtol=1e-12)

    def test_invalid_yield(self):
        with pytest.raises(ValueError, match="yield must be in"):
            downstream.polishing_step("cex", 100.0, {}, yield_fraction=0.0)

    def test_negative_impurity(self):
        with pytest.raises(ValueError, match="negative impurity"):
            downstream.polishing_step("cex", 100.0, {"hcp_ng_per_mg": -1.0}, yield_fraction=0.9)


class TestUFDF:
    def test_diafiltration_follows_exp_minus_n(self):
        result = downstream.ufdf_step(100.0, 10.0, target_concentration_g_per_L=50.0, diavolumes=7.0)
        assert verification.verify_diafiltration(result["buffer_exchange_fraction"], 7.0).passed

    def test_zero_diavolumes_exchanges_nothing(self):
        result = downstream.ufdf_step(100.0, 10.0, target_concentration_g_per_L=50.0, diavolumes=0.0)
        assert result["buffer_exchange_fraction"] == 0.0

    def test_concentration_factor(self):
        result = downstream.ufdf_step(100.0, 10.0, target_concentration_g_per_L=50.0, yield_fraction=1.0)
        np.testing.assert_allclose(result["final_volume_L"], 2.0, rtol=1e-12)
        np.testing.assert_allclose(result["concentration_factor"], 5.0, rtol=1e-12)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="must be > 0"):
            downstream.ufdf_step(100.0, 10.0, target_concentration_g_per_L=0.0)
        with pytest.raises(ValueError, match="yield must be in"):
            downstream.ufdf_step(100.0, 10.0, target_concentration_g_per_L=50.0, yield_fraction=1.5)


class TestTrain:
    def test_cumulative_yield_is_the_product_of_steps(self):
        steps = [
            {"step": "harvest", "yield_fraction": 1.0, "product_g": 100.0},
            {"step": "capture", "yield_fraction": 0.9, "product_g": 90.0},
            {"step": "polish", "yield_fraction": 0.8, "product_g": 72.0},
        ]
        train = downstream.train_summary(steps)
        np.testing.assert_allclose(train["cumulative_yield_fraction"].iloc[-1], 0.72, rtol=1e-12)
        assert verification.verify_product_balance(train).passed

    def test_broken_balance_is_caught(self):
        train = pd.DataFrame(
            [
                {"step": "harvest", "yield_fraction": 1.0, "product_g": 100.0},
                {"step": "capture", "yield_fraction": 0.9, "product_g": 95.0},  # wrong
            ]
        )
        assert not verification.verify_product_balance(train).passed

    def test_empty_train(self):
        with pytest.raises(ValueError, match="no steps"):
            downstream.train_summary([])

    def test_missing_columns(self):
        with pytest.raises(ValueError, match="missing"):
            downstream.train_summary([{"step": "x"}])
