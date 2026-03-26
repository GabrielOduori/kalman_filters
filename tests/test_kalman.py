import numpy as np
import pytest
from src import KalmanFilter1D


def make_kf(**kwargs):
    defaults = dict(x0=0.0, P0=1.0, A=1, H=1, Q=0.1, R=1.0)
    return KalmanFilter1D(**{**defaults, **kwargs})


# ------------------------------------------------------------------
# predict()
# ------------------------------------------------------------------

class TestPredict:
    def test_state_propagates(self):
        kf = make_kf(x0=5.0, A=2)
        kf.predict()
        assert kf.x == pytest.approx(10.0)

    def test_covariance_grows(self):
        kf = make_kf(P0=1.0, Q=0.5)
        kf.predict()
        assert kf.P == pytest.approx(1.5)

    def test_covariance_never_negative(self):
        kf = make_kf(P0=0.0, Q=0.1)
        kf.predict()
        assert kf.P >= 0


# ------------------------------------------------------------------
# update()
# ------------------------------------------------------------------

class TestUpdate:
    def test_perfect_measurement_trusted_fully(self):
        """With zero sensor noise, estimate should equal measurement."""
        kf = make_kf(x0=0.0, P0=1.0, R=1e-10)
        kf.predict()
        kf.update(42.0)
        assert kf.x == pytest.approx(42.0, abs=1e-5)

    def test_zero_gain_when_sensor_very_noisy(self):
        """With huge sensor noise, estimate barely moves from prediction."""
        kf = make_kf(x0=10.0, P0=0.01, R=1e10)
        kf.predict()
        x_pred = kf.x
        kf.update(999.0)
        assert kf.x == pytest.approx(x_pred, abs=0.01)

    def test_covariance_shrinks_after_update(self):
        kf = make_kf(P0=1.0)
        kf.predict()
        P_pred = kf.P
        kf.update(0.0)
        assert kf.P < P_pred

    def test_custom_R_overrides_default(self):
        kf1 = make_kf(x0=0.0, P0=1.0, R=1.0)
        kf2 = make_kf(x0=0.0, P0=1.0, R=1.0)
        kf1.predict(); kf1.update(10.0, R=1.0)
        kf2.predict(); kf2.update(10.0, R=100.0)
        # lower R → more trust → larger update
        assert kf1.x > kf2.x


# ------------------------------------------------------------------
# step() and filter()
# ------------------------------------------------------------------

class TestStepAndFilter:
    def test_filter_output_length(self):
        kf = make_kf()
        meas = np.ones(50)
        out = kf.filter(meas)
        assert len(out) == 50

    def test_estimates_property_matches_filter_output(self):
        kf = make_kf()
        out = kf.filter(np.ones(20))
        assert np.array_equal(out, kf.estimates)

    def test_uncertainty_is_positive(self):
        kf = make_kf()
        kf.filter(np.random.randn(30))
        assert np.all(kf.uncertainty > 0)

    def test_converges_on_constant_signal(self):
        """Filter should converge close to the true constant value."""
        true_val = 40.0
        np.random.seed(0)
        meas = true_val + np.random.randn(200) * 3.0
        kf = make_kf(x0=0.0, P0=100.0, Q=0.1, R=9.0)
        kf.filter(meas)
        assert abs(kf.estimates[-1] - true_val) < 2.0

    def test_multi_sensor_step(self):
        """Two sequential updates should reduce uncertainty more than one."""
        kf1 = make_kf(P0=10.0)
        kf1.predict()
        kf1.update(5.0, R=1.0)
        P_single = kf1.P

        kf2 = make_kf(P0=10.0)
        kf2.predict()
        kf2.update(5.0, R=1.0)
        kf2.update(5.0, R=1.0)
        P_double = kf2.P

        assert P_double < P_single

    def test_step_single_value(self):
        kf = make_kf()
        kf.step(5.0)
        assert len(kf.estimates) == 1

    def test_step_list_of_tuples(self):
        kf = make_kf()
        kf.step([(5.0, 1.0), (5.0, 2.0)])
        assert len(kf.estimates) == 1


# ------------------------------------------------------------------
# smooth()
# ------------------------------------------------------------------

class TestSmooth:
    def test_smooth_output_length(self):
        kf = make_kf()
        kf.filter(np.ones(40))
        smoothed = kf.smooth()
        assert len(smoothed) == 40

    def test_smooth_reduces_rmse(self):
        """Smoothed estimates should be at least as accurate as filtered."""
        np.random.seed(7)
        true = np.cumsum(np.random.randn(150) * 0.3) + 40
        meas = true + np.random.randn(150) * 5.0

        kf = make_kf(x0=true[0], P0=10.0, Q=0.3, R=25.0)
        kf.filter(meas)
        smoothed = kf.smooth()

        rmse_filtered = np.sqrt(np.mean((kf.estimates - true) ** 2))
        rmse_smoothed = np.sqrt(np.mean((smoothed       - true) ** 2))

        assert rmse_smoothed <= rmse_filtered

    def test_smooth_must_be_called_after_filter(self):
        kf = make_kf()
        with pytest.raises(RuntimeError):
            kf.smooth()
