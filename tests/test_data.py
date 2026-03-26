import numpy as np
import pandas as pd
import pytest
from src.data import calibrate_sensor


class TestCalibrateSensor:
    def test_output_length_matches_input(self):
        sensor = pd.Series([100.0, 200.0, 300.0])
        ref    = pd.Series([10.0,  20.0,  30.0])
        out = calibrate_sensor(sensor, ref)
        assert len(out) == len(sensor)

    def test_perfect_linear_relationship(self):
        """If sensor = 2*ref + 5, calibration should recover ref exactly."""
        ref    = pd.Series(np.linspace(10, 60, 100))
        sensor = 2 * ref + 5
        out = calibrate_sensor(sensor, ref)
        np.testing.assert_allclose(out.values, ref.values, atol=1e-6)

    def test_output_in_reference_range(self):
        np.random.seed(1)
        ref    = pd.Series(np.random.uniform(20, 80, 200))
        sensor = 3 * ref + np.random.randn(200) * 2
        out = calibrate_sensor(sensor, ref)
        assert out.min() > 0
        assert out.max() < 200
