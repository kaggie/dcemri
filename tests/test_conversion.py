import unittest
import numpy as np
from core.conversion import signal_to_concentration, signal_tc_to_concentration_tc

class TestConversion(unittest.TestCase):

    def setUp(self):
        # Common parameters for tests
        self.r1_relaxivity = 0.0045  # s^-1 mM^-1
        self.TR = 4.5 / 1000  # TR in seconds (4.5 ms)
        
        # For signal_to_concentration
        self.dce_shape = (2, 2, 2, 10) # x, y, z, time
        self.t10_shape = (2, 2, 2)     # x, y, z
        self.dce_series_data = np.ones(self.dce_shape) * 100 # Base signal
        # Simulate signal increase post-contrast
        self.dce_series_data[..., 5:] *= 1.5 
        self.t10_map_data = np.ones(self.t10_shape) * 1.4 # T10 in seconds (e.g., 1400 ms)

        # For signal_tc_to_concentration_tc
        self.signal_tc_data = np.array([100, 100, 100, 100, 100, 150, 160, 155, 150, 145], dtype=float)
        self.t10_scalar = 1.4 # seconds

    # --- Tests for signal_to_concentration ---

    def test_s2c_valid_inputs(self):
        Ct_data = signal_to_concentration(
            self.dce_series_data,
            self.t10_map_data,
            self.r1_relaxivity,
            self.TR,
            baseline_time_points=5
        )
        self.assertEqual(Ct_data.shape, self.dce_shape)
        # Concentrations should be roughly zero at baseline
        np.testing.assert_array_almost_equal(Ct_data[..., :4], 0, decimal=1)
        # Concentrations should be positive after signal increase (assuming T1 shortening)
        self.assertTrue(np.all(Ct_data[..., 5:] > 0))

    def test_s2c_non_4d_dce_data(self):
        with self.assertRaises(ValueError):
            signal_to_concentration(np.ones((2,2,10)), self.t10_map_data, self.r1_relaxivity, self.TR)

    def test_s2c_non_3d_t10_data(self):
        with self.assertRaises(ValueError):
            signal_to_concentration(self.dce_series_data, np.ones((2,2)), self.r1_relaxivity, self.TR)

    def test_s2c_mismatched_spatial_dims(self):
        t10_mismatched = np.ones((2, 2, 1)) # Different Z dimension
        with self.assertRaises(ValueError):
            signal_to_concentration(self.dce_series_data, t10_mismatched, self.r1_relaxivity, self.TR)

    def test_s2c_non_positive_tr(self):
        with self.assertRaises(ValueError):
            signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1_relaxivity, 0)
        with self.assertRaises(ValueError):
            signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1_relaxivity, -1)

    def test_s2c_non_positive_r1(self):
        with self.assertRaises(ValueError):
            signal_to_concentration(self.dce_series_data, self.t10_map_data, 0, self.TR)
        with self.assertRaises(ValueError):
            signal_to_concentration(self.dce_series_data, self.t10_map_data, -0.001, self.TR)

    def test_s2c_invalid_baseline_points(self):
        with self.assertRaises(ValueError): # baseline_time_points <= 0
            signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1_relaxivity, self.TR, 0)
        with self.assertRaises(ValueError): # baseline_time_points >= num_time_points
            signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1_relaxivity, self.TR, 10)
        with self.assertRaises(ValueError): # baseline_time_points > num_time_points
            signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1_relaxivity, self.TR, 11)

    def test_s2c_t10_zeros(self):
        t10_with_zeros = self.t10_map_data.copy()
        t10_with_zeros[0,0,0] = 0
        Ct_data = signal_to_concentration(self.dce_series_data, t10_with_zeros, self.r1_relaxivity, self.TR)
        self.assertEqual(Ct_data.shape, self.dce_shape)
        # Expectation is that it runs without math errors due to epsilon addition for R1_0

    def test_s2c_s_pre_zeros(self):
        dce_s_pre_zero = self.dce_series_data.copy()
        dce_s_pre_zero[0,0,0,:5] = 0 # Make baseline zero for one voxel
        Ct_data = signal_to_concentration(dce_s_pre_zero, self.t10_map_data, self.r1_relaxivity, self.TR)
        self.assertEqual(Ct_data.shape, self.dce_shape)
        # Expectation is that it runs without math errors due to S_pre_safe

    # --- Tests for signal_tc_to_concentration_tc ---

    def test_s2c_tc_valid_inputs(self):
        Ct_tc = signal_tc_to_concentration_tc(
            self.signal_tc_data,
            self.t10_scalar,
            self.r1_relaxivity,
            self.TR,
            baseline_time_points=5
        )
        self.assertEqual(Ct_tc.shape, self.signal_tc_data.shape)
        # Concentrations should be roughly zero at baseline
        np.testing.assert_array_almost_equal(Ct_tc[:4], 0, decimal=1)
        # Concentrations should be positive after signal increase
        self.assertTrue(np.all(Ct_tc[5:] > 0))

    def test_s2c_tc_non_1d_signal_tc(self):
        with self.assertRaises(ValueError):
            signal_tc_to_concentration_tc(np.ones((5,2)), self.t10_scalar, self.r1_relaxivity, self.TR)

    def test_s2c_tc_non_positive_t10(self):
        with self.assertRaises(ValueError):
            signal_tc_to_concentration_tc(self.signal_tc_data, 0, self.r1_relaxivity, self.TR)
        with self.assertRaises(ValueError):
            signal_tc_to_concentration_tc(self.signal_tc_data, -1.0, self.r1_relaxivity, self.TR)

    def test_s2c_tc_non_positive_r1(self):
        with self.assertRaises(ValueError):
            signal_tc_to_concentration_tc(self.signal_tc_data, self.t10_scalar, 0, self.TR)
        with self.assertRaises(ValueError):
            signal_tc_to_concentration_tc(self.signal_tc_data, self.t10_scalar, -0.001, self.TR)

    def test_s2c_tc_non_positive_tr(self):
        with self.assertRaises(ValueError):
            signal_tc_to_concentration_tc(self.signal_tc_data, self.t10_scalar, self.r1_relaxivity, 0)
        with self.assertRaises(ValueError):
            signal_tc_to_concentration_tc(self.signal_tc_data, self.t10_scalar, self.r1_relaxivity, -1.0)

    def test_s2c_tc_invalid_baseline_points(self):
        with self.assertRaises(ValueError): # baseline_time_points <= 0
            signal_tc_to_concentration_tc(self.signal_tc_data, self.t10_scalar, self.r1_relaxivity, self.TR, 0)
        with self.assertRaises(ValueError): # baseline_time_points >= num_time_points
            signal_tc_to_concentration_tc(self.signal_tc_data, self.t10_scalar, self.r1_relaxivity, self.TR, 10)
        with self.assertRaises(ValueError): # baseline_time_points > num_time_points
            signal_tc_to_concentration_tc(self.signal_tc_data, self.t10_scalar, self.r1_relaxivity, self.TR, 11)

    def test_s2c_tc_empty_signal_tc(self):
        with self.assertRaises(ValueError):
            signal_tc_to_concentration_tc(np.array([]), self.t10_scalar, self.r1_relaxivity, self.TR)

    def test_s2c_tc_s_pre_tc_zeros(self):
        signal_tc_s_pre_zero = self.signal_tc_data.copy()
        signal_tc_s_pre_zero[:5] = 0 # Make baseline zero
        Ct_tc = signal_tc_to_concentration_tc(signal_tc_s_pre_zero, self.t10_scalar, self.r1_relaxivity, self.TR)
        self.assertEqual(Ct_tc.shape, signal_tc_s_pre_zero.shape)
        # Expectation is that it runs without math errors due to S_pre_tc adjustment

if __name__ == "__main__":
    unittest.main()
