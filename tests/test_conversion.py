import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises_regex, assert_allclose # Added assert_allclose

# Add the project root to the Python path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import conversion

class TestSignalToConcentration4D(unittest.TestCase): # Renamed
    def setUp(self):
        """Prepare common test data for 4D signal_to_concentration."""
        self.dce_shape = (2, 2, 2, 10)  # Small x,y,z for manageable data
        self.spatial_shape = self.dce_shape[:3]
        
        self.t10_map_data = np.full(self.spatial_shape, 1.0)  # T10 of 1s
        self.r1 = 4.5  # s^-1 mM^-1
        self.TR = 0.005  # 5 ms (seconds)
        
        self.dce_series_data = np.ones(self.dce_shape) * 100  # Baseline signal 100
        self.dce_series_data[..., 5:] *= 1.5 
        self.baseline_time_points = 5

    def test_conversion_basic_case_4d(self): # Renamed
        """Test basic signal to concentration conversion for 4D data."""
        Ct_data = conversion.signal_to_concentration(
            self.dce_series_data, 
            self.t10_map_data, 
            self.r1, 
            self.TR, 
            baseline_time_points=self.baseline_time_points
        )
        self.assertEqual(Ct_data.shape, self.dce_series_data.shape)
        
        # Manually calculate for one voxel to verify
        S_pre_expected = 100.0
        R1_0_expected = 1.0 / self.t10_map_data[0,0,0] 
        assert_array_almost_equal(Ct_data[0,0,0,:self.baseline_time_points], 0, decimal=5,
                                  err_msg="Baseline concentration should be zero.")
        
        S_t_enhanced = 150.0
        signal_ratio_term_enhanced = S_t_enhanced / S_pre_expected
        E1_0_term = (1.0 - np.exp(-self.TR * R1_0_expected)) 
        log_arg_enhanced = 1.0 - (signal_ratio_term_enhanced * E1_0_term) 
        R1_t_enhanced = (-1.0 / self.TR) * np.log(log_arg_enhanced) 
        delta_R1_t_enhanced = R1_t_enhanced - R1_0_expected 
        Ct_t_enhanced_expected = delta_R1_t_enhanced / self.r1 
        
        assert_array_almost_equal(Ct_data[0,0,0,self.baseline_time_points:], Ct_t_enhanced_expected, decimal=5,
                                  err_msg="Concentration in enhanced phase mismatch.")

    def test_input_validation_dimensions_4d(self): # Renamed
        """Test input validation for incorrect data dimensions for 4D data."""
        with assert_raises_regex(self, ValueError, "dce_series_data must be a 4D NumPy array."):
            conversion.signal_to_concentration(np.zeros((2,2,10)), self.t10_map_data, self.r1, self.TR, self.baseline_time_points)
        with assert_raises_regex(self, ValueError, "t10_map_data must be a 3D NumPy array."):
            conversion.signal_to_concentration(self.dce_series_data, np.zeros((2,2,2,10)), self.r1, self.TR, self.baseline_time_points)
        with assert_raises_regex(self, ValueError, "Spatial dimensions of dce_series_data .* must match t10_map_data."):
            conversion.signal_to_concentration(self.dce_series_data, np.zeros((3,3,3)), self.r1, self.TR, self.baseline_time_points)

    def test_input_validation_parameters_4d(self): # Renamed
        """Test input validation for incorrect parameter values for 4D data."""
        with assert_raises_regex(self, ValueError, "TR must be positive."):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1, 0, self.baseline_time_points)
        with assert_raises_regex(self, ValueError, "r1_relaxivity must be positive."):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, 0, self.TR, self.baseline_time_points)
        with assert_raises_regex(self, ValueError, "baseline_time_points must be positive."):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1, self.TR, baseline_time_points=0)
        with assert_raises_regex(self, ValueError, "baseline_time_points must be less than the number of time points"):
            conversion.signal_to_concentration(self.dce_series_data, self.t10_map_data, self.r1, self.TR, baseline_time_points=self.dce_shape[3])

    def test_zero_t10_handling_4d(self): # Renamed
        """Test handling of T10=0 for 4D data (should use epsilon)."""
        t10_map_data_with_zero = self.t10_map_data.copy()
        t10_map_data_with_zero[0,0,0] = 0 # Set one voxel's T10 to zero
        Ct_data = conversion.signal_to_concentration(self.dce_series_data, t10_map_data_with_zero, self.r1, self.TR, baseline_time_points=self.baseline_time_points)
        self.assertEqual(Ct_data.shape, self.dce_series_data.shape)
        self.assertTrue(np.isfinite(Ct_data[0,0,0,0]), "Ct should be finite even with T10=0 due to epsilon.")
        assert_array_almost_equal(Ct_data[0,0,0,:self.baseline_time_points], 0, decimal=3, 
                                  err_msg="Baseline with T10=0 not handled as expected.")


    def test_zero_S_pre_handling_4d(self): # Renamed
        """Test handling of S_pre=0 for 4D data (should use epsilon)."""
        dce_data_zero_baseline = np.zeros_like(self.dce_series_data) 
        dce_data_zero_baseline[..., self.baseline_time_points:] = 50 
        
        Ct_data = conversion.signal_to_concentration(dce_data_zero_baseline, self.t10_map_data, self.r1, self.TR, baseline_time_points=self.baseline_time_points)
        self.assertEqual(Ct_data.shape, self.dce_series_data.shape)
        self.assertTrue(np.all(np.abs(Ct_data[..., self.baseline_time_points:]) > 1e3) | np.all(np.isinf(Ct_data[..., self.baseline_time_points:])) | np.all(np.isnan(Ct_data[..., self.baseline_time_points:])))


    def test_log_arg_clipping_4d(self): # Renamed
        """Test log_arg clipping for specific voxel in 4D data."""
        dce_high_signal = self.dce_series_data.copy()
        dce_high_signal[0,0,0,:self.baseline_time_points] = 10 
        dce_high_signal[0,0,0,self.baseline_time_points] = 10 * 2100 
        
        t10_for_clipping_test = self.t10_map_data.copy()
        t10_for_clipping_test[0,0,0] = 10.0 
        
        Ct_data = conversion.signal_to_concentration(
            dce_high_signal, t10_for_clipping_test, self.r1, self.TR, 
            baseline_time_points=self.baseline_time_points
        )
        self.assertTrue(np.isfinite(Ct_data[0,0,0,self.baseline_time_points]), "Clipped value should be finite")
        self.assertGreater(Ct_data[0,0,0,self.baseline_time_points], 0, "Clipped value should result in positive conc if signal increases")

    def test_nan_inf_handling_4d(self): # New test
        """Test NaN/Inf in input DCE data for the 4D version."""
        dce_data_with_nan_baseline = self.dce_series_data.copy()
        dce_data_with_nan_baseline[0,0,0,0] = np.nan 
        Ct_nan_baseline = conversion.signal_to_concentration(dce_data_with_nan_baseline, self.t10_map_data, self.r1, self.TR, self.baseline_time_points)
        self.assertTrue(np.all(np.isnan(Ct_nan_baseline[0,0,0,:])), "Voxel with NaN in baseline should be all NaN Ct")

        dce_data_with_nan_signal = self.dce_series_data.copy()
        dce_data_with_nan_signal[0,0,0,self.baseline_time_points+1] = np.nan 
        Ct_nan_signal = conversion.signal_to_concentration(dce_data_with_nan_signal, self.t10_map_data, self.r1, self.TR, self.baseline_time_points)
        self.assertTrue(np.isnan(Ct_nan_signal[0,0,0,self.baseline_time_points+1]), "NaN signal should propagate to NaN Ct")
        self.assertFalse(np.isnan(Ct_nan_signal[0,0,0,self.baseline_time_points-1]))

        dce_data_with_inf = self.dce_series_data.copy()
        dce_data_with_inf[0,0,0,self.baseline_time_points+1] = np.inf 
        Ct_inf = conversion.signal_to_concentration(dce_data_with_inf, self.t10_map_data, self.r1, self.TR, self.baseline_time_points)
        self.assertTrue(np.isinf(Ct_inf[0,0,0,self.baseline_time_points+1]) or \
                        np.isnan(Ct_inf[0,0,0,self.baseline_time_points+1]),
                        "Inf signal did not propagate as expected")


class TestSignalTcToConcentrationTc(unittest.TestCase):
    def setUp(self):
        """Prepare common test data for 1D signal_tc_to_concentration_tc."""
        self.num_time_points = 10
        self.signal_tc_ideal = np.ones(self.num_time_points) * 100  
        self.signal_tc_ideal[5:] *= 1.5 
        self.t10_scalar = 1.0 
        self.r1_scalar = 4.5 
        self.TR_scalar = 0.005 
        self.baseline_pts_ideal = 5

    def test_tc_conversion_basic_1d(self): # Renamed
        """Test basic signal to concentration conversion for 1D TC data."""
        Ct_tc = conversion.signal_tc_to_concentration_tc(
            self.signal_tc_ideal, self.t10_scalar, self.r1_scalar, self.TR_scalar, self.baseline_pts_ideal
        )
        self.assertEqual(Ct_tc.shape, self.signal_tc_ideal.shape)
        
        S_pre_tc_expected = 100.0
        R1_0_tc_expected = 1.0 / self.t10_scalar
        
        assert_array_almost_equal(Ct_tc[:self.baseline_pts_ideal], 0, decimal=5, 
                                  err_msg="Baseline concentration in 1D TC should be zero.")
        
        S_t_enhanced = 150.0
        signal_ratio_term_enhanced = S_t_enhanced / S_pre_tc_expected
        E1_0_term = (1.0 - np.exp(-self.TR_scalar * R1_0_tc_expected))
        log_arg_enhanced = 1.0 - (signal_ratio_term_enhanced * E1_0_term)
        R1_t_enhanced = (-1.0 / self.TR_scalar) * np.log(log_arg_enhanced)
        delta_R1_t_enhanced = R1_t_enhanced - R1_0_tc_expected
        Ct_t_enhanced_expected = delta_R1_t_enhanced / self.r1_scalar
        
        assert_array_almost_equal(Ct_tc[self.baseline_pts_ideal:], Ct_t_enhanced_expected, decimal=5,
                                  err_msg="Enhanced concentration in 1D TC mismatch.")

    def test_tc_baseline_points_variations(self): # New
        """Test different numbers of baseline points for 1D TC."""
        signal_tc = np.array([90., 95., 100., 105., 110., 150., 160., 170., 180., 190.])
        
        Ct_1_baseline = conversion.signal_tc_to_concentration_tc(signal_tc, self.t10_scalar, self.r1_scalar, self.TR_scalar, 1)
        S0_1 = 90.0
        R1_0 = 1.0 / self.t10_scalar
        E0_term = (1.0 - np.exp(-self.TR_scalar * R1_0))
        expected_Ct_1_idx0 = ((-1.0/self.TR_scalar) * np.log(1.0 - (signal_tc[0]/S0_1)*E0_term) - R1_0) / self.r1_scalar
        assert_array_almost_equal(Ct_1_baseline[0], expected_Ct_1_idx0, decimal=5)

        almost_all_baseline_count = len(signal_tc) - 1
        Ct_almost_all_baseline = conversion.signal_tc_to_concentration_tc(signal_tc, self.t10_scalar, self.r1_scalar, self.TR_scalar, almost_all_baseline_count)
        self.assertTrue(np.all(np.isfinite(Ct_almost_all_baseline)))


    def test_tc_conversion_input_validation(self): # Existing
        """Test input validation for 1D TC conversion."""
        with assert_raises_regex(self, ValueError, "signal_tc must be a 1D NumPy array."):
            conversion.signal_tc_to_concentration_tc(np.zeros((2,2)), self.t10_scalar, self.r1_scalar, self.TR_scalar, self.baseline_pts_ideal)
        with assert_raises_regex(self, ValueError, "t10_scalar must be a positive number."):
            conversion.signal_tc_to_concentration_tc(self.signal_tc_ideal, 0, self.r1_scalar, self.TR_scalar, self.baseline_pts_ideal)
        with assert_raises_regex(self, ValueError, "r1_relaxivity must be a positive number."):
            conversion.signal_tc_to_concentration_tc(self.signal_tc_ideal, self.t10_scalar, 0, self.TR_scalar, self.baseline_pts_ideal)
        with assert_raises_regex(self, ValueError, "TR must be a positive number."):
            conversion.signal_tc_to_concentration_tc(self.signal_tc_ideal, self.t10_scalar, self.r1_scalar, 0, self.baseline_pts_ideal)
        with assert_raises_regex(self, ValueError, "baseline_time_points must be a positive integer."):
            conversion.signal_tc_to_concentration_tc(self.signal_tc_ideal, self.t10_scalar, self.r1_scalar, self.TR_scalar, 0)
        with assert_raises_regex(self, ValueError, "baseline_time_points must be less than the number of time points"):
            conversion.signal_tc_to_concentration_tc(self.signal_tc_ideal, self.t10_scalar, self.r1_scalar, self.TR_scalar, len(self.signal_tc_ideal))
        with assert_raises_regex(self, ValueError, "signal_tc cannot be empty."):
            conversion.signal_tc_to_concentration_tc(np.array([]), self.t10_scalar, self.r1_scalar, self.TR_scalar, 1)

    def test_tc_input_data_issues(self): # New
        """Test 1D TC with problematic input signal data (NaN, Inf)."""
        signal_nan_baseline = np.array([100., np.nan, 100., 150., 160.])
        Ct_nan_baseline = conversion.signal_tc_to_concentration_tc(signal_nan_baseline, self.t10_scalar, self.r1_scalar, self.TR_scalar, 3)
        self.assertTrue(np.all(np.isnan(Ct_nan_baseline)), "NaN in baseline should make all Ct NaN due to S0 being NaN.")

        signal_nan_post_baseline = np.array([100., 100., 100., np.nan, 160.])
        Ct_nan_post_baseline = conversion.signal_tc_to_concentration_tc(signal_nan_post_baseline, self.t10_scalar, self.r1_scalar, self.TR_scalar, 3)
        self.assertTrue(np.isnan(Ct_nan_post_baseline[3]), "NaN signal should propagate to NaN Ct at that point.")
        self.assertFalse(np.isnan(Ct_nan_post_baseline[0]))
        self.assertFalse(np.isnan(Ct_nan_post_baseline[4]))

        signal_inf_post_baseline = np.array([100., 100., 100., np.inf, 160.])
        Ct_inf_post_baseline = conversion.signal_tc_to_concentration_tc(signal_inf_post_baseline, self.t10_scalar, self.r1_scalar, self.TR_scalar, 3)
        self.assertTrue(np.isinf(Ct_inf_post_baseline[3]) or np.isnan(Ct_inf_post_baseline[3]), 
                        "Inf signal should lead to Inf or NaN Ct.")


    def test_tc_conversion_edge_cases(self): # Existing, verified and slightly enhanced
        """Test edge cases for 1D TC conversion."""
        signal_tc_zero_baseline = np.zeros_like(self.signal_tc_ideal)
        signal_tc_zero_baseline[self.baseline_pts_ideal:] = 50 
        
        Ct_tc_zero_baseline = conversion.signal_tc_to_concentration_tc(
            signal_tc_zero_baseline, self.t10_scalar, self.r1_scalar, self.TR_scalar, self.baseline_pts_ideal
        )
        self.assertTrue(np.all(np.isfinite(Ct_tc_zero_baseline)))
        expected_ct_baseline_for_zero_signal_zero_S0 = (- (1.0/self.t10_scalar)) / self.r1_scalar
        assert_array_almost_equal(Ct_tc_zero_baseline[:self.baseline_pts_ideal], expected_ct_baseline_for_zero_signal_zero_S0, decimal=5,
                                  err_msg="Ct for zero signal with zero S0 (epsilon) incorrect.")

        signal_tc_high = np.ones_like(self.signal_tc_ideal) * 10 
        signal_tc_high[self.baseline_pts_ideal] = 10 * 210 
        
        t10_for_clip = 1.0 
        Ct_tc_high_signal = conversion.signal_tc_to_concentration_tc(
            signal_tc_high, t10_for_clip, self.r1_scalar, self.TR_scalar, self.baseline_pts_ideal
        )
        self.assertTrue(np.isfinite(Ct_tc_high_signal[self.baseline_pts_ideal]),
                        "Concentration where log_arg was clipped should be finite.")
        self.assertAlmostEqual(Ct_tc_high_signal[self.baseline_pts_ideal], 920.805, places=3,
                               err_msg="Concentration after log_arg clipping is not as expected.")


if __name__ == '__main__':
    unittest.main()
