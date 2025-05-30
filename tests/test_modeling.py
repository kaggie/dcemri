import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose, assert_raises

import sys
import os

# Add project root for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import modeling
from core import aif # For AIF data generation if needed
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

# --- Helper data ---
T_TISSUE_SHORT = np.linspace(0, 1, 10) # 10 points, up to 1 time unit (e.g. min)
T_TISSUE_LONG = np.linspace(0, 5, 50) # 50 points, up to 5 time units (e.g. min)

# Simple AIF for testing: instant rise, exponential decay
T_AIF = np.linspace(0, 5, 100)
AIF_CONC = 10 * np.exp(-T_AIF / 0.5)
AIF_CONC[0] = 0

CP_INTERP_FUNC = interp1d(T_AIF, AIF_CONC, kind='linear', bounds_error=False, fill_value=0.0)
INTEGRAL_CP_DT_AIF = cumtrapz(AIF_CONC, T_AIF, initial=0)
INTEGRAL_CP_DT_INTERP_FUNC = interp1d(T_AIF, INTEGRAL_CP_DT_AIF, kind='linear', bounds_error=False, fill_value=0.0)
T_AIF_MAX = T_AIF[-1]


class TestPharmacokineticModels(unittest.TestCase):
    """Tests for individual pharmacokinetic model functions."""

    def test_standard_tofts_model_conv_ideal(self):
        Ktrans, ve = 0.2, 0.3
        Ct_expected = modeling._convolve_Cp_with_exp(T_TISSUE_LONG, Ktrans, ve, CP_INTERP_FUNC)
        Ct_model = modeling.standard_tofts_model_conv(T_TISSUE_LONG, Ktrans, ve, CP_INTERP_FUNC)
        assert_array_almost_equal(Ct_model, Ct_expected)

    def test_extended_tofts_model_conv_ideal(self):
        Ktrans, ve, vp = 0.2, 0.3, 0.05
        vp_comp = vp * CP_INTERP_FUNC(T_TISSUE_LONG)
        ees_comp = modeling._convolve_Cp_with_exp(T_TISSUE_LONG, Ktrans, ve, CP_INTERP_FUNC)
        Ct_expected = vp_comp + ees_comp
        Ct_model = modeling.extended_tofts_model_conv(T_TISSUE_LONG, Ktrans, ve, vp, CP_INTERP_FUNC)
        assert_array_almost_equal(Ct_model, Ct_expected)

    def test_patlak_model_ideal(self):
        Ktrans, vp = 0.1, 0.02
        Ct_expected = Ktrans * INTEGRAL_CP_DT_INTERP_FUNC(T_TISSUE_LONG) + vp * CP_INTERP_FUNC(T_TISSUE_LONG)
        Ct_model = modeling.patlak_model(T_TISSUE_LONG, Ktrans, vp, CP_INTERP_FUNC, INTEGRAL_CP_DT_INTERP_FUNC)
        assert_array_almost_equal(Ct_model, Ct_expected)

    def test_solve_2cxm_ode_model_ideal(self):
        Fp, PS, vp, ve = 0.5, 0.1, 0.05, 0.25
        Ct_model = modeling.solve_2cxm_ode_model(T_TISSUE_LONG, Fp, PS, vp, ve, CP_INTERP_FUNC, t_span_max=T_AIF_MAX)
        self.assertEqual(Ct_model.shape, T_TISSUE_LONG.shape)
        self.assertTrue(np.all(np.isfinite(Ct_model)))

    def test_model_parameter_validation(self):
        models_to_test = {
            "standard_tofts": lambda t, p: modeling.standard_tofts_model_conv(t, p.get('Ktrans',0.1), p.get('ve',0.1), CP_INTERP_FUNC),
            "extended_tofts": lambda t, p: modeling.extended_tofts_model_conv(t, p.get('Ktrans',0.1), p.get('ve',0.1), p.get('vp',0.1), CP_INTERP_FUNC),
            "patlak": lambda t, p: modeling.patlak_model(t, p.get('Ktrans',0.1), p.get('vp',0.1), CP_INTERP_FUNC, INTEGRAL_CP_DT_INTERP_FUNC),
            "2cxm": lambda t, p: modeling.solve_2cxm_ode_model(t, p.get('Fp',0.1), p.get('PS',0.1), p.get('vp',0.1), p.get('ve',0.1), CP_INTERP_FUNC, T_AIF_MAX)
        }
        params_per_model = {
            "standard_tofts": ['Ktrans', 've'], "extended_tofts": ['Ktrans', 've', 'vp'],
            "patlak": ['Ktrans', 'vp'], "2cxm": ['Fp', 'PS', 'vp', 've']
        }

        for name, model_func in models_to_test.items():
            for param_name in params_per_model[name]:
                invalid_params_dict = {pn: 0.1 for pn in params_per_model[name]} # Base valid params

                current_test_val = -0.1
                if name == "2cxm" and (param_name == "vp" or param_name == "ve"):
                    current_test_val = 1e-8

                invalid_params_dict[param_name] = current_test_val

                with self.subTest(model=name, param=param_name, value=current_test_val):
                    res = model_func(T_TISSUE_SHORT, invalid_params_dict)
                    self.assertTrue(np.all(np.isinf(res)), f"{name} with {param_name}={current_test_val} did not return all inf.")

    def test_model_edge_case_time_points(self):
        t_empty = np.array([])
        t_single = np.array([1.0])
        # Test Standard Tofts (as representative for convolution based)
        self.assertEqual(modeling.standard_tofts_model_conv(t_empty, 0.1, 0.1, CP_INTERP_FUNC).shape, t_empty.shape)
        # _convolve_Cp_with_exp returns zeros_like for len(t) < 2
        self.assertEqual(modeling.standard_tofts_model_conv(t_single, 0.1, 0.1, CP_INTERP_FUNC).shape, t_single.shape)
        self.assertTrue(np.all(modeling.standard_tofts_model_conv(t_single, 0.1, 0.1, CP_INTERP_FUNC)==0))


        # Test 2CXM
        self.assertEqual(modeling.solve_2cxm_ode_model(t_empty, 0.1,0.1,0.1,0.1,CP_INTERP_FUNC, T_AIF_MAX).shape, t_empty.shape)
        self.assertEqual(modeling.solve_2cxm_ode_model(t_single, 0.1,0.1,0.1,0.1,CP_INTERP_FUNC, T_AIF_MAX).shape, t_single.shape)

    @patch('core.modeling.solve_ivp')
    def test_solve_2cxm_ode_solver_failure(self, mock_solve_ivp):
        mock_sol = MagicMock()
        mock_sol.status = -1
        mock_sol.message = "Solver failed"
        mock_solve_ivp.return_value = mock_sol
        
        Ct_model = modeling.solve_2cxm_ode_model(T_TISSUE_SHORT, 0.5, 0.1, 0.05, 0.25, CP_INTERP_FUNC, T_AIF_MAX)
        self.assertTrue(np.all(np.isinf(Ct_model)))


class TestSingleVoxelFitting(unittest.TestCase):
    """Tests for single-voxel fitting functions."""
    noise_level_relative = 0.05 # Relative noise level for Tofts models
    noise_level_absolute = 0.001 # Absolute noise level for Patlak/2CXM (assuming concentrations are small)


    def _generate_synthetic_data(self, model_name, true_params):
        Ct_clean = None
        if model_name == "Standard Tofts":
            Ktrans, ve = true_params
            Ct_clean = modeling.standard_tofts_model_conv(T_TISSUE_LONG, Ktrans, ve, CP_INTERP_FUNC)
        elif model_name == "Extended Tofts":
            Ktrans, ve, vp = true_params
            Ct_clean = modeling.extended_tofts_model_conv(T_TISSUE_LONG, Ktrans, ve, vp, CP_INTERP_FUNC)
        elif model_name == "Patlak":
            Ktrans, vp = true_params
            Ct_clean = modeling.patlak_model(T_TISSUE_LONG, Ktrans, vp, CP_INTERP_FUNC, INTEGRAL_CP_DT_INTERP_FUNC)
        elif model_name == "2CXM":
            Fp, PS, vp, ve = true_params
            Ct_clean = modeling.solve_2cxm_ode_model(T_TISSUE_LONG, Fp, PS, vp, ve, CP_INTERP_FUNC, T_AIF_MAX)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        current_noise_level = self.noise_level_absolute
        if model_name in ["Standard Tofts", "Extended Tofts"] and np.max(Ct_clean) > 0:
            current_noise_level = self.noise_level_relative * np.max(Ct_clean)
        
        noise = np.random.normal(0, current_noise_level, size=Ct_clean.shape)
        return T_TISSUE_LONG, Ct_clean + noise, Ct_clean

    def _test_fitting_function(self, fit_func_name, model_name, true_params, initial_guess, bounds):
        fit_func = getattr(modeling, fit_func_name)
        t_tissue, Ct_noisy, _ = self._generate_synthetic_data(model_name, true_params)

        args_for_fit = [t_tissue, Ct_noisy]
        if model_name == "Patlak":
            args_for_fit.extend([CP_INTERP_FUNC, INTEGRAL_CP_DT_INTERP_FUNC])
        elif model_name == "2CXM":
            args_for_fit.extend([CP_INTERP_FUNC, T_AIF_MAX])
        else: # Tofts models
            args_for_fit.append(CP_INTERP_FUNC)
        args_for_fit.extend([initial_guess, bounds])
        
        params_fitted, curve_fitted = fit_func(*args_for_fit)
        
        assert_allclose(params_fitted, true_params, rtol=0.6, atol=0.15, err_msg=f"{fit_func_name} failed to recover parameters accurately.") # Relaxed tolerance for noisy data
        
        # Check fitted curve matches popt
        args_for_model = [t_tissue] + list(params_fitted)
        if model_name == "Patlak":
            args_for_model.extend([CP_INTERP_FUNC, INTEGRAL_CP_DT_INTERP_FUNC])
        elif model_name == "2CXM":
            args_for_model.extend([CP_INTERP_FUNC, T_AIF_MAX])
        else:
            args_for_model.append(CP_INTERP_FUNC)

        ref_curve_func = getattr(modeling, model_name.lower().replace(" ", "_") + "_model_conv" if "tofts" in model_name.lower() else model_name.lower()+"_model" if "patlak" in model_name.lower() else "solve_2cxm_ode_model")
        ref_curve = ref_curve_func(*args_for_model)
        
        assert_array_almost_equal(curve_fitted, ref_curve, err_msg=f"{fit_func_name} returned curve does not match model with popt.")

        # Test insufficient data (length of data < number of parameters)
        num_params = len(initial_guess)
        t_short = t_tissue[:num_params] # This will cause error in curve_fit
        Ct_short = Ct_noisy[:num_params]
        
        args_for_short_fit = [t_short, Ct_short] + args_for_fit[2:] # Keep other args same
        
        params_nan, curve_nan = fit_func(*args_for_short_fit)
        self.assertTrue(all(np.isnan(p) for p in params_nan), f"{fit_func_name} did not return all NaNs for params with insufficient data.")
        self.assertTrue(np.all(np.isnan(curve_nan)), f"{fit_func_name} did not return all NaNs for curve with insufficient data.")
        
        # Test all zeros/NaNs Ct_tissue
        Ct_zeros = np.zeros_like(t_tissue)
        args_for_zeros_fit = [t_tissue, Ct_zeros] + args_for_fit[2:]
        params_zeros, _ = fit_func(*args_for_zeros_fit)
        # curve_fit might return initial guess or bounds if it can't compute Jacobian, or NaN.
        # Check for NaNs as an indicator of fit failure or inability to proceed.
        self.assertTrue(all(np.isnan(p) for p in params_zeros), f"{fit_func_name} with zero Ct did not result in NaN parameters.")

        Ct_nans = np.full_like(t_tissue, np.nan)
        args_for_nans_fit = [t_tissue, Ct_nans] + args_for_fit[2:]
        params_nans, _ = fit_func(*args_for_nans_fit)
        self.assertTrue(all(np.isnan(p) for p in params_nans), f"{fit_func_name} with NaN Ct did not result in NaN parameters.")


    def test_fit_standard_tofts(self):
        self._test_fitting_function("fit_standard_tofts", "Standard Tofts", (0.25, 0.35), (0.1, 0.2), ([0,0],[1,1]))
        
    def test_fit_extended_tofts(self):
        self._test_fitting_function("fit_extended_tofts", "Extended Tofts", (0.25, 0.35, 0.03), (0.1,0.2,0.05), ([0,0,0],[1,1,0.5]))

    def test_fit_patlak_model(self):
         self._test_fitting_function("fit_patlak_model", "Patlak", (0.15, 0.04), (0.1,0.05), ([0,0],[1,0.5]))

    def test_fit_2cxm_model(self):
        # Note: 2CXM fitting can be sensitive. Tolerances might need adjustment.
        self._test_fitting_function("fit_2cxm_model", "2CXM", (0.6, 0.15, 0.04, 0.20), (0.5,0.1,0.05,0.1), ([0,0,1e-3,1e-3],[2,1,0.5,0.7]))


class TestFitVoxelWorker(unittest.TestCase):
    """Tests for the _fit_voxel_worker function."""

    def setUp(self):
        self.voxel_idx = (0,0,0)
        self.t_tissue = T_TISSUE_LONG
        self.t_aif = T_AIF
        self.Cp_aif = AIF_CONC
        self.true_st_params = (0.2, 0.3) # Ktrans, ve
        # Generate clean data using TestSingleVoxelFitting's helper (or a simplified one here)
        Ct_st_clean = modeling.standard_tofts_model_conv(self.t_tissue, *self.true_st_params, CP_INTERP_FUNC)
        self.Ct_st = Ct_st_clean + np.random.normal(0, 0.001, Ct_st_clean.shape)


    def test_worker_successful_fit_standard_tofts(self): # Specify model
        args = (self.voxel_idx, self.Ct_st, self.t_tissue, self.t_aif, self.Cp_aif,
                "Standard Tofts", (0.1,0.1), ([0,0],[1,1]))
        idx, model_name, result = modeling._fit_voxel_worker(args)
        self.assertEqual(idx, self.voxel_idx)
        self.assertEqual(model_name, "Standard Tofts")
        self.assertNotIn("error", result, f"Fit worker returned error: {result.get('error')}")
        self.assertTrue("Ktrans" in result and "ve" in result)
        assert_allclose([result["Ktrans"], result["ve"]], self.true_st_params, rtol=0.6, atol=0.15) # Relaxed due to noise & fit

    def test_worker_skipped_all_nan(self):
        Ct_nan = np.full_like(self.t_tissue, np.nan)
        args = (self.voxel_idx, Ct_nan, self.t_tissue, self.t_aif, self.Cp_aif,
                "Standard Tofts", (0.1,0.1), ([0,0],[1,1]))
        _, _, result = modeling._fit_voxel_worker(args)
        self.assertIn("error", result)
        self.assertIn("Skipped (all NaN or all zero data)", result["error"])

    def test_worker_skipped_insufficient_data(self):
        t_short = self.t_tissue[:3] # Standard Tofts needs at least 2 params * 2 = 4 points typically
        Ct_short = self.Ct_st[:3]
        args = (self.voxel_idx, Ct_short, t_short, self.t_aif, self.Cp_aif,
                "Standard Tofts", (0.1,0.1), ([0,0],[1,1]))
        _, _, result = modeling._fit_voxel_worker(args)
        self.assertIn("error", result)
        self.assertTrue("Skipped (insufficient valid data" in result["error"] or "Skipped (valid data points" in result["error"])


    def test_worker_unknown_model(self):
        args = (self.voxel_idx, self.Ct_st, self.t_tissue, self.t_aif, self.Cp_aif,
                "Unknown Model Type", (0.1,0.1), ([0,0],[1,1]))
        _, _, result = modeling._fit_voxel_worker(args)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Unknown model: Unknown Model Type")

    @patch('core.modeling.fit_standard_tofts')
    def test_worker_fit_failure_nan_params(self, mock_fit_st):
        mock_fit_st.return_value = ((np.nan, np.nan), np.full_like(self.t_tissue, np.nan))
        args = (self.voxel_idx, self.Ct_st, self.t_tissue, self.t_aif, self.Cp_aif,
                "Standard Tofts", (0.1,0.1), ([0,0],[1,1]))
        _, _, result = modeling._fit_voxel_worker(args)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Fit failed (returned None or NaN parameters)")


class TestVoxelWiseFitting(unittest.TestCase):
    """Tests for voxel-wise fitting functions (e.g., fit_standard_tofts_voxelwise)."""

    def setUp(self):
        self.Ct_data_shape = (2,2,1,20) # X, Y, Z, Time
        self.Ct_data = np.zeros(self.Ct_data_shape)
        self.t_tissue = np.linspace(0, 2, 20)
        self.t_aif = T_AIF[:50]
        self.Cp_aif = AIF_CONC[:50]
        
        self.true_st_params = (0.25, 0.3) # Ktrans, ve
        Ct_clean_voxel0 = modeling.standard_tofts_model_conv(self.t_tissue, *self.true_st_params,
                                                             interp1d(self.t_aif, self.Cp_aif, kind='linear', bounds_error=False, fill_value=0.0))
        self.Ct_data[0,0,0,:] = Ct_clean_voxel0 + np.random.normal(0, 0.001 * np.max(Ct_clean_voxel0), size=Ct_clean_voxel0.shape)
        self.Ct_data[0,1,0,:] = Ct_clean_voxel0 * 0.5 + np.random.normal(0, 0.001 * np.max(Ct_clean_voxel0), size=Ct_clean_voxel0.shape)
        self.Ct_data[1,0,0,:] = np.nan
        self.Ct_data[1,1,0,:] = 0

        self.mask = np.ones(self.Ct_data_shape[:3], dtype=bool)
        self.mask[1,0,0] = False # Mask out the NaN voxel for some tests

    @patch('multiprocessing.Pool')
    def test_fit_standard_tofts_voxelwise_mocked_pool(self, mock_pool_constructor):
        mock_pool_instance = MagicMock()
        
        # Simulate worker results for unmasked voxels
        # Voxel (0,0,0) - good fit
        # Voxel (0,1,0) - good fit (different params)
        # Voxel (1,1,0) - all zero Ct, worker should return error or specific values indicating failure
        worker_results = [
            ((0,0,0), "Standard Tofts", {"Ktrans": 0.24, "ve": 0.29}),
            ((0,1,0), "Standard Tofts", {"Ktrans": 0.12, "ve": 0.14}),
            ((1,1,0), "Standard Tofts", {"error": "Skipped (all NaN or all zero data)"}) # Worker handles zero data
        ]
        # Note: Voxel (1,0,0) is masked out by self.mask, so it won't be in tasks_args_list
        # and thus not in worker_results if the mask is applied correctly before forming tasks.

        mock_pool_instance.map.return_value = worker_results
        mock_pool_constructor.return_value.__enter__.return_value = mock_pool_instance

        param_maps = modeling.fit_standard_tofts_voxelwise(
            self.Ct_data, self.t_tissue, self.t_aif, self.Cp_aif,
            mask=self.mask, num_processes=2
        )
        
        self.assertIn("Ktrans", param_maps)
        self.assertIn("ve", param_maps)
        self.assertEqual(param_maps["Ktrans"].shape, self.Ct_data_shape[:3])
        
        assert_allclose(param_maps["Ktrans"][0,0,0], 0.24)
        assert_allclose(param_maps["ve"][0,0,0], 0.29)
        assert_allclose(param_maps["Ktrans"][0,1,0], 0.12)
        assert_allclose(param_maps["ve"][0,1,0], 0.14)
        
        self.assertTrue(np.isnan(param_maps["Ktrans"][1,0,0]), "Masked out voxel (1,0,0) should be NaN")
        self.assertTrue(np.isnan(param_maps["Ktrans"][1,1,0]), "Voxel (1,1,0) (all zeros) which worker skipped should be NaN")


    def test_fit_standard_tofts_voxelwise_serial(self):
        """Test voxel-wise fitting in serial mode (num_processes=1)."""
        mask = self.mask.copy() # Use the mask that excludes the all-NaN voxel

        param_maps = modeling.fit_standard_tofts_voxelwise(
            self.Ct_data, self.t_tissue, self.t_aif, self.Cp_aif,
            mask=mask, num_processes=1
        )
        self.assertIn("Ktrans", param_maps); self.assertIn("ve", param_maps)
        # Voxel (0,0,0) - should fit
        assert_allclose(param_maps["Ktrans"][0,0,0], self.true_st_params[0], rtol=0.6, atol=0.15)
        assert_allclose(param_maps["ve"][0,0,0], self.true_st_params[1], rtol=0.6, atol=0.15)
        # Voxel (0,1,0) - should fit to different params (data was Ct_clean_voxel0 * 0.5)
        # This is a rough check, actual fit depends on noise and optimizer path
        self.assertTrue(0 < param_maps["Ktrans"][0,1,0] < self.true_st_params[0])
        
        # Voxel (1,0,0) - masked out
        self.assertTrue(np.isnan(param_maps["Ktrans"][1,0,0]))
        # Voxel (1,1,0) - all zeros Ct, should result in error from worker, thus NaN
        self.assertTrue(np.isnan(param_maps["Ktrans"][1,1,0]))


    def test_voxelwise_all_masked_out(self):
        """Test behavior when the mask excludes all voxels."""
        mask_all_false = np.zeros(self.Ct_data_shape[:3], dtype=bool)
        param_maps = modeling.fit_standard_tofts_voxelwise(
            self.Ct_data, self.t_tissue, self.t_aif, self.Cp_aif,
            mask=mask_all_false, num_processes=1
        )
        self.assertTrue(np.all(np.isnan(param_maps["Ktrans"])))
        self.assertTrue(np.all(np.isnan(param_maps["ve"])))

    def test_voxelwise_ct_all_nans(self):
        """Test voxel-wise fitting when input Ct_data is all NaNs."""
        Ct_all_nans = np.full_like(self.Ct_data, np.nan)
        param_maps = modeling.fit_standard_tofts_voxelwise(
            Ct_all_nans, self.t_tissue, self.t_aif, self.Cp_aif,
            mask=None, num_processes=1 # No mask, process all
        )
        self.assertTrue(np.all(np.isnan(param_maps["Ktrans"])))
        self.assertTrue(np.all(np.isnan(param_maps["ve"])))


if __name__ == '__main__':
    unittest.main()
