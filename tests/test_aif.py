import unittest
import numpy as np
import os
import tempfile
import shutil
import csv
import json 
from unittest.mock import patch, MagicMock

# Add project root for imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import aif
# No need to import core.conversion directly in test_aif if only mocking its functions that aif.py uses.

class TestAIFFileIO(unittest.TestCase):
    """Tests for AIF file loading and saving."""
    def setUp(self):
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        # print(f"TestAIFFileIO: Test directory created at {self.test_dir}")

    def tearDown(self):
        """Remove the temporary directory after tests."""
        # print(f"TestAIFFileIO: Removing test directory {self.test_dir}")
        shutil.rmtree(self.test_dir)

    def _create_test_aif_file(self, filename, header_cols, data_rows, delimiter=','):
        """Helper to create AIF files for testing."""
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            if header_cols:
                writer.writerow(header_cols)
            writer.writerows(data_rows)
        return filepath

    def test_load_aif_from_file_csv_comma_header(self):
        """Test loading a comma-separated CSV with header."""
        times = np.array([0, 1, 2], dtype=float)
        concentrations = np.array([0.0, 0.1, 0.2], dtype=float)
        filepath = self._create_test_aif_file("test_aif_comma.csv",
                                              ["Time (s)", "Concentration (mM)"],
                                              zip(times, concentrations), delimiter=',')
        loaded_times, loaded_concs = aif.load_aif_from_file(filepath)
        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_concs, concentrations)

    def test_load_aif_from_file_csv_semicolon_no_header(self):
        """Test loading a semicolon-separated CSV without header."""
        times = np.array([0.1, 1.1, 2.1], dtype=float)
        concentrations = np.array([0.05, 0.15, 0.25], dtype=float)
        filepath = self._create_test_aif_file("test_aif_semi.csv",
                                              None, # No header
                                              zip(times, concentrations), delimiter=';')
        loaded_times, loaded_concs = aif.load_aif_from_file(filepath)
        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_concs, concentrations)

    def test_load_aif_from_file_txt_tab_header(self):
        """Test loading a tab-separated TXT file with header."""
        times = np.array([10, 20, 30], dtype=float)
        concentrations = np.array([0.5, 1.5, 1.0], dtype=float)
        filepath = self._create_test_aif_file("test_aif_tab.txt",
                                              ["Time", "Concentration"],
                                              zip(times, concentrations), delimiter='\t')
        loaded_times, loaded_concs = aif.load_aif_from_file(filepath)
        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_concs, concentrations)

    def test_load_aif_from_file_txt_space_no_header(self):
        """Test loading a space-separated TXT file without header."""
        times = np.array([0, 5, 10], dtype=float)
        concentrations = np.array([0.0, 0.2, 0.4], dtype=float)
        filepath = self._create_test_aif_file("test_aif_space.txt",
                                              None, # No header
                                              zip(times, concentrations), delimiter=' ')
        loaded_times, loaded_concs = aif.load_aif_from_file(filepath)
        np.testing.assert_array_almost_equal(loaded_times, times)
        np.testing.assert_array_almost_equal(loaded_concs, concentrations)

    def test_load_aif_from_file_not_found(self):
        """Test FileNotFoundError for a non-existent file."""
        filepath = os.path.join(self.test_dir, "non_existent_aif.txt")
        with self.assertRaises(FileNotFoundError):
            aif.load_aif_from_file(filepath)

    def test_load_aif_from_file_incorrect_columns(self):
        """Test ValueError for file with incorrect number of columns."""
        filepath = self._create_test_aif_file("bad_cols.txt", None, [[0, 0.1, 0.05]], delimiter='\t')
        with self.assertRaisesRegex(ValueError, "Expected 2 columns"):
            aif.load_aif_from_file(filepath)

    def test_load_aif_from_file_non_numeric_data(self):
        """Test ValueError for file with non-numeric data."""
        filepath = self._create_test_aif_file("non_numeric.txt", None, [[0, "text"]], delimiter='\t')
        with self.assertRaisesRegex(ValueError, "Non-numeric data found"):
            aif.load_aif_from_file(filepath)
            
    def test_load_aif_from_file_empty_content(self):
        """Test ValueError for an empty AIF file."""
        filepath = os.path.join(self.test_dir, "empty_aif.txt")
        with open(filepath, 'w') as f: # Create an empty file
            pass
        with self.assertRaisesRegex(ValueError, "AIF file is empty"):
            aif.load_aif_from_file(filepath)

    def test_load_aif_from_file_header_only(self):
        """Test ValueError for AIF file containing only a header."""
        filepath = self._create_test_aif_file("header_only.csv", ["Time", "Conc"], [])
        with self.assertRaisesRegex(ValueError, "No numeric data found"):
            aif.load_aif_from_file(filepath)

    def test_save_aif_curve_csv_and_txt(self):
        """Test saving AIF curve to both CSV and TXT and verify content."""
        times = np.array([0, 1.1, 2.2, 3.55], dtype=float)
        concentrations = np.array([0.0, 0.155, 0.255, 0.105], dtype=float)

        for ext, delimiter_expected in [(".csv", ","), (".txt", "\t")]:
            with self.subTest(extension=ext):
                # Use a unique filename for each subtest within the test_dir
                filepath = os.path.join(self.test_dir, f"saved_aif{ext}")

                aif.save_aif_curve(times, concentrations, filepath)

                with open(filepath, 'r', newline='') as f_read:
                    reader = csv.reader(f_read, delimiter=delimiter_expected)
                    header = next(reader)
                    self.assertEqual(header, ['Time', 'Concentration'])
                    loaded_data = list(reader)
                    self.assertEqual(len(loaded_data), len(times))
                    for i, row in enumerate(loaded_data):
                        self.assertAlmostEqual(float(row[0]), times[i], places=5)
                        self.assertAlmostEqual(float(row[1]), concentrations[i], places=5)

    def test_save_aif_curve_invalid_inputs(self):
        """Test save_aif_curve with invalid inputs."""
        with self.assertRaisesRegex(ValueError, "Time points and concentrations arrays must have the same length."):
            aif.save_aif_curve(np.array([1,2]), np.array([1,2,3]), "test_len.csv")
        with self.assertRaisesRegex(ValueError, "Time points and concentrations must be 1D arrays."):
            aif.save_aif_curve(np.array([[1],[2]]), np.array([[1],[2]]), "test_dim.csv")


class TestPopulationAIFs(unittest.TestCase):
    """Tests for population AIF model functions."""
    def setUp(self):
        """Set up common time points for AIF tests (assume minutes for parameter consistency)."""
        self.time_points_minutes = np.array([0, 0.1, 0.5, 1.0, 2.0, 5.0])
        self.time_points_empty = np.array([])
        self.time_points_single = np.array([1.0])

    def _test_aif_model(self, model_func, expected_values_func, model_name):
        """Helper to test an AIF model with default and custom D_scaler."""
        # Test with default D_scaler = 1.0
        generated_concs = model_func(self.time_points_minutes)
        expected_concs = expected_values_func(self.time_points_minutes, D_scaler=1.0)
        np.testing.assert_array_almost_equal(generated_concs, expected_concs, decimal=5,
                                              err_msg=f"{model_name} default D_scaler mismatch.")

        # Test with custom D_scaler
        custom_D_scaler = 0.5
        generated_concs_scaled = model_func(self.time_points_minutes, D_scaler=custom_D_scaler)
        expected_concs_scaled = expected_values_func(self.time_points_minutes, D_scaler=custom_D_scaler)
        np.testing.assert_array_almost_equal(generated_concs_scaled, expected_concs_scaled, decimal=5,
                                              err_msg=f"{model_name} custom D_scaler mismatch.")
        # Also check direct scaling relationship
        np.testing.assert_array_almost_equal(generated_concs_scaled, generated_concs * custom_D_scaler, decimal=5,
                                              err_msg=f"{model_name} D_scaler did not scale output correctly.")

        # Test edge case time inputs
        self.assertEqual(len(model_func(self.time_points_empty)), 0, f"{model_name} with empty time_points failed.")
        self.assertEqual(len(model_func(self.time_points_single)), 1, f"{model_name} with single time_point failed.")

        # Test parameter validation (negative values)
        param_meta = aif.AIF_PARAMETER_METADATA.get(model_name.lower().replace(" ", "_"), []) # a bit hacky for key
        if param_meta:
            # Test one of the primary amplitude or rate parameters for negativity
            # Assuming A1 or m1 are generally the second/third parameter in metadata after D_scaler
            param_to_test_negativity = param_meta[1][0] if len(param_meta) > 1 else param_meta[0][0]
            with self.assertRaisesRegex(ValueError, "AIF parameters must be non-negative",
                                       msg=f"{model_name} did not raise ValueError for negative {param_to_test_negativity}"):
                model_func(self.time_points_minutes, **{param_to_test_negativity: -0.1})
        
        # Test TypeError for invalid time_points type
        with self.assertRaisesRegex(TypeError, "time_points must be a NumPy array",
                                   msg=f"{model_name} did not raise TypeError for list time_points"):
            model_func([0, 1, 2])


    def test_parker_aif(self):
        """Test Parker AIF model."""
        def expected_parker(t, D_scaler): # Parker params: A1=0.809, m1=0.171, A2=0.330, m2=2.05
            return D_scaler * (0.809 * np.exp(-0.171*t) + 0.330 * np.exp(-2.05*t))
        self._test_aif_model(aif.parker_aif, expected_parker, "Parker")

    def test_weinmann_aif(self):
        """Test Weinmann AIF model."""
        def expected_weinmann(t, D_scaler): # Weinmann params: A1=3.99, m1=0.144, A2=4.78, m2=0.0111
            return D_scaler * (3.99 * np.exp(-0.144*t) + 4.78 * np.exp(-0.0111*t))
        self._test_aif_model(aif.weinmann_aif, expected_weinmann, "Weinmann")

    def test_fast_biexponential_aif(self):
        """Test Fast Bi-exponential AIF model."""
        def expected_fast_biexp(t, D_scaler): # FastBiexp params: A1=0.6, m1=3.0, A2=0.4, m2=0.3
            return D_scaler * (0.6 * np.exp(-3.0*t) + 0.4 * np.exp(-0.3*t))
        self._test_aif_model(aif.fast_biexponential_aif, expected_fast_biexp, "Fast Bi-exponential")


class TestGeneratePopulationAIF(unittest.TestCase):
    """Tests for the `generate_population_aif` factory function."""
    def setUp(self):
        self.time_points = np.array([0, 0.5, 1.0, 2.0]) # Assume minutes

    def test_generate_known_models(self):
        """Test generating all known AIF models by name."""
        for model_name in aif.POPULATION_AIFS.keys():
            with self.subTest(model=model_name):
                generated_aif = aif.generate_population_aif(model_name, self.time_points)
                self.assertIsNotNone(generated_aif, f"Generation of {model_name} returned None.")
                self.assertEqual(len(generated_aif), len(self.time_points), f"{model_name} output length mismatch.")
                # Compare with direct call
                direct_call_aif = aif.POPULATION_AIFS[model_name](self.time_points)
                np.testing.assert_array_almost_equal(generated_aif, direct_call_aif,
                                                      err_msg=f"{model_name} generated by factory differs from direct call.")

    def test_generate_with_custom_params(self):
        """Test generating a model with custom parameters."""
        custom_params = {'D_scaler': 0.5, 'A1': 0.7} # Example for Parker
        generated_aif = aif.generate_population_aif("parker", self.time_points, params=custom_params)
        direct_call_aif = aif.parker_aif(self.time_points, **custom_params)
        np.testing.assert_array_almost_equal(generated_aif, direct_call_aif, decimal=5)

    def test_generate_unknown_model(self):
        """Test generating an unknown model name."""
        generated_aif = aif.generate_population_aif("unknown_model_name", self.time_points)
        self.assertIsNone(generated_aif, "Generating an unknown model should return None.")

    def test_generate_with_invalid_params_type(self):
        """Test generating a model with invalid parameter types."""
        with self.assertRaisesRegex(ValueError, "Error calling AIF model"): # Or TypeError depending on model
            aif.generate_population_aif("parker", self.time_points, params={'A1': "not_a_float"})

    def test_generate_with_incorrect_param_names(self):
        """Test generating a model with incorrect parameter names for that model."""
        # This should raise a ValueError due to TypeError in the underlying model call (unexpected kwarg)
        with self.assertRaisesRegex(ValueError, "Error calling AIF model"):
             aif.generate_population_aif("parker", self.time_points, params={'InvalidParam': 1.0})


class TestExtractAIFFromROI(unittest.TestCase):
    """Tests for `extract_aif_from_roi` function."""

    @patch('core.aif.conversion.signal_tc_to_concentration_tc') # Patch within core.aif's scope
    def test_extract_aif_valid_roi(self, mock_signal_to_conc):
        """Test AIF extraction with a valid ROI and mocked conversion."""
        # Mock setup
        mock_return_concentration = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_signal_to_conc.return_value = mock_return_concentration

        # Test data
        dce_data = np.zeros((10, 10, 5, 5)) # X, Y, Z, Time
        # Populate ROI area with specific signal values
        # ROI: x=2-3, y=2-3, z=1. Expected mean: (10+20+30+40)/4 = 25
        dce_data[2, 2, 1, :] = 10
        dce_data[3, 2, 1, :] = 20
        dce_data[2, 3, 1, :] = 30
        dce_data[3, 3, 1, :] = 40

        expected_mean_signal_in_roi = np.full(5, 25.0) # (10+20+30+40)/4 = 25, for all 5 time points

        roi_coords = (2, 2, 2, 2) # x_start, y_start, width, height
        slice_index_z = 1
        t10_blood, r1_blood, TR, baseline_pts = 1.4, 4.5, 0.005, 2

        aif_time, aif_conc = aif.extract_aif_from_roi(
            dce_data, roi_coords, slice_index_z,
            t10_blood, r1_blood, TR, baseline_pts
        )

        # Verify mock was called correctly
        mock_signal_to_conc.assert_called_once()
        call_args = mock_signal_to_conc.call_args[0]
        np.testing.assert_array_almost_equal(call_args[0], expected_mean_signal_in_roi,
                                              err_msg="Mean signal passed to conversion mock is incorrect.")
        self.assertEqual(call_args[1], t10_blood)
        self.assertEqual(call_args[2], r1_blood)
        self.assertEqual(call_args[3], TR)
        self.assertEqual(call_args[4], baseline_pts)

        # Verify output
        expected_time_vector = np.arange(dce_data.shape[3]) * TR
        np.testing.assert_array_almost_equal(aif_time, expected_time_vector)
        np.testing.assert_array_almost_equal(aif_conc, mock_return_concentration)

    def test_extract_aif_invalid_inputs(self):
        """Test `extract_aif_from_roi` with various invalid inputs."""
        dce_3d_data = np.zeros((10, 10, 5)) # Not 4D
        dce_4d_data = np.zeros((10, 10, 5, 5))

        with self.assertRaisesRegex(ValueError, "dce_4d_data must be a 4D array"):
            aif.extract_aif_from_roi(dce_3d_data, (0,0,1,1), 0, 1.4, 4.5, 0.005)

        with self.assertRaisesRegex(ValueError, "ROI start coordinates or Z-slice index are out of bounds"):
            aif.extract_aif_from_roi(dce_4d_data, (0,0,1,1), 10, 1.4, 4.5, 0.005) # Z out of bounds

        with self.assertRaisesRegex(ValueError, "ROI dimensions exceed data boundaries"):
            aif.extract_aif_from_roi(dce_4d_data, (8,8,5,5), 0, 1.4, 4.5, 0.005) # ROI exceeds X,Y

        with self.assertRaisesRegex(ValueError, "ROI width and height must be positive"):
            aif.extract_aif_from_roi(dce_4d_data, (0,0,0,1), 0, 1.4, 4.5, 0.005) # Zero width


class TestAIFROIDefinitionIO(unittest.TestCase):
    """Tests for saving and loading AIF ROI definitions."""
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_roi_props = {"slice_index": 10, "pos_x": 20.5, "pos_y": 30.0, "size_w": 15.2, "size_h": 10.8, "image_ref_name": "Mean DCE"}
    def tearDown(self): shutil.rmtree(self.test_dir)
    def test_save_load_aif_roi_definition(self):
        tmp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=self.test_dir, mode='w'); tmp_file_path = tmp_file.name; tmp_file.close() 
        aif.save_aif_roi_definition(self.sample_roi_props, tmp_file_path); loaded_props = aif.load_aif_roi_definition(tmp_file_path)
        self.assertIsNotNone(loaded_props); self.assertEqual(loaded_props, self.sample_roi_props); os.remove(tmp_file_path)
    def test_load_aif_roi_definition_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "no_such_roi.json")
        with self.assertRaises(FileNotFoundError): aif.load_aif_roi_definition(non_existent_path)
    def test_load_aif_roi_definition_bad_json(self):
        tmp_file_path = os.path.join(self.test_dir, "bad_roi.json")
        with open(tmp_file_path, 'w') as f: f.write("{'slice_index': 10, ...") 
        with self.assertRaisesRegex(ValueError, "Error decoding JSON"): aif.load_aif_roi_definition(tmp_file_path)
    def test_load_aif_roi_definition_missing_keys(self):
        tmp_file_path = os.path.join(self.test_dir, "missing_keys_roi.json"); bad_props = {"slice_index": 5, "pos_x": 10.0} 
        with open(tmp_file_path, 'w') as f: json.dump(bad_props, f)
        with self.assertRaisesRegex(ValueError, "Missing required key"): aif.load_aif_roi_definition(tmp_file_path)
    def test_load_aif_roi_definition_wrong_types(self):
        tmp_file_path = os.path.join(self.test_dir, "wrong_types_roi.json"); wrong_type_props = self.sample_roi_props.copy(); wrong_type_props["slice_index"] = "not_an_integer" 
        with open(tmp_file_path, 'w') as f: json.dump(wrong_type_props, f)
        with self.assertRaisesRegex(ValueError, "slice_index must be an integer"): aif.load_aif_roi_definition(tmp_file_path)
        wrong_type_props_2 = self.sample_roi_props.copy(); wrong_type_props_2["pos_x"] = "not_a_float" 
        with open(tmp_file_path, 'w') as f: json.dump(wrong_type_props_2, f)
        with self.assertRaisesRegex(ValueError, "ROI position/size values must be numeric"): aif.load_aif_roi_definition(tmp_file_path)

if __name__ == '__main__':
    unittest.main()
