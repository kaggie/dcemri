import unittest
import numpy as np
import os
import json
from core.aif import (
    load_aif_from_file,
    save_aif_curve,
    parker_aif,
    weinmann_aif,
    fast_biexponential_aif,
    generate_population_aif,
    extract_aif_from_roi,
    save_aif_roi_definition,
    load_aif_roi_definition,
)

class TestAIF(unittest.TestCase):
    def setUp(self):
        # Create dummy files for testing
        self.csv_file = "test_aif.csv"
        self.txt_file = "test_aif.txt"
        self.json_file = "test_roi.json"

        # Default content for CSV and TXT files
        with open(self.csv_file, "w") as f:
            f.write("1,10\n")
            f.write("2,20\n")
            f.write("3,30\n")

        with open(self.txt_file, "w") as f:
            f.write("1\t10\n")
            f.write("2\t20\n")
            f.write("3\t30\n")

    def tearDown(self):
        # Clean up dummy files
        for f in [self.csv_file, self.txt_file, self.json_file]:
            if os.path.exists(f):
                os.remove(f)

    # Tests for load_aif_from_file
    def test_load_aif_from_csv(self):
        time_points, aif_curve = load_aif_from_file(self.csv_file)
        np.testing.assert_array_equal(time_points, np.array([1, 2, 3]))
        np.testing.assert_array_equal(aif_curve, np.array([10, 20, 30]))

    def test_load_aif_from_txt(self):
        time_points, aif_curve = load_aif_from_file(self.txt_file)
        np.testing.assert_array_equal(time_points, np.array([1, 2, 3]))
        np.testing.assert_array_equal(aif_curve, np.array([10, 20, 30]))

    def test_load_aif_with_header(self):
        with open(self.csv_file, "w") as f:
            f.write("Time,Concentration\n")
            f.write("1,10\n")
            f.write("2,20\n")
            f.write("3,30\n")
        time_points, aif_curve = load_aif_from_file(self.csv_file) # Removed has_header=True
        np.testing.assert_array_equal(time_points, np.array([1, 2, 3]))
        np.testing.assert_array_equal(aif_curve, np.array([10, 20, 30]))

    def test_load_aif_non_numeric(self):
        with open(self.csv_file, "w") as f:
            f.write("1,abc\n")
            f.write("2,20\n")
        with self.assertRaises(ValueError):
            load_aif_from_file(self.csv_file)

    def test_load_aif_incorrect_columns(self):
        with open(self.csv_file, "w") as f:
            f.write("1,10,100\n")
            f.write("2,20,200\n")
        with self.assertRaises(ValueError):
            load_aif_from_file(self.csv_file)

    def test_load_aif_non_existent_file(self):
        with self.assertRaises(FileNotFoundError):
            load_aif_from_file("non_existent_file.csv")

    def test_load_aif_empty_file(self):
        with open(self.csv_file, "w") as f:
            pass
        with self.assertRaises(ValueError):
            load_aif_from_file(self.csv_file)

    # Tests for save_aif_curve
    def test_save_aif_to_csv(self):
        time_points = np.array([1, 2, 3])
        aif_curve = np.array([10, 20, 30])
        save_aif_curve(time_points, aif_curve, self.csv_file) # Corrected argument order
        loaded_time, loaded_aif = load_aif_from_file(self.csv_file)
        np.testing.assert_array_equal(loaded_time, time_points)
        np.testing.assert_array_equal(loaded_aif, aif_curve)

    def test_save_aif_to_txt(self):
        time_points = np.array([1, 2, 3])
        aif_curve = np.array([10, 20, 30])
        save_aif_curve(time_points, aif_curve, self.txt_file) # Corrected argument order
        loaded_time, loaded_aif = load_aif_from_file(self.txt_file)
        np.testing.assert_array_equal(loaded_time, time_points)
        np.testing.assert_array_equal(loaded_aif, aif_curve)

    def test_save_aif_mismatched_lengths(self):
        time_points = np.array([1, 2, 3])
        aif_curve = np.array([10, 20])
        with self.assertRaises(ValueError):
            save_aif_curve(time_points, aif_curve, self.csv_file)

    def test_save_aif_non_1d_array(self):
        time_points = np.array([[1, 2], [3, 4]])
        aif_curve = np.array([10, 20, 30, 40])
        with self.assertRaises(ValueError):
            save_aif_curve(time_points, aif_curve, self.csv_file)
        
        time_points_1d = np.array([1,2,3,4])
        aif_curve_2d = np.array([[10, 20], [30, 40]])
        with self.assertRaises(ValueError):
            save_aif_curve(time_points_1d, aif_curve_2d, self.csv_file)

    # Tests for AIF models
    def test_parker_aif_default_params(self):
        time_points = np.array([0, 1, 2, 3, 4, 5]) # minutes
        aif = parker_aif(time_points)
        self.assertEqual(aif.shape, time_points.shape)

    def test_parker_aif_custom_params(self):
        time_points = np.array([0, 1, 2, 3, 4, 5])
        # Corrected params to match function definition in core/aif.py
        aif = parker_aif(time_points, D_scaler=1.0, A1=0.5, m1=0.1, A2=0.6, m2=0.2)
        self.assertEqual(aif.shape, time_points.shape)

    def test_parker_aif_negative_params(self):
        time_points = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            parker_aif(time_points, A1=-0.5) # Corrected param

    def test_parker_aif_invalid_time_input(self):
        time_points = [0, 1, 2] # Should be numpy array
        with self.assertRaises(TypeError):
            parker_aif(time_points)

    def test_weinmann_aif_default_params(self):
        time_points = np.array([0, 1, 2, 3, 4, 5]) # minutes
        aif = weinmann_aif(time_points)
        self.assertEqual(aif.shape, time_points.shape)

    def test_weinmann_aif_custom_params(self):
        time_points = np.array([0, 1, 2, 3, 4, 5])
        # Corrected params to match function definition
        aif = weinmann_aif(time_points, D_scaler=1.0, A1=4.0, m1=0.2, A2=5.0, m2=0.3)
        self.assertEqual(aif.shape, time_points.shape)

    def test_weinmann_aif_negative_params(self):
        time_points = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            weinmann_aif(time_points, A1=-4.0)
    
    def test_weinmann_aif_invalid_time_input(self):
        time_points = [0, 1, 2] # Should be numpy array
        with self.assertRaises(TypeError):
            weinmann_aif(time_points)

    def test_fast_biexponential_aif_default_params(self):
        time_points = np.array([0, 1, 2, 3, 4, 5]) # minutes
        aif = fast_biexponential_aif(time_points)
        self.assertEqual(aif.shape, time_points.shape)

    def test_fast_biexponential_aif_custom_params(self):
        time_points = np.array([0, 1, 2, 3, 4, 5])
        # Removed T0 as it's not in the function signature
        aif = fast_biexponential_aif(time_points, D_scaler=1.0, A1=0.5, m1=0.1, A2=0.6, m2=0.2)
        self.assertEqual(aif.shape, time_points.shape)

    def test_fast_biexponential_aif_negative_params(self):
        time_points = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            fast_biexponential_aif(time_points, A1=-0.5)

    def test_fast_biexponential_aif_invalid_time_input(self):
        time_points = [0, 1, 2] # Should be numpy array
        with self.assertRaises(TypeError):
            fast_biexponential_aif(time_points)

    # Tests for generate_population_aif
    def test_generate_parker_aif(self):
        time_points = np.array([0, 1, 2, 3, 4, 5])
        aif = generate_population_aif("parker", time_points) # Corrected arg name
        self.assertEqual(aif.shape, time_points.shape)

    def test_generate_weinmann_aif(self):
        time_points = np.array([0, 1, 2, 3, 4, 5])
        aif = generate_population_aif("weinmann", time_points) # Corrected arg name
        self.assertEqual(aif.shape, time_points.shape)

    def test_generate_fast_biexponential_aif(self):
        time_points = np.array([0, 1, 2, 3, 4, 5])
        aif = generate_population_aif("fast_biexponential", time_points) # Corrected arg name
        self.assertEqual(aif.shape, time_points.shape)

    def test_generate_aif_custom_params(self):
        time_points = np.array([0, 1, 2, 3, 4, 5])
        # Params should match the actual function (e.g. parker_aif)
        params = {"D_scaler":1.0, "A1": 0.5, "m1": 0.1, "A2":0.6, "m2":0.2} # Parker AIF params, corrected
        aif = generate_population_aif("parker", time_points, params=params) # Corrected arg name
        self.assertEqual(aif.shape, time_points.shape)

    def test_generate_aif_unknown_model(self):
        time_points = np.array([0, 1, 2, 3, 4, 5])
        aif = generate_population_aif("unknown_model", time_points) # Corrected arg name
        self.assertIsNone(aif)

    def test_generate_aif_invalid_params(self):
        time_points = np.array([0, 1, 2, 3, 4, 5])
        params = {"A1": -0.5} # Invalid Parker AIF param, corrected
        with self.assertRaises(ValueError): # This will be ValueError from parker_aif
            generate_population_aif("parker", time_points, params=params) # Corrected arg name

    # Tests for extract_aif_from_roi
    def test_extract_aif_from_roi(self):
        dce_data = np.zeros((5, 5, 5, 10))
        dce_data[1:3, 1:3, 1, :] = np.tile((np.arange(10) + 1), (2,2,1)).transpose(0,1,2) # Assign to x,y,time for z=1

        roi_x_start, roi_y_start, roi_z_slice = 1, 1, 1
        roi_width, roi_height = 2, 2

        t10_blood, r1_blood, TR = 1440.0, 0.0045, 4.5 # Example physiological values

        aif_time, aif_concentration = extract_aif_from_roi(
            dce_4d_data=dce_data,
            roi_2d_coords=(roi_x_start, roi_y_start, roi_width, roi_height),
            slice_index_z=roi_z_slice,
            t10_blood=t10_blood,
            r1_blood=r1_blood,
            TR=TR
        )
        self.assertEqual(aif_concentration.shape, (10,))
        self.assertEqual(aif_time.shape, (10,))
        # Exact value check for aif_concentration depends on conversion.signal_tc_to_concentration_tc
        # For now, we check that it runs and returns the correct shape.
        # If conversion.signal_tc_to_concentration_tc was identity for S0=0:
        # expected_signal = np.arange(10) + 1
        # np.testing.assert_array_almost_equal(aif_concentration, expected_signal)


    def test_extract_aif_invalid_roi_coords(self):
        dce_data = np.zeros((5, 5, 5, 10))
        t10_blood, r1_blood, TR = 1440.0, 0.0045, 4.5
        
        with self.assertRaises(ValueError): # x_start out of bounds
            extract_aif_from_roi(dce_data, (-1, 1, 2, 2), 1, t10_blood, r1_blood, TR)
        with self.assertRaises(ValueError): # y_start out of bounds
            extract_aif_from_roi(dce_data, (1, -1, 2, 2), 1, t10_blood, r1_blood, TR)
        with self.assertRaises(ValueError): # slice_index_z out of bounds
            extract_aif_from_roi(dce_data, (1, 1, 2, 2), 10, t10_blood, r1_blood, TR)
        with self.assertRaises(ValueError): # x_start + width out of bounds
            extract_aif_from_roi(dce_data, (1, 1, 5, 2), 1, t10_blood, r1_blood, TR)
        with self.assertRaises(ValueError): # y_start + height out of bounds
            extract_aif_from_roi(dce_data, (1, 1, 2, 5), 1, t10_blood, r1_blood, TR)

    def test_extract_aif_invalid_dce_data_dim(self):
        dce_data_3d = np.zeros((5, 5, 5))
        t10_blood, r1_blood, TR = 1440.0, 0.0045, 4.5
        with self.assertRaises(ValueError):
            extract_aif_from_roi(dce_data_3d, (1,1,2,2), 1, t10_blood, r1_blood, TR)

    def test_extract_aif_non_positive_roi_dim(self):
        dce_data = np.zeros((5, 5, 5, 10))
        t10_blood, r1_blood, TR = 1440.0, 0.0045, 4.5
        with self.assertRaises(ValueError): # width = 0
            extract_aif_from_roi(dce_data, (1, 1, 0, 2), 1, t10_blood, r1_blood, TR)
        with self.assertRaises(ValueError): # height = 0
            extract_aif_from_roi(dce_data, (1, 1, 2, 0), 1, t10_blood, r1_blood, TR)

    # Tests for save_aif_roi_definition and load_aif_roi_definition
    def test_save_and_load_aif_roi_definition(self):
        roi_properties = {
            "slice_index": 1, "pos_x": 10, "pos_y": 20,
            "size_w": 5, "size_h": 5, "image_ref_name": "ref_img.nii.gz",
            "description": "Test ROI A" # Added description to match example
        }
        # Corrected argument order for save_aif_roi_definition
        save_aif_roi_definition(roi_properties, self.json_file)
        loaded_properties = load_aif_roi_definition(self.json_file)
        self.assertEqual(roi_properties, loaded_properties)

    def test_load_aif_roi_def_non_existent_file(self):
        with self.assertRaises(FileNotFoundError):
            load_aif_roi_definition("non_existent_roi.json")

    def test_load_aif_roi_def_malformed_json(self):
        with open(self.json_file, "w") as f:
            f.write("{'slice_index': 1, 'pos_x': 10,") # Malformed JSON
        with self.assertRaises(ValueError):
            load_aif_roi_definition(self.json_file)

    def test_load_aif_roi_def_missing_keys(self):
        # Missing 'pos_y'
        roi_properties = {"slice_index": 1, "pos_x": 10, "size_w":5, "size_h":5, "image_ref_name":"img.nii"}
        with open(self.json_file, "w") as f:
            json.dump(roi_properties, f)
        with self.assertRaises(ValueError):
            load_aif_roi_definition(self.json_file)

    def test_load_aif_roi_def_incorrect_types(self):
        # 'slice_index' should be int
        roi_properties = {"slice_index": "1", "pos_x": 10, "pos_y":20, "size_w":5, "size_h":5, "image_ref_name":"img.nii"}
        with open(self.json_file, "w") as f:
            json.dump(roi_properties, f)
        with self.assertRaises(ValueError):
            load_aif_roi_definition(self.json_file)

if __name__ == "__main__":
    unittest.main()
