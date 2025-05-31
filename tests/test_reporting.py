import unittest
import numpy as np
import os
import csv
from core.reporting import (
    calculate_roi_statistics,
    format_roi_statistics_to_string,
    save_multiple_roi_statistics_csv,
)

try:
    from core.reporting import save_roi_statistics_csv
    SAVE_ROI_STATS_CSV_EXISTS = True
except ImportError:
    SAVE_ROI_STATS_CSV_EXISTS = False


class TestReporting(unittest.TestCase):
    def setUp(self):
        self.data_map_slice_valid = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=float)

        self.roi_mask_slice_A = np.array([
            [True, True, False],
            [True, False, False],
            [False, False, False]
        ], dtype=bool)

        self.roi_mask_slice_B = np.array([
            [False, False, False],
            [False, True, True],
            [False, True, True]
        ], dtype=bool)
        
        self.roi_mask_all_false = np.zeros_like(self.data_map_slice_valid, dtype=bool)

        self.data_map_with_nans = self.data_map_slice_valid.copy()
        self.data_map_with_nans[0, 0] = np.nan

        self.roi_mask_only_nans = np.array([
            [True, False, False],
            [False, False, False],
            [False, False, False]
        ], dtype=bool)

        self.test_csv_filename = "test_stats.csv"
        self.test_multi_csv_filename = "test_multi_stats.csv"


    def tearDown(self):
        for filename in [self.test_csv_filename, self.test_multi_csv_filename]:
            if os.path.exists(filename):
                os.remove(filename)

    # --- Tests for calculate_roi_statistics ---
    def test_calc_stats_valid_roi(self):
        stats = calculate_roi_statistics(self.data_map_slice_valid, self.roi_mask_slice_A)
        self.assertEqual(stats['N'], 3); self.assertEqual(stats['N_valid'], 3)
        self.assertAlmostEqual(stats['Mean'], np.mean([1.0, 2.0, 4.0]))
        self.assertAlmostEqual(stats['StdDev'], np.std([1.0, 2.0, 4.0]))

    def test_calc_stats_roi_all_false(self):
        stats = calculate_roi_statistics(self.data_map_slice_valid, self.roi_mask_all_false)
        self.assertEqual(stats['N'], 0); self.assertEqual(stats['N_valid'], 0)
        self.assertTrue(np.isnan(stats['Mean']))

    def test_calc_stats_roi_with_only_nans(self):
        stats = calculate_roi_statistics(self.data_map_with_nans, self.roi_mask_only_nans)
        self.assertEqual(stats['N'], 1); self.assertEqual(stats['N_valid'], 0)
        self.assertTrue(np.isnan(stats['Mean']))

    def test_calc_stats_roi_mix_valid_and_nans(self):
        stats = calculate_roi_statistics(self.data_map_with_nans, self.roi_mask_slice_A)
        self.assertEqual(stats['N'], 3); self.assertEqual(stats['N_valid'], 2)
        self.assertAlmostEqual(stats['Mean'], np.mean([2.0, 4.0]))

    def test_calc_stats_empty_arrays_matching_shape(self):
        empty_data = np.empty((0,0), dtype=float)
        empty_roi = np.empty((0,0), dtype=bool)
        stats = calculate_roi_statistics(empty_data, empty_roi)
        self.assertEqual(stats['N'], 0)
        self.assertEqual(stats['N_valid'], 0)
        self.assertTrue(np.isnan(stats['Mean']))

    def test_calc_stats_empty_arrays_mismatch_shape_value_error(self):
        empty_data = np.empty((0,1), dtype=float) # Shape (0,1)
        empty_roi = np.empty((0,0), dtype=bool)  # Shape (0,0)
        with self.assertRaises(ValueError):
             calculate_roi_statistics(empty_data, empty_roi)


    def test_calc_stats_mismatched_shapes(self):
        roi_mismatch = np.array([[True, False],[False,True]], dtype=bool)
        with self.assertRaises(ValueError):
            calculate_roi_statistics(self.data_map_slice_valid, roi_mismatch)

    def test_calc_stats_non_2d_inputs(self):
        with self.assertRaises(ValueError):
            calculate_roi_statistics(np.array([1,2,3]), self.roi_mask_slice_A)
        with self.assertRaises(ValueError):
            calculate_roi_statistics(self.data_map_slice_valid, np.array([True,False,True]))

    # --- Tests for format_roi_statistics_to_string ---
    def test_format_stats_valid_dict(self):
        stats = {'N': 3, 'N_valid': 3, 'Mean': 2.333, 'StdDev': 1.247, 'Median': 2.0, 'Min': 1.0, 'Max': 4.0}
        formatted_str = format_roi_statistics_to_string(stats, "Ktrans", "ROI_A")
        self.assertIn("parameter map 'Ktrans'", formatted_str) # Check for map name
        self.assertIn("ROI_A", formatted_str) # Check for ROI name
        self.assertIn("Mean: 2.3330", formatted_str)

    def test_format_stats_n_valid_zero(self):
        stats = {'N': 0, 'N_valid': 0, 'Mean': np.nan, 'StdDev': np.nan, 'Median': np.nan, 'Min': np.nan, 'Max': np.nan}
        formatted_str = format_roi_statistics_to_string(stats, "Ve", "ROI_None")
        self.assertIn("No valid data points found", formatted_str)
        self.assertIn("'Ve'", formatted_str)
        self.assertIn("ROI_None", formatted_str)


    def test_format_stats_dict_none(self):
        formatted_str = format_roi_statistics_to_string(None, "Ktrans", "ROI_X")
        self.assertIn("No valid data points found", formatted_str)
        self.assertIn("'Ktrans'", formatted_str)
        self.assertIn("ROI_X", formatted_str)
        
    def test_format_stats_different_names(self):
        stats = {'N': 1, 'N_valid': 1, 'Mean': 5.0, 'StdDev': 0.0, 'Median': 5.0, 'Min': 5.0, 'Max': 5.0}
        formatted_str = format_roi_statistics_to_string(stats, "MyMap", "MyROI")
        self.assertIn("parameter map 'MyMap'", formatted_str)
        self.assertIn("MyROI", formatted_str)
        self.assertIn("Mean: 5.0000", formatted_str)

    # --- Tests for save_multiple_roi_statistics_csv ---
    def test_save_multi_stats_basic(self):
        stats1 = {'N': 3, 'N_valid': 3, 'Mean': 1.0, 'StdDev': 0.1, 'Median': 1.0, 'Min': 0.8, 'Max': 1.2}
        stats2 = {'N': 5, 'N_valid': 4, 'Mean': 2.5, 'StdDev': 0.5, 'Median': 2.5, 'Min': 2.0, 'Max': 3.0, 'CoV': 0.2}
        stats_results_list = [
            ("Ktrans", 0, "ROI_A", stats1), ("Ve", 0, "ROI_A", stats2),
            ("Ktrans", 1, "ROI_B", calculate_roi_statistics(self.data_map_slice_valid, self.roi_mask_slice_B))
        ]
        save_multiple_roi_statistics_csv(stats_results_list, self.test_multi_csv_filename)
        self.assertTrue(os.path.exists(self.test_multi_csv_filename))
        with open(self.test_multi_csv_filename, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertIn("CoV", header)
            rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0][header.index("CoV")], "nan") # Corrected: expect "nan" for missing keys
            self.assertEqual(rows[1][header.index("CoV")], "0.2")

    def test_save_multi_stats_empty_list(self):
        save_multiple_roi_statistics_csv([], self.test_multi_csv_filename)
        self.assertFalse(os.path.exists(self.test_multi_csv_filename)) # File should not be created

    def test_save_multi_stats_with_nans_or_empty_stats(self):
        stats_nan = {'N': 0, 'N_valid': 0, 'Mean': np.nan, 'StdDev': np.nan, 'Median': np.nan, 'Min': np.nan, 'Max': np.nan}
        stats_results_list = [("Ktrans", 0, "ROI_Empty", stats_nan), ("Ve", 1, "ROI_None", None)]
        save_multiple_roi_statistics_csv(stats_results_list, self.test_multi_csv_filename)
        self.assertTrue(os.path.exists(self.test_multi_csv_filename))
        with open(self.test_multi_csv_filename, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][header.index("Mean")], "nan")
            self.assertEqual(rows[1][header.index("Mean")], "nan") # Corrected to expect "nan" for None stats_dict as well

    # --- Tests for save_roi_statistics_csv (if it exists) ---
    @unittest.skipUnless(SAVE_ROI_STATS_CSV_EXISTS, "save_roi_statistics_csv function not found in core.reporting")
    def test_save_single_roi_stats_basic(self):
        stats = {'N': 3, 'N_valid': 3, 'Mean': 1.0, 'StdDev': 0.1, 'Median': 1.0, 'Min': 0.8, 'Max': 1.2}
        # Corrected argument order: stats_dict, filepath, map_name, roi_name
        save_roi_statistics_csv(stats, self.test_csv_filename, "Ktrans_map", "ROI_Test")
        self.assertTrue(os.path.exists(self.test_csv_filename))
        with open(self.test_csv_filename, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["MapName", "ROIName", "Statistic", "Value"])
            rows = list(reader)
            self.assertTrue(any(row == ["Ktrans_map", "ROI_Test", "Mean", "1.0"] for row in rows))

    @unittest.skipUnless(SAVE_ROI_STATS_CSV_EXISTS, "save_roi_statistics_csv function not found in core.reporting")
    def test_save_single_roi_stats_empty_or_none_stats(self):
        with self.assertRaises(ValueError): # This function raises ValueError for None/empty stats
            save_roi_statistics_csv(None, self.test_csv_filename, "Ktrans_map", "ROI_Empty") # Corrected
        stats_nan = {'N': 0, 'N_valid': 0, 'Mean': np.nan} # Simplified
        save_roi_statistics_csv(stats_nan, self.test_csv_filename, "Ktrans_map", "ROI_NaNs") # Corrected
        with open(self.test_csv_filename, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader) # Skip header for this check
            self.assertTrue(any(row == ["Ktrans_map", "ROI_NaNs", "Mean", "nan"] for row in rows if row))

if __name__ == '__main__':
    unittest.main()
