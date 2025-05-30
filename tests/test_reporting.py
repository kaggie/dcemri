import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal # Not used directly, but good to have for np arrays
import csv
import tempfile
import os
import shutil

# Add project root for imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import reporting

class TestCalculateRoiStatistics(unittest.TestCase): # New class for this function
    def setUp(self):
        self.data_map_slice = np.array([
            [1, 2, 3, np.nan],
            [4, 5, np.inf, 7],
            [8, 9, 10, 11]
        ], dtype=float) # 3x4 slice

    def test_valid_roi_all_numeric(self):
        """Test with a valid ROI containing only numeric data."""
        roi_mask = np.array([
            [True, True, False, False],
            [True, True, False, False],
            [False, False, False, False]
        ], dtype=bool)
        # Expected values from [1, 2, 4, 5]
        stats = reporting.calculate_roi_statistics(self.data_map_slice, roi_mask)
        self.assertEqual(stats["N"], 4)
        self.assertEqual(stats["N_valid"], 4)
        self.assertAlmostEqual(stats["Mean"], np.mean([1,2,4,5])) 
        self.assertAlmostEqual(stats["StdDev"], np.std([1,2,4,5])) 
        self.assertAlmostEqual(stats["Median"], np.median([1,2,4,5])) 
        self.assertAlmostEqual(stats["Min"], 1.0)
        self.assertAlmostEqual(stats["Max"], 5.0)

    def test_roi_with_nans(self):
        """Test ROI including NaN values."""
        roi_mask = np.array([
            [True, True, True, True], # Includes a NaN
            [False, False, False, False],
            [False, False, False, False]
        ], dtype=bool)
        stats = reporting.calculate_roi_statistics(self.data_map_slice, roi_mask)
        self.assertEqual(stats["N"], 4) 
        self.assertEqual(stats["N_valid"], 3) 
        self.assertAlmostEqual(stats["Mean"], np.nanmean([1,2,3,np.nan])) 
        self.assertAlmostEqual(stats["StdDev"], np.nanstd([1,2,3,np.nan])) 
        self.assertAlmostEqual(stats["Median"], np.nanmedian([1,2,3,np.nan]))
        self.assertAlmostEqual(stats["Min"], np.nanmin([1,2,3,np.nan])) 
        self.assertAlmostEqual(stats["Max"], np.nanmax([1,2,3,np.nan])) 

    def test_roi_with_infs(self): # New test
        """Test ROI including Inf values."""
        roi_mask = np.array([
            [False, False, False, False],
            [False, True, True, True], # Values: [5, np.inf, 7]
            [False, False, False, False]
        ], dtype=bool)
        stats = reporting.calculate_roi_statistics(self.data_map_slice, roi_mask)
        self.assertEqual(stats["N"], 3)
        self.assertEqual(stats["N_valid"], 3) 
        self.assertEqual(stats["Mean"], np.inf)
        self.assertTrue(np.isnan(stats["StdDev"]) or np.isinf(stats["StdDev"])) 
        self.assertAlmostEqual(stats["Median"], np.median([5, np.inf, 7])) 
        self.assertAlmostEqual(stats["Min"], 5.0)
        self.assertEqual(stats["Max"], np.inf)
        
    def test_empty_roi_all_false_mask(self):
        """Test with an ROI mask that is all False."""
        roi_mask = np.zeros_like(self.data_map_slice, dtype=bool)
        stats = reporting.calculate_roi_statistics(self.data_map_slice, roi_mask)
        self.assertEqual(stats["N"], 0)
        self.assertEqual(stats["N_valid"], 0)
        self.assertTrue(all(np.isnan(stats[key]) for key in ["Mean", "StdDev", "Median", "Min", "Max"]))

    def test_map_all_nans(self):
        """Test with a data map slice that is all NaNs."""
        data_all_nans = np.full_like(self.data_map_slice, np.nan)
        roi_mask = np.ones_like(self.data_map_slice, dtype=bool) 
        stats = reporting.calculate_roi_statistics(data_all_nans, roi_mask)
        self.assertEqual(stats["N"], self.data_map_slice.size)
        self.assertEqual(stats["N_valid"], 0)
        self.assertTrue(all(np.isnan(stats[key]) for key in ["Mean", "StdDev", "Median", "Min", "Max"]))

    def test_input_validation(self): # New test for input validation
        """Test input validation for shapes and dimensions."""
        with self.assertRaisesRegex(ValueError, "data_map_slice must be a 2D NumPy array."):
            reporting.calculate_roi_statistics(np.zeros((2,2,2)), np.zeros((2,2,2), dtype=bool))
        with self.assertRaisesRegex(ValueError, "roi_mask_slice must be a 2D NumPy array."):
            reporting.calculate_roi_statistics(np.zeros((2,2)), np.zeros((2,2,2), dtype=bool))
        with self.assertRaisesRegex(ValueError, "data_map_slice and roi_mask_slice must have the same shape."):
            reporting.calculate_roi_statistics(np.zeros((2,2)), np.zeros((3,3), dtype=bool))


class TestFormatRoiStatisticsToString(unittest.TestCase): # New Class
    """Tests for the format_roi_statistics_to_string function."""

    def test_format_valid_stats(self):
        stats = {"N": 10, "N_valid": 8, "Mean": 12.34567, "StdDev": 2.0, "Median": 12.0, "Min": 5.0, "Max": 20.0}
        formatted_str = reporting.format_roi_statistics_to_string(stats, "Ktrans", "Tumor Core")
        self.assertIn("Statistics for Tumor Core on parameter map 'Ktrans':", formatted_str) # Updated expected string
        self.assertIn("Mean: 12.3457", formatted_str) 
        self.assertIn("StdDev: 2.0000", formatted_str)
        self.assertIn("N: 10", formatted_str)
        self.assertIn("N_valid: 8", formatted_str)

    def test_format_stats_with_nans(self):
        stats = {"N": 5, "N_valid": 0, "Mean": np.nan, "StdDev": np.nan, "Median": np.nan, "Min": np.nan, "Max": np.nan}
        formatted_str = reporting.format_roi_statistics_to_string(stats, "Ve", "Necrotic Area")
        self.assertEqual(formatted_str, "No valid data points found in Necrotic Area for parameter map 'Ve'.") # Updated expected string

    def test_format_empty_or_none_stats(self):
        self.assertEqual(reporting.format_roi_statistics_to_string(None, "Ktrans"), 
                         "No valid data points found in ROI for parameter map 'Ktrans'.") # Updated
        self.assertEqual(reporting.format_roi_statistics_to_string({}, "Ktrans"), 
                         "No valid data points found in ROI for parameter map 'Ktrans'.") # Updated
        self.assertEqual(reporting.format_roi_statistics_to_string({"N_valid":0}, "Ktrans"), 
                         "No valid data points found in ROI for parameter map 'Ktrans'.") # Updated


class TestSaveMultipleRoiStatisticsCsv(unittest.TestCase): # New Class
    """Tests for the save_multiple_roi_statistics_csv function."""
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.test_dir, "roi_stats.csv")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_valid_data(self):
        stats_list = [
            ("Ktrans", 0, "ROI1_Slice0", {"N": 10, "N_valid": 10, "Mean": 0.25, "StdDev": 0.05, "Median": 0.24, "Min": 0.1, "Max": 0.4}),
            ("Ve", 0, "ROI1_Slice0", {"N": 10, "N_valid": 8, "Mean": 0.5, "StdDev": 0.1, "Median": 0.49, "Min": 0.3, "Max": 0.7, "ExtraStat": 1.0}),
            ("Ktrans", 1, "ROI2_Slice1", {"N": 5, "N_valid": 0, "Mean": np.nan, "StdDev": np.nan, "Median": np.nan, "Min": np.nan, "Max": np.nan})
        ]
        reporting.save_multiple_roi_statistics_csv(stats_list, self.filepath)
        
        self.assertTrue(os.path.exists(self.filepath))
        with open(self.filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            # Fieldnames should be dynamically determined by first valid dict + ExtraStat
            self.assertEqual(reader.fieldnames, ['MapName', 'SliceIndex', 'ROIName', 'N', 'N_valid', 'Mean', 'StdDev', 'Median', 'Min', 'Max', 'ExtraStat'])
            rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['MapName'], "Ktrans")
            self.assertEqual(rows[0]['ROIName'], "ROI1_Slice0")
            self.assertEqual(float(rows[0]['Mean']), 0.25)
            # ExtraStat should be nan or empty for the first row as it's not in its dict
            self.assertTrue(rows[0]['ExtraStat'] == str(np.nan) or rows[0]['ExtraStat'] == "")
            
            self.assertEqual(rows[1]['MapName'], "Ve")
            self.assertEqual(float(rows[1]['ExtraStat']), 1.0) # This row defined ExtraStat

            self.assertEqual(rows[2]['MapName'], "Ktrans")
            self.assertEqual(rows[2]['ROIName'], "ROI2_Slice1")
            self.assertTrue(rows[2]['Mean'] == str(np.nan) or rows[2]['Mean'] == "")

    def test_save_no_statistics(self):
        """Test saving when the input list is empty."""
        reporting.save_multiple_roi_statistics_csv([], self.filepath)
        self.assertFalse(os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0, 
                         "CSV file should not be created or should be empty if no stats are provided.")

    def test_save_all_rois_empty_nan_stats(self): # New test
        """Test saving when all ROIs have no valid data."""
        stats_list = [
            ("Ktrans", 0, "ROI1", {"N": 5, "N_valid": 0, "Mean": np.nan, "StdDev": np.nan, "Median": np.nan, "Min": np.nan, "Max": np.nan}),
            ("Ve", 1, "ROI2", {"N": 0, "N_valid": 0, "Mean": np.nan, "StdDev": np.nan, "Median": np.nan, "Min": np.nan, "Max": np.nan})
        ]
        reporting.save_multiple_roi_statistics_csv(stats_list, self.filepath)
        self.assertTrue(os.path.exists(self.filepath)) 
        with open(self.filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            # Default keys because no dict had N_valid > 0 to get keys from
            expected_headers = ['MapName', 'SliceIndex', 'ROIName', 'N', 'N_valid', 'Mean', 'StdDev', 'Median', 'Min', 'Max']
            self.assertEqual(reader.fieldnames, expected_headers)
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            for row in rows:
                for key in ['Mean', 'StdDev', 'Median', 'Min', 'Max']:
                    # np.nan is written as an empty string by csv.DictWriter when the value is float(np.nan)
                    # If it was a string "nan" it would be "nan"
                    self.assertTrue(row[key] == "" or row[key] == str(np.nan)) 

    def test_save_invalid_filepath(self): # New test
        """Test saving to an invalid filepath (e.g., a directory)."""
        stats_list = [("Ktrans", 0, "ROI1", {"N": 10, "Mean": 0.25})]
        # On Unix-like systems, saving to a directory path raises IsADirectoryError, a subclass of IOError/OSError.
        # On Windows, it might raise PermissionError, also a subclass of OSError.
        with self.assertRaises(IOError): 
            reporting.save_multiple_roi_statistics_csv(stats_list, self.test_dir)


class TestSaveRoiStatisticsCsvOld(unittest.TestCase): # New Class for the old function
    """Tests for the older save_roi_statistics_csv function."""
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.test_dir, "single_roi_stats.csv")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_single_roi_valid(self):
        stats_dict = {"N": 10, "N_valid": 8, "Mean": 0.5, "StdDev": 0.1, "Median": 0.45}
        reporting.save_roi_statistics_csv(stats_dict, self.filepath, "Ktrans_Map", "Tumor_ROI")
        
        self.assertTrue(os.path.exists(self.filepath))
        with open(self.filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ['MapName', 'ROIName', 'Statistic', 'Value'])
            rows = list(reader)
            self.assertEqual(len(rows), len(stats_dict))
            
            expected_data_dict = {stat: str(val) for stat, val in stats_dict.items()}
            for row in rows:
                self.assertEqual(row['MapName'], "Ktrans_Map")
                self.assertEqual(row['ROIName'], "Tumor_ROI")
                self.assertEqual(row['Value'], expected_data_dict[row['Statistic']])
                
    def test_save_single_roi_empty_stats_valueerror(self): # Renamed for clarity
        with self.assertRaisesRegex(ValueError, "No statistics data to save."): # Check specific error message
            reporting.save_roi_statistics_csv({}, self.filepath, "TestMap", "TestROI")


if __name__ == '__main__':
    unittest.main()
