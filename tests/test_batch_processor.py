import unittest
import subprocess
import os
import tempfile
import shutil
import sys
import nibabel as nib
import numpy as np
import csv

# Determine project root to construct path to batch_processor.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BATCH_PROCESSOR_SCRIPT_PATH = os.path.join(project_root, "batch_processor.py")
# Ensure batch_processor.py is executable by the python interpreter used for tests
if not os.path.exists(BATCH_PROCESSOR_SCRIPT_PATH):
    raise FileNotFoundError(f"Batch processor script not found at: {BATCH_PROCESSOR_SCRIPT_PATH}")


class TestBatchProcessorArgs(unittest.TestCase):
    """Tests for command-line argument parsing of batch_processor.py."""
    def setUp(self):
        self.script_path = BATCH_PROCESSOR_SCRIPT_PATH
        self.test_dir = tempfile.mkdtemp()

        # Create minimal dummy files needed for argparse to pass file existence checks if any are implicitly done by type=
        # For most argparse checks, files don't strictly need to exist unless type=argparse.FileType
        self.dce_dummy = os.path.join(self.test_dir, "dce.nii")
        self.t1_dummy = os.path.join(self.test_dir, "t1.nii")
        self.aif_dummy = os.path.join(self.test_dir, "aif.txt")
        self.mask_dummy = os.path.join(self.test_dir, "mask.nii")

        for f_path in [self.dce_dummy, self.t1_dummy, self.aif_dummy, self.mask_dummy]:
            with open(f_path, 'w') as f: f.write("dummy content") # Minimal content

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _run_script(self, args_list):
        """Helper to run the batch_processor.py script with given arguments."""
        return subprocess.run(
            [sys.executable, self.script_path] + args_list,
            capture_output=True, text=True, check=False # check=False to allow non-zero exit codes
        )

    def test_help_message(self):
        """Test that the script provides a help message and exits cleanly."""
        result = self._run_script(["--help"])
        self.assertEqual(result.returncode, 0, "Running with --help should exit with 0.")
        self.assertIn("usage: batch_processor.py", result.stdout.lower())
        self.assertIn("--dce", result.stdout)
        self.assertIn("--t1map", result.stdout)
        self.assertIn("--model", result.stdout)
        self.assertIn("--out_dir", result.stdout)
        self.assertIn("--aif_file", result.stdout)
        self.assertIn("--aif_pop_model", result.stdout)


    def test_missing_required_args(self):
        """Test script failure when required arguments are missing."""
        required_args_sets = [
            ["--t1map", self.t1_dummy, "--tr", "0.005", "--r1_relaxivity", "4.5", "--aif_file", self.aif_dummy, "--model", "Standard Tofts", "--out_dir", self.test_dir], # Missing --dce
            ["--dce", self.dce_dummy, "--tr", "0.005", "--r1_relaxivity", "4.5", "--aif_file", self.aif_dummy, "--model", "Standard Tofts", "--out_dir", self.test_dir], # Missing --t1map
            ["--dce", self.dce_dummy, "--t1map", self.t1_dummy, "--r1_relaxivity", "4.5", "--aif_file", self.aif_dummy, "--model", "Standard Tofts", "--out_dir", self.test_dir], # Missing --tr
            # ... and so on for each required argument
        ]
        for i, arg_set in enumerate(required_args_sets):
            with self.subTest(missing_arg_index=i):
                result = self._run_script(arg_set)
                self.assertEqual(result.returncode, 2, f"Script should fail with argparse error (2). Stderr: {result.stderr}")
                self.assertIn("the following arguments are required", result.stderr.lower())
        
        # Test missing AIF specification
        result = self._run_script(["--dce", self.dce_dummy, "--t1map", self.t1_dummy, "--tr", "0.005", "--r1_relaxivity", "4.5", "--model", "Standard Tofts", "--out_dir", self.test_dir])
        self.assertEqual(result.returncode, 2)
        self.assertIn("one of the arguments --aif_file --aif_pop_model is required", result.stderr.lower())


    def test_mutually_exclusive_aif_args(self):
        """Test providing both --aif_file and --aif_pop_model fails."""
        args = ["--dce", self.dce_dummy, "--t1map", self.t1_dummy, "--tr", "0.005", "--r1_relaxivity", "4.5",
                "--aif_file", self.aif_dummy, "--aif_pop_model", "parker",
                "--model", "Standard Tofts", "--out_dir", self.test_dir]
        result = self._run_script(args)
        self.assertEqual(result.returncode, 2)
        self.assertIn("argument --aif_pop_model: not allowed with argument --aif_file", result.stderr)

    def test_invalid_choices(self):
        """Test invalid choices for arguments with predefined choices."""
        invalid_model_args = ["--dce", self.dce_dummy, "--t1map", self.t1_dummy, "--tr", "0.005", "--r1_relaxivity", "4.5",
                              "--aif_file", self.aif_dummy, "--model", "InvalidModelName", "--out_dir", self.test_dir]
        result = self._run_script(invalid_model_args)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice: 'InvalidModelName'", result.stderr)

        invalid_aif_model_args = ["--dce", self.dce_dummy, "--t1map", self.t1_dummy, "--tr", "0.005", "--r1_relaxivity", "4.5",
                                  "--aif_pop_model", "InvalidAIFName", "--model", "Standard Tofts", "--out_dir", self.test_dir]
        result = self._run_script(invalid_aif_model_args)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice: 'InvalidAIFName'", result.stderr)

    def test_invalid_type_for_numeric_args(self):
        """Test providing non-numeric values for numeric arguments."""
        args_sets = [
            (["--tr", "not_a_float"], "argument --tr: invalid float value: 'not_a_float'"),
            (["--r1_relaxivity", "abc"], "argument --r1_relaxivity: invalid float value: 'abc'"),
            (["--baseline_points", "xyz"], "argument --baseline_points: invalid int value: 'xyz'"),
            (["--num_processes", "def"], "argument --num_processes: invalid int value: 'def'"),
        ]
        base_args = ["--dce", self.dce_dummy, "--t1map", self.t1_dummy, "--aif_file", self.aif_dummy,
                       "--model", "Standard Tofts", "--out_dir", self.test_dir]
        
        for invalid_arg_pair, error_msg_fragment in args_sets:
            with self.subTest(invalid_arg=invalid_arg_pair[0]):
                result = self._run_script(base_args + invalid_arg_pair)
                self.assertEqual(result.returncode, 2, f"Stderr: {result.stderr}")
                self.assertIn(error_msg_fragment, result.stderr)

    def test_aif_param_parsing(self):
        """Test parsing of --aif_param key-value pairs."""
        args = ["--dce", self.dce_dummy, "--t1map", self.t1_dummy, "--tr", "0.005", "--r1_relaxivity", "4.5",
                "--aif_pop_model", "parker",
                "--aif_param", "D_scaler", "0.8",
                "--aif_param", "A1", "0.7",
                "--model", "Standard Tofts", "--out_dir", os.path.join(self.test_dir, "aif_param_test")]
        result = self._run_script(args)
        # Expect it to fail after argparse, during data loading or AIF generation with dummy files
        self.assertNotEqual(result.returncode, 2, f"Argparse should pass. Stderr: {result.stderr}")
        # Check if the script's printout includes the parsed AIF params
        # This relies on the script printing args, which it does.
        self.assertIn("Population AIF Params: [['D_scaler', '0.8'], ['A1', '0.7']]", result.stdout.replace(" ", ""))


class TestBatchProcessorEndToEnd(unittest.TestCase):
    """Basic end-to-end tests for batch_processor.py."""

    def setUp(self):
        self.base_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.base_dir, "inputs")
        self.output_dir = os.path.join(self.base_dir, "outputs")
        os.makedirs(self.input_dir)
        # Output dir creation is tested, so don't create it here initially for one test.

        # Create mock NIfTI files
        self.dce_path = self._create_mock_nifti("dce.nii.gz", (3,3,2,10), np.float32, is_dce=True) # X,Y,Z,Time
        self.t1_path = self._create_mock_nifti("t1map.nii.gz", (3,3,2), np.float32)
        self.mask_path = self._create_mock_nifti("mask.nii.gz", (3,3,2), np.uint8, is_mask=True)
        
        self.aif_path = os.path.join(self.input_dir, "mock_aif.txt")
        with open(self.aif_path, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["Time", "Concentration"]) # Header
            for i in range(10): # 10 time points for AIF
                # Time in seconds, matching TR*num_dce_timepoints typically
                writer.writerow([i * (2.0), 0.5 * np.exp(-i*0.1) + 0.1*i*0.1])

    def tearDown(self):
        shutil.rmtree(self.base_dir)

    def _create_mock_nifti(self, filename, shape, dtype, is_dce=False, is_mask=False, affine=np.eye(4)):
        filepath = os.path.join(self.input_dir, filename)
        data = np.random.rand(*shape).astype(dtype) * 100 # Scale to typical signal values
        if is_dce:
            data[..., shape[-1]//2:] *= 1.5 # Simulate contrast enhancement
        if is_mask:
            data = (data > np.mean(data)).astype(dtype) # Binary mask from random data

        img = nib.Nifti1Image(data, affine)
        if is_dce: # Set TR in header for dce
            img.header['pixdim'][4] = 2.0 # Example TR of 2s
        nib.save(img, filepath)
        return filepath

    def _run_script(self, args_list):
        return subprocess.run(
            [sys.executable, BATCH_PROCESSOR_SCRIPT_PATH] + args_list,
            capture_output=True, text=True, check=False
        )

    def test_successful_run_tofts_aif_file(self):
        """Test a basic successful run with Standard Tofts model and AIF file."""
        args = [
            "--dce", self.dce_path, "--t1map", self.t1_path, "--mask", self.mask_path,
            "--tr", "2.0", # Match TR used in mock DCE header if that's used by script (it isn't currently)
            "--r1_relaxivity", "3.7", "--baseline_points", "3",
            "--aif_file", self.aif_path,
            "--model", "Standard Tofts",
            "--out_dir", self.output_dir,
            "--num_processes", "1"
        ]
        # Create output dir for this test
        os.makedirs(self.output_dir, exist_ok=True)
        result = self._run_script(args)

        print("STDOUT (test_successful_run_tofts_aif_file):", result.stdout)
        print("STDERR (test_successful_run_tofts_aif_file):", result.stderr)
        self.assertEqual(result.returncode, 0, f"Script failed. Stderr: {result.stderr}\nStdout: {result.stdout}")
        
        expected_ktrans_path = os.path.join(self.output_dir, "Ktrans.nii.gz")
        expected_ve_path = os.path.join(self.output_dir, "ve.nii.gz")
        self.assertTrue(os.path.exists(expected_ktrans_path), "Ktrans map not created.")
        self.assertTrue(os.path.exists(expected_ve_path), "ve map not created.")

        ktrans_map_img = nib.load(expected_ktrans_path)
        self.assertEqual(ktrans_map_img.shape, (3,3,2))
        ktrans_data = ktrans_map_img.get_fdata()
        mask_data = nib.load(self.mask_path).get_fdata().astype(bool)
        self.assertFalse(np.all(np.isnan(ktrans_data[mask_data])),
                         "Ktrans map is all NaNs within the mask.")

    def test_output_dir_creation(self):
        """Test that the output directory is created if it doesn't exist."""
        new_output_dir = os.path.join(self.test_dir, "newly_created_output")
        # DO NOT create new_output_dir here

        args = [
            "--dce", self.dce_path, "--t1map", self.t1_path,
            "--tr", "2.0", "--r1_relaxivity", "3.7",
            "--aif_file", self.aif_path,
            "--model", "Standard Tofts",
            "--out_dir", new_output_dir,
            "--num_processes", "1"
        ]
        result = self._run_script(args)
        self.assertEqual(result.returncode, 0, f"Script failed. Stderr: {result.stderr}\nStdout: {result.stdout}")
        self.assertTrue(os.path.isdir(new_output_dir), "Output directory was not created by the script.")

    def test_output_dir_is_file_error(self): # Renamed for clarity
        """Test error if output directory path is an existing file."""
        output_is_file_path = os.path.join(self.test_dir, "i_am_a_file.txt")
        with open(output_is_file_path, 'w') as f:
            f.write("This is a file, not a directory.")

        args = [
            "--dce", self.dce_path, "--t1map", self.t1_path,
            "--tr", "2.0", "--r1_relaxivity", "3.7",
            "--aif_file", self.aif_path,
            "--model", "Standard Tofts",
            "--out_dir", output_is_file_path,
            "--num_processes", "1"
        ]
        result = self._run_script(args)
        self.assertNotEqual(result.returncode, 0)
        # Script prints its own error message for this
        self.assertIn("error creating output directory", result.stdout.lower())


if __name__ == '__main__':
    unittest.main()
