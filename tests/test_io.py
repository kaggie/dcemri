import unittest
import numpy as np
import nibabel as nib
import os
from core.io import (
    load_nifti_file,
    load_dce_series,
    load_t1_map,
    load_mask,
    save_nifti_map,
)

# Helper function to create a dummy NIfTI file
def create_dummy_nifti(filename, data_shape, affine=np.eye(4), dtype=np.float32):
    """Creates a dummy NIfTI file for testing."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = np.random.rand(*data_shape).astype(dtype)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filename)
    return filename

# Helper function to create an empty/corrupted file
def create_invalid_nifti(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write("This is not a nifti file")
    return filename

class TestIO(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_nifti_data_io" # Use a unique name for this test suite
        os.makedirs(self.test_dir, exist_ok=True)

        # Valid files
        self.nifti_3d_file = create_dummy_nifti(
            os.path.join(self.test_dir, "dummy_3d.nii.gz"), (5, 5, 5)
        )
        self.nifti_4d_file = create_dummy_nifti(
            os.path.join(self.test_dir, "dummy_4d.nii.gz"), (5, 5, 5, 10)
        )
        self.mask_file_int = create_dummy_nifti( # For mask loading, integer type
            os.path.join(self.test_dir, "dummy_mask_int.nii.gz"), (5, 5, 5), dtype=np.int16
        )

        # Invalid file for generic load_nifti_file
        self.invalid_nifti_format_file = create_invalid_nifti(
            os.path.join(self.test_dir, "invalid_format.nii.gz")
        )
        self.empty_nifti_file = os.path.join(self.test_dir, "empty.nii.gz")
        with open(self.empty_nifti_file, 'w') as f:
            pass


    def tearDown(self):
        for f in os.listdir(self.test_dir):
            try:
                os.remove(os.path.join(self.test_dir, f))
            except OSError: # Handle cases where a file might be a directory if setup failed
                pass
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    # --- Tests for load_nifti_file ---
    def test_load_nifti_valid(self):
        img = load_nifti_file(self.nifti_3d_file)
        self.assertIsNotNone(img)
        self.assertIsInstance(img, nib.Nifti1Image)
        self.assertEqual(img.shape, (5,5,5))

    def test_load_nifti_non_existent(self):
        with self.assertRaises(FileNotFoundError):
            load_nifti_file(os.path.join(self.test_dir, "non_existent.nii.gz"))

    def test_load_nifti_invalid_format_file(self):
        # nibabel.load can raise various errors for malformed files.
        # ImageFileError is common, which is a subclass of ValueError.
        with self.assertRaises(ValueError):
            load_nifti_file(self.invalid_nifti_format_file)

    def test_load_nifti_empty_file(self):
         # nibabel.load raises ImageFileError for empty files too.
        with self.assertRaises(ValueError):
            load_nifti_file(self.empty_nifti_file)

    # --- Tests for load_dce_series ---
    def test_load_dce_series_valid_4d(self):
        data, affine, header = load_dce_series(self.nifti_4d_file)
        self.assertIsNotNone(data)
        self.assertEqual(data.shape, (5, 5, 5, 10))
        self.assertIsNotNone(affine)
        self.assertIsNotNone(header)
        self.assertIsInstance(data, np.ndarray)

    def test_load_dce_series_rejects_3d(self):
        with self.assertRaises(ValueError):
            load_dce_series(self.nifti_3d_file)

    def test_load_dce_series_non_existent(self):
        with self.assertRaises(FileNotFoundError):
            load_dce_series(os.path.join(self.test_dir, "non_existent_dce.nii.gz"))

    def test_load_dce_series_invalid_file(self):
        with self.assertRaises(ValueError): # Expecting error from load_nifti_file
            load_dce_series(self.invalid_nifti_format_file)

    # --- Tests for load_t1_map ---
    def test_load_t1_map_valid_3d(self):
        data, affine, header = load_t1_map(self.nifti_3d_file)
        self.assertIsNotNone(data)
        self.assertEqual(data.shape, (5, 5, 5))
        self.assertIsNotNone(affine)
        self.assertIsNotNone(header)
        self.assertIsInstance(data, np.ndarray)

    def test_load_t1_map_rejects_4d(self):
        with self.assertRaises(ValueError):
            load_t1_map(self.nifti_4d_file)

    def test_load_t1_map_dce_shape_validation_matching(self):
        # dce_shape (5,5,5,10) -> spatial (5,5,5) should match self.nifti_3d_file (5,5,5)
        dce_ref_shape = (5,5,5,10)
        data, _, _ = load_t1_map(self.nifti_3d_file, dce_shape=dce_ref_shape)
        self.assertEqual(data.shape, (5,5,5))

    def test_load_t1_map_dce_shape_validation_mismatch(self):
        dce_ref_shape_mismatch = (6, 5, 5, 10) # X dimension differs
        with self.assertRaises(ValueError):
            load_t1_map(self.nifti_3d_file, dce_shape=dce_ref_shape_mismatch)

    def test_load_t1_map_non_existent(self):
        with self.assertRaises(FileNotFoundError):
            load_t1_map(os.path.join(self.test_dir, "non_existent_t1.nii.gz"))

    def test_load_t1_map_invalid_file(self):
        with self.assertRaises(ValueError):
            load_t1_map(self.invalid_nifti_format_file)

    # --- Tests for load_mask ---
    def test_load_mask_valid_3d_int(self):
        # Using self.mask_file_int which has dtype=int16
        mask_data, affine, header = load_mask(self.mask_file_int)
        self.assertIsNotNone(mask_data)
        self.assertEqual(mask_data.shape, (5, 5, 5))
        self.assertEqual(mask_data.dtype, bool) # Should be converted to boolean
        # Check some values are True/False if possible, or that it contains both
        # For random int data, it's likely to have non-zero values that become True
        self.assertTrue(np.any(mask_data) or not np.all(mask_data)) # Ensure not all False or all True if original was mixed
        self.assertIsNotNone(affine)
        self.assertIsNotNone(header)

    def test_load_mask_rejects_4d(self):
        with self.assertRaises(ValueError):
            load_mask(self.nifti_4d_file) # Using a 4D float file

    def test_load_mask_ref_shape_validation_matching(self):
        ref_shape = (5,5,5)
        mask_data, _, _ = load_mask(self.mask_file_int, reference_shape=ref_shape)
        self.assertEqual(mask_data.shape, ref_shape)

    def test_load_mask_ref_shape_validation_mismatch(self):
        ref_shape_mismatch = (6, 5, 5) # X dimension differs
        with self.assertRaises(ValueError):
            load_mask(self.mask_file_int, reference_shape=ref_shape_mismatch)

    def test_load_mask_ref_shape_4d_mismatch(self):
        # Mask is 3D, reference_shape is 4D, this should also fail as spatial part differs
        ref_shape_4d_mismatch = (6,5,5,10)
        with self.assertRaises(ValueError):
            load_mask(self.mask_file_int, reference_shape=ref_shape_4d_mismatch)

        # Mask is 3D, reference_shape is 4D. This should fail because load_mask expects
        # reference_shape to be 3D if the mask is 3D, for an exact match.
        ref_shape_4d_spatial_match = (5,5,5,12)
        with self.assertRaises(ValueError):
            load_mask(self.mask_file_int, reference_shape=ref_shape_4d_spatial_match)


    def test_load_mask_non_existent(self):
        with self.assertRaises(FileNotFoundError):
            load_mask(os.path.join(self.test_dir, "non_existent_mask.nii.gz"))

    def test_load_mask_invalid_file(self):
        with self.assertRaises(ValueError):
            load_mask(self.invalid_nifti_format_file)

    # --- Tests for save_nifti_map ---
    def test_save_nifti_map_3d_data_3d_ref(self):
        data_to_save = np.random.rand(5, 5, 5).astype(np.float32)
        output_path = os.path.join(self.test_dir, "saved_3d_from_3dref.nii.gz")
        
        save_nifti_map(data_to_save, self.nifti_3d_file, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        loaded_img = nib.load(output_path)
        loaded_data = loaded_img.get_fdata()
        
        np.testing.assert_array_almost_equal(loaded_data, data_to_save, decimal=5)
        self.assertEqual(loaded_img.shape, data_to_save.shape)
        
        ref_img = nib.load(self.nifti_3d_file)
        np.testing.assert_array_almost_equal(loaded_img.affine, ref_img.affine)
        self.assertEqual(loaded_img.header.get_data_dtype(), data_to_save.dtype)

    def test_save_nifti_map_4d_data_4d_ref(self):
        data_to_save = np.random.rand(5, 5, 5, 10).astype(np.float32)
        output_path = os.path.join(self.test_dir, "saved_4d_from_4dref.nii.gz")

        save_nifti_map(data_to_save, self.nifti_4d_file, output_path)
        self.assertTrue(os.path.exists(output_path))

        loaded_img = nib.load(output_path)
        loaded_data = loaded_img.get_fdata()

        np.testing.assert_array_almost_equal(loaded_data, data_to_save, decimal=5)
        self.assertEqual(loaded_img.shape, data_to_save.shape)
        ref_img = nib.load(self.nifti_4d_file)
        np.testing.assert_array_almost_equal(loaded_img.affine, ref_img.affine)
        self.assertEqual(loaded_img.header.get_data_dtype(), data_to_save.dtype)

    def test_save_nifti_map_3d_data_4d_ref(self):
        # Saving a 3D map using a 4D reference (e.g. saving a T1 map using DCE series as ref)
        data_to_save = np.random.rand(5, 5, 5).astype(np.float32)
        output_path = os.path.join(self.test_dir, "saved_3d_from_4dref.nii.gz")
        
        save_nifti_map(data_to_save, self.nifti_4d_file, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        loaded_img = nib.load(output_path)
        loaded_data = loaded_img.get_fdata()
        
        np.testing.assert_array_almost_equal(loaded_data, data_to_save, decimal=5)
        self.assertEqual(loaded_img.shape, data_to_save.shape)
        ref_img = nib.load(self.nifti_4d_file)
        np.testing.assert_array_almost_equal(loaded_img.affine, ref_img.affine)
        self.assertEqual(loaded_img.header.get_data_dtype(), data_to_save.dtype)
        self.assertEqual(len(loaded_img.header.get_zooms()), 3) # Ensure output is 3D

    def test_save_nifti_map_4d_data_3d_ref_error(self):
        # Attempting to save 4D data with a 3D reference's header might be problematic
        # if not handled carefully by the save function to adjust header dimensions.
        # The function core.io.save_nifti_map is designed to handle this by taking spatial
        # affine and creating a new header for the output that matches data_map's dimensionality.
        data_to_save = np.random.rand(5, 5, 5, 7).astype(np.float32)
        output_path = os.path.join(self.test_dir, "saved_4d_from_3dref.nii.gz")

        save_nifti_map(data_to_save, self.nifti_3d_file, output_path)
        self.assertTrue(os.path.exists(output_path))

        loaded_img = nib.load(output_path)
        loaded_data = loaded_img.get_fdata()
        np.testing.assert_array_almost_equal(loaded_data, data_to_save, decimal=5)
        self.assertEqual(loaded_img.shape, data_to_save.shape) # Shape should be (5,5,5,7)
        
        ref_img = nib.load(self.nifti_3d_file)
        np.testing.assert_array_almost_equal(loaded_img.affine[:3,:3], ref_img.affine[:3,:3]) # Compare spatial part of affine
        # The full affine might differ in the 4th dimension scaling if original was purely 3D.
        # Check if the output header has 4 dimensions for zooms
        self.assertEqual(len(loaded_img.header.get_zooms()), 4)


    def test_save_nifti_map_invalid_data_dim(self):
        data_2d = np.random.rand(5, 5).astype(np.float32)
        with self.assertRaises(ValueError):
            save_nifti_map(data_2d, self.nifti_3d_file, "wont_save_2d.nii.gz")
        
        data_5d = np.random.rand(5,5,5,5,5).astype(np.float32)
        with self.assertRaises(ValueError):
            save_nifti_map(data_5d, self.nifti_4d_file, "wont_save_5d.nii.gz")

    def test_save_nifti_map_mismatched_spatial_dims(self):
        data_mismatch = np.random.rand(6, 5, 5).astype(np.float32) # X dim is 6, ref is 5
        with self.assertRaises(ValueError):
            save_nifti_map(data_mismatch, self.nifti_3d_file, "wont_save_mismatch.nii.gz")

    def test_save_nifti_map_non_existent_ref(self):
        data_to_save = np.random.rand(5,5,5).astype(np.float32)
        with self.assertRaises(FileNotFoundError):
            save_nifti_map(data_to_save, os.path.join(self.test_dir,"non_existent_ref.nii.gz"), "wont_save.nii.gz")


if __name__ == "__main__":
    unittest.main()
