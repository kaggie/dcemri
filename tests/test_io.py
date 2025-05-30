import unittest
import numpy as np
import nibabel as nib
import os
import tempfile
import shutil
import sys
from numpy.testing import assert_array_equal, assert_raises # Using assert_raises from numpy.testing for consistency if needed, else unittest.assertRaises

# Add the project root to the Python path to allow direct import of dce_mri_analyzer
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import io

class TestNiftiLoading(unittest.TestCase): # Renamed from TestIoFunctions
    def setUp(self):
        """Create a temporary directory for fake NIfTI files."""
        self.test_dir = tempfile.mkdtemp()
        self.default_affine = np.eye(4)


    def tearDown(self):
        """Clean up the temporary directory and its contents."""
        shutil.rmtree(self.test_dir)

    def _create_mock_nifti_file(self, filename, data_shape, dtype=np.float32, affine_matrix=None, is_corrupted=False):
        """Helper function to create a mock NIfTI file."""
        filepath = os.path.join(self.test_dir, filename)
        if affine_matrix is None:
            affine_matrix = self.default_affine

        if is_corrupted:
            with open(filepath, 'wb') as f:
                f.write(b"corrupted_data_not_a_nifti_file")
        else:
            data = np.arange(np.prod(data_shape), dtype=dtype).reshape(data_shape)
            img = nib.Nifti1Image(data, affine_matrix)
            nib.save(img, filepath)
        return filepath

    def test_load_nifti_file_success(self):
        """Test successful loading of a NIfTI file."""
        shape = (10, 10, 10)
        expected_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        fake_nifti_path = self._create_mock_nifti_file(filename="test_success.nii.gz", data_shape=shape)

        img = io.load_nifti_file(fake_nifti_path)
        self.assertIsInstance(img, nib.Nifti1Image)
        self.assertEqual(img.shape, shape)
        assert_array_equal(img.get_fdata(), expected_data)
        assert_array_equal(img.affine, self.default_affine)


    def test_load_nifti_file_not_found(self):
        """Test loading a non-existent NIfTI file."""
        non_existent_file = os.path.join(self.test_dir, "non_existent_file.nii")
        with self.assertRaises(FileNotFoundError):
            io.load_nifti_file(non_existent_file)

    def test_load_nifti_file_invalid_corrupted(self): # Renamed for clarity
        """Test loading an invalid/corrupted NIfTI file."""
        invalid_file_path = self._create_mock_nifti_file("invalid.nii.gz", (2,2,2), is_corrupted=True)
        with self.assertRaises(ValueError) as context:
            io.load_nifti_file(invalid_file_path)
        self.assertTrue("Invalid NIfTI file" in str(context.exception))


    def test_load_dce_series_success(self):
        """Test successful loading of a 4D DCE series."""
        shape = (10, 10, 10, 20)
        expected_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        fake_dce_path = self._create_mock_nifti_file(shape=shape, filename="dce.nii.gz")

        dce_data = io.load_dce_series(fake_dce_path)
        self.assertIsInstance(dce_data, np.ndarray)
        self.assertEqual(dce_data.shape, shape)
        assert_array_equal(dce_data, expected_data)


    def test_load_dce_series_wrong_dim(self):
        """Test loading a DCE series with incorrect (3D) dimensions."""
        fake_3d_path = self._create_mock_nifti_file(shape=(10, 10, 10), filename="3d_for_dce.nii.gz")
        with self.assertRaisesRegex(ValueError, "DCE series must be a 4D NIfTI image."):
            io.load_dce_series(fake_3d_path)

    def test_load_t1_map_success(self):
        """Test successful loading of a 3D T1 map."""
        shape = (10,10,10)
        expected_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        fake_t1_path = self._create_mock_nifti_file(shape=shape, filename="t1.nii.gz")

        t1_data = io.load_t1_map(fake_t1_path)
        self.assertIsInstance(t1_data, np.ndarray)
        self.assertEqual(t1_data.shape, shape)
        assert_array_equal(t1_data, expected_data)

    def test_load_t1_map_wrong_dim(self):
        """Test loading a T1 map with incorrect (4D) dimensions."""
        fake_4d_path = self._create_mock_nifti_file(shape=(10, 10, 10, 5), filename="4d_for_t1.nii.gz")
        with self.assertRaisesRegex(ValueError, "T1 map must be a 3D NIfTI image."):
            io.load_t1_map(fake_4d_path)

    def test_load_t1_map_dim_mismatch_with_dce(self):
        """Test T1 map loading with spatial dimensions mismatch against DCE shape."""
        dce_shape_ref = (10, 10, 10, 5)
        fake_t1_mismatch_path = self._create_mock_nifti_file(shape=(5, 5, 5), filename="t1_mismatch.nii.gz")
        with self.assertRaisesRegex(ValueError, "T1 map dimensions do not match DCE series spatial dimensions."):
            io.load_t1_map(fake_t1_mismatch_path, dce_shape=dce_shape_ref)

    def test_load_mask_success(self):
        """Test successful loading of a 3D mask file, ensuring boolean output."""
        mask_data_orig = np.random.randint(0, 3, size=(10, 10, 10)).astype(np.uint8) # Include non-binary values
        fake_mask_path = os.path.join(self.test_dir, "mask_input.nii.gz")
        mask_img_for_ref = nib.Nifti1Image(mask_data_orig, self.default_affine)
        nib.save(mask_img_for_ref, fake_mask_path)

        mask_data_loaded = io.load_mask(fake_mask_path)
        self.assertIsInstance(mask_data_loaded, np.ndarray)
        self.assertEqual(mask_data_loaded.dtype, bool) # Crucial check
        self.assertEqual(mask_data_loaded.shape, (10, 10, 10))
        # Verify that non-zero values became True, zero values became False
        np.testing.assert_array_equal(mask_data_loaded, mask_data_orig.astype(bool))


    def test_load_mask_wrong_dim(self):
        """Test loading a mask with incorrect (4D) dimensions."""
        fake_4d_mask_path = self._create_mock_nifti_file(shape=(10, 10, 10, 3), filename="4d_mask.nii.gz")
        with self.assertRaisesRegex(ValueError, "Mask must be a 3D NIfTI image."):
            io.load_mask(fake_4d_mask_path)

    def test_load_mask_dim_mismatch_with_ref(self):
        """Test mask loading with spatial dimensions mismatch against reference shape."""
        reference_shape_ref = (10, 10, 10)
        fake_mask_mismatch_path = self._create_mock_nifti_file(shape=(5, 5, 5), filename="mask_mismatch.nii.gz")
        with self.assertRaisesRegex(ValueError, "Mask dimensions do not match the reference image dimensions."):
            io.load_mask(fake_mask_mismatch_path, reference_shape=reference_shape_ref)

class TestSaveNiftiMap(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ref_affine = np.array([
            [-2.,  0.,  0.,  128.],
            [ 0.,  2.,  0., -128.],
            [ 0.,  0.,  2., -128.],
            [ 0.,  0.,  0.,    1.]
        ])
        self.ref_shape_3d = (5,5,5)
        self.ref_shape_4d = (5,5,5,10)

        self.ref_3d_filepath = os.path.join(self.test_dir, "ref_3d.nii.gz")
        ref_3d_data = np.zeros(self.ref_shape_3d, dtype=np.int16)
        ref_3d_img = nib.Nifti1Image(ref_3d_data, self.ref_affine)
        nib.save(ref_3d_img, self.ref_3d_filepath)
        
        self.ref_4d_filepath = os.path.join(self.test_dir, "ref_4d.nii.gz")
        ref_4d_data = np.zeros(self.ref_shape_4d, dtype=np.float32)
        ref_4d_img = nib.Nifti1Image(ref_4d_data, self.ref_affine)
        ref_4d_img.header['pixdim'][4] = 2.5 # Example TR value
        nib.save(ref_4d_img, self.ref_4d_filepath)

        self.map_to_save_valid = np.random.rand(*self.ref_shape_3d).astype(np.float32)


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_nifti_map_success_3d_ref(self):
        """Test saving a 3D map using a 3D reference NIfTI."""
        output_filepath = os.path.join(self.test_dir, "output_map_3d_ref.nii.gz")
        
        io.save_nifti_map(self.map_to_save_valid, self.ref_3d_filepath, output_filepath)
        
        self.assertTrue(os.path.exists(output_filepath))
        loaded_img = nib.load(output_filepath)
        
        self.assertEqual(loaded_img.shape, self.ref_shape_3d)
        np.testing.assert_array_almost_equal(loaded_img.get_fdata(), self.map_to_save_valid, decimal=5)
        np.testing.assert_array_almost_equal(loaded_img.affine, self.ref_affine)
        self.assertEqual(loaded_img.header.get_data_dtype(), np.float32)

    def test_save_nifti_map_success_4d_ref(self):
        """Test saving a 3D map using a 4D reference NIfTI."""
        output_filepath = os.path.join(self.test_dir, "output_map_4d_ref.nii.gz")
        
        io.save_nifti_map(self.map_to_save_valid, self.ref_4d_filepath, output_filepath)
        
        self.assertTrue(os.path.exists(output_filepath))
        loaded_img = nib.load(output_filepath)
        
        self.assertEqual(loaded_img.shape, self.ref_shape_3d)
        np.testing.assert_array_almost_equal(loaded_img.get_fdata(), self.map_to_save_valid, decimal=5)
        np.testing.assert_array_almost_equal(loaded_img.affine, self.ref_affine)
        self.assertEqual(loaded_img.header.get_data_dtype(), np.float32)
        self.assertEqual(loaded_img.header['dim'][0], 3)
        self.assertEqual(loaded_img.header['dim'][4], 1) # 4th dim size should be 1 for a 3D map
        self.assertNotEqual(loaded_img.header['pixdim'][4], nib.load(self.ref_4d_filepath).header['pixdim'][4],
                            "pixdim[4] from 4D ref should not be directly copied to 3D map header if it implies temporal spacing")


    def test_save_nifti_map_ref_not_found(self):
        """Test FileNotFoundError if reference NIfTI does not exist."""
        non_existent_ref = os.path.join(self.test_dir, "non_existent_ref.nii.gz")
        output_filepath = os.path.join(self.test_dir, "output_map.nii.gz")
        with self.assertRaises(FileNotFoundError):
            io.save_nifti_map(self.map_to_save_valid, non_existent_ref, output_filepath)

    def test_save_nifti_map_data_not_3d(self):
        """Test ValueError if data_map is not 3D."""
        data_map_2d = np.random.rand(5,5)
        output_filepath = os.path.join(self.test_dir, "output_map_2d.nii.gz")
        with self.assertRaisesRegex(ValueError, "data_map must be a 3D array"):
            io.save_nifti_map(data_map_2d, self.ref_3d_filepath, output_filepath)

        data_map_4d = np.random.rand(5,5,5,2) # Attempting to save a 4D map with this function
        output_filepath_4d = os.path.join(self.test_dir, "output_map_4d.nii.gz")
        with self.assertRaisesRegex(ValueError, "data_map must be a 3D array"):
            io.save_nifti_map(data_map_4d, self.ref_4d_filepath, output_filepath_4d)


    def test_save_nifti_map_shape_mismatch(self):
        """Test ValueError if data_map shape mismatches reference NIfTI spatial shape."""
        data_map_wrong_shape = np.random.rand(3,3,3)
        output_filepath = os.path.join(self.test_dir, "output_map_wrong_shape.nii.gz")
        # self.ref_3d_filepath is 5x5x5, this data_map is 3x3x3
        with self.assertRaisesRegex(ValueError, "data_map shape .* does not match reference NIfTI spatial shape"):
            io.save_nifti_map(data_map_wrong_shape, self.ref_3d_filepath, output_filepath)

    def test_save_nifti_map_invalid_output_path(self):
        """Test IOError for an invalid output filepath (e.g., a directory)."""
        # Attempting to save to a path that is a directory should raise an error.
        # Nibabel's save function typically raises FileNotFoundError or IsADirectoryError,
        # which the wrapper in core.io might catch and re-raise as IOError or ValueError.
        # Based on core.io.save_nifti_map, it re-raises nibabel errors as generic Exception or specific FileNotFoundError.
        # Let's check for IOError as per prompt.
        with self.assertRaises(IOError): # Or specific error like IsADirectoryError if not wrapped
            io.save_nifti_map(self.map_to_save_valid, self.ref_3d_filepath, self.test_dir)


if __name__ == '__main__':
    unittest.main()
