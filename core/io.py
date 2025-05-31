import os
import nibabel as nib
import numpy as np

"""
This module provides utility functions for input/output operations, primarily
focused on loading and saving NIfTI (Neuroimaging Informatics Technology Initiative)
files. NIfTI is a common file format for storing neuroimaging data, including
DCE-MRI series, T1 maps, and segmentation masks.

The functions handle:
- Loading generic NIfTI files.
- Specialized loading for 4D DCE series, 3D T1 maps, and 3D masks,
  including basic validation (e.g., dimensions, data type conversion for masks).
- Saving 3D or 4D NumPy arrays as NIfTI files, using a reference NIfTI image
  to preserve affine transformation and header information.
"""

def load_nifti_file(filepath: str):
    """
    Loads a NIfTI file.

    Args:
        filepath (str): Path to the NIfTI file.

    Returns:
        nibabel.nifti1.Nifti1Image: The loaded NIfTI image object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid NIfTI file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NIfTI file not found at: {filepath}")
    try:
        img = nib.load(filepath)
        return img
    except Exception as e:
        raise ValueError(f"Invalid NIfTI file: {filepath}. Error: {e}")

def load_dce_series(filepath: str):
    """
    Loads a 4D DCE NIfTI series.

    Args:
        filepath (str): Path to the 4D NIfTI file.

    Returns:
        np.ndarray: The DCE series data as a NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid 4D NIfTI file.
    """
    img = load_nifti_file(filepath)
    if img.ndim != 4:
        raise ValueError("DCE series must be a 4D NIfTI image.")
    return img.get_fdata(), img.affine, img.header

def load_t1_map(filepath: str, dce_shape: tuple = None):
    """
    Loads a 3D T1 map NIfTI file.

    Args:
        filepath (str): Path to the 3D NIfTI T1 map file.
        dce_shape (tuple, optional): The shape of the corresponding 4D DCE series
                                     (e.g., (nx, ny, nz, nt)). If provided,
                                     the spatial dimensions (nx, ny, nz) of the
                                     T1 map are validated against those of the
                                     DCE series. Defaults to None (no validation).

    Returns:
        np.ndarray: The T1 map data as a NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid 3D NIfTI file or if dimensions 
                    do not match dce_shape.
    """
    img = load_nifti_file(filepath)
    if img.ndim != 3:
        raise ValueError("T1 map must be a 3D NIfTI image.")
    if dce_shape is not None:
        if img.shape != dce_shape[:3]:
            raise ValueError(
                "T1 map dimensions do not match DCE series spatial dimensions."
            )
    return img.get_fdata(), img.affine, img.header

def load_mask(filepath: str, reference_shape: tuple = None):
    """
    Loads a 3D mask NIfTI file and converts it to a boolean array.

    The mask is expected to contain integer values, where non-zero values
    indicate regions within the mask.

    Args:
        filepath (str): Path to the 3D NIfTI mask file.
        reference_shape (tuple, optional): The spatial shape (e.g., (nx, ny, nz))
                                           of a reference image (like a T1 map or
                                           one frame of a DCE series). If provided,
                                           the dimensions of the mask are validated
                                           against this reference shape.
                                           Defaults to None (no validation).

    Returns:
        np.ndarray: The mask data as a boolean NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid 3D NIfTI file or if dimensions
                    do not match reference_shape.
    """
    img = load_nifti_file(filepath)
    if img.ndim != 3:
        raise ValueError("Mask must be a 3D NIfTI image.")
    if reference_shape is not None:
        if img.shape != reference_shape:
            raise ValueError(
                "Mask dimensions do not match the reference image dimensions."
            )
    data = img.get_fdata()
    return data.astype(bool), img.affine, img.header

def save_nifti_map(data_map: np.ndarray, original_nifti_ref_path: str, output_filepath: str):
    """
    Saves a 3D or 4D data map as a NIfTI file, using a reference NIfTI file
    to provide the affine transformation and header information.

    This ensures that the saved map is spatially aligned with the reference image.
    The data type of the saved map is set to float32.

    Args:
        data_map (np.ndarray): The 3D or 4D NumPy array containing the map data
                               (e.g., Ktrans map, Ve map, AUC map).
        original_nifti_ref_path (str): Path to an original NIfTI file (e.g.,
                                       the T1 map, a single frame of the DCE series,
                                       or the full DCE series if saving a 4D map).
                                       This file is used to source the affine
                                       matrix and header.
        output_filepath (str): The path where the new NIfTI map file will be saved.

    Raises:
        FileNotFoundError: If the `original_nifti_ref_path` does not exist.
        ValueError: If `data_map` is not 3D or 4D, or if its spatial dimensions
                    do not match the spatial dimensions of the reference NIfTI image.
        Exception: Can re-raise exceptions from `nibabel.load` (for the reference image)
                   or `nibabel.save` (for the output image).
    """
    if not os.path.exists(original_nifti_ref_path):
        raise FileNotFoundError(f"Reference NIfTI file not found at: {original_nifti_ref_path}")

    try:
        ref_nifti_img = nib.load(original_nifti_ref_path)
    except Exception as e:
        raise ValueError(f"Could not load reference NIfTI file: {original_nifti_ref_path}. Error: {e}")

    # Validate dimensions of the input data_map
    if data_map.ndim not in [3, 4]:
        raise ValueError(f"data_map must be a 3D or 4D array. Got {data_map.ndim} dimensions.")

    # Validate spatial dimensions against the reference NIfTI
    # ref_nifti_img.shape[:3] gives the first 3 dimensions (spatial) of the reference
    if data_map.shape[:3] != ref_nifti_img.shape[:3]:
        raise ValueError(
            f"Spatial dimensions of data_map {data_map.shape[:3]} do not match "
            f"reference NIfTI spatial dimensions {ref_nifti_img.shape[:3]}."
        )

    # If data_map is 4D, and reference is 3D, this is problematic for direct header copy.
    # However, if ref is 4D and data_map is 3D, it's common (e.g. saving a param map from DCE).
    # If both are 4D, ensure the 4th dimension matches if strict header usage is implied,
    # or allow different 4th dim if it's just for affine/basic header.
    # For this function, the primary goal is spatial alignment and correct affine.
    # Header modification handles data type and shape.

    # Create a new NIfTI header by copying from the reference.
    # This preserves most of the metadata including affine, orientation, etc.
    new_header = ref_nifti_img.header.copy()

    # Set the data type for the output map (float32 is common for parametric maps).
    new_header.set_data_dtype(np.float32)

    # Update the header with the shape of the new data_map.
    # This is crucial if data_map's dimensionality (3D vs 4D) or
    # 4th dimension size differs from the reference image.
    new_header.set_data_shape(data_map.shape)

    # If the reference image was 4D (e.g., a DCE series) and the new data_map is 3D
    # (e.g., a Ktrans map), some 4D-specific header fields might need adjustment
    # or removal. `set_data_shape` often handles the `dim` field correctly.
    # For instance, `dim[0]` (number of dimensions) will be set to 3 or 4.
    # `dim[4]` (size of 4th dim) will be set according to data_map.shape.
    # `pixdim` for the 4th dimension might also be implicitly handled or might
    # inherit a value that's irrelevant for a 3D map.
    # Nibabel generally manages these details well when creating the Nifti1Image object
    # with the new header and data.

    # Create the new NIfTI image object using the data_map, the reference affine,
    # and the modified header.
    new_nifti_image = nib.Nifti1Image(data_map.astype(np.float32), ref_nifti_img.affine, header=new_header)

    try:
        nib.save(new_nifti_image, output_filepath)
        # Consider using logging module instead of print for application use
        print(f"NIfTI map saved to: {output_filepath}")
    except Exception as e:
        # Provide more context for saving errors
        raise IOError(f"Could not save NIfTI map to {output_filepath}. Error: {e}")
