import numpy as np
import csv
import os
import json
from ..core import conversion

"""
This module provides functions for Arterial Input Function (AIF) generation,
loading, saving, and manipulation. AIFs are crucial in pharmacokinetic
modeling, representing the concentration of a contrast agent in arterial blood
over time.

The module includes:
- Definitions of population-averaged AIF models (Parker, Weinmann, Fast Bie-exponential).
- Metadata for AIF model parameters.
- Functions to load AIF data from text/CSV files.
- Functions to save AIF data to text/CSV files.
- Functions to generate AIF curves using population models.
- Functions to extract AIFs from a Region of Interest (ROI) in 4D DCE MRI data.
- Functions to save and load AIF ROI definitions.
"""

# --- AIF Parameter Metadata ---
AIF_PARAMETER_METADATA = {
    "parker": [
        # Parameter name, Default value, Min value, Max value, Tooltip (optional)
        # Units for Parker: time in minutes. A1/A2 in mM*min, m1/m2 in min^-1
        ('D_scaler', 1.0, 0.0, 10.0, "Overall scaling factor (e.g., dose adjustment)"),
        ('A1', 0.809, 0.0, 5.0, "Amplitude of first exponential (mM*min)"),
        ('m1', 0.171, 0.0, 5.0, "Decay rate of first exponential (min^-1)"),
        ('A2', 0.330, 0.0, 5.0, "Amplitude of second exponential (mM*min)"),
        ('m2', 2.05, 0.0, 10.0, "Decay rate of second exponential (min^-1)")
    ],
    "weinmann": [
        # Units for Weinmann: time in minutes. A1/A2 in mM*min, m1/m2 in min^-1
        ('D_scaler', 1.0, 0.0, 10.0, "Overall scaling factor"),
        ('A1', 3.99, 0.0, 10.0, "Amplitude of first exponential (mM*min)"), 
        ('m1', 0.144, 0.0, 2.0, "Decay rate of first exponential (min^-1)"), 
        ('A2', 4.78, 0.0, 10.0, "Amplitude of second exponential (mM*min)"),
        ('m2', 0.0111, 0.0, 1.0, "Decay rate of second exponential (min^-1)")
    ],
    "fast_biexponential": [
        # Units: time in minutes. A1/A2 unitless proportions, m1/m2 in min^-1
        ('D_scaler', 1.0, 0.0, 10.0, "Overall scaling factor"),
        ('A1', 0.6, 0.0, 1.0, "Proportion of first exponential"),
        ('m1', 3.0, 0.0, 10.0, "Decay rate of first exponential (min^-1)"),
        ('A2', 0.4, 0.0, 1.0, "Proportion of second exponential"),
        ('m2', 0.3, 0.0, 5.0, "Decay rate of second exponential (min^-1)")
    ]
}
"""A dictionary containing metadata for parameters of different AIF models.
The keys are model names (e.g., "parker", "weinmann") and the values are lists
of tuples, where each tuple defines a parameter:
(parameter_name, default_value, min_value, max_value, tooltip_string).
Units for time are generally in minutes. Concentration units depend on the model
but are often mM or arbitrary units scaled by D_scaler.
"""

def load_aif_from_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads AIF data (time and concentration) from a CSV or text file.

    The file should contain two columns: time and concentration.
    It can optionally have a header row. The function attempts to automatically
    detect CSV dialect and header presence.

    Args:
        filepath (str): Path to the AIF file.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - time_points: 1D array of time values.
            - concentrations: 1D array of concentration values.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is incorrect, data is non-numeric,
                    or the file is empty/contains no numeric data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"AIF file not found at: {filepath}")

    time_points, concentrations = [], []
    try:
        with open(filepath, 'r', newline='') as f:
            # Read a chunk to sniff for CSV dialect; reset pointer afterwards
            content_to_sniff = f.read(1024)
            f.seek(0)
            is_csv = False

            # Check if the file is explicitly a CSV
            if filepath.lower().endswith(".csv"):
                try:
                    dialect = csv.Sniffer().sniff(content_to_sniff)
                    reader = csv.reader(f, dialect)
                    is_csv = True
                except csv.Error:
                    # If sniffing fails for a .csv, proceed as a plain text file
                    pass # Fall through to non-CSV handling

            if not is_csv: # Handle as plain text or CSV where sniffing failed
                lines = f.readlines()
                if not lines:
                    raise ValueError(f"AIF file is empty: {filepath}")

                # Attempt to detect and skip header for plain text files
                start_line_index = 0
                first_line_parts = lines[0].strip().split()
                if first_line_parts:
                    try:
                        float(first_line_parts[0]) # Check if the first part of the first line is numeric
                    except ValueError:
                        start_line_index = 1 # Assume header if not numeric

                # Check if there's any data after a potential header
                if start_line_index >= len(lines) and len(lines) > 0 :
                    raise ValueError(f"No numeric data found after header in AIF file: {filepath}")
                if not lines[start_line_index:]:
                     raise ValueError(f"No numeric data found in AIF file: {filepath}")

                for line_num, line_content in enumerate(lines[start_line_index:]):
                    parts = line_content.strip().split()
                    if not parts: # Skip empty lines
                        continue
                    if len(parts) != 2:
                        raise ValueError(
                            f"Incorrect format in AIF file: {filepath} at line {line_num + start_line_index + 1}. "
                            f"Expected 2 columns, got {len(parts)}."
                        )
                    try:
                        time_points.append(float(parts[0]))
                        concentrations.append(float(parts[1]))
                    except ValueError:
                        raise ValueError(
                            f"Non-numeric data found in AIF file: {filepath} at line {line_num + start_line_index + 1}."
                        )
            else: # Handle as CSV (dialect was successfully sniffed)
                header_skipped = False
                for i, row in enumerate(reader):
                    if not row: # Skip empty rows
                        continue
                    if not header_skipped:
                        # Attempt to detect header in CSV by checking if first element is non-numeric
                        try:
                            float(row[0])
                        except ValueError:
                            header_skipped = True
                            continue # Skip header row
                    
                    if len(row) != 2:
                        raise ValueError(
                            f"Incorrect format in AIF file: {filepath} at line {i + 1}. "
                            f"Expected 2 columns, got {len(row)}."
                        )
                    try:
                        time_points.append(float(row[0]))
                        concentrations.append(float(row[1]))
                    except ValueError:
                        raise ValueError(
                            f"Non-numeric data found in AIF file: {filepath} at line {i + 1} after potential header."
                        )
            
            if not time_points: # Ensure some data was actually loaded
                raise ValueError(f"No numeric data found in AIF file: {filepath}")

    except Exception as e:
        # Catch-all for other read errors, re-raising as ValueError for consistency
        raise ValueError(f"Error reading AIF file {filepath}: {e}")

    return np.array(time_points), np.array(concentrations)

def save_aif_curve(time_points: np.ndarray, concentrations: np.ndarray, filepath: str):
    """
    Saves an AIF curve (time and concentration arrays) to a CSV or text file.

    The output file will have a header "Time,Concentration".
    The delimiter is a comma (`,`) for `.csv` files and a tab (`\t`) for `.txt` files.
    A warning is printed if an unknown file extension is used, and it defaults to CSV format.

    Args:
        time_points (np.ndarray): 1D array of time values.
        concentrations (np.ndarray): 1D array of concentration values.
        filepath (str): Path to save the AIF file.

    Raises:
        ValueError: If time_points and concentrations arrays have different lengths
                    or are not 1D arrays.
        IOError: If an error occurs during file writing.
    """
    if len(time_points) != len(concentrations):
        raise ValueError("Time points and concentrations arrays must have the same length.")
    if time_points.ndim != 1 or concentrations.ndim != 1:
        raise ValueError("Time points and concentrations must be 1D arrays.")

    # Prepare data for writing: transpose to get (N, 2) shape
    data = np.vstack((time_points, concentrations)).T

    try:
        # Determine delimiter based on file extension
        delimiter = ',' if filepath.lower().endswith('.csv') else '\t'
        if not (filepath.lower().endswith('.csv') or filepath.lower().endswith('.txt')):
            print(
                f"Warning: Unknown file extension for AIF curve ('{os.path.splitext(filepath)[1]}'), "
                f"saving as CSV with delimiter ','."
            )
            delimiter = ',' # Default to CSV for unknown extensions

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(['Time', 'Concentration']) # Write header
            writer.writerows(data) # Write data rows
    except Exception as e:
        raise IOError(f"Failed to save AIF curve to {filepath}: {e}")

def parker_aif(time_points: np.ndarray, D_scaler: float = 1.0, A1: float = 0.809, m1: float = 0.171, A2: float = 0.330, m2: float = 2.05) -> np.ndarray:
    """
    Implements a bi-exponential Parker Arterial Input Function (AIF).
    Based on Parker et al. (2006), Magn Reson Med, 56(5), 993-1000.
    Default parameters (A1, m1, A2, m2) assume time_points are in minutes.
    Cp(t) = D_scaler * (A1 * exp(-m1 * t) + A2 * exp(-m2 * t))

    Args:
        time_points (np.ndarray): Array of time points (typically in minutes).
        D_scaler (float, optional): Overall scaling factor. Defaults to 1.0.
        A1 (float, optional): Amplitude of the first exponential term (mM*min). Defaults to 0.809.
        m1 (float, optional): Decay rate of the first exponential term (min⁻¹). Defaults to 0.171.
        A2 (float, optional): Amplitude of the second exponential term (mM*min). Defaults to 0.330.
        m2 (float, optional): Decay rate of the second exponential term (min⁻¹). Defaults to 2.05.
    Returns:
        np.ndarray: AIF concentration values at the given time points.
    Raises:
        TypeError: If time_points is not a NumPy array.
        ValueError: If any of the AIF parameters (D_scaler, A1, m1, A2, m2) are negative.
    """
    if not isinstance(time_points, np.ndarray):
        raise TypeError("time_points must be a NumPy array.")
    if D_scaler < 0 or A1 < 0 or A2 < 0 or m1 < 0 or m2 < 0:
        raise ValueError("AIF parameters must be non-negative.")

    # Ensure time is non-negative for the exponential calculation
    valid_time_points = np.maximum(time_points, 0)
    term1 = A1 * np.exp(-m1 * valid_time_points)
    term2 = A2 * np.exp(-m2 * valid_time_points)
    return D_scaler * (term1 + term2)

def weinmann_aif(time_points: np.ndarray, D_scaler: float = 1.0, A1: float = 3.99, m1: float = 0.144, A2: float = 4.78, m2: float = 0.0111) -> np.ndarray:
    """
    Implements a bi-exponential Weinmann Arterial Input Function (AIF).
    Based on Weinmann et al. (1982), Am J Roentgenol, 142(3), 619-624.
    Default parameters (A1, m1, A2, m2) assume time_points are in minutes.
    Cp(t) = D_scaler * (A1 * exp(-m1 * t) + A2 * exp(-m2 * t))

    Args:
        time_points (np.ndarray): Array of time points (typically in minutes).
        D_scaler (float, optional): Overall scaling factor. Defaults to 1.0.
        A1 (float, optional): Amplitude of the first exponential term (mM*min). Defaults to 3.99.
        m1 (float, optional): Decay rate of the first exponential term (min⁻¹). Defaults to 0.144.
        A2 (float, optional): Amplitude of the second exponential term (mM*min). Defaults to 4.78.
        m2 (float, optional): Decay rate of the second exponential term (min⁻¹). Defaults to 0.0111.

    Returns:
        np.ndarray: AIF concentration values at the given time points.
    Raises:
        TypeError: If time_points is not a NumPy array.
        ValueError: If any of the AIF parameters (D_scaler, A1, m1, A2, m2) are negative.
    """
    if not isinstance(time_points, np.ndarray):
        raise TypeError("time_points must be a NumPy array.")
    if D_scaler < 0 or A1 < 0 or A2 < 0 or m1 < 0 or m2 < 0:
        raise ValueError("AIF parameters must be non-negative.")

    valid_time_points = np.maximum(time_points, 0)
    term1 = A1 * np.exp(-m1 * valid_time_points)
    term2 = A2 * np.exp(-m2 * valid_time_points)
    return D_scaler * (term1 + term2)

def fast_biexponential_aif(time_points: np.ndarray, D_scaler: float = 1.0, A1: float = 0.6, m1: float = 3.0, A2: float = 0.4, m2: float = 0.3) -> np.ndarray:
    """
    Implements a 'Fast' bi-exponential population-averaged Arterial Input Function (AIF).
    This model uses different default parameters for faster decay, potentially
    suitable for specific contrast agents or imaging protocols.
    Cp(t) = D_scaler * (A1 * exp(-m1 * t) + A2 * exp(-m2 * t))
    Default parameters assume time_points are in minutes.

    Args:
        time_points (np.ndarray): Array of time points (typically in minutes).
        D_scaler (float, optional): Overall scaling factor. Defaults to 1.0.
        A1 (float, optional): Amplitude or proportion of the first exponential term. Defaults to 0.6.
        m1 (float, optional): Decay rate of the first exponential term (min⁻¹). Defaults to 3.0.
        A2 (float, optional): Amplitude or proportion of the second exponential term. Defaults to 0.4.
        m2 (float, optional): Decay rate of the second exponential term (min⁻¹). Defaults to 0.3.

    Returns:
        np.ndarray: AIF concentration values at the given time points.
    Raises:
        TypeError: If time_points is not a NumPy array.
        ValueError: If any of the AIF parameters (D_scaler, A1, m1, A2, m2) are negative.
    """
    if not isinstance(time_points, np.ndarray):
        raise TypeError("time_points must be a NumPy array.")
    if D_scaler < 0 or A1 < 0 or A2 < 0 or m1 < 0 or m2 < 0:
        raise ValueError("AIF parameters must be non-negative.")

    valid_time_points = np.maximum(time_points, 0)
    term1 = A1 * np.exp(-m1 * valid_time_points)
    term2 = A2 * np.exp(-m2 * valid_time_points)
    return D_scaler * (term1 + term2)

POPULATION_AIFS = {
    "parker": parker_aif,
    "weinmann": weinmann_aif,
    "fast_biexponential": fast_biexponential_aif,
}
"""A dictionary mapping names of population AIF models to their respective functions."""

def generate_population_aif(name: str, time_points: np.ndarray, params: dict = None) -> np.ndarray | None:
    """
    Generates an AIF curve using a specified population model.

    Args:
        name (str): The name of the population AIF model to use (e.g., "parker", "weinmann").
        time_points (np.ndarray): Array of time points for which to generate the AIF.
        params (dict, optional): A dictionary of parameters to override the defaults
                                 for the selected AIF model. Defaults to None, which
                                 uses the model's default parameters.

    Returns:
        np.ndarray | None: A NumPy array representing the AIF concentrations,
                           or None if the specified model name is not found or an
                           unexpected error occurs during generation.

    Raises:
        ValueError: If there's an error calling the AIF model function with the
                    provided parameters (e.g., incorrect parameter names or types).
    """
    if name in POPULATION_AIFS:
        model_function = POPULATION_AIFS[name]
        try:
            if params:
                return model_function(time_points, **params)
            else:
                return model_function(time_points)
        except TypeError as e:
            # More specific error for parameter issues
            raise ValueError(f"Error calling AIF model '{name}' with provided parameters: {e}")
        except Exception as e:
            # Catch other unexpected errors during AIF generation
            print(f"Unexpected error generating population AIF '{name}': {e}")
            return None
    else:
        # Model name not found
        return None

def extract_aif_from_roi(
    dce_4d_data: np.ndarray,
    roi_2d_coords: tuple[int, int, int, int],
    slice_index_z: int,
    t10_blood: float,
    r1_blood: float,
    TR: float,
    baseline_time_points_aif: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts an Arterial Input Function (AIF) from a specified Region of Interest (ROI)
    within 4D DCE (Dynamic Contrast-Enhanced) MRI data.

    The process involves:
    1. Selecting a 3D patch from the 4D data based on ROI coordinates and slice index.
    2. Averaging the signal intensity within this patch across the spatial dimensions (x, y)
       for each time point, resulting in a mean signal time course.
    3. Converting this signal time course to a concentration time course using
       pharmacokinetic principles (e.g., T1 mapping, baseline signal).
    4. Generating a corresponding time array based on the TR (Repetition Time).

    Args:
        dce_4d_data (np.ndarray): The 4D DCE MRI data array with dimensions
                                  (X, Y, Z, Time).
        roi_2d_coords (tuple[int, int, int, int]): A tuple defining the 2D ROI
                                                in the XY plane: (x_start, y_start, width, height).
        slice_index_z (int): The Z-slice index where the ROI is located.
        t10_blood (float): Longitudinal relaxation time (T1) of blood before
                           contrast agent administration (in milliseconds or seconds,
                           ensure consistency with TR and r1_blood).
        r1_blood (float): Longitudinal relaxivity of the contrast agent in blood
                          (in s⁻¹mM⁻¹ or similar units, ensure consistency).
        TR (float): Repetition Time of the MRI sequence (in milliseconds or seconds,
                    ensure consistency with t10_blood).
        baseline_time_points_aif (int, optional): Number of initial time points
                                                 to average for establishing the
                                                 baseline signal (S0). Defaults to 5.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - aif_time_tc (np.ndarray): 1D array of time points for the AIF.
            - aif_conc_tc (np.ndarray): 1D array of AIF concentration values.

    Raises:
        ValueError: If input dimensions are incorrect, ROI coordinates are out of bounds,
                    ROI dimensions are non-positive, or the ROI patch is empty.
    """
    if dce_4d_data.ndim != 4:
        raise ValueError("dce_4d_data must be a 4D array (X, Y, Z, Time).")

    x_start, y_start, width, height = roi_2d_coords

    # Validate ROI coordinates and dimensions
    if not (0 <= x_start < dce_4d_data.shape[0] and \
            0 <= y_start < dce_4d_data.shape[1] and \
            0 <= slice_index_z < dce_4d_data.shape[2]):
        raise ValueError(f"ROI start coordinates or Z-slice index are out of bounds for the DCE data shape {dce_4d_data.shape}.")
    if not (x_start + width <= dce_4d_data.shape[0] and \
            y_start + height <= dce_4d_data.shape[1]):
        raise ValueError(f"ROI dimensions exceed data boundaries.")
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive.")

    # Extract the 3D patch corresponding to the ROI over time
    roi_patch_3d = dce_4d_data[x_start : x_start + width,
                               y_start : y_start + height,
                               slice_index_z,
                               :]
    if roi_patch_3d.size == 0: # Should be caught by previous checks, but as a safeguard
        raise ValueError("ROI patch is empty. Check ROI coordinates and data dimensions.")

    # Calculate the mean signal time course within the ROI
    mean_roi_signal_tc = np.mean(roi_patch_3d, axis=(0, 1)) # Average over X and Y
    if len(mean_roi_signal_tc) == 0: # Should not happen if roi_patch_3d is not empty
        raise ValueError("Mean ROI signal time course is empty.")

    # Convert signal time course to concentration time course
    aif_conc_tc = conversion.signal_tc_to_concentration_tc(
        signal_tc=mean_roi_signal_tc,
        T10=t10_blood,
        r1=r1_blood,
        TR=TR,
        n_baseline_points=baseline_time_points_aif
    )

    # Generate time axis for the AIF
    aif_time_tc = np.arange(len(mean_roi_signal_tc)) * TR

    return aif_time_tc, aif_conc_tc

def save_aif_roi_definition(roi_properties: dict, filepath: str):
    """
    Saves AIF ROI (Region of Interest) definition properties to a JSON file.

    Args:
        roi_properties (dict): A dictionary containing ROI properties.
                               Expected keys might include "slice_index", "pos_x",
                               "pos_y", "size_w", "size_h", "image_ref_name", etc.
        filepath (str): The path to the JSON file where the ROI definition will be saved.

    Raises:
        IOError: If an error occurs during file writing (e.g., permission issues)
                 or if `roi_properties` is not a serializable dictionary.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(roi_properties, f, indent=4) # Use indent for readability
    except (IOError, TypeError) as e: # Catch potential errors during JSON serialization or file I/O
        raise IOError(f"Error saving AIF ROI definition to {filepath}: {e}")

def load_aif_roi_definition(filepath: str) -> dict | None:
    """
    Loads AIF ROI (Region of Interest) definition properties from a JSON file.

    Performs validation to ensure essential keys are present and have correct types.

    Args:
        filepath (str): The path to the JSON file containing the ROI definition.

    Returns:
        dict | None: A dictionary containing the loaded ROI properties if successful.
                     Returns None if the file is not found (though FileNotFoundError is raised).
                     This function primarily raises errors for invalid content.

    Raises:
        FileNotFoundError: If the specified `filepath` does not exist.
        ValueError: If the file content is not a valid JSON object, if required keys
                    are missing, or if data types of values are incorrect.
        IOError: If an error occurs during file reading (e.g., permission issues).
    """
    required_keys = ["slice_index", "pos_x", "pos_y", "size_w", "size_h", "image_ref_name"]
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("ROI definition file is not a valid JSON object.")

        # Validate presence of required keys
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key in AIF ROI definition: '{key}'")

        # Validate data types for specific keys
        if not isinstance(data["slice_index"], int):
            raise ValueError("ROI 'slice_index' must be an integer.")
        if not all(isinstance(data[k], (int, float)) for k in ["pos_x", "pos_y", "size_w", "size_h"]):
            raise ValueError("ROI position (pos_x, pos_y) and size (size_w, size_h) values must be numeric (int or float).")
        if not isinstance(data["image_ref_name"], str):
            raise ValueError("ROI 'image_ref_name' must be a string.")
        # Add more specific validations as needed for other keys

        return data
    except FileNotFoundError:
        raise # Re-raise FileNotFoundError to be handled by the caller
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from AIF ROI file {filepath}: {e}")
    except IOError as e: # Catch other file reading issues
        raise IOError(f"Error reading AIF ROI definition from {filepath}: {e}")
