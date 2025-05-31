import numpy as np
import csv

"""
This module provides functions for generating and saving reports based on
pharmacokinetic model fitting results, particularly focusing on Region of Interest (ROI)
statistics.

It includes utilities to:
- Calculate descriptive statistics (mean, median, stddev, etc.) for parameter
  map values within specified ROIs.
- Format these statistics into human-readable strings.
- Save aggregated ROI statistics from multiple parameter maps and ROIs into
  a structured CSV file.
"""

def calculate_roi_statistics(data_map_slice: np.ndarray, roi_mask_slice: np.ndarray) -> dict:
    """
    Calculates basic statistics for values within an ROI on a 2D data slice.

    The statistics include total number of pixels in ROI ('N'), number of valid (non-NaN)
    pixels ('N_valid'), mean, standard deviation, median, minimum, and maximum of
    the valid pixel values.

    Args:
        data_map_slice (np.ndarray): The 2D NumPy array of the parameter map slice.
        roi_mask_slice (np.ndarray): A 2D boolean NumPy array of the same shape as
                                     `data_map_slice`, where True indicates pixels
                                     within the ROI.

    Returns:
        dict: A dictionary containing the calculated statistics:
              {
                  "N": int,             # Total pixels in ROI
                  "N_valid": int,       # Number of non-NaN pixels in ROI
                  "Mean": float,
                  "StdDev": float,
                  "Median": float,
                  "Min": float,
                  "Max": float
              }
              If the ROI is empty or all values within the ROI are NaN, 'N' and 'N_valid'
              will be 0, and other statistics will be np.nan.

    Raises:
        ValueError: If input arrays are not 2D or have mismatched shapes.
    """
    if not isinstance(data_map_slice, np.ndarray) or data_map_slice.ndim != 2:
        raise ValueError("data_map_slice must be a 2D NumPy array.")
    if not isinstance(roi_mask_slice, np.ndarray) or roi_mask_slice.ndim != 2:
        raise ValueError("roi_mask_slice must be a 2D NumPy array.")
    if data_map_slice.shape != roi_mask_slice.shape:
        raise ValueError("data_map_slice and roi_mask_slice must have the same shape.")

    # Ensure roi_mask_slice is boolean, as it might be passed as int/float
    roi_mask_slice_bool = roi_mask_slice.astype(bool)
    
    # Extract values from the data map slice that fall within the ROI
    roi_values = data_map_slice[roi_mask_slice_bool]

    if roi_values.size == 0: # No pixels in the ROI
        return {"N": 0, "N_valid": 0, "Mean": np.nan, "StdDev": np.nan, 
                "Median": np.nan, "Min": np.nan, "Max": np.nan}
    
    # Calculate statistics, ignoring NaNs where appropriate
    stats = {
        "N": roi_values.size,                        # Total number of pixels in the ROI
        "N_valid": np.sum(~np.isnan(roi_values)),    # Number of valid (non-NaN) pixels
        "Mean": np.nanmean(roi_values),              # Mean of non-NaN values
        "StdDev": np.nanstd(roi_values),             # Standard deviation of non-NaN values
        "Median": np.nanmedian(roi_values),          # Median of non-NaN values
        "Min": np.nanmin(roi_values),                # Minimum of non-NaN values
        "Max": np.nanmax(roi_values)                 # Maximum of non-NaN values
    }
    return stats

def format_roi_statistics_to_string(stats_dict: dict | None, map_name: str, roi_name: str = "ROI") -> str:
    """
    Formats ROI statistics from a dictionary into a human-readable string.

    Args:
        stats_dict (dict | None): Dictionary of statistics, typically from
                                  `calculate_roi_statistics`. If None or if
                                  'N_valid' is 0, a message indicating no
                                  valid data is returned.
        map_name (str): Name of the parameter map for which statistics were calculated
                        (e.g., "Ktrans", "Ve").
        roi_name (str, optional): Name of the ROI (e.g., "Tumor Core", "Whole Lesion").
                                  Defaults to "ROI".

    Returns:
        str: A multi-line formatted string of the statistics, or a message
             if no valid data is available.
    """
    # Check if the statistics dictionary is valid and contains data
    if stats_dict is None or not isinstance(stats_dict, dict) or stats_dict.get("N_valid", 0) == 0:
        return f"No valid data points found in {roi_name} for parameter map '{map_name}'."

    lines = [f"Statistics for {roi_name} on parameter map '{map_name}':"]
    for key, value in stats_dict.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)

def save_multiple_roi_statistics_csv(
    stats_results_list: list[tuple[str, int, str, dict]], 
    filepath: str
    ):
    '''
    Saves statistics for multiple ROIs to a single CSV file.
    Args:
        stats_results_list (list[tuple[str, int, str, dict]]):
            A list of tuples, where each tuple contains:
            - map_name (str): Name of the parameter map (e.g., "Ktrans").
            - slice_index (int): Z-slice index from which the ROI was taken.
            - roi_name (str): Name of the ROI (e.g., "Tumor_Slice10").
            - stats_dict (dict): Dictionary of statistics for this ROI/map/slice,
                                 as returned by `calculate_roi_statistics`.
        filepath (str): Path to the CSV file where the statistics will be saved.

    Raises:
        IOError: If an error occurs during file writing.
        Exception: For other unexpected errors during the process.
    '''
    if not stats_results_list:
        # It might be desirable to log this or inform the user.
        # For now, if the list is empty, we simply don't create a file.
        print("No statistics provided to save_multiple_roi_statistics_csv. CSV file will not be created.")
        return

    # Determine the fieldnames for the CSV file.
    # These include contextual information (MapName, SliceIndex, ROIName)
    # and the actual statistical measures (Mean, Median, etc.).
    # Determine the fieldnames for the CSV file by collecting all unique stat keys.
    # These include contextual information (MapName, SliceIndex, ROIName)
    # and all unique actual statistical measures found across all stats_dicts.
    all_stat_keys = set()
    has_any_valid_stats = False
    for _, _, _, s_dict in stats_results_list:
        if s_dict: # s_dict could be None
            all_stat_keys.update(s_dict.keys())
            if s_dict.get("N_valid", 0) > 0:
                has_any_valid_stats = True
    
    if not all_stat_keys and not has_any_valid_stats :
         # If all ROIs are empty/None and no keys could be gathered (e.g. all stats_dicts were None)
         # use a default set of stat keys.
         stat_keys = ["N", "N_valid", "Mean", "StdDev", "Median", "Min", "Max"]
    else:
         # Sort for consistent column order, though not strictly necessary for DictWriter
         stat_keys = sorted(list(all_stat_keys))

    # Standard headers for context, followed by the dynamically obtained statistic keys
    fieldnames = ['MapName', 'SliceIndex', 'ROIName'] + stat_keys

    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader() # Write the header row

            for map_name, slice_idx, roi_name, stats_dict_for_roi in stats_results_list:
                row_data = {
                    'MapName': map_name,
                    'SliceIndex': slice_idx,
                    'ROIName': roi_name
                }
                if stats_dict_for_roi: # If stats_dict is not None
                    # Populate the row with actual statistics
                    # Use .get(key, np.nan) to handle cases where a specific stat might be missing
                    # (though calculate_roi_statistics should be consistent).
                    for skey in stat_keys:
                        row_data[skey] = stats_dict_for_roi.get(skey, np.nan)
                else:
                    # If stats_dict_for_roi is None (e.g., an error occurred for this ROI earlier),
                    # fill stat columns with NaN or a placeholder.
                    for skey in stat_keys:
                        row_data[skey] = np.nan # Or "Error" / "N/A"
                
                writer.writerow(row_data)
        print(f"Multiple ROI statistics saved to: {filepath}")
    except IOError as e:
        raise IOError(f"Error writing ROI statistics to CSV file {filepath}: {e}")
    except Exception as e: # Catch any other unexpected errors
        raise Exception(f"An unexpected error occurred while saving multiple ROI statistics to CSV: {e}")

# Keep the old function for now, or mark as deprecated.
# For this task, we'll just leave it. If it's not used, it can be removed later.
def save_roi_statistics_csv(stats_dict: dict, filepath: str, map_name: str, roi_name: str = "ROI"):
    """
    Saves statistics for a single ROI to a CSV file.
    (Note: This is an older version, consider using `save_multiple_roi_statistics_csv`
     for more comprehensive reporting across multiple ROIs/maps.)

    The CSV format is: MapName, ROIName, Statistic, Value.

    Args:
        stats_dict (dict): Dictionary of statistics (e.g., from `calculate_roi_statistics`).
        filepath (str): Path to the CSV file.
        map_name (str): Name of the parameter map.
        roi_name (str, optional): Name of the ROI. Defaults to "ROI".

    Raises:
        ValueError: If `stats_dict` is empty or None.
        IOError: If an error occurs during file writing.
    """
    if not stats_dict: # Check if the dictionary is empty or None
        raise ValueError("No statistics data provided to save.")
        
    fieldnames = ['MapName', 'ROIName', 'Statistic', 'Value']
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for stat_name, stat_value in stats_dict.items():
                writer.writerow({
                    'MapName': map_name, 
                    'ROIName': roi_name, 
                    'Statistic': stat_name, 
                    'Value': stat_value
                })
    except IOError as e:
        raise IOError(f"Error writing ROI statistics to CSV file {filepath}: {e}")
