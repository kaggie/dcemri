"""
Command-line batch processor for DCE-MRI analysis.

This script allows users to process a single DCE-MRI dataset through a defined
pipeline including data loading, AIF definition (file or population model),
signal-to-concentration conversion, and pharmacokinetic model fitting.
Results (parameter maps) are saved to a specified output directory.

Key functionalities:
-   Parses command-line arguments for input files (DCE, T1 map, mask),
    processing parameters (TR, relaxivity, baseline points), AIF configuration,
    model selection, and output settings.
-   Loads NIfTI data using functions from `core.io`.
-   Performs signal-to-concentration conversion using `core.conversion`.
-   Handles AIF definition either from a file or by generating a population AIF
    using `core.aif`, allowing parameter overrides via CLI.
-   Fits selected pharmacokinetic models (Standard Tofts, Extended Tofts, Patlak, 2CXM)
    to the data voxel-wise using parallel processing via `core.modeling`.
-   Saves the resulting parameter maps as NIfTI files.

Example Usage:
python batch_processor.py \
    --dce /path/to/dce_series.nii.gz \
    --t1map /path/to/t1_map.nii.gz \
    --mask /path/to/mask.nii.gz \
    --tr 0.005 --r1_relaxivity 4.5 --baseline_points 5 \
    --aif_pop_model parker --aif_param D_scaler 1.0 --aif_param A1 0.81 \
    --model "Extended Tofts" \
    --out_dir /path/to/output_results \
    --num_processes 4

Note on AIF time units:
The script currently assumes that the time vector for AIF generation (derived from
DCE TR and number of time points) is in seconds. Population AIF model parameters
(e.g., decay rates m1, m2 from AIF_PARAMETER_METADATA) are often defined with time
in minutes. Users should ensure that provided AIF parameters are consistent with
the time unit used (seconds) or that the AIF model functions in `core.aif`
internally handle any necessary unit conversions if parameters are strictly per minute.
The current script does not perform automatic unit conversion for AIF model parameters based on TR units.
"""
import argparse
import os
import sys # Added for sys.path modification explanation
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz
import nibabel as nib # Not strictly needed if io functions handle all NIfTI aspects

# Add project root to sys.path to allow direct import of core modules
    # This assumes the script is in a directory like 'dce_mri_analyzer/scripts/'
    # and 'core' is in 'dce_mri_analyzer/core/'.
    # If running as a script (__package__ is None or empty), this line adds the
    # parent directory of the current script's parent directory (i.e., 'dce_mri_analyzer/')
    # to the Python path. This allows direct imports like `from core import io`.
if __package__ is None or __package__ == '':
        # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Original
        # More robust way to get project root if script is in project_root/scripts/batch_processor.py
        # and core is in project_root/core
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        sys.path.insert(0, project_root)


from core import io
from core import aif
from core import conversion
from core import modeling

def main():
    """
    Main function for the batch processing script.

    Parses command-line arguments, loads data, preprocesses AIF,
    performs signal-to-concentration conversion, fits the selected
    pharmacokinetic model, and saves the resulting parameter maps.
    """
    parser = argparse.ArgumentParser(description="DCE-MRI Batch Processor - Single Dataset")

    # --- Argument Definitions ---
    # Input Files
    parser.add_argument("--dce", required=True, help="Path to the 4D DCE NIfTI file.")
    parser.add_argument("--t1map", required=True, help="Path to the 3D T1 map NIfTI file (pre-contrast T1 values).")
    parser.add_argument("--mask", help="Path to a 3D Mask NIfTI file (optional). Processing will be limited to mask region if provided.")

    # Processing Parameters
    parser.add_argument("--tr", required=True, type=float, help="Repetition Time (TR) of the DCE sequence in seconds.")
    parser.add_argument("--r1_relaxivity", required=True, type=float, help="r1 relaxivity of the contrast agent (e.g., in L/mmol/s or s⁻¹mM⁻¹). Ensure units are consistent with concentration units desired.")
    parser.add_argument("--baseline_points", type=int, default=5, help="Number of initial (pre-contrast) time points in the DCE series used for baseline signal calculation (S0). Default is 5.")

    # AIF Configuration - mutually exclusive group: user must provide either a file or choose a population model
    aif_group = parser.add_mutually_exclusive_group(required=True)
    aif_group.add_argument("--aif_file", help="Path to an AIF file (CSV or TXT format with two columns: time, concentration).")
    aif_group.add_argument("--aif_pop_model", choices=list(aif.POPULATION_AIFS.keys()), help="Name of the population AIF model to use (e.g., 'parker', 'weinmann').")
    
    # Population AIF Parameters - allows overriding default model parameters
    # Example usage: --aif_param D_scaler 1.0 --aif_param A1 0.8
    # This uses action='append' and nargs=2 to collect key-value pairs.
    parser.add_argument('--aif_param', action='append', nargs=2, metavar=('PARAM_KEY', 'PARAM_VALUE'),
                        help="Set a specific parameter for the chosen population AIF model (e.g., D_scaler 1.0). Can be used multiple times for multiple parameters.")

    # Model Fitting Configuration
    parser.add_argument("--model", required=True, choices=["Standard Tofts", "Extended Tofts", "Patlak", "2CXM"], help="Pharmacokinetic model to apply for fitting.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of CPU cores to use for parallel model fitting. Defaults to all available cores.")

    # Output Configuration
    parser.add_argument("--out_dir", required=True, help="Output directory where the generated parameter maps will be saved.")

    args = parser.parse_args() # Parse the command-line arguments

    # --- Print Summary of Inputs ---
    print("--- DCE-MRI Batch Processor Configuration ---")
    print(f"  DCE File: {args.dce}")
    print(f"  T1 Map File: {args.t1map}")
    print(f"  Mask File: {args.mask if args.mask else 'Not provided'}")
    print(f"  TR: {args.tr} s, r1 Relaxivity: {args.r1_relaxivity}, Baseline Points: {args.baseline_points}")
    if args.aif_file:
        print(f"  AIF Source: File - {args.aif_file}")
    if args.aif_pop_model:
        print(f"  AIF Source: Population Model - {args.aif_pop_model}")
        if args.aif_param:
            print(f"    Custom Population AIF Parameters: {dict(args.aif_param)}") # Convert list of pairs to dict for printing
    print(f"  Pharmacokinetic Model: {args.model}")
    print(f"  Output Directory: {args.out_dir}")
    print(f"  Number of Processes for Fitting: {args.num_processes}")
    print("-------------------------------------------")

    # --- Ensure Output Directory Exists ---
    try:
        os.makedirs(args.out_dir, exist_ok=True) # Create output directory if it doesn't exist
        print(f"Output directory '{args.out_dir}' ensured.")
    except Exception as e:
        print(f"Fatal Error: Could not create output directory '{args.out_dir}'. Reason: {e}")
        exit(1) # Exit if output directory cannot be created

    # --- 1. Load Input Data ---
    try:
        print("Step 1: Loading input data...")
        dce_data, dce_affine, dce_header = io.load_dce_series(args.dce)
        print(f"  DCE data loaded successfully. Shape: {dce_data.shape}")

        t10_data, t10_affine, t10_header = io.load_t1_map(args.t1map, dce_shape=dce_data.shape) # Validate T1 map against DCE shape
        print(f"  T1 map loaded successfully. Shape: {t10_data.shape}")

        mask_data = None
        if args.mask:
            mask_data, mask_affine, mask_header = io.load_mask(args.mask, reference_shape=dce_data.shape[:3]) # Validate mask against DCE spatial shape
            print(f"  Mask loaded successfully. Shape: {mask_data.shape}")
        else:
            print("  No mask provided. Processing will be applied to all voxels.")
    except FileNotFoundError as fnf_error:
        print(f"Fatal Error: Input file not found. {fnf_error}")
        exit(1)
    except ValueError as val_error:
        print(f"Fatal Error: Invalid input file or mismatched dimensions. {val_error}")
        exit(1)
    except Exception as e:
        print(f"Fatal Error: An unexpected error occurred during data loading: {e}")
        exit(1)

    # --- 2. Signal-to-Concentration Conversion ---
    try:
        print("Step 2: Performing signal-to-concentration conversion...")
        Ct_data = conversion.signal_to_concentration(
            dce_series_data=dce_data,
            t10_map_data=t10_data,
            r1_relaxivity=args.r1_relaxivity,
            TR=args.tr,
            baseline_time_points=args.baseline_points
        )
        print(f"  Signal-to-concentration conversion successful. Ct_data shape: {Ct_data.shape}")
    except ValueError as val_error:
        print(f"Fatal Error: Invalid parameters for signal-to-concentration conversion. {val_error}")
        exit(1)
    except Exception as e:
        print(f"Fatal Error: An unexpected error occurred during signal-to-concentration conversion: {e}")
        exit(1)

    # --- 3. Prepare Arterial Input Function (AIF) ---
    aif_time_arr, aif_conc_arr = None, None # Initialize AIF arrays
    try:
        print("Step 3: Preparing AIF...")
        if args.aif_file:
            # Load AIF from the specified file
            print(f"  Loading AIF from file: {args.aif_file}")
            aif_time_arr, aif_conc_arr = aif.load_aif_from_file(args.aif_file)
            print(f"  AIF loaded from file. Time points: {len(aif_time_arr)}")
        elif args.aif_pop_model:
            # Generate AIF using a population model
            print(f"  Generating population AIF using model: {args.aif_pop_model}")
            num_time_points_dce = dce_data.shape[3]
            
            # Create a time vector for the AIF, typically matching the DCE acquisition duration and TR.
            # The units of this time vector (seconds vs. minutes) must be consistent with
            # how the population AIF model parameters (e.g., decay rates m1, m2) are defined.
            # Current AIF models in `core.aif` (e.g., Parker, Weinmann) have default parameters
            # assuming time is in minutes. If `args.tr` is in seconds, this time vector will be in seconds.
            # This discrepancy needs careful handling: either AIF models should adapt, or time vector
            # should be converted, or parameters provided via --aif_param should be in units
            # consistent with the time vector's units (seconds).
            # Current implementation: time_vector_for_aif is in seconds.
            # User needs to ensure --aif_param values are appropriate for time in seconds if overriding defaults.
            time_vector_for_aif = np.arange(num_time_points_dce) * args.tr # Time in seconds
            
            pop_aif_params_from_cli = {}
            if args.aif_param: # If user provided --aif_param arguments
                for key, value_str in args.aif_param:
                    try:
                        pop_aif_params_from_cli[key] = float(value_str)
                    except ValueError:
                        print(f"Warning: Could not convert population AIF parameter '{key}' value '{value_str}' to float. This parameter will be ignored or default will be used if applicable.")
            
            # Fetch default parameters from metadata for the selected model
            final_pop_aif_params = {}
            if args.aif_pop_model in aif.AIF_PARAMETER_METADATA:
                for p_name_meta, p_default_meta, _, _, _ in aif.AIF_PARAMETER_METADATA[args.aif_pop_model]:
                    final_pop_aif_params[p_name_meta] = p_default_meta # Start with model's default
            else:
                 print(f"Warning: No parameter metadata found for AIF model '{args.aif_pop_model}'. Using only parameters specified via --aif_param or model's hardcoded defaults.")

            # Override defaults with any parameters provided via CLI
            for p_name_cli, p_val_cli in pop_aif_params_from_cli.items():
                if p_name_cli in final_pop_aif_params:
                    final_pop_aif_params[p_name_cli] = p_val_cli
                else:
                    # If a CLI param is not in metadata, it might be a typo or an undocumented param.
                    # Still, allow it to be passed to the AIF function, which might handle it.
                    final_pop_aif_params[p_name_cli] = p_val_cli
                    print(f"Warning: AIF parameter '{p_name_cli}' provided via CLI is not found in standard metadata for model '{args.aif_pop_model}'. It will still be passed to the model function.")

            print(f"  Using effective AIF parameters for '{args.aif_pop_model}': {final_pop_aif_params}")
            aif_conc_arr = aif.generate_population_aif(args.aif_pop_model, time_vector_for_aif, params=final_pop_aif_params)
            aif_time_arr = time_vector_for_aif # Time array matches the generated concentration array
            
        if aif_time_arr is None or aif_conc_arr is None:
            print("Fatal Error: AIF could not be defined or generated. Check AIF file path or population model parameters.")
            exit(1)
        print(f"  AIF prepared successfully. Time points: {len(aif_time_arr)}, Max Concentration: {np.max(aif_conc_arr):.4f}")

    except FileNotFoundError as fnf_error:
        print(f"Fatal Error: AIF file not found. {fnf_error}")
        exit(1)
    except ValueError as val_error:
        print(f"Fatal Error: Invalid AIF data or parameters. {val_error}")
        exit(1)
    except Exception as e:
        print(f"Fatal Error: An unexpected error occurred during AIF preparation: {e}")
        exit(1)

    # --- 4. Pharmacokinetic Model Fitting ---
    try:
        print(f"Step 4: Performing {args.model} model fitting...")
        # Time vector for tissue concentration curves (Ct_data)
        # This should be in the same units as aif_time_arr for consistency in modeling functions.
        # Since aif_time_arr (if generated from pop model) used args.tr (seconds), t_tissue should also be in seconds.
        t_tissue = np.arange(Ct_data.shape[3]) * args.tr # Time in seconds
        
        parameter_maps = {} # Dictionary to store output parameter maps
        
        print(f"  Starting fitting with {args.num_processes} processes...")
        if args.model == "Standard Tofts":
            parameter_maps = modeling.fit_standard_tofts_voxelwise(
                Ct_data, t_tissue, aif_time_arr, aif_conc_arr, 
                mask=mask_data, num_processes=args.num_processes
            )
        elif args.model == "Extended Tofts":
            parameter_maps = modeling.fit_extended_tofts_voxelwise(
                Ct_data, t_tissue, aif_time_arr, aif_conc_arr, 
                mask=mask_data, num_processes=args.num_processes
            )
        elif args.model == "Patlak":
            parameter_maps = modeling.fit_patlak_model_voxelwise(
                Ct_data, t_tissue, aif_time_arr, aif_conc_arr, 
                mask=mask_data, num_processes=args.num_processes
            )
        elif args.model == "2CXM":
            parameter_maps = modeling.fit_2cxm_model_voxelwise(
                Ct_data, t_tissue, aif_time_arr, aif_conc_arr, 
                mask=mask_data, num_processes=args.num_processes
            )
        else:
            print(f"Model {args.model} not implemented in batch mode.")
            exit(1)
        print(f"{args.model} fitting completed.")

    except Exception as e:
        print(f"Error during model fitting: {e}")
        # Consider printing traceback for debugging:
        # import traceback
        # traceback.print_exc()
        exit(1)

    # 5. Save Maps
    try:
        if not parameter_maps:
            print("No parameter maps were generated by the model fitting.")
        else:
            print("Saving parameter maps...")
            for map_name, map_data in parameter_maps.items():
                if map_data is not None:
                    output_filepath = os.path.join(args.out_dir, f"{map_name}.nii.gz")
                    # Use T1 map as reference for saving header/affine
                    # This assumes args.t1map is a valid path to a NIfTI file
                    io.save_nifti_map(map_data, args.t1map, output_filepath)
                    print(f"Saved {map_name} to {output_filepath}")
                else:
                    print(f"Map data for '{map_name}' is None, not saving.")
    except Exception as e:
        print(f"Error saving parameter maps: {e}")
        exit(1)

    print("--- Batch processing completed successfully! ---")

if __name__ == "__main__":
    # For multiprocessing safety, especially when bundled
    if sys.platform.startswith('win'):
        import multiprocessing
        multiprocessing.freeze_support()
    main()
