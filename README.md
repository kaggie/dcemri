# DCE-MRI Analysis Tool

## Overview

This tool is being developed to enable researchers and clinicians to load and manage DCE-MRI (Dynamic Contrast-Enhanced Magnetic Resonance Imaging) time-series data, convert raw signal intensity to contrast agent concentration, perform pharmacokinetic modeling, and visualize and report the results.

## Current Features

*   **Data Loading & Management:**
    *   Loading of 4D DCE NIfTI series (`.nii`, `.nii.gz`).
    *   Loading of 3D T1 maps (NIfTI).
    *   Loading of 3D Masks (NIfTI, optional).
    *   Validation of input file integrity (basic NIfTI format check) and dimensions (e.g., DCE is 4D, T1 map is 3D, spatial dimensions match).
*   **Signal-to-Concentration Conversion:**
    *   Conversion of raw signal intensity to contrast agent concentration using user-provided r1 relaxivity, TR (Repetition Time), and number of baseline time points.
*   **AIF Management:**
    *   Loading AIF from TXT/CSV files.
    *   Selection of population-based AIF models (e.g., Parker, Weinmann, Fast Bi-exponential).
    *   User interface for adjusting parameters (e.g., amplitudes, rate constants) of selected population AIF models.
    *   Interactive AIF definition by drawing an ROI on the displayed image (mean signal from ROI converted to concentration).
    *   Input fields for AIF-specific parameters (T10_blood, r1_blood, AIF baseline points).
    *   Saving and loading of user-defined AIF ROI definitions (slice, position, size, reference image) to/from JSON files.
    *   Saving of derived AIF curves (time & concentration data) to CSV/TXT files.
*   **Pharmacokinetic Model Fitting:**
    *   Implementation of Standard Tofts model, yielding Ktrans (volume transfer constant) and ve (extravascular extracellular space volume fraction).
    *   Implementation of Extended Tofts model, yielding Ktrans, ve, and vp (plasma volume fraction).
    *   Implementation of Patlak model, yielding Ktrans_patlak (Patlak Ktrans) and vp_patlak (Patlak vp).
    *   Implementation of Two-Compartment Exchange Model (2CXM), yielding Fp (plasma flow), PS (permeability-surface area product), vp (plasma volume fraction), and ve (extravascular extracellular space volume fraction).
    *   Voxel-wise fitting of the selected model to tissue concentration curves, optionally constrained by a loaded mask.
    *   Parallelized voxel-wise pharmacokinetic model fitting using Python's multiprocessing to leverage multiple CPU cores, significantly speeding up processing.
*   **Parameter Map Generation & Export:**
    *   Generation of 3D parameter maps for Ktrans, ve, vp (from Tofts models), Ktrans_patlak, vp_patlak (from Patlak model), and Fp_2cxm, PS_2cxm, vp_2cxm, ve_2cxm (from 2CXM).
    *   Export of these maps as NIfTI files, using a reference NIfTI (e.g., T1 map or original DCE) for spatial alignment and header information.
*   **Visualization:**
    *   Display of loaded 3D/4D volumes (DCE, T1, Mask), generated concentration maps (mean over time), and pharmacokinetic parameter maps as 2D slices.
    *   Slice navigation using a slider.
    *   Interactive plotting of concentration-time curves for any selected voxel by double-clicking on the image viewer (plots tissue concentration, AIF, and the fitted model curve if available).
    *   Overlay of parameter maps: Display parameter maps semi-transparently on top of a selected anatomical base image (e.g., T1 map, Mean DCE), with controls for overlay map selection, alpha (transparency), and colormap.
*   **ROI Analysis & Reporting:**
    *   Tools to draw multiple ROIs on displayed parameter maps or other images for statistical analysis. Each ROI is assigned a unique name and a distinct color.
    *   Calculation of basic statistics (mean, std, median, min, max, N, N_valid) for these ROIs.
    *   Statistics for all defined ROIs are displayed in the UI, updating dynamically based on the currently viewed map and slice. ROIs defined on other views are also listed.
    *   Saving of statistics for all currently defined (and valid) ROIs to a single CSV file.
*   **Output and Reporting:**
    *   Ability to save the currently displayed AIF/Concentration/Fit plot to image files (PNG, SVG, etc.).
*   **User Interface:**
    *   Basic Graphical User Interface (GUI) for all functionalities.
    *   Logging of operations, loaded file details, and any errors encountered.

## Documentation and Guides

For detailed information on using the tool, understanding the implemented models, and advanced topics, please refer to the documentation in the `wiki/` directory of this repository.

Key documents include:
*   `wiki/01_Basic_Usage.md`: Step-by-step guide for GUI and CLI.
*   `wiki/02_Advanced_Topics.md`: In-depth information on AIF management, parallel processing, and troubleshooting.
*   `wiki/03_Model_Descriptions.md`: Descriptions of the pharmacokinetic models.
*   `wiki/04_Literature_References.md`: Citations for the algorithms and models used.

## Batch Processing
*   Initial command-line interface (`batch_processor.py`) for processing a single dataset without the GUI.
*   Supports specification of input files, AIF (file or population model with parameters), processing parameters, model choice, and output directory via CLI arguments.

## Technical Stack

*   Python 3.x
*   NumPy: For numerical operations and array handling.
*   SciPy: For scientific computing, including optimization (curve fitting), integration (ODE solving, numerical integration), and interpolation.
*   NiBabel: For loading and interacting with NIfTI files.
*   PyQt5: For the graphical user interface.
*   PyQtGraph: For 2D image visualization and plotting.

## Setup and Running

1.  **Clone the repository:**
    ```bash
    # git clone <repository_url> # (Placeholder for when hosted)
    # cd dce-mri-analyzer 
    ```
    (Assuming the repository root will be named `dce-mri-analyzer`)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies from `requirements.txt`:**
    Navigate into the project's root directory (e.g., `dce-mri-analyzer`, where `requirements.txt` is located) and run:
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary Python packages.

4.  **Installing from PyPI (Future):**
    Once the package is published on the Python Package Index (PyPI), you will be able to install it directly using pip:
    ```bash
    # pip install dce-mri-analyzer 
    ```
    (Note: This is a placeholder for future distribution.)

5.  **Run the application:**
    From the project's root directory (e.g., `dce_mri_analyzer`, where `main.py` is located):
    ```bash
    python main.py
    ```
    On Windows, if using multiprocessing, it's good practice to ensure the script is run in a way that `multiprocessing.freeze_support()` can be effective (this is included in `main.py`).

## Quick Start

This section provides a minimal example to get started with the tool.

### Using the GUI (`main.py`)

1.  Launch the application:
    ```bash
    python main.py
    ```
2.  Load your 4D DCE NIfTI image (e.g., `dce_series.nii.gz`).
3.  Load your 3D T1 map (e.g., `t1_map.nii.gz`).
4.  Optionally, load a 3D mask file (e.g., `mask.nii.gz`).
5.  Select or define an Arterial Input Function (AIF):
    *   Load from a TXT/CSV file.
    *   Choose a population-based model (e.g., Parker).
    *   Draw an ROI on the image to define an AIF.
6.  Set necessary parameters for signal-to-concentration conversion (r1 relaxivity, TR, AIF T10_blood, etc.) and select the number of baseline time points.
7.  Choose a pharmacokinetic model from the available options (e.g., Standard Tofts).
8.  Click "Run Fitting" to perform voxel-wise model fitting.
9.  View the generated parameter maps (e.g., Ktrans, ve).
10. Save desired parameter maps as NIfTI files.

### Using the Command-Line Interface (`batch_processor.py`)

1.  Run the batch processor with your data. For example:
    ```bash
    python batch_processor.py \
        --dce path/to/your/dce_series.nii.gz \
        --t1map path/to/your/t1_map.nii.gz \
        --aif-type population \
        --aif-model Parker \
        --output path/to/your/output_directory \
        --model Tofts \
        --r1 0.0045 \
        --tr 4.5 \
        --aif-t1-blood 1680 \
        --baseline-points 5 
    ```
2.  To see all available command-line options and their descriptions:
    ```bash
    python batch_processor.py --help
    ```

## Performance Note
Voxel-wise operations (like pharmacokinetic model fitting) can be time-consuming. The application now supports parallel processing for these operations to leverage multiple CPU cores, which can significantly reduce processing time. The number of cores can be selected in the UI. The 2CXM, due to its complexity (ODE solving per iteration), is notably slower than other models.

## To Do / Future Enhancements

*   **Advanced AIF Management:**
    *   Saving user-defined ROIs for AIF (currently saves definition, not the derived AIF curve itself).
    *   Integration of more population-based AIF models with UI for parameter adjustment.
*   **More Pharmacokinetic Models:**
    *   Implementation of other models (e.g., shutter-speed model).
*   **Improved Visualization:**
    *   ROI drawing tools for statistics (currently RectROI, could be more complex shapes).
    *   Direct display of NIfTI files without loading into NumPy arrays first for large datasets (memory efficiency).
*   **Batch Processing:**
    *   Ability to process multiple datasets via a script or batch interface (e.g., from a CSV manifest file).
*   **Output and Reporting:**
    *   More comprehensive export options (e.g., aggregated reports, saving plots from batch mode).
    *   Saving and loading of analysis "sessions" or "projects".

For a list of planned features and potential areas for future development, please see the `improvements.md` file in the project root.
Summaries of key scientific literature underpinning the models and AIFs implemented can be found in `literature.md`.

## Contributing

Contributions are welcome! We appreciate any effort to improve the tool, whether it's fixing a bug, adding a new feature, or improving documentation.

Here are some ways to contribute:

*   **Reporting Issues:** If you encounter a bug or have a suggestion, please open an issue on the project's issue tracker (link to be added once available).
*   **Submitting Pull Requests:**
    1.  Fork the repository.
    2.  Create a new branch for your feature or bug fix (e.g., `feature/new-model` or `fix/aif-loading-bug`).
    3.  Make your changes.
    4.  Add or update tests as appropriate to cover your changes.
    5.  Ensure your code adheres to general Python best practices (e.g., consider PEP 8 for style).
    6.  Write clear commit messages.
    7.  Push your branch to your fork and submit a pull request to the main repository.
*   **Proposing Major Changes:** For significant changes or new features, it's a good idea to open an issue first to discuss your ideas with the maintainers. This helps ensure your contribution aligns with the project's goals.

This project aims to provide a user-friendly and modular tool for DCE-MRI analysis.
