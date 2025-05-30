# Basic Usage Guide

This guide provides instructions on how to use the DCE-MRI Analysis Tool, covering both the Graphical User Interface (GUI) and the Command-Line Interface (CLI).

## 1. GUI Usage (`main.py`)

The GUI provides an interactive way to load data, perform analysis, and visualize results.

### 1.1. Launching the Application

1.  Navigate to the root directory of the project in your terminal.
2.  Ensure your Python virtual environment is activated.
3.  Run the main application script:
    ```bash
    python main.py
    ```
    This will open the main window of the application.

### 1.2. Loading Data

The first step is to load your DCE-MRI data:

1.  **Load 4D DCE NIfTI:**
    *   Click the "Load DCE Series" button.
    *   Select your 4D NIfTI file (e.g., `dce_series.nii.gz`).
    *   The application will validate the file and display its dimensions.
2.  **Load 3D T1 Map:**
    *   Click the "Load T1 Map" button.
    *   Select your 3D T1 map NIfTI file (e.g., `t1_map.nii.gz`).
    *   This map is used for T1 correction and as a reference for spatial alignment. Its dimensions must match the spatial dimensions of the DCE series.
3.  **Load 3D Mask (Optional):**
    *   Click the "Load Mask" button.
    *   Select your 3D mask NIfTI file (e.g., `mask.nii.gz`).
    *   If a mask is loaded, pharmacokinetic model fitting will only be performed on voxels within the mask, which can significantly speed up processing. The mask dimensions must match the spatial dimensions of the DCE series.

### 1.3. Arterial Input Function (AIF)

An AIF is crucial for pharmacokinetic modeling. You can define or load an AIF in several ways:

1.  **Load AIF from File:**
    *   Click the "Load AIF from File" button.
    *   Select a TXT or CSV file containing two columns: time and concentration.
    *   The AIF plot will be displayed.
2.  **Select Population-Based AIF Model:**
    *   Choose a model from the "Population AIF Model" dropdown (e.g., Parker, Weinmann).
    *   Adjust the model parameters (amplitudes, decay rates, scaler) using the input fields. The AIF plot will update dynamically.
3.  **Draw AIF from ROI:**
    *   Ensure your DCE series is loaded.
    *   Click the "Define AIF from ROI" button. This will activate ROI drawing mode on the DCE image viewer.
    *   Draw a rectangle over an area representing an artery (e.g., femoral artery, carotid artery).
    *   The mean signal intensity from this ROI over time will be converted to concentration and used as the AIF.
    *   You can save the ROI definition (slice, coordinates, image reference) using "Save AIF ROI" and load it later using "Load AIF ROI".

### 1.4. Setting Parameters

Before running the analysis, configure the necessary parameters:

*   **Signal to Concentration Parameters:**
    *   `r1 Relaxivity (tissue)`: Enter the r1 relaxivity of the contrast agent in tissue (e.g., 0.0045 s⁻¹mM⁻¹).
    *   `TR (Repetition Time)`: Enter the TR of your DCE sequence in seconds.
    *   `Baseline Time Points (Tissue)`: Specify the number of initial time points in the DCE series to be averaged for baseline signal (S0) calculation in tissue.
*   **AIF Specific Parameters (if applicable):**
    *   `T10 Blood`: Enter the baseline T1 value of blood in milliseconds (e.g., 1680 ms).
    *   `r1 Relaxivity (blood)`: Enter the r1 relaxivity for blood (e.g., 0.0045 s⁻¹mM⁻¹).
    *   `Baseline Time Points (AIF)`: Specify the number of baseline points if deriving AIF from an ROI.
*   **Pharmacokinetic Model:**
    *   Select the desired model from the "Pharmacokinetic Model" dropdown (e.g., Standard Tofts, Extended Tofts, Patlak, 2CXM).
*   **Parallel Processing:**
    *   `Number of Processes`: Choose the number of CPU cores to use for parallelized voxel-wise fitting. Using more cores can significantly speed up processing, especially for large datasets or complex models like 2CXM.

### 1.5. Running Model Fitting

1.  Once all data is loaded and parameters are set, click the "Run Fitting" button.
2.  The application will first convert the DCE signal to concentration on a voxel-by-voxel basis.
3.  Then, it will fit the selected pharmacokinetic model to the tissue concentration curves for all voxels (or those within the mask).
4.  Progress will be indicated in the log panel. This step can take time depending on the data size, model complexity, and number of processes selected.

### 1.6. Visualizing Results

*   **Parameter Maps:**
    *   After fitting, the generated parameter maps (e.g., Ktrans, ve, vp) will be listed in the "Available Maps" dropdown under the "View Maps" tab.
    *   Select a map to display it as 2D slices. Use the slider to navigate through slices.
    *   You can overlay parameter maps on an anatomical base image (e.g., T1 map) by selecting the base image and the overlay map, then adjusting transparency (alpha) and colormap.
*   **Concentration-Time Curves:**
    *   Double-click on any voxel in an image viewer (DCE, T1 map, parameter map) to display the concentration-time curve for that voxel.
    *   If model fitting has been performed, the plot will also show the AIF used and the fitted model curve for that voxel.

### 1.7. Saving Outputs

*   **Parameter Maps:**
    *   Select a parameter map from the "Available Maps" list.
    *   Click the "Save Current Map" button.
    *   Choose a filename and location. Maps are saved in NIfTI format.
*   **Plots:**
    *   The concentration-time curve plot can be saved by clicking the "Save Plot" button below the plot area. Choose the desired image format (PNG, SVG, etc.).
*   **AIF Curves:**
    *   If an AIF is loaded from a file or generated, it can be saved using the "Save AIF Curve" button.

### 1.8. ROI Analysis

1.  Display a parameter map or other image you want to analyze.
2.  Under the "ROI Statistics" tab, click "Add ROI".
3.  Draw a rectangular ROI on the image viewer. The ROI will be assigned a name and color.
4.  Statistics (mean, std, median, min, max, N, N_valid) for this ROI on the currently displayed map and slice will be shown in the table.
5.  You can define multiple ROIs.
6.  To save all calculated ROI statistics to a CSV file, click "Save All ROI Stats".

## 2. Command-Line Interface (CLI) Usage (`batch_processor.py`)

The `batch_processor.py` script allows for automated processing of a single dataset without the GUI. This is useful for batch scripting or integrating into larger analysis pipelines.

### 2.1. Purpose

The CLI takes paths to input NIfTI files, AIF information, model selection, and processing parameters as command-line arguments and saves the resulting parameter maps to a specified output directory.

### 2.2. Getting Help

To see all available command-line options and their descriptions, run:
```bash
python batch_processor.py --help
```

### 2.3. Example Command

Here's an example of a command to run the batch processor:

```bash
python batch_processor.py \
    --dce path/to/your/dce_series.nii.gz \
    --t1map path/to/your/t1_map.nii.gz \
    --mask path/to/your/mask.nii.gz \
    --aif-type population \
    --aif-model Parker \
    --aif-param D_scaler 1.0 \
    --aif-param A1 0.809 \
    --model Tofts \
    --r1 0.0045 \
    --tr 4.5 \
    --t1-blood 1680 \
    --r1-blood 0.0045 \
    --baseline-points-tissue 5 \
    --baseline-points-aif 3 \
    --output path/to/your/output_directory \
    --num-processes 4
```

**Explanation of common arguments:**

*   `--dce`: Path to the 4D DCE NIfTI file.
*   `--t1map`: Path to the 3D T1 map NIfTI file.
*   `--mask` (optional): Path to the 3D mask NIfTI file.
*   `--aif-type`: Method for AIF definition.
    *   `population`: Use a population-based model. Requires `--aif-model`.
    *   `file`: Load AIF from a TXT/CSV file. Requires `--aif-file`.
    *   `roi` (if supported): Define AIF from an ROI in a JSON definition file. Requires `--aif-roi-file`.
*   `--aif-model`: Name of the population AIF model (e.g., `Parker`, `Weinmann`).
*   `--aif-param name value`: Sets a parameter for the population AIF model (can be used multiple times).
*   `--aif-file`: Path to the AIF TXT/CSV file.
*   `--model`: Pharmacokinetic model to use (e.g., `Tofts`, `ExtendedTofts`, `Patlak`, `2CXM`).
*   `--r1`: r1 relaxivity of contrast agent in tissue.
*   `--tr`: Repetition Time of the DCE sequence.
*   `--t1-blood`: Baseline T1 of blood (for AIF from ROI or some population models if applicable).
*   `--r1-blood`: r1 relaxivity for blood.
*   `--baseline-points-tissue`: Number of baseline time points for tissue S0.
*   `--baseline-points-aif`: Number of baseline time points for AIF S0 (if AIF from ROI).
*   `--output`: Path to the directory where output parameter maps will be saved.
*   `--num-processes` (optional): Number of CPU cores for parallel processing.

### 2.4. Expected Inputs and Outputs

*   **Inputs:** Valid NIfTI files for DCE, T1 map, and optionally mask. AIF data via file or population model parameters.
*   **Outputs:** The script will save the calculated pharmacokinetic parameter maps (e.g., `Ktrans.nii.gz`, `ve.nii.gz`) as NIfTI files in the specified output directory.

Refer to the `--help` output for a complete and up-to-date list of all arguments and their descriptions.
