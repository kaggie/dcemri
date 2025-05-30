# Advanced Topics

This section delves into more specific aspects of the DCE-MRI Analysis Tool, providing detailed information on AIF management, parallel processing, and common troubleshooting tips.

## 1. AIF Management Deep Dive

Accurate AIF definition is critical for reliable pharmacokinetic modeling.

### 1.1. AIF File Formats (TXT/CSV)

When loading an AIF from a file, the tool expects a simple text-based format:

*   **Structure:** Two columns of numerical data.
    *   Column 1: Time (typically in minutes or seconds).
    *   Column 2: Contrast agent concentration (typically in mM).
*   **Delimiter:**
    *   For `.csv` files, a comma (`,`) is usually expected as the delimiter.
    *   For `.txt` files, tabs or spaces are common delimiters.
    *   The tool attempts to auto-detect the delimiter.
*   **Header:** An optional header line can be present. The tool will attempt to automatically skip it if it doesn't parse as numeric data.
*   **Example (`aif.txt`):**
    ```
    Time(min) Concentration(mM)
    0.0       0.0
    0.1       0.5
    0.2       2.5
    0.3       5.0
    0.4       4.5
    ...       ...
    ```

**Important Considerations:**
*   Ensure the time units of your AIF file are consistent with the time units expected by the pharmacokinetic models and other time-related parameters (like TR, if converting from seconds to minutes implicitly or explicitly). The population AIF models (Parker, Weinmann, Fast Bi-exponential) assume time in **minutes** for their default parameters.
*   The concentration units should also be consistent (typically mM).

### 1.2. Population AIF Model Parameters

The tool includes several common population-based AIF models. Their parameters can be adjusted in the GUI.

*   **Parker et al. (2006)**
    *   `D_scaler`: Overall scaling factor (dimensionless). Default: 1.0. Applied to the entire AIF. Useful for dose adjustments.
    *   `A1`: Amplitude of the first exponential term (mM*min). Default: 0.809.
    *   `m1`: Decay rate of the first exponential term (min⁻¹). Default: 0.171.
    *   `A2`: Amplitude of the second exponential term (mM*min). Default: 0.330.
    *   `m2`: Decay rate of the second exponential term (min⁻¹). Default: 2.05.
    *   *Formula*: `Cp(t) = D_scaler * (A1 * exp(-m1 * t) + A2 * exp(-m2 * t))`
    *   *Time unit for t*: minutes.

*   **Weinmann et al. (1982)**
    *   `D_scaler`: Overall scaling factor (dimensionless). Default: 1.0.
    *   `A1`: Amplitude of first exponential (mM*min). Default: 3.99.
    *   `m1`: Decay rate of first exponential (min⁻¹). Default: 0.144.
    *   `A2`: Amplitude of second exponential (mM*min). Default: 4.78.
    *   `m2`: Decay rate of second exponential (min⁻¹). Default: 0.0111.
    *   *Formula*: `Cp(t) = D_scaler * (A1 * exp(-m1 * t) + A2 * exp(-m2 * t))`
    *   *Time unit for t*: minutes.

*   **Fast Bi-exponential (Generic)**
    *   `D_scaler`: Overall scaling factor (dimensionless). Default: 1.0.
    *   `A1`: Proportion/amplitude of first exponential. Default: 0.6.
    *   `m1`: Decay rate of first exponential (min⁻¹). Default: 3.0.
    *   `A2`: Proportion/amplitude of second exponential. Default: 0.4.
    *   `m2`: Decay rate of second exponential (min⁻¹). Default: 0.3.
    *   *Formula*: `Cp(t) = D_scaler * (A1 * exp(-m1 * t) + A2 * exp(-m2 * t))`
    *   *Time unit for t*: minutes.

**Parameter Adjustment Tips:**
*   `D_scaler` can be used to adjust for variations in injected contrast agent dose compared to what the population model assumes.
*   The `A` (amplitude) and `m` (decay rate) parameters control the shape of the AIF curve (peak height, wash-in/wash-out rates). Adjust these cautiously and ideally with reference to literature or known AIF characteristics for your imaging protocol.

### 1.3. Defining AIF via ROI

Using an image-derived AIF can be more patient-specific.

*   **Choosing the ROI Location:**
    *   Select a major artery that is clearly visible within the dynamic scan range and relatively free from motion artifacts. Common choices include the femoral artery (for pelvic/abdominal scans) or carotid artery (for neck/brain scans). For preclinical studies, the aorta or a large cardiac vessel might be used.
    *   The ROI should be placed in a region where the signal is bright and uniform, avoiding vessel walls or partial volume effects where possible.
*   **ROI Size:**
    *   The ROI should be large enough to provide a stable mean signal but small enough to fit within the artery and avoid surrounding tissue.
*   **Slice Selection:** Choose a slice where the artery is well-defined and has minimal through-plane motion.
*   **Required Parameters:** When defining AIF from an ROI, you **must** provide:
    *   `T10 Blood`: The pre-contrast T1 relaxation time of blood (in ms). This is crucial for accurate signal-to-concentration conversion.
    *   `r1 Relaxivity (blood)`: The r1 relaxivity of the contrast agent in blood (e.g., 0.0045 s⁻¹mM⁻¹).
    *   `Baseline Time Points (AIF)`: The number of initial time points from the ROI's signal curve to average for S0 (baseline signal) calculation.

### 1.4. Saving and Loading AIF ROI Definitions

To ensure reproducibility, you can save and load the definition of an AIF ROI.

*   **Saving (`Save AIF ROI`):**
    *   This saves a JSON file containing the ROI's position (x, y coordinates of the top-left corner), size (width, height), the slice index (z), and the filename of the DCE series from which it was derived (for reference).
    *   **Example JSON (`my_aif_roi.json`):**
      ```json
      {
          "slice_index": 10,
          "pos_x": 120,
          "pos_y": 150,
          "size_w": 5,
          "size_h": 5,
          "image_ref_name": "dce_series_patient01.nii.gz"
      }
      ```
*   **Loading (`Load AIF ROI`):**
    *   This loads the ROI definition from the JSON file. The ROI will be re-applied if the currently loaded DCE image matches the `image_ref_name` or if you choose to apply it to a different compatible image.
    *   The AIF curve is then re-extracted using the saved ROI and the currently set AIF parameters (T10 blood, r1 blood, etc.).

## 2. Parallel Processing

Voxel-wise pharmacokinetic model fitting can be computationally intensive. The tool utilizes Python's `multiprocessing` module to distribute calculations across multiple CPU cores.

*   **How it Works:** The image volume (or the masked region) is divided into smaller chunks or individual voxels, and each available process works on fitting the model to the data from these chunks/voxels independently.
*   **Controlling Core Usage (GUI):** The "Number of Processes" dropdown in the GUI allows you to select how many CPU cores to use.
    *   Choosing a higher number (up to your system's core count) generally leads to faster processing.
    *   However, using all cores might make the system less responsive for other tasks.
*   **CLI (`--num-processes`):** The `batch_processor.py` script accepts the `--num-processes` argument to specify core usage.
*   **Performance Note:**
    *   Models like the Standard Tofts and Patlak are relatively fast to fit.
    *   The Extended Tofts model is moderately more complex.
    *   The **2CXM (Two-Compartment Exchange Model)** is significantly more computationally demanding due to the need to solve a system of ordinary differential equations (ODEs) for each voxel at each iteration of the fitting process. Parallel processing is highly recommended for 2CXM.
    *   There's a small overhead associated with setting up parallel processes. For very small datasets or extremely simple models, serial processing (1 process) might occasionally be faster, but this is rare for typical DCE-MRI analyses.

## 3. Troubleshooting

Here are some common issues and potential solutions:

*   **Issue: File Not Found errors.**
    *   **Solution:** Double-check the file paths provided for DCE, T1 map, mask, or AIF files. Ensure the files exist at the specified locations and that you have read permissions.

*   **Issue: Dimension Mismatch errors (e.g., "DCE and T1 map dimensions do not match").**
    *   **Solution:** The spatial dimensions (width, height, number of slices) of the DCE series, T1 map, and mask must be identical. Verify this using a NIfTI viewer or header information.

*   **Issue: AIF plot looks incorrect or has unexpected values.**
    *   **Solution (AIF from file):** Check the AIF file for correct formatting (two columns, numeric data, appropriate delimiter). Ensure time and concentration units are correct.
    *   **Solution (AIF from ROI):** Ensure `T10 Blood`, `r1 Relaxivity (blood)`, and `Baseline Time Points (AIF)` are set correctly. The ROI might be placed incorrectly (e.g., including non-arterial tissue or affected by artifacts). Try redefining the ROI in a cleaner arterial segment.
    *   **Solution (Population AIF):** Check if the default parameters are appropriate for your study. Time units for population models are assumed to be in minutes.

*   **Issue: Pharmacokinetic model fitting fails or produces all NaN maps.**
    *   **Solution:**
        *   **Check AIF:** A poor quality or incorrect AIF is a common cause.
        *   **Check Tissue Data:** Ensure the loaded DCE data is valid and shows contrast enhancement.
        *   **Check Parameters:** Verify that `r1 Relaxivity (tissue)`, `TR`, and `Baseline Time Points (Tissue)` are correctly set.
        *   **Model Choice:** The chosen model might not be appropriate for the data.
        *   **Mask Issues:** If using a mask, ensure it correctly delineates the tissue of interest. An overly restrictive or misaligned mask can lead to no valid voxels for fitting.
        *   **Insufficient Baseline:** If `Baseline Time Points (Tissue)` is too small or includes points after contrast arrival, S0 calculation will be incorrect, leading to poor concentration conversion and fitting.

*   **Issue: Application is slow or unresponsive during fitting.**
    *   **Solution:** This is expected for large datasets or complex models like 2CXM.
        *   Utilize parallel processing by selecting an appropriate number of cores.
        *   If using a mask, ensure it's as tight as possible to the region of interest to reduce the number of voxels processed.
        *   Consider processing smaller regions or a subset of slices for initial testing if feasible.

*   **Issue: "Log of non-positive value" or similar math errors during concentration conversion.**
    *   **Solution:** This can happen if the signal intensity S(t) is less than or equal to the baseline signal S0, or if S0 is zero or negative. The tool includes safeguards (clipping S(t)/S0 to a small positive epsilon), but extreme data issues might still cause problems.
        *   Check the number of `Baseline Time Points (Tissue)` to ensure S0 is calculated correctly from pre-contrast images.
        *   Inspect the raw signal data for anomalies.

*   **Issue (CLI): `batch_processor.py` exits with an error or gives unexpected results.**
    *   **Solution:** Carefully check all command-line arguments against the `--help` output. Ensure file paths are correct and AIF parameters are properly specified (e.g., using `--aif-param name value` for population models). Examine any error messages printed to the console.
