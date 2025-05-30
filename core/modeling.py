import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, solve_ivp
import multiprocessing
import os
from functools import partial # Not strictly used in the final version, but good for potential future use

"""
This module implements various pharmacokinetic (PK) models used for analyzing
Dynamic Contrast-Enhanced (DCE) MRI data. It provides functions for:

1.  **Model Definitions**:
    *   Standard Tofts Model (convolution-based)
    *   Extended Tofts Model (convolution-based)
    *   Patlak Model (linear regression after transformation)
    *   Two-Compartment Exchange Model (2CXM) (solved via ODE integration)

2.  **Single-Voxel Fitting**:
    *   Functions to fit each model to a single voxel's tissue concentration
        time-course (Ct_tissue) given an Arterial Input Function (AIF, Cp_aif).
    *   These use `scipy.optimize.curve_fit` for non-linear least squares
        optimization.

3.  **Voxel-wise Fitting (Parallelized)**:
    *   Functions to apply model fitting to entire 3D volumes (4D Ct_data)
        on a voxel-by-voxel basis.
    *   Utilizes `multiprocessing` for parallel execution to speed up the
        computationally intensive fitting process.
    *   Includes a generic worker function `_fit_voxel_worker` and a base
        dispatcher `_base_fit_voxelwise`.

The module relies on interpolated AIF data for model calculations to handle
potential mismatches in time points between the AIF and tissue curves.
"""

# --- Helper for 2CXM ODE System ---
def _ode_system_2cxm(t: float, y: list[float], Fp: float, PS: float, vp: float, ve: float, Cp_aif_interp_func: callable) -> list[float]:
    """
    Defines the system of ordinary differential equations for the 2-Compartment Exchange Model (2CXM).

    This system describes the rate of change of contrast agent concentration in the
    plasma compartment (C_p_tis) and the extravascular extracellular space (EES)
    compartment (C_e_tis) within a tissue voxel.

    Args:
        t (float): Current time point for ODE solver.
        y (list[float]): List containing current concentrations [C_p_tis, C_e_tis].
        Fp (float): Plasma flow (ml/100g/min or similar units).
        PS (float): Permeability-surface area product (ml/100g/min or similar units).
        vp (float): Fractional volume of plasma compartment (unitless, 0 to 1).
        ve (float): Fractional volume of EES compartment (unitless, 0 to 1).
        Cp_aif_interp_func (callable): An interpolation function that returns the
                                       AIF concentration Cp_aif(t) at any time t.

    Returns:
        list[float]: List of derivatives [dC_p_tis/dt, dC_e_tis/dt].
    """
    C_p_tis, C_e_tis = y
    Cp_aif_val = Cp_aif_interp_func(t) # Get AIF concentration at current time t

    # Ensure fractional volumes are positive to avoid division by zero or instability
    vp_eff = vp if vp > 1e-6 else 1e-6 # Effective vp, bounded to a small positive number
    ve_eff = ve if ve > 1e-6 else 1e-6 # Effective ve, bounded

    # Differential equation for plasma compartment concentration in tissue
    dC_p_tis_dt = (Fp / vp_eff) * (Cp_aif_val - C_p_tis) - (PS / vp_eff) * (C_p_tis - C_e_tis)
    # Differential equation for EES compartment concentration in tissue
    dC_e_tis_dt = (PS / ve_eff) * (C_p_tis - C_e_tis)

    return [dC_p_tis_dt, dC_e_tis_dt]

# --- Model Definitions (Convolution-based and Patlak) ---
def _convolve_Cp_with_exp(t: np.ndarray, Ktrans: float, ve: float, Cp_t_interp_func: callable) -> np.ndarray:
    """
    Helper function to convolve an interpolated AIF (Cp_t) with an exponential decay kernel.
    This is a core component of the Tofts models.

    The convolution integral is: Ktrans * integral[Cp(tau) * exp(-(Ktrans/ve)*(t-tau))] dtau
    from 0 to t.

    Args:
        t (np.ndarray): Array of time points for which the convolution is calculated.
                        Must be sorted and have at least 2 points for dt calculation.
        Ktrans (float): Transfer constant (e.g., min^-1).
        ve (float): Fractional volume of extravascular extracellular space (EES) (unitless).
        Cp_t_interp_func (callable): Interpolation function for the AIF, Cp_aif(t).

    Returns:
        np.ndarray: The result of the convolution, scaled by Ktrans. Represents the
                    contribution from the EES to the total tissue concentration.

    Raises:
        ValueError: If time points `t` are not sorted or have less than 2 points.
    """
    if len(t) < 2:
        # Not enough points for convolution or dt calculation
        return np.zeros_like(t)

    dt = t[1] - t[0] # Assume uniform time steps for simplicity in np.convolve
    if dt <= 0:
        raise ValueError("Time points 't' for convolution must be sorted and strictly increasing.")

    # Add a small epsilon to ve to prevent division by zero if ve is very small or zero.
    # This makes k_exp very large, leading to rapid decay, effectively zeroing out the term.
    k_exp = Ktrans / (ve + 1e-9)

    # Exponential decay kernel: exp(-k_exp * t)
    exp_decay_kernel = np.exp(-k_exp * t)

    # Get AIF values at the specified time points using the interpolation function
    Cp_values_at_t = Cp_t_interp_func(t)

    # Perform convolution
    # 'full' mode computes the convolution at all points of overlap.
    # We then take the first 'len(t)' points, corresponding to the causal part.
    # The result is scaled by dt to approximate the integral.
    convolution_result = np.convolve(Cp_values_at_t, exp_decay_kernel, mode='full')[:len(t)] * dt

    return Ktrans * convolution_result

def standard_tofts_model_conv(t: np.ndarray, Ktrans: float, ve: float, Cp_t_interp_func: callable) -> np.ndarray:
    """
    Standard Tofts Model calculated via convolution.
    Ct(t) = Ktrans * integral[Cp(tau) * exp(-(Ktrans/ve)*(t-tau))] dtau

    Args:
        t (np.ndarray): Time points.
        Ktrans (float): Transfer constant (e.g., min^-1).
        ve (float): Fractional EES volume (unitless).
        Cp_t_interp_func (callable): Interpolated AIF function Cp_aif(t).

    Returns:
        np.ndarray: Tissue concentration time-course predicted by the Standard Tofts model.
                    Returns np.inf array if parameters are negative (for fitting).
    """
    # Parameter constraints for fitting stability
    if Ktrans < 0 or ve < 0:
        return np.full_like(t, np.inf) # Return array of infinities if params are invalid
    return _convolve_Cp_with_exp(t, Ktrans, ve, Cp_t_interp_func)

def extended_tofts_model_conv(t: np.ndarray, Ktrans: float, ve: float, vp: float, Cp_t_interp_func: callable) -> np.ndarray:
    """
    Extended Tofts Model calculated via convolution.
    Ct(t) = vp * Cp(t) + Ktrans * integral[Cp(tau) * exp(-(Ktrans/ve)*(t-tau))] dtau

    Args:
        t (np.ndarray): Time points.
        Ktrans (float): Transfer constant (e.g., min^-1).
        ve (float): Fractional EES volume (unitless).
        vp (float): Fractional plasma volume (unitless).
        Cp_t_interp_func (callable): Interpolated AIF function Cp_aif(t).

    Returns:
        np.ndarray: Tissue concentration time-course predicted by the Extended Tofts model.
                    Returns np.inf array if parameters are negative (for fitting).
    """
    # Parameter constraints for fitting stability
    if Ktrans < 0 or ve < 0 or vp < 0:
        return np.full_like(t, np.inf)

    vp_component = vp * Cp_t_interp_func(t)
    tofts_component = _convolve_Cp_with_exp(t, Ktrans, ve, Cp_t_interp_func)
    return vp_component + tofts_component

def patlak_model(t_points: np.ndarray, Ktrans: float, vp: float,
                 Cp_t_interp_func: callable, integral_Cp_dt_interp_func: callable) -> np.ndarray:
    """
    Patlak Model (linearized form for fitting).
    Ct(t) / Cp(t) = Ktrans * (Integral[Cp(tau)dtau from 0 to t] / Cp(t)) + vp
    This function returns Ct(t) directly for fitting against measured Ct(t):
    Ct(t) = Ktrans * Integral[Cp(tau)dtau from 0 to t] + vp * Cp(t)

    Args:
        t_points (np.ndarray): Time points.
        Ktrans (float): Transfer constant (slope in Patlak plot, related to PS).
        vp (float): Fractional plasma volume (intercept in Patlak plot).
        Cp_t_interp_func (callable): Interpolated AIF function Cp_aif(t).
        integral_Cp_dt_interp_func (callable): Interpolated function for the
                                               cumulative integral of AIF, Int[Cp_aif(tau)dtau].

    Returns:
        np.ndarray: Tissue concentration time-course predicted by the Patlak model.
                    Returns np.inf array if parameters are negative (for fitting).
    """
    # Parameter constraints for fitting stability
    if Ktrans < 0 or vp < 0:
        return np.full_like(t_points, np.inf)

    Cp_values = Cp_t_interp_func(t_points)
    integral_Cp_values = integral_Cp_dt_interp_func(t_points)
    return Ktrans * integral_Cp_values + vp * Cp_values

def solve_2cxm_ode_model(t_eval_points: np.ndarray, Fp: float, PS: float, vp: float, ve: float,
                         Cp_aif_interp_func: callable, t_span_max: float = None) -> np.ndarray:
    """
    Solves the 2-Compartment Exchange Model (2CXM) using an ODE solver.

    The model output is Ct(t) = vp * C_p_tis(t) + ve * C_e_tis(t), where C_p_tis and
    C_e_tis are the concentrations in the tissue plasma and EES compartments,
    respectively, obtained by solving the ODE system defined in `_ode_system_2cxm`.

    Args:
        t_eval_points (np.ndarray): Time points at which to evaluate the model solution.
        Fp (float): Plasma flow.
        PS (float): Permeability-surface area product.
        vp (float): Fractional plasma volume. Must be > 0.
        ve (float): Fractional EES volume. Must be > 0.
        Cp_aif_interp_func (callable): Interpolated AIF function Cp_aif(t).
        t_span_max (float, optional): Maximum time for the ODE solver integration span.
                                     If None, uses the last point in `t_eval_points`.
                                     This can be useful if AIF data extends beyond
                                     tissue time points. Defaults to None.

    Returns:
        np.ndarray: Tissue concentration time-course Ct(t) predicted by the 2CXM.
                    Returns np.inf array if parameters are invalid or solver fails.
    """
    # Parameter constraints, especially for volumes which are denominators in ODE
    if Fp < 0 or PS < 0 or vp <= 1e-7 or ve <= 1e-7: # vp, ve must be strictly positive
        return np.full_like(t_eval_points, np.inf)

    y0 = [0, 0] # Initial conditions: C_p_tis(0) = 0, C_e_tis(0) = 0

    # Define the time span for the ODE solver
    t_span_solve = [t_eval_points[0], t_eval_points[-1]]
    if t_span_max is not None:
         t_span_solve = [t_eval_points[0], max(t_span_max, t_eval_points[-1])] # Ensure span covers eval points

    try:
        sol = solve_ivp(
            fun=_ode_system_2cxm,
            t_span=t_span_solve,
            y0=y0,
            t_eval=t_eval_points, # Points where solution is desired
            args=(Fp, PS, vp, ve, Cp_aif_interp_func), # Arguments for _ode_system_2cxm
            method='RK45', # Standard explicit Runge-Kutta method
            dense_output=False # Not needing continuous solution here
        )

        if sol.status != 0: # Check if solver was successful
            # sol.message might contain more info, could be logged
            return np.full_like(t_eval_points, np.inf)

        C_p_tis_solved = sol.y[0, :] # Solution for C_p_tis
        C_e_tis_solved = sol.y[1, :] # Solution for C_e_tis

        # Total tissue concentration
        Ct_model = vp * C_p_tis_solved + ve * C_e_tis_solved
        return Ct_model
    except Exception: # Catch any other errors during ODE solution
        return np.full_like(t_eval_points, np.inf)


# --- Single-Voxel Fitting Functions ---
def fit_standard_tofts(t_tissue: np.ndarray, Ct_tissue: np.ndarray, Cp_interp_func: callable,
                       initial_params: tuple = (0.1, 0.2),
                       bounds_params: tuple = ([0, 0], [1.0, 1.0])) -> tuple[tuple, np.ndarray]:
    """
    Fits the Standard Tofts model to a single voxel's tissue concentration data.

    Args:
        t_tissue (np.ndarray): Time points for the tissue curve.
        Ct_tissue (np.ndarray): Tissue concentration values at t_tissue.
        Cp_interp_func (callable): Interpolated AIF function Cp_aif(t).
        initial_params (tuple, optional): Initial guess for [Ktrans, ve].
                                          Defaults to (0.1, 0.2).
        bounds_params (tuple, optional): Bounds for [Ktrans, ve] as ([min_vals], [max_vals]).
                                         Defaults to ([0, 0], [1.0, 1.0]).

    Returns:
        tuple: (parameters, fitted_curve)
            - parameters (tuple): Fitted (Ktrans, ve) or (np.nan, np.nan) on failure.
            - fitted_curve (np.ndarray): Model prediction with fitted parameters,
                                         or NaN array on failure/insufficient data.
    """
    # Basic check for valid input data for fitting
    if not (isinstance(t_tissue, np.ndarray) and isinstance(Ct_tissue, np.ndarray) and \
            len(t_tissue) > 1 and len(Ct_tissue) == len(t_tissue)):
        return (np.nan, np.nan), np.full_like(Ct_tissue if isinstance(Ct_tissue, np.ndarray) else t_tissue, np.nan)

    try:
        # Define the objective function for curve_fit (Standard Tofts model)
        objective_func = lambda t_obj, Ktrans, ve: standard_tofts_model_conv(t_obj, Ktrans, ve, Cp_interp_func)

        popt, pcov = curve_fit(
            objective_func, t_tissue, Ct_tissue,
            p0=initial_params, bounds=bounds_params,
            method='trf', # Trust Region Reflective algorithm, good for bounds
            ftol=1e-4, xtol=1e-4, gtol=1e-4 # Tolerances for termination
        )
        # Generate the fitted curve using the optimized parameters
        fitted_curve = standard_tofts_model_conv(t_tissue, popt[0], popt[1], Cp_interp_func)
        return tuple(popt), fitted_curve
    except Exception: # Catch errors during fitting (e.g., RuntimeError from curve_fit)
        return (np.nan, np.nan), np.full_like(t_tissue, np.nan)

def fit_extended_tofts(t_tissue: np.ndarray, Ct_tissue: np.ndarray, Cp_interp_func: callable,
                       initial_params: tuple = (0.1, 0.2, 0.05),
                       bounds_params: tuple = ([0, 0, 0], [1.0, 1.0, 0.5])) -> tuple[tuple, np.ndarray]:
    """
    Fits the Extended Tofts model to a single voxel's tissue concentration data.

    Args:
        t_tissue (np.ndarray): Time points for the tissue curve.
        Ct_tissue (np.ndarray): Tissue concentration values at t_tissue.
        Cp_interp_func (callable): Interpolated AIF function Cp_aif(t).
        initial_params (tuple, optional): Initial guess for [Ktrans, ve, vp].
                                          Defaults to (0.1, 0.2, 0.05).
        bounds_params (tuple, optional): Bounds for [Ktrans, ve, vp].
                                         Defaults to ([0, 0, 0], [1.0, 1.0, 0.5]).

    Returns:
        tuple: (parameters, fitted_curve)
            - parameters (tuple): Fitted (Ktrans, ve, vp) or (np.nan, ..., np.nan) on failure.
            - fitted_curve (np.ndarray): Model prediction with fitted parameters.
    """
    if not (isinstance(t_tissue, np.ndarray) and isinstance(Ct_tissue, np.ndarray) and \
            len(t_tissue) > 1 and len(Ct_tissue) == len(t_tissue)):
        return (np.nan, np.nan, np.nan), np.full_like(Ct_tissue if isinstance(Ct_tissue, np.ndarray) else t_tissue, np.nan)

    try:
        # Objective function for curve_fit (Extended Tofts model)
        objective_func = lambda t_obj, Ktrans, ve, vp: extended_tofts_model_conv(t_obj, Ktrans, ve, vp, Cp_interp_func)

        popt, pcov = curve_fit(
            objective_func, t_tissue, Ct_tissue,
            p0=initial_params, bounds=bounds_params,
            method='trf', ftol=1e-4, xtol=1e-4, gtol=1e-4
        )
        fitted_curve = extended_tofts_model_conv(t_tissue, popt[0], popt[1], popt[2], Cp_interp_func)
        return tuple(popt), fitted_curve
    except Exception:
        return (np.nan, np.nan, np.nan), np.full_like(t_tissue, np.nan)

def fit_patlak_model(t_tissue: np.ndarray, Ct_tissue: np.ndarray,
                     Cp_interp_func: callable, integral_Cp_dt_interp_func: callable,
                     initial_params: tuple = (0.05, 0.05),
                     bounds_params: tuple = ([0, 0], [1.0, 0.5])) -> tuple[tuple, np.ndarray]:
    """
    Fits the Patlak model to a single voxel's tissue concentration data.

    Args:
        t_tissue (np.ndarray): Time points for the tissue curve.
        Ct_tissue (np.ndarray): Tissue concentration values at t_tissue.
        Cp_interp_func (callable): Interpolated AIF function Cp_aif(t).
        integral_Cp_dt_interp_func (callable): Interpolated cumulative integral of AIF.
        initial_params (tuple, optional): Initial guess for [Ktrans_patlak, vp_patlak].
                                          Defaults to (0.05, 0.05).
        bounds_params (tuple, optional): Bounds for [Ktrans_patlak, vp_patlak].
                                         Defaults to ([0, 0], [1.0, 0.5]).

    Returns:
        tuple: (parameters, fitted_curve)
            - parameters (tuple): Fitted (Ktrans_patlak, vp_patlak) or (np.nan, np.nan) on failure.
            - fitted_curve (np.ndarray): Model prediction with fitted parameters.
    """
    if not (isinstance(t_tissue, np.ndarray) and isinstance(Ct_tissue, np.ndarray) and \
            len(t_tissue) > 1 and len(Ct_tissue) == len(t_tissue)):
        return (np.nan, np.nan), np.full_like(Ct_tissue if isinstance(Ct_tissue, np.ndarray) else t_tissue, np.nan)

    # Objective function for curve_fit (Patlak model)
    def objective_func(t_obj, Ktrans_patlak, vp_patlak):
        return patlak_model(t_obj, Ktrans_patlak, vp_patlak, Cp_interp_func, integral_Cp_dt_interp_func)

    try:
        popt, pcov = curve_fit(
            objective_func, t_tissue, Ct_tissue,
            p0=initial_params, bounds=bounds_params,
            method='trf', xtol=1e-4, ftol=1e-4, gtol=1e-4
        )
        fitted_curve = patlak_model(t_tissue, popt[0], popt[1], Cp_interp_func, integral_Cp_dt_interp_func)
        return tuple(popt), fitted_curve
    except Exception:
        return (np.nan, np.nan), np.full_like(t_tissue, np.nan)

def fit_2cxm_model(t_tissue: np.ndarray, Ct_tissue: np.ndarray, Cp_aif_interp_func: callable, t_aif_max: float,
                   initial_params: tuple = (0.1, 0.05, 0.05, 0.1),
                   bounds_params: tuple = ([0, 0, 1e-3, 1e-3], [2.0, 1.0, 0.5, 0.7])) -> tuple[tuple, np.ndarray]:
    """
    Fits the 2-Compartment Exchange Model (2CXM) to a single voxel's tissue concentration data.

    Args:
        t_tissue (np.ndarray): Time points for the tissue curve.
        Ct_tissue (np.ndarray): Tissue concentration values at t_tissue.
        Cp_aif_interp_func (callable): Interpolated AIF function Cp_aif(t).
        t_aif_max (float): Maximum time of the AIF data, used to set ODE solver span.
        initial_params (tuple, optional): Initial guess for [Fp, PS, vp, ve].
                                          Defaults to (0.1, 0.05, 0.05, 0.1).
                                          Note: vp and ve bounds should be strictly > 0.
        bounds_params (tuple, optional): Bounds for [Fp, PS, vp, ve].
                                         Defaults to ([0, 0, 1e-3, 1e-3], [2.0, 1.0, 0.5, 0.7]).

    Returns:
        tuple: (parameters, fitted_curve)
            - parameters (tuple): Fitted (Fp, PS, vp, ve) or (np.nan, ..., np.nan) on failure.
            - fitted_curve (np.ndarray): Model prediction with fitted parameters.
    """
    if not (isinstance(t_tissue, np.ndarray) and isinstance(Ct_tissue, np.ndarray) and \
            len(t_tissue) > 1 and len(Ct_tissue) == len(t_tissue)):
        return (np.nan, np.nan, np.nan, np.nan), np.full_like(Ct_tissue if isinstance(Ct_tissue, np.ndarray) else t_tissue, np.nan)

    # Objective function for curve_fit (2CXM model)
    def objective_func(t_obj, Fp, PS, vp, ve):
        return solve_2cxm_ode_model(t_obj, Fp, PS, vp, ve, Cp_aif_interp_func, t_span_max=t_aif_max)

    try:
        popt, pcov = curve_fit(
            objective_func, t_tissue, Ct_tissue,
            p0=initial_params, bounds=bounds_params,
            method='trf',
            ftol=1e-3, xtol=1e-3, gtol=1e-3 # Looser tolerances for potentially complex ODE fits
        )
        fitted_curve = solve_2cxm_ode_model(t_tissue, popt[0], popt[1], popt[2], popt[3], Cp_aif_interp_func, t_span_max=t_aif_max)
        return tuple(popt), fitted_curve
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan), np.full_like(t_tissue, np.nan)


# --- Multiprocessing Worker Function (Top-Level) ---
def _fit_voxel_worker(args_tuple: tuple) -> tuple:
    """
    Worker function for fitting a pharmacokinetic model to a single voxel's data.
    Designed to be called by `multiprocessing.Pool.map`.

    Performs data validation, AIF interpolation setup, model selection, fitting,
    and result packaging.

    Args:
        args_tuple (tuple): A tuple containing all necessary arguments:
            - voxel_idx_xyz (tuple): (x, y, z) coordinates of the voxel.
            - Ct_voxel (np.ndarray): 1D array of tissue concentration for the voxel.
            - t_tissue (np.ndarray): 1D array of time points for tissue data.
            - t_aif (np.ndarray): 1D array of time points for AIF data.
            - Cp_aif (np.ndarray): 1D array of AIF concentrations.
            - model_name (str): Name of the model to fit (e.g., "Standard Tofts").
            - initial_params_for_model (tuple): Initial parameters for the selected model.
            - bounds_params_for_model (tuple): Bounds for parameters for the selected model.

    Returns:
        tuple: (voxel_idx_xyz, model_name, result_dict)
            - voxel_idx_xyz (tuple): (x,y,z) coordinates of the voxel.
            - model_name (str): Name of the model processed.
            - result_dict (dict): Dictionary containing fitted parameters (e.g.,
                                  {"Ktrans": 0.1, "ve": 0.2}) or an error message
                                  (e.g., {"error": "Fit failed"}).
    """
    voxel_idx_xyz, Ct_voxel, t_tissue, t_aif, Cp_aif, model_name, \
    initial_params_for_model, bounds_params_for_model = args_tuple

    # --- Voxel Data Sanity Checks ---
    # Check for all NaNs, all zeros, or too few valid data points
    if np.all(np.isnan(Ct_voxel)) or np.all(Ct_voxel == 0):
        return voxel_idx_xyz, model_name, {"error": "Skipped (all NaN or all zero data)"}

    valid_indices = ~np.isnan(Ct_voxel)
    Ct_voxel_clean = Ct_voxel[valid_indices]
    t_tissue_clean = t_tissue[valid_indices]

    # Check if enough data points remain after cleaning for the number of parameters
    # Generally, need more data points than parameters; curve_fit might need at least P+1 or more.
    # A common heuristic is at least 2*P points.
    min_points_needed = len(initial_params_for_model) * 2 
    if len(Ct_voxel_clean) < min_points_needed:
        return voxel_idx_xyz, model_name, {"error": f"Skipped (insufficient valid data: got {len(Ct_voxel_clean)}, need at least {min_points_needed})"}
    if len(Ct_voxel_clean) < len(initial_params_for_model): # Stricter check for curve_fit
         return voxel_idx_xyz, model_name, {"error": f"Skipped (valid data points {len(Ct_voxel_clean)} < n_params {len(initial_params_for_model)})"}


    try:
        # --- AIF Interpolation Setup ---
        # Create an interpolation function for the AIF (Cp_aif)
        # This allows the model functions to evaluate Cp_aif at any time point t.
        # 'linear' kind is common, 'bounds_error=False' prevents errors for t outside t_aif range,
        # 'fill_value=0.0' (or "extrapolate") handles values outside original AIF time.
        Cp_interp_func = interp1d(t_aif, Cp_aif, kind='linear', bounds_error=False, fill_value=0.0)

        params_tuple = None
        param_names = [] # To store names of parameters for the specific model

        # --- Model Selection and Fitting ---
        if model_name == "Standard Tofts":
            params_tuple, _ = fit_standard_tofts(t_tissue_clean, Ct_voxel_clean, Cp_interp_func,
                                                 initial_params_for_model, bounds_params_for_model)
            param_names = ["Ktrans", "ve"]
        elif model_name == "Extended Tofts":
            params_tuple, _ = fit_extended_tofts(t_tissue_clean, Ct_voxel_clean, Cp_interp_func,
                                                 initial_params_for_model, bounds_params_for_model)
            param_names = ["Ktrans", "ve", "vp"]
        elif model_name == "Patlak":
            # Patlak model requires the integral of Cp_aif. Pre-calculate and interpolate it.
            integral_Cp_dt_aif = cumtrapz(Cp_aif, t_aif, initial=0)
            integral_Cp_dt_interp_func = interp1d(t_aif, integral_Cp_dt_aif, kind='linear',
                                                  bounds_error=False, fill_value=0.0)
            params_tuple, _ = fit_patlak_model(t_tissue_clean, Ct_voxel_clean, Cp_interp_func,
                                               integral_Cp_dt_interp_func,
                                               initial_params_for_model, bounds_params_for_model)
            param_names = ["Ktrans_patlak", "vp_patlak"]
        elif model_name == "2CXM":
            # Determine t_aif_max for ODE solver span; use last AIF time or last tissue time.
            t_aif_max = t_aif[-1] if len(t_aif) > 0 else t_tissue_clean[-1]
            if len(t_aif) == 0 and len(t_tissue_clean) == 0: # Should be caught by earlier checks
                 return voxel_idx_xyz, model_name, {"error": "Cannot determine t_aif_max for 2CXM (no time data)"}

            params_tuple, _ = fit_2cxm_model(t_tissue_clean, Ct_voxel_clean, Cp_interp_func, t_aif_max,
                                             initial_params_for_model, bounds_params_for_model)
            param_names = ["Fp_2cxm", "PS_2cxm", "vp_2cxm", "ve_2cxm"]
        else:
            return voxel_idx_xyz, model_name, {"error": f"Unknown model: {model_name}"}
        
        # --- Result Packaging ---
        if params_tuple is None or np.any(np.isnan(params_tuple)):
            # Fit failed or returned NaNs
            return voxel_idx_xyz, model_name, {"error": "Fit failed (returned None or NaN parameters)"}

        # Successfully fitted, return parameters as a dictionary
        return voxel_idx_xyz, model_name, dict(zip(param_names, params_tuple))

    except Exception as e:
        # Catch any unexpected errors during the worker's execution
        return voxel_idx_xyz, model_name, {"error": f"Unexpected error in worker: {str(e)}"}


# --- Voxel-wise Fitting Functions (Parallelized) ---
def _base_fit_voxelwise(
    Ct_data: np.ndarray, t_tissue: np.ndarray, t_aif: np.ndarray, Cp_aif: np.ndarray,
    model_name: str, param_names_map: dict,
    initial_params: tuple, bounds_params: tuple,
    mask: np.ndarray = None, num_processes: int = None
) -> dict[str, np.ndarray]:
    """
    Base function for voxel-wise fitting of a pharmacokinetic model using multiprocessing.

    This function prepares tasks for each voxel (optionally within a mask),
    distributes these tasks to a pool of worker processes (_fit_voxel_worker),
    and aggregates the results into parameter maps.

    Args:
        Ct_data (np.ndarray): 4D array (x,y,z,time) of tissue concentration data.
        t_tissue (np.ndarray): 1D array of time points for tissue data.
        t_aif (np.ndarray): 1D array of time points for AIF data.
        Cp_aif (np.ndarray): 1D array of AIF concentrations.
        model_name (str): Name of the model to fit (e.g., "Standard Tofts").
        param_names_map (dict): Dictionary mapping internal parameter names (from fitting functions)
                                to output map names (e.g., {"Ktrans": "Ktrans_map_name"}).
        initial_params (tuple): Initial parameter guesses for the model.
        bounds_params (tuple): Parameter bounds for the model.
        mask (np.ndarray, optional): 3D boolean array. If provided, fitting is
                                     only performed for voxels where mask is True.
                                     Defaults to None (fit all voxels).
        num_processes (int, optional): Number of CPU processes to use.
                                       Defaults to None (os.cpu_count()).

    Returns:
        dict[str, np.ndarray]: A dictionary where keys are parameter map names
                               (from `param_names_map.values()`) and values are
                               3D NumPy arrays of the fitted parameters.
                               Voxels with errors or outside the mask will have NaN.
    """
    # --- Input Validations ---
    if not isinstance(Ct_data, np.ndarray) or Ct_data.ndim != 4:
        raise ValueError("Ct_data must be a 4D NumPy array (x, y, z, time).")
    if not isinstance(t_tissue, np.ndarray) or t_tissue.ndim != 1 or Ct_data.shape[3] != len(t_tissue):
        raise ValueError("t_tissue must be a 1D array matching the time dimension of Ct_data.")
    if not isinstance(t_aif, np.ndarray) or not isinstance(Cp_aif, np.ndarray) or \
       t_aif.ndim != 1 or Cp_aif.ndim != 1 or len(t_aif) != len(Cp_aif):
        raise ValueError("t_aif and Cp_aif must be 1D NumPy arrays of the same length.")
    if mask is not None and (not isinstance(mask, np.ndarray) or mask.ndim != 3 or mask.shape != Ct_data.shape[:3]):
        raise ValueError("Mask must be a 3D NumPy array matching spatial dimensions of Ct_data.")

    # Determine number of processes to use
    num_proc_to_use = num_processes if num_processes and num_processes > 0 else os.cpu_count()
    if num_proc_to_use is None: num_proc_to_use = 1 # Fallback if os.cpu_count() is None

    spatial_dims = Ct_data.shape[:3]
    # Initialize result maps with NaNs
    result_maps = {map_key_name: np.full(spatial_dims, np.nan, dtype=np.float32)
                   for map_key_name in param_names_map.values()}

    # --- Prepare Tasks for Multiprocessing ---
    tasks_args_list = []
    for x in range(spatial_dims[0]):
        for y in range(spatial_dims[1]):
            for z in range(spatial_dims[2]):
                if mask is not None and not mask[x, y, z]:
                    continue # Skip voxel if outside mask
                Ct_voxel = Ct_data[x, y, z, :]
                # Arguments for _fit_voxel_worker
                tasks_args_list.append(
                    ((x,y,z), Ct_voxel, t_tissue, t_aif, Cp_aif, model_name, initial_params, bounds_params)
                )

    if not tasks_args_list:
        # Log this or handle as appropriate for the application
        print(f"Warning: No voxels to process for {model_name} fitting (e.g., mask is all False or data empty). Returning empty/NaN maps.")
        return result_maps

    print(f"Starting {model_name} fitting for {len(tasks_args_list)} voxels using up to {num_proc_to_use} processes...")

    results_list = []
    # --- Execute Fitting: Parallel or Serial ---
    if num_proc_to_use > 1 and len(tasks_args_list) > 1: # Use multiprocessing
        try:
            with multiprocessing.Pool(processes=num_proc_to_use) as pool:
                results_list = pool.map(_fit_voxel_worker, tasks_args_list)
        except Exception as e:
            print(f"Error during multiprocessing pool for {model_name}: {e}. Falling back to serial processing.")
            num_proc_to_use = 1 # Force serial execution if pool fails

    if not results_list or num_proc_to_use == 1: # Use serial processing (fallback or if num_proc_to_use is 1)
        if num_proc_to_use > 1 and not results_list : # This means pool failed, and we are in fallback
             print(f"Processing {model_name} serially due to previous multiprocessing error...")
        else: # Standard serial execution
             print(f"Processing {model_name} serially...")
        results_list = [_fit_voxel_worker(args) for args in tasks_args_list]

    # --- Aggregate Results ---
    for result_item in results_list:
        if result_item is None:
            # This might happen if a worker process itself crashes unexpectedly
            # Or if _fit_voxel_worker returns None (which it shouldn't based on current code)
            # Log this occurrence if possible
            continue

        voxel_idx_xyz, model_name_out, result_data_dict = result_item

        # Ensure the result is for the correct model (sanity check)
        if model_name_out == model_name:
            if "error" not in result_data_dict:
                # Successfully fitted parameters
                for p_name_internal, p_name_map_key in param_names_map.items():
                    if p_name_internal in result_data_dict:
                        result_maps[p_name_map_key][voxel_idx_xyz] = result_data_dict[p_name_internal]
                    else:
                        # This case should ideally not happen if param_names_map is correct
                        result_maps[p_name_map_key][voxel_idx_xyz] = np.nan
            else:
                # An error occurred for this voxel, already filled with NaN.
                # Optionally, log the error: print(f"Error fitting voxel {voxel_idx_xyz}: {result_data_dict['error']}")
                pass

    print(f"{model_name} voxel-wise fitting completed.")
    return result_maps

# --- Public Voxel-wise Fitting Functions ---

def fit_standard_tofts_voxelwise(Ct_data: np.ndarray, t_tissue: np.ndarray, t_aif: np.ndarray, Cp_aif: np.ndarray,
                                 mask: np.ndarray = None,
                                 initial_params: tuple = (0.1, 0.2),
                                 bounds_params: tuple = ([0.001, 0.001], [1.0, 1.0]),
                                 num_processes: int = None) -> dict[str, np.ndarray]:
    """
    Performs voxel-wise fitting of the Standard Tofts model.

    Args:
        Ct_data (np.ndarray): 4D array (x,y,z,time) of tissue concentration.
        t_tissue (np.ndarray): 1D array of time points for tissue data.
        t_aif (np.ndarray): 1D array of time points for AIF data.
        Cp_aif (np.ndarray): 1D array of AIF concentrations.
        mask (np.ndarray, optional): 3D boolean mask. Defaults to None.
        initial_params (tuple, optional): Initial guess for [Ktrans, ve].
        bounds_params (tuple, optional): Bounds for [Ktrans, ve].
        num_processes (int, optional): Number of CPU processes. Defaults to os.cpu_count().

    Returns:
        dict[str, np.ndarray]: Dictionary with keys "Ktrans" and "ve",
                               containing 3D maps of the fitted parameters.
    """
    param_names_map = {"Ktrans": "Ktrans", "ve": "ve"} # Maps internal names to output map names
    return _base_fit_voxelwise(Ct_data, t_tissue, t_aif, Cp_aif,
                               "Standard Tofts", param_names_map,
                               initial_params, bounds_params,
                               mask, num_processes)

def fit_extended_tofts_voxelwise(Ct_data: np.ndarray, t_tissue: np.ndarray, t_aif: np.ndarray, Cp_aif: np.ndarray,
                                 mask: np.ndarray = None,
                                 initial_params: tuple = (0.1, 0.2, 0.05),
                                 bounds_params: tuple = ([0.001, 0.001, 0.001], [1.0, 1.0, 0.5]),
                                 num_processes: int = None) -> dict[str, np.ndarray]:
    """
    Performs voxel-wise fitting of the Extended Tofts model.

    Args:
        Ct_data, t_tissue, t_aif, Cp_aif, mask, num_processes: (as in Standard Tofts)
        initial_params (tuple, optional): Initial guess for [Ktrans, ve, vp].
        bounds_params (tuple, optional): Bounds for [Ktrans, ve, vp].

    Returns:
        dict[str, np.ndarray]: Dictionary with keys "Ktrans", "ve", "vp",
                               containing 3D maps of the fitted parameters.
    """
    param_names_map = {"Ktrans": "Ktrans", "ve": "ve", "vp": "vp"}
    return _base_fit_voxelwise(Ct_data, t_tissue, t_aif, Cp_aif,
                               "Extended Tofts", param_names_map,
                               initial_params, bounds_params,
                               mask, num_processes)

def fit_patlak_model_voxelwise(Ct_data: np.ndarray, t_tissue: np.ndarray, t_aif: np.ndarray, Cp_aif: np.ndarray,
                               mask: np.ndarray = None,
                               initial_params: tuple = (0.05, 0.05),
                               bounds_params: tuple = ([0, 0], [1.0, 0.5]),
                               num_processes: int = None) -> dict[str, np.ndarray]:
    """
    Performs voxel-wise fitting of the Patlak model.

    Args:
        Ct_data, t_tissue, t_aif, Cp_aif, mask, num_processes: (as in Standard Tofts)
        initial_params (tuple, optional): Initial guess for [Ktrans_patlak, vp_patlak].
        bounds_params (tuple, optional): Bounds for [Ktrans_patlak, vp_patlak].

    Returns:
        dict[str, np.ndarray]: Dictionary with keys "Ktrans_patlak", "vp_patlak",
                               containing 3D maps of the fitted parameters.
    """
    param_names_map = {"Ktrans_patlak": "Ktrans_patlak", "vp_patlak": "vp_patlak"}
    return _base_fit_voxelwise(Ct_data, t_tissue, t_aif, Cp_aif,
                               "Patlak", param_names_map,
                               initial_params, bounds_params,
                               mask, num_processes)

def fit_2cxm_model_voxelwise(Ct_data: np.ndarray, t_tissue: np.ndarray, t_aif: np.ndarray, Cp_aif: np.ndarray,
                             mask: np.ndarray = None,
                             initial_params: tuple = (0.1, 0.05, 0.05, 0.1),
                             bounds_params: tuple = ([0, 0, 1e-3, 1e-3], [2.0, 1.0, 0.5, 0.7]),
                             num_processes: int = None) -> dict[str, np.ndarray]:
    """
    Performs voxel-wise fitting of the 2-Compartment Exchange Model (2CXM).

    Args:
        Ct_data, t_tissue, t_aif, Cp_aif, mask, num_processes: (as in Standard Tofts)
        initial_params (tuple, optional): Initial guess for [Fp, PS, vp, ve].
                                          vp and ve must be > 0.
        bounds_params (tuple, optional): Bounds for [Fp, PS, vp, ve].
                                         Ensure lower bounds for vp, ve are > 0.

    Returns:
        dict[str, np.ndarray]: Dictionary with keys "Fp_2cxm", "PS_2cxm", "vp_2cxm", "ve_2cxm",
                               containing 3D maps of the fitted parameters.
    """
    # Map internal parameter names from fit_2cxm_model to output map keys
    param_names_map = {"Fp_2cxm": "Fp_2cxm", "PS_2cxm": "PS_2cxm", "vp_2cxm": "vp_2cxm", "ve_2cxm": "ve_2cxm"}
    return _base_fit_voxelwise(Ct_data, t_tissue, t_aif, Cp_aif,
                               "2CXM", param_names_map,
                               initial_params, bounds_params,
                               mask, num_processes)
