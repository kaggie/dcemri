"""
Module defining the main window of the DCE-MRI analysis application.

This module contains the `MainWindow` class, which orchestrates the user interface,
data processing workflows, and visualization for analyzing DCE-MRI data. It integrates
various components from the `core` and `utils` packages to provide a cohesive
application experience.
"""
import sys
import traceback
import numpy as np
import nibabel as nib 
from scipy.interpolate import interp1d 
from scipy.integrate import cumtrapz 
import os 
import functools 

import pyqtgraph as pg 
import pyqtgraph.exporters as pg_exporters # Import for plot saving
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QTextEdit,
    QGroupBox,
    QFormLayout,
    QComboBox,
    QRadioButton,
    QSizePolicy, 
    QSlider, 
    QSpinBox, 
)
from PyQt5.QtCore import Qt, QPointF 

from ..core import io
from ..core import conversion
from ..core import aif 
from ..core import modeling 
from ..core import reporting

class MainWindow(QMainWindow):
    """
    Main application window for DCE-MRI analysis.

    This class sets up the UI, manages data loading, AIF definition,
    pharmacokinetic model fitting, results visualization, and ROI statistics.
    It uses PyQt5 for the GUI and pyqtgraph for plotting and image display.

    Attributes:
        dce_data (np.ndarray | None): Loaded 4D DCE image data.
        t10_data (np.ndarray | None): Loaded 3D T10 map data.
        mask_data (np.ndarray | None): Loaded 3D mask data (boolean).
        Ct_data (np.ndarray | None): Tissue concentration curves (4D).
        dce_shape_for_validation (tuple | None): Shape of the loaded DCE data, used for validating
                                             other inputs like T1 maps and masks.
        dce_time_vector (np.ndarray | None): Time vector for DCE series based on TR.
        tr_float (float | None): Repetition Time in seconds.
        aif_time (np.ndarray | None): Time vector for the current AIF.
        aif_concentration (np.ndarray | None): Concentration values for the current AIF.
        Cp_interp_func (callable | None): Interpolation function for AIF concentration.
        integral_Cp_dt_interp_func (callable | None): Interpolation function for AIF integral.
        selected_model_name (str | None): Name of the currently selected PK model.
        parameter_maps (dict): Dictionary storing generated parameter maps (e.g., Ktrans, ve).
                               Keys are map names, values are 3D np.ndarrays.
        displayable_volumes (dict): Dictionary of all volumes available for display in the
                                   image view (original data, Ct, parameter maps).
                                   Keys are descriptive names, values are 3D/4D np.ndarrays.
        current_display_key (str | None): Key of the volume currently shown in the main image view.
        current_slice_index (int): Current slice index being displayed.
        aif_roi_object (pg.RectROI | None): pyqtgraph ROI object for defining AIF from image.
        stats_roi_list (list): List of dictionaries, each representing an ROI for statistics.
        overlay_image_item (pg.ImageItem): ImageItem for displaying overlays on the main view.
        # Add other important attributes here as they are reviewed.
    """
    def __init__(self):
        """
        Initializes the MainWindow, sets up UI layout, and internal data structures.
        """
        super().__init__()

        self.setWindowTitle("DCE-MRI Analysis Tool")
        self.setGeometry(100, 100, 1250, 900) # Set initial window size and position

        # Configure pyqtgraph global options for white background and black foreground
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # --- Initialize Data Attributes ---
        # These will store loaded data, processed data, and AIF/model information.
        self.dce_data = None; self.t10_data = None; self.mask_data = None
        self.Ct_data = None; self.dce_shape_for_validation = None
        self.dce_time_vector = None 
        self.dce_filepath = None; self.t1_filepath = None # Store filepaths for reference (e.g. saving maps)
        self.tr_float = None # TR in seconds

        # AIF related attributes
        self.aif_time = None; self.aif_concentration = None
        self.Cp_interp_func = None # Interpolated AIF: Cp(t)
        self.integral_Cp_dt_interp_func = None # Interpolated integral of AIF: Int(Cp(tau)dtau)
        self.population_aif_time_vector = np.linspace(0, 10, 121) # Default time for pop AIFs (mins)
        self.aif_param_input_widgets = {} # For dynamic population AIF parameter inputs

        # Model fitting related attributes
        self.selected_model_name = None # Name of the PK model selected by the user
        self.parameter_maps = {} # Stores results of model fitting (e.g., {'Ktrans': Ktrans_map_data, ...})

        # Display related attributes
        self.displayable_volumes = {}  # Stores all data that can be shown in image_view
        self.current_display_key = None # Key of the currently displayed volume in map_selector_combo
        self.current_slice_index = 0 # Current slice index of the image_view
        self.aif_roi_object = None # pyqtgraph.RectROI object for AIF definition from ROI

        # Statistics ROI attributes
        self.stats_roi_list = [] # List to store multiple pg.RectROI objects for statistics
        self.stats_roi_counter = 0 # Counter for unique Stats ROI names

        # Overlay attributes
        self.overlay_image_item = pg.ImageItem() # For displaying parameter maps as overlays
        self.current_overlay_map_key = "None" # Key of the map selected for overlay
        self.overlay_alpha = 0.5 # Opacity of the overlay
        self.overlay_cmap_name = 'viridis' # Colormap for the overlay

        # --- Setup Main UI Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget) # Main horizontal layout

        # --- Create Left Panel (Controls) ---
        self.left_panel_layout = QVBoxLayout()
        self._create_file_io_section() # File loading buttons and labels
        self._create_conversion_settings_section() # Inputs for S->C conversion params
        self._create_aif_section() # AIF definition options (file, population, ROI)
        self._create_model_fitting_section() # Model selection and export buttons
        self._create_processing_section() # Run analysis button, CPU core selection
        self._create_roi_stats_section() # ROI statistics controls and display

        # Add all group boxes to the left panel
        self.left_panel_layout.addWidget(self.file_io_group)
        self.left_panel_layout.addWidget(self.conversion_settings_group)
        self.left_panel_layout.addWidget(self.aif_group)
        self.left_panel_layout.addWidget(self.model_fitting_group)
        self.left_panel_layout.addWidget(self.processing_section)
        self.left_panel_layout.addWidget(self.roi_stats_group)
        self.left_panel_layout.addStretch(1) # Add stretch to push controls to the top

        # --- Create Right Panel (Display, Plot, Log) ---
        self.right_panel_layout = QHBoxLayout()
        self._create_display_area() # Main image display and slice controls
        self._create_log_console_and_plot_area() # Plot widget and log console

        # Add left and right panels to the main layout
        self.main_layout.addLayout(self.left_panel_layout, stretch=1) # Left panel takes 1/4 space
        self.main_layout.addLayout(self.right_panel_layout, stretch=3) # Right panel takes 3/4 space

        # --- Connect UI Signals to Slots ---
        self._connect_signals()
        
        # --- Initial UI State Updates ---
        self.update_aif_ui_state() # Set initial enabled/disabled state of AIF widgets
        self.handle_model_selection() # Set initial model selection and update related UI
        self.handle_population_aif_selection_changed() # Populate AIF params if needed

    def _connect_signals(self):
        """Connects all UI element signals (e.g., button clicks, combo box changes) to their respective handler methods."""
        # File I/O
        self.load_dce_button.clicked.connect(self.load_dce_file)
        self.load_t1_button.clicked.connect(self.load_t1_file)
        self.load_mask_button.clicked.connect(self.load_mask_file)

        # Processing
        self.run_button.clicked.connect(self.run_analysis)
        self.tr_input.textChanged.connect(self.handle_tr_changed) # Connect TR input changes
        
        # AIF controls
        self.load_aif_button.clicked.connect(self.handle_load_aif_file)
        self.save_derived_aif_button.clicked.connect(self.handle_save_derived_aif)
        self.select_population_aif_button.clicked.connect(self.handle_apply_population_aif)
        self.aif_load_file_radio.toggled.connect(self.update_aif_ui_state)
        self.aif_population_radio.toggled.connect(self.update_aif_ui_state)
        self.aif_roi_radio.toggled.connect(self.update_aif_ui_state)
        self.draw_aif_roi_button.clicked.connect(self.handle_draw_aif_roi_button)
        self.save_aif_roi_button.clicked.connect(self.handle_save_aif_roi_def)
        self.load_aif_roi_button.clicked.connect(self.handle_load_aif_roi_def)
        self.population_aif_combo.currentIndexChanged.connect(self.handle_population_aif_selection_changed)

        # Model selection and export
        self.model_standard_tofts_radio.toggled.connect(self.handle_model_selection)
        self.model_extended_tofts_radio.toggled.connect(self.handle_model_selection)
        self.model_patlak_radio.toggled.connect(self.handle_model_selection)
        self.model_2cxm_radio.toggled.connect(self.handle_model_selection)
        self.model_none_radio.toggled.connect(self.handle_model_selection)
        
        # Export buttons for different parameters (using lambdas for concise connection)
        self.export_ktrans_button.clicked.connect(lambda: self.export_map("Ktrans"))
        self.export_ve_button.clicked.connect(lambda: self.export_map("ve"))
        self.export_vp_button.clicked.connect(lambda: self.export_map("vp"))
        self.export_ktrans_patlak_button.clicked.connect(lambda: self.export_map("Ktrans_patlak"))
        self.export_vp_patlak_button.clicked.connect(lambda: self.export_map("vp_patlak"))
        self.export_fp_2cxm_button.clicked.connect(lambda: self.export_map("Fp_2cxm"))
        self.export_ps_2cxm_button.clicked.connect(lambda: self.export_map("PS_2cxm"))
        self.export_vp_2cxm_button.clicked.connect(lambda: self.export_map("vp_2cxm"))
        self.export_ve_2cxm_button.clicked.connect(lambda: self.export_map("ve_2cxm"))

        # Display controls
        self.map_selector_combo.currentIndexChanged.connect(self.handle_map_selection_changed)
        self.slice_slider.valueChanged.connect(self.handle_slice_changed)
        self.image_view.getView().scene().sigMouseClicked.connect(self.handle_voxel_clicked) # Voxel click for plotting

        # Statistics ROI controls
        self.add_stats_roi_button.clicked.connect(self.handle_add_stats_roi)
        self.clear_last_stats_roi_button.clicked.connect(self.handle_clear_last_stats_roi)
        self.clear_all_stats_rois_button.clicked.connect(self.handle_clear_all_stats_rois)
        self.save_stats_button.clicked.connect(self.handle_save_all_roi_stats)

        # Overlay controls
        self.overlay_map_selector_combo.currentIndexChanged.connect(self.handle_overlay_controls_changed)
        self.overlay_alpha_slider.valueChanged.connect(self.handle_overlay_controls_changed)
        self.overlay_cmap_combo.currentIndexChanged.connect(self.handle_overlay_controls_changed)
        
        # Plot saving
        self.save_plot_button.clicked.connect(self.handle_save_plot)

    # --- UI Creation Methods ---
    def _create_file_io_section(self):
        """Creates the 'File I/O' group box with widgets for loading DCE, T1, and Mask files."""
        self.file_io_group = QGroupBox("File I/O")
        layout = QVBoxLayout()
        # DCE Series
        self.load_dce_button = QPushButton("Load DCE Series")
        self.dce_path_label = QLabel("Not loaded") # To display the path of the loaded DCE file
        layout.addWidget(self.load_dce_button)
        layout.addWidget(self.dce_path_label)
        # T1 Map
        self.load_t1_button = QPushButton("Load T1 Map")
        self.t1_path_label = QLabel("Not loaded") # To display T1 map path
        layout.addWidget(self.load_t1_button)
        layout.addWidget(self.t1_path_label)
        # Mask
        self.load_mask_button = QPushButton("Load Mask")
        self.mask_path_label = QLabel("Not loaded") # To display mask path
        layout.addWidget(self.load_mask_button)
        layout.addWidget(self.mask_path_label)
        self.file_io_group.setLayout(layout)

    def _create_conversion_settings_section(self):
        """Creates the 'Conversion Settings (Tissue)' group box for S->C parameters."""
        self.conversion_settings_group = QGroupBox("Conversion Settings (Tissue)")
        layout = QFormLayout()
        self.r1_input = QLineEdit("4.5") # Default r1 relaxivity
        layout.addRow(QLabel("r1 Relaxivity (s⁻¹mM⁻¹):"), self.r1_input)
        self.tr_input = QLineEdit("0.005") # Default TR in seconds
        layout.addRow(QLabel("TR (s):"), self.tr_input)
        self.te_input = QLineEdit("0.002") # Default TE in seconds (Note: TE currently not used in core.conversion)
        layout.addRow(QLabel("TE (s):"), self.te_input)
        self.baseline_points_input = QLineEdit("5") # Default number of baseline points for S0 calculation
        layout.addRow(QLabel("Baseline Pts (Tissue):"), self.baseline_points_input)
        self.conversion_settings_group.setLayout(layout)

    def _create_aif_section(self):
        """
        Creates the 'Arterial Input Function (AIF)' group box.
        This section allows users to load an AIF from a file, select a population AIF model,
        or define an AIF from an ROI drawn on the image.
        """
        self.aif_group = QGroupBox("Arterial Input Function (AIF)")
        v_layout = QVBoxLayout() # Main vertical layout for AIF section

        # Radio buttons for AIF mode selection
        self.aif_load_file_radio = QRadioButton("Load AIF from File")
        self.aif_load_file_radio.setChecked(True) # Default AIF mode
        self.aif_population_radio = QRadioButton("Select Population AIF")
        self.aif_roi_radio = QRadioButton("Define AIF from ROI")
        self.aif_roi_radio.setEnabled(True) # Assuming DCE data might be loaded to enable this
        v_layout.addWidget(self.aif_load_file_radio)

        # Section for loading AIF from file
        file_load_h_layout = QHBoxLayout()
        self.load_aif_button = QPushButton("Load AIF File...")
        file_load_h_layout.addWidget(self.load_aif_button)
        self.aif_file_label = QLabel("No AIF file loaded.")
        self.aif_file_label.setWordWrap(True)
        file_load_h_layout.addWidget(self.aif_file_label, 1) # Label takes remaining space
        v_layout.addLayout(file_load_h_layout)

        # Section for selecting population AIF
        v_layout.addWidget(self.aif_population_radio)
        pop_aif_h_layout = QHBoxLayout()
        self.population_aif_combo = QComboBox()
        pop_aif_h_layout.addWidget(self.population_aif_combo)
        if aif.POPULATION_AIFS: # Populate combo box if models are defined
            self.population_aif_combo.addItems(sorted(aif.POPULATION_AIFS.keys()))
        self.select_population_aif_button = QPushButton("Apply")
        pop_aif_h_layout.addWidget(self.select_population_aif_button)
        v_layout.addLayout(pop_aif_h_layout)

        # Group for dynamically generated population AIF parameters
        self.aif_params_group = QGroupBox("Population AIF Parameters")
        self.aif_params_form_layout = QFormLayout()
        self.aif_params_group.setLayout(self.aif_params_form_layout)
        v_layout.addWidget(self.aif_params_group)
        self.aif_params_group.setVisible(False) # Initially hidden

        # Section for defining AIF from ROI
        v_layout.addWidget(self.aif_roi_radio)
        roi_aif_controls_layout = QVBoxLayout() # Sub-layout for ROI AIF controls
        roi_aif_buttons_layout = QHBoxLayout()
        self.draw_aif_roi_button = QPushButton("Define/Redraw AIF ROI")
        roi_aif_buttons_layout.addWidget(self.draw_aif_roi_button)
        self.save_aif_roi_button = QPushButton("Save AIF ROI Def.")
        self.save_aif_roi_button.setEnabled(False) # Enabled when ROI is drawn
        roi_aif_buttons_layout.addWidget(self.save_aif_roi_button)
        self.load_aif_roi_button = QPushButton("Load AIF ROI Def.")
        roi_aif_buttons_layout.addWidget(self.load_aif_roi_button)
        roi_aif_controls_layout.addLayout(roi_aif_buttons_layout)

        # Input fields for AIF ROI parameters
        roi_aif_form_layout = QFormLayout()
        self.aif_t10_blood_input = QLineEdit("1.4") # Default T10 of blood
        roi_aif_form_layout.addRow(QLabel("AIF T10 Blood (s):"), self.aif_t10_blood_input)
        self.aif_r1_blood_input = QLineEdit("4.5") # Default r1 of blood
        roi_aif_form_layout.addRow(QLabel("AIF r1 Blood (s⁻¹mM⁻¹):"), self.aif_r1_blood_input)
        self.aif_baseline_points_input = QLineEdit("5") # Default baseline points for AIF signal
        roi_aif_form_layout.addRow(QLabel("AIF Baseline Points:"), self.aif_baseline_points_input)
        roi_aif_controls_layout.addLayout(roi_aif_form_layout)
        v_layout.addLayout(roi_aif_controls_layout)

        # Button to save the derived AIF curve (from population or ROI)
        self.save_derived_aif_button = QPushButton("Save Derived AIF Curve")
        self.save_derived_aif_button.setEnabled(False) # Enabled when an AIF is successfully derived
        v_layout.addWidget(self.save_derived_aif_button)

        self.aif_status_label = QLabel("AIF: Not defined") # Displays current AIF status
        v_layout.addWidget(self.aif_status_label)
        self.aif_group.setLayout(v_layout)

    def _create_model_fitting_section(self):
        """
        Creates the 'Model Fitting' group box.
        Users can select a pharmacokinetic model (e.g., Tofts, Patlak) and initiate
        voxel-wise fitting. Buttons to export resulting parameter maps are also included here.
        """
        self.model_fitting_group = QGroupBox("Model Fitting")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select Model:"))
        self.model_standard_tofts_radio = QRadioButton("Standard Tofts")
        self.model_extended_tofts_radio = QRadioButton("Extended Tofts")
        self.model_patlak_radio = QRadioButton("Patlak Model") 
        self.model_2cxm_radio = QRadioButton("2CXM (Fp,PS,vp,ve)") 
        self.model_none_radio = QRadioButton("No Model Fitting"); self.model_none_radio.setChecked(True)
        layout.addWidget(self.model_standard_tofts_radio); layout.addWidget(self.model_extended_tofts_radio); layout.addWidget(self.model_patlak_radio); layout.addWidget(self.model_2cxm_radio); layout.addWidget(self.model_none_radio)
        export_layout_tofts = QHBoxLayout()
        self.export_ktrans_button = QPushButton("Ktrans (Tofts)"); self.export_ktrans_button.setEnabled(False)
        self.export_ve_button = QPushButton("ve (Tofts)"); self.export_ve_button.setEnabled(False)
        self.export_vp_button = QPushButton("vp (Ext.Tofts)"); self.export_vp_button.setEnabled(False)
        export_layout_tofts.addWidget(self.export_ktrans_button); export_layout_tofts.addWidget(self.export_ve_button); export_layout_tofts.addWidget(self.export_vp_button)
        layout.addLayout(export_layout_tofts)
        export_layout_patlak = QHBoxLayout()
        self.export_ktrans_patlak_button = QPushButton("Ktrans (Patlak)"); self.export_ktrans_patlak_button.setEnabled(False) 
        self.export_vp_patlak_button = QPushButton("vp (Patlak)"); self.export_vp_patlak_button.setEnabled(False)       
        export_layout_patlak.addWidget(self.export_ktrans_patlak_button); export_layout_patlak.addWidget(self.export_vp_patlak_button)
        layout.addLayout(export_layout_patlak)
        export_layout_2cxm = QHBoxLayout() 
        self.export_fp_2cxm_button = QPushButton("Fp (2CXM)"); self.export_fp_2cxm_button.setEnabled(False)
        self.export_ps_2cxm_button = QPushButton("PS (2CXM)"); self.export_ps_2cxm_button.setEnabled(False)
        self.export_vp_2cxm_button = QPushButton("vp (2CXM)"); self.export_vp_2cxm_button.setEnabled(False)
        self.export_ve_2cxm_button = QPushButton("ve (2CXM)"); self.export_ve_2cxm_button.setEnabled(False)
        export_layout_2cxm.addWidget(self.export_fp_2cxm_button); export_layout_2cxm.addWidget(self.export_ps_2cxm_button); export_layout_2cxm.addWidget(self.export_vp_2cxm_button); export_layout_2cxm.addWidget(self.export_ve_2cxm_button)
        layout.addLayout(export_layout_2cxm)
        self.model_fitting_group.setLayout(layout)

    def _create_roi_stats_section(self):
        """
        Creates the 'ROI Statistics' group box.
        This section allows users to draw ROIs on the displayed maps/images,
        view statistics for these ROIs, and save the statistics.
        """
        self.roi_stats_group = QGroupBox("ROI Statistics")
        layout = QVBoxLayout()
        roi_buttons_layout = QHBoxLayout()
        self.add_stats_roi_button = QPushButton("Add Stats ROI"); roi_buttons_layout.addWidget(self.add_stats_roi_button)
        self.clear_last_stats_roi_button = QPushButton("Clear Last ROI"); roi_buttons_layout.addWidget(self.clear_last_stats_roi_button)
        self.clear_all_stats_rois_button = QPushButton("Clear All ROIs"); roi_buttons_layout.addWidget(self.clear_all_stats_rois_button)
        layout.addLayout(roi_buttons_layout)
        self.stats_results_display = QTextEdit(); self.stats_results_display.setReadOnly(True); self.stats_results_display.setPlaceholderText("ROI statistics for currently displayed map & slice will appear here."); self.stats_results_display.setFixedHeight(150); layout.addWidget(self.stats_results_display)
        self.save_stats_button = QPushButton("Save All Visible ROI Stats"); self.save_stats_button.setEnabled(False); layout.addWidget(self.save_stats_button)
        self.roi_stats_group.setLayout(layout)

    def _create_processing_section(self):
        """
        Creates the 'Processing' group box.
        Contains controls for initiating the main analysis pipeline (S->C conversion
        and model fitting) and setting the number of CPU cores for parallel processing.
        """
        self.selected_model_name = None
        if self.model_standard_tofts_radio.isChecked(): self.selected_model_name = "Standard Tofts"
        elif self.model_extended_tofts_radio.isChecked(): self.selected_model_name = "Extended Tofts"
        elif self.model_patlak_radio.isChecked(): self.selected_model_name = "Patlak" 
        elif self.model_2cxm_radio.isChecked(): self.selected_model_name = "2CXM" 
        self.processing_section = QGroupBox("Processing")
        layout = QFormLayout()
        self.num_processes_input = QSpinBox()
        self.num_processes_input.setMinimum(1)
        num_cpus = os.cpu_count() # Get number of available CPU cores
        self.num_processes_input.setMaximum(num_cpus if num_cpus else 1) # Max to num cores
        self.num_processes_input.setValue(num_cpus if num_cpus else 1) # Default to all cores
        layout.addRow(QLabel("Number of Cores for Fitting:"), self.num_processes_input)
        self.run_button = QPushButton("Run Full Analysis (S->C & Voxel-wise Fitting)")
        layout.addRow(self.run_button)
        self.processing_section.setLayout(layout)

    def _create_display_area(self):
        """
        Creates the main image display area.
        This includes a dropdown to select which map/image to display, a slider for
        navigating through slices, the main image view (pyqtgraph.ImageView),
        and controls for overlaying another map.
        """
        self.display_area_group = QGroupBox("Image Display") 
        display_layout = QVBoxLayout(self.display_area_group)
        self.map_selector_combo = QComboBox(); display_layout.addWidget(self.map_selector_combo)
        slider_layout = QHBoxLayout()
        self.slice_slider_label = QLabel("Slice: 0/0")
        self.slice_slider = QSlider(Qt.Horizontal); self.slice_slider.setEnabled(False) 
        slider_layout.addWidget(QLabel("Slice:")); slider_layout.addWidget(self.slice_slider, 1); slider_layout.addWidget(self.slice_slider_label)
        display_layout.addLayout(slider_layout)
        self.image_view = pg.ImageView(); self.image_view.getView().addItem(self.overlay_image_item); self.overlay_image_item.setZValue(10); self.overlay_image_item.setVisible(False); display_layout.addWidget(self.image_view)
        overlay_controls_layout = QHBoxLayout(); overlay_controls_layout.addWidget(QLabel("Overlay:"))
        self.overlay_map_selector_combo = QComboBox(); self.overlay_map_selector_combo.addItem("None"); overlay_controls_layout.addWidget(self.overlay_map_selector_combo)
        self.overlay_alpha_slider = QSlider(Qt.Horizontal); self.overlay_alpha_slider.setRange(0, 100); self.overlay_alpha_slider.setValue(int(self.overlay_alpha * 100)); overlay_controls_layout.addWidget(self.overlay_alpha_slider)
        self.overlay_alpha_label = QLabel(f"{self.overlay_alpha*100:.0f}%"); overlay_controls_layout.addWidget(self.overlay_alpha_label)
        overlay_controls_layout.addWidget(QLabel("Cmap:")); self.overlay_cmap_combo = QComboBox(); self.overlay_cmap_combo.addItems(['viridis', 'jet', 'hot', 'coolwarm', 'magma', 'plasma', 'inferno', 'cividis']); self.overlay_cmap_combo.setCurrentText(self.overlay_cmap_name); overlay_controls_layout.addWidget(self.overlay_cmap_combo)
        display_layout.addLayout(overlay_controls_layout) 
        self.right_panel_layout.addWidget(self.display_area_group, stretch=2)

    def _create_log_console_and_plot_area(self):
        """
        Creates the right-most part of the UI, containing the plot widget for
        time-course visualization and the log console for messages.
        Also includes a button to save the current plot.
        """
        plot_log_group = QGroupBox("Plots and Log")
        plot_log_layout = QVBoxLayout(plot_log_group)

        # Plot Widget and Save Button
        plot_area_layout = QVBoxLayout() # Use QVBoxLayout for plot and its button
        self.plot_widget = pg.PlotWidget(); self.plot_widget.setLabel('bottom', 'Time'); self.plot_widget.setLabel('left', 'Concentration'); self.plot_widget.addLegend(offset=(-10,10)) 
        plot_area_layout.addWidget(self.plot_widget, stretch=1) # Plot takes most space in this sub-layout
        
        self.save_plot_button = QPushButton("Save Current Plot")
        self.save_plot_button.setEnabled(False) # Initially disabled
        plot_area_layout.addWidget(self.save_plot_button) # Add button below plot
        
        plot_log_layout.addLayout(plot_area_layout, stretch=1) # Add plot area to main group

        # Log Console
        self.log_console = QTextEdit(); self.log_console.setReadOnly(True); plot_log_layout.addWidget(self.log_console, stretch=1)
        self.right_panel_layout.addWidget(plot_log_group, stretch=1)

    # --- AIF Handling Methods ---
    def update_aif_ui_state(self):
        """
        Updates the enabled/disabled state of AIF-related UI widgets based on the
        selected AIF definition method (file, population, ROI) and data availability.
        Manages visibility of parameter input fields for population AIFs.
        Removes AIF ROI from display if ROI mode is deselected.
        """
        if data is None: return
        self.displayable_volumes[name] = data; current_base_selection = self.map_selector_combo.currentText()
        self.map_selector_combo.blockSignals(True); self.map_selector_combo.clear(); self.map_selector_combo.addItems(self.displayable_volumes.keys())
        idx_base = self.map_selector_combo.findText(name) 
        if idx_base != -1: self.map_selector_combo.setCurrentIndex(idx_base)
        elif current_base_selection and self.map_selector_combo.findText(current_base_selection) != -1: self.map_selector_combo.setCurrentText(current_base_selection)
        self.map_selector_combo.blockSignals(False)
        current_overlay_selection = self.overlay_map_selector_combo.currentText()
        self.overlay_map_selector_combo.blockSignals(True); self.overlay_map_selector_combo.clear(); self.overlay_map_selector_combo.addItem("None")
        for vol_name, vol_data in self.displayable_volumes.items():
            if vol_data.ndim == 3: self.overlay_map_selector_combo.addItem(vol_name)
        idx_overlay = self.overlay_map_selector_combo.findText(current_overlay_selection)
        if idx_overlay != -1: self.overlay_map_selector_combo.setCurrentIndex(idx_overlay)
        else: self.overlay_map_selector_combo.setCurrentIndex(0) 
        self.overlay_map_selector_combo.blockSignals(False)
        if self.map_selector_combo.currentText() == name: self.handle_map_selection_changed() 

    def handle_overlay_controls_changed(self): # Unchanged
        self.current_overlay_map_key = self.overlay_map_selector_combo.currentText(); self.overlay_alpha = self.overlay_alpha_slider.value() / 100.0
        self.overlay_alpha_label.setText(f"{self.overlay_alpha*100:.0f}%"); self.overlay_cmap_name = self.overlay_cmap_combo.currentText()
        self.update_overlay_image_display()

    def update_overlay_image_display(self): # Unchanged
        if self.current_overlay_map_key is None or self.current_overlay_map_key == "None": self.overlay_image_item.clear(); self.overlay_image_item.setVisible(False); return
        overlay_volume_data = self.displayable_volumes.get(self.current_overlay_map_key); base_image_item = self.image_view.getImageItem()
        if overlay_volume_data is None or base_image_item is None or base_image_item.image is None: self.overlay_image_item.clear(); self.overlay_image_item.setVisible(False); return
        current_slice_idx = self.image_view.currentIndex 
        if overlay_volume_data.ndim == 3: overlay_data_permuted = overlay_volume_data.transpose(2,1,0) 
        else: self.overlay_image_item.clear(); self.overlay_image_item.setVisible(False); return
        if not (0 <= current_slice_idx < overlay_data_permuted.shape[0]): self.overlay_image_item.clear(); self.overlay_image_item.setVisible(False); return 
        overlay_slice_to_display = overlay_data_permuted[current_slice_idx]
        self.overlay_image_item.setImage(overlay_slice_to_display, autoLevels=False) 
        cmap = pg.colormap.get(self.overlay_cmap_name); lut = cmap.getLookupTable(alpha=True) 
        min_val = np.nanmin(overlay_slice_to_display); max_val = np.nanmax(overlay_slice_to_display)
        if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val: 
            min_val_vol = np.nanmin(overlay_volume_data); max_val_vol = np.nanmax(overlay_volume_data)
            if np.isnan(min_val_vol) or np.isnan(max_val_vol) or min_val_vol == max_val_vol: self.overlay_image_item.clear(); self.overlay_image_item.setVisible(False); return
            min_val, max_val = min_val_vol, max_val_vol
        self.overlay_image_item.setLookupTable(lut); self.overlay_image_item.setLevels([min_val, max_val]); self.overlay_image_item.setOpacity(self.overlay_alpha); self.overlay_image_item.setVisible(True)

    def handle_map_selection_changed(self): # Unchanged
        selected_key = self.map_selector_combo.currentText()
        if not selected_key: self.image_view.clear(); self.slice_slider.setEnabled(False); self.slice_slider_label.setText("Slice: 0/0"); self.overlay_image_item.clear(); self.overlay_image_item.setVisible(False); self.update_all_rois_stats_display(); return
        self.current_display_key = selected_key; volume_data = self.displayable_volumes.get(self.current_display_key)
        if volume_data is None: self.image_view.clear(); self.slice_slider.setEnabled(False); self.slice_slider_label.setText("Slice: 0/0"); self.overlay_image_item.clear(); self.overlay_image_item.setVisible(False); self.update_all_rois_stats_display(); return
        display_data = None
        if volume_data.ndim == 3: display_data = volume_data.transpose(2, 1, 0)
        elif volume_data.ndim == 4: mean_over_time = np.mean(volume_data, axis=3); display_data = mean_over_time.transpose(2, 1, 0)
        else: self.log_console.append(f"Volume '{selected_key}' unsupported dim: {volume_data.ndim}."); self.image_view.clear(); self.slice_slider.setEnabled(False); self.slice_slider_label.setText("Slice: 0/0"); self.overlay_image_item.clear(); self.overlay_image_item.setVisible(False); self.update_all_rois_stats_display(); return
        self.image_view.setImage(display_data, autoRange=True, autoLevels=True, autoHistogramRange=True)
        num_slices = display_data.shape[0]
        self.slice_slider.setEnabled(True); self.slice_slider.setMinimum(0); self.slice_slider.setMaximum(num_slices - 1)
        current_idx = self.image_view.currentIndex 
        if not (0 <= current_idx < num_slices): current_idx = 0 
        self.image_view.setCurrentIndex(current_idx); self.slice_slider.setValue(current_idx); self.slice_slider_label.setText(f"Slice: {current_idx + 1}/{num_slices}")
        self.update_overlay_image_display(); self.update_all_rois_stats_display() 

    def handle_slice_changed(self, value): # Unchanged
        self.current_slice_index = value
        if self.image_view.image is not None:
            num_slices = self.image_view.image.shape[0]; safe_value = np.clip(value, 0, num_slices - 1)
            self.image_view.setCurrentIndex(safe_value); self.slice_slider_label.setText(f"Slice: {safe_value + 1}/{num_slices}")
            if value != safe_value and self.slice_slider.value() != safe_value : self.slice_slider.setValue(safe_value)
            self.update_all_rois_stats_display(); self.update_overlay_image_display() 

    def handle_voxel_clicked(self, mouse_click_event): # Modified to update save plot button
        if not mouse_click_event.double(): return 
        self.save_plot_button.setEnabled(False) # Disable first
        image_item = self.image_view.getImageItem(); 
        if image_item is None or image_item.image is None: return
        scene_pos = mouse_click_event.scenePos(); img_coords_float = image_item.mapFromScene(scene_pos)
        y_in_slice = int(round(img_coords_float.y())); x_in_slice = int(round(img_coords_float.x())); current_z_index_in_display = self.image_view.currentIndex 
        current_slice_shape = self.image_view.image[current_z_index_in_display].shape
        if not (0 <= y_in_slice < current_slice_shape[0] and 0 <= x_in_slice < current_slice_shape[1]): self.log_console.append(f"Clicked outside current slice boundaries."); return
        z_orig = current_z_index_in_display; y_orig = y_in_slice; x_orig = x_in_slice
        self.log_console.append(f"Image double-clicked. Mapped to original (X:{x_orig}, Y:{y_orig}, Z:{z_orig})")
        if self.Ct_data is None or not (0 <= x_orig < self.Ct_data.shape[0] and 0 <= y_orig < self.Ct_data.shape[1] and 0 <= z_orig < self.Ct_data.shape[2]): self.log_console.append(f"Clicked coords ({x_orig},{y_orig},{z_orig}) outside Ct_data bounds."); return
        self.plot_selected_voxel_curves(x_orig, y_orig, z_orig)

    def plot_selected_voxel_curves(self, x_idx, y_idx, z_idx): # Modified to update save plot button
        self.plot_widget.clear(); self.plot_widget.setTitle(f"Curves for Voxel (X:{x_idx}, Y:{y_idx}, Z:{z_idx})")
        if self.Ct_data is None: self.log_console.append("Ct data not available for plotting."); self.save_plot_button.setEnabled(False); return
        Ct_voxel = self.Ct_data[x_idx, y_idx, z_idx, :]; t_values = self.dce_time_vector
        if t_values is None: 
            try: tr_val = float(self.tr_input.text()); num_time_points = self.Ct_data.shape[3]; t_values = np.arange(num_time_points) * tr_val
            except ValueError: self.log_console.append("TR value invalid for plotting time axis."); self.save_plot_button.setEnabled(False); return
        self.plot_widget.plot(t_values, Ct_voxel, pen=pg.mkPen('b', width=2), name='Tissue Conc.')
        if self.aif_time is not None and self.aif_concentration is not None: self.plot_widget.plot(self.aif_time, self.aif_concentration, pen='r', name='AIF')
        if self.selected_model_name and self.parameter_maps and self.Cp_interp_func: 
            model_params = {}; valid_params = True; fitted_curve = None; param_str = "N/A"
            if self.selected_model_name == "Standard Tofts":
                if "Ktrans" in self.parameter_maps and "ve" in self.parameter_maps: Ktrans_val = self.parameter_maps["Ktrans"][x_idx, y_idx, z_idx]; ve_val = self.parameter_maps["ve"][x_idx, y_idx, z_idx]; 
                if np.isnan(Ktrans_val) or np.isnan(ve_val): valid_params = False; else: model_params['Ktrans'] = Ktrans_val; model_params['ve'] = ve_val
                else: valid_params = False
                if valid_params: fitted_curve = modeling.standard_tofts_model_conv(t_values, model_params['Ktrans'], model_params['ve'], self.Cp_interp_func)
            elif self.selected_model_name == "Extended Tofts":
                if "Ktrans" in self.parameter_maps and "ve" in self.parameter_maps and "vp" in self.parameter_maps: Ktrans_val = self.parameter_maps["Ktrans"][x_idx, y_idx, z_idx]; ve_val = self.parameter_maps["ve"][x_idx, y_idx, z_idx]; vp_val = self.parameter_maps["vp"][x_idx, y_idx, z_idx];
                if np.isnan(Ktrans_val) or np.isnan(ve_val) or np.isnan(vp_val): valid_params = False; else: model_params['Ktrans'] = Ktrans_val; model_params['ve'] = ve_val; model_params['vp'] = vp_val
                else: valid_params = False
                if valid_params: fitted_curve = modeling.extended_tofts_model_conv(t_values, model_params['Ktrans'], model_params['ve'], model_params['vp'], self.Cp_interp_func)
            elif self.selected_model_name == "Patlak": 
                if "Ktrans_patlak" in self.parameter_maps and "vp_patlak" in self.parameter_maps and self.integral_Cp_dt_interp_func: Ktrans_val = self.parameter_maps["Ktrans_patlak"][x_idx, y_idx, z_idx]; vp_val = self.parameter_maps["vp_patlak"][x_idx, y_idx, z_idx];
                if np.isnan(Ktrans_val) or np.isnan(vp_val): valid_params = False; else: model_params['Ktrans_patlak'] = Ktrans_val; model_params['vp_patlak'] = vp_val
                else: valid_params = False
                if valid_params: fitted_curve = modeling.patlak_model(t_values, model_params['Ktrans_patlak'], model_params['vp_patlak'], self.Cp_interp_func, self.integral_Cp_dt_interp_func)
            elif self.selected_model_name == "2CXM": 
                map_keys = ["Fp_2cxm", "PS_2cxm", "vp_2cxm", "ve_2cxm"]
                if all(key in self.parameter_maps for key in map_keys): params = [self.parameter_maps[key][x_idx, y_idx, z_idx] for key in map_keys];
                if any(np.isnan(p) for p in params): valid_params = False; else: model_params = dict(zip(map_keys, params)); Fp_val, PS_val, vp_val, ve_val = params[0], params[1], params[2], params[3]
                else: valid_params = False
                if valid_params: t_aif_max_for_plot = self.aif_time[-1] if self.aif_time is not None and len(self.aif_time) > 0 else t_values[-1]; fitted_curve = modeling.solve_2cxm_ode_model(t_values, Fp_val, PS_val, vp_val, ve_val, self.Cp_interp_func, t_span_max=t_aif_max_for_plot)
            if valid_params and fitted_curve is not None: self.plot_widget.plot(t_values, fitted_curve, pen='g', name=f'{self.selected_model_name} Fit'); param_str = ", ".join([f"{k}={v:.4f}" for k,v in model_params.items()]); self.log_console.append(f"Plotted fit for ({x_idx},{y_idx},{z_idx}). Params: {param_str}")
            elif valid_params and fitted_curve is None: self.log_console.append(f"Fit parameters valid but curve generation failed for {self.selected_model_name} at ({x_idx},{y_idx},{z_idx}).")
            else: self.log_console.append(f"No valid pre-fitted parameters for voxel ({x_idx},{y_idx},{z_idx}) for {self.selected_model_name}.")
        self.plot_widget.autoRange()
        self.save_plot_button.setEnabled(bool(self.plot_widget.getPlotItem().listDataItems()))

    def update_aif_ui_state(self): # Unchanged
        is_file_mode = self.aif_load_file_radio.isChecked()
        is_pop_mode = self.aif_population_radio.isChecked()
        is_roi_mode = self.aif_roi_radio.isChecked()

        # Enable/disable file loading widgets
        self.load_aif_button.setEnabled(is_file_mode)
        self.aif_file_label.setEnabled(is_file_mode)

        # Enable/disable population AIF widgets
        self.population_aif_combo.setEnabled(is_pop_mode)
        self.select_population_aif_button.setEnabled(is_pop_mode)
        self.aif_params_group.setVisible(is_pop_mode) # Show/hide parameter input group
        if is_pop_mode:
            self.handle_population_aif_selection_changed() # Update params if pop mode selected

        # Enable/disable ROI AIF widgets
        self.draw_aif_roi_button.setEnabled(is_roi_mode)
        self.save_aif_roi_button.setEnabled(is_roi_mode and self.aif_roi_object is not None)
        self.load_aif_roi_button.setEnabled(is_roi_mode)
        self.aif_t10_blood_input.setEnabled(is_roi_mode)
        self.aif_r1_blood_input.setEnabled(is_roi_mode)
        self.aif_baseline_points_input.setEnabled(is_roi_mode)

        # Enable "Save Derived AIF" if AIF data is available
        self.save_derived_aif_button.setEnabled(self.aif_time is not None and self.aif_concentration is not None)

        # If ROI mode is deselected and an AIF ROI exists, remove it
        if not is_roi_mode and self.aif_roi_object:
            if self.image_view.getView() and self.aif_roi_object in self.image_view.getView().items:
                 self.image_view.removeItem(self.aif_roi_object)
            self.aif_roi_object = None
            self.log_console.append("AIF ROI removed from image as ROI mode was deselected.")
            self.save_aif_roi_button.setEnabled(False)

    def _create_aif_interpolators(self) -> bool:
        """
        Creates interpolation functions for the AIF time-concentration curve and its integral.
        Stores them in `self.Cp_interp_func` and `self.integral_Cp_dt_interp_func`.

        This is crucial for model fitting, as models often require AIF values at arbitrary
        time points (matching tissue curve time points).

        Returns:
            bool: True if interpolators were successfully created, False otherwise.
        """
        if self.aif_time is not None and self.aif_concentration is not None and len(self.aif_time) > 1:
            try:
                # Interpolator for Cp(t)
                self.Cp_interp_func = interp1d(
                    self.aif_time, self.aif_concentration,
                    kind='linear',      # Linear interpolation is common
                    bounds_error=False, # Allow evaluation outside original time range
                    fill_value=0.0      # Fill with 0 outside range (or "extrapolate")
                )
                # Calculate cumulative integral of AIF: Int_0^t Cp(tau) dtau
                integral_Cp_dt_aif = cumtrapz(self.aif_concentration, self.aif_time, initial=0)
                # Interpolator for the integral of Cp(t)
                self.integral_Cp_dt_interp_func = interp1d(
                    self.aif_time, integral_Cp_dt_aif,
                    kind='linear', bounds_error=False, fill_value=0.0
                )
                self.log_console.append("AIF interpolators created successfully.")
                return True
            except Exception as e:
                self.log_console.append(f"Error creating AIF interpolators: {e}")
                self.Cp_interp_func = None
                self.integral_Cp_dt_interp_func = None
                return False
        else:
            # Not enough data points or AIF not loaded
            self.Cp_interp_func = None
            self.integral_Cp_dt_interp_func = None
            if self.aif_time is not None: # Only log if AIF was loaded but insufficient
                self.log_console.append("AIF time or concentration has insufficient points for interpolation.")
            return False

    def handle_population_aif_selection_changed(self):
        """
        Handles changes in the selected population AIF model from the combo box.
        Dynamically creates input fields for the parameters of the selected AIF model.
        """
        # Clear existing parameter input widgets
        for i in reversed(range(self.aif_params_form_layout.rowCount())):
            self.aif_params_form_layout.removeRow(i)
        self.aif_param_input_widgets.clear()

        selected_aif_name = self.population_aif_combo.currentText()

        # Only proceed if population AIF mode is active and a model is selected
        if not selected_aif_name or not self.aif_population_radio.isChecked():
            self.aif_params_group.setVisible(False)
            return

        self.aif_params_group.setVisible(True)
        param_meta_list = aif.AIF_PARAMETER_METADATA.get(selected_aif_name, [])

        if not param_meta_list:
            self.log_console.append(f"No parameter metadata found for AIF model: {selected_aif_name}")
            return

        # Create QLineEdit widgets for each parameter
        for param_name, default_val, min_val, max_val, tooltip_text in param_meta_list:
            label = QLabel(f"{param_name}:")
            line_edit = QLineEdit(str(default_val))
            if tooltip_text: # Add tooltip if provided in metadata
                label.setToolTip(tooltip_text)
                line_edit.setToolTip(tooltip_text)
            self.aif_params_form_layout.addRow(label, line_edit)
            self.aif_param_input_widgets[param_name] = line_edit # Store for later access
        self.log_console.append(f"Parameter input fields generated for '{selected_aif_name}' AIF.")

    def handle_load_aif_file(self):
        """Handles loading an AIF from a user-selected text or CSV file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load AIF File", "", "AIF Files (*.txt *.csv);;All Files (*)")
        if filepath:
            try:
                self.log_console.append(f"Attempting to load AIF from: {filepath}")
                self.aif_time, self.aif_concentration = aif.load_aif_from_file(filepath)

                if self._create_aif_interpolators():
                    self.aif_file_label.setText(os.path.basename(filepath))
                    self.aif_status_label.setText(f"AIF: Loaded from file. Points: {len(self.aif_time)}")
                    self.log_console.append("AIF loaded successfully from file.")
                    # Optionally plot the loaded AIF
                    self.plot_widget.clear()
                    self.plot_widget.plot(self.aif_time, self.aif_concentration, pen='r', name='Loaded AIF')
                    self.plot_widget.autoRange()
                    self.save_plot_button.setEnabled(True)
                else:
                    # Failed to create interpolators, reset AIF data
                    self.aif_time, self.aif_concentration = None, None
                    self.aif_file_label.setText("Error creating AIF interpolator.")
                    self.aif_status_label.setText("AIF: Error processing loaded file.")
            except Exception as e:
                # Reset AIF data on any error during loading or processing
                self.aif_time, self.aif_concentration = None, None
                self.Cp_interp_func, self.integral_Cp_dt_interp_func = None, None
                self.aif_file_label.setText("Error loading AIF file.")
                self.aif_status_label.setText("AIF: Error loading file.")
                self.log_console.append(f"Error loading AIF from file: {e}\n{traceback.format_exc()}")
            self.update_aif_ui_state() # Update UI elements based on new AIF state

    def handle_apply_population_aif(self):
        """
        Applies the selected population AIF model using parameters entered by the user.
        Generates the AIF curve and updates interpolators.
        """
        selected_aif_name = self.population_aif_combo.currentText()
        if not selected_aif_name or selected_aif_name == "None": # "None" check might be redundant if not added to combo
            self.log_console.append("Please select a population AIF model.")
            return

        aif_params = {}
        try:
            # Retrieve parameters from the dynamically created input fields
            for param_name, line_edit_widget in self.aif_param_input_widgets.items():
                aif_params[param_name] = float(line_edit_widget.text())
        except ValueError:
            self.log_console.append("Invalid character in AIF parameters. Please enter numbers only.")
            return

        # Use DCE time vector if available, otherwise fallback to a default population AIF time vector
        time_vector_for_aif = self.dce_time_vector if self.dce_time_vector is not None else self.population_aif_time_vector
        if self.dce_time_vector is None:
            self.log_console.append(f"Warning: Using default/fallback time vector for population AIF generation ({len(time_vector_for_aif)} points). Ensure TR is set if DCE data is loaded.")

        self.log_console.append(f"Applying population AIF: {selected_aif_name} with parameters: {aif_params}")
        try:
            # Generate AIF concentration curve using the selected model and parameters
            aif_c_generated = aif.generate_population_aif(selected_aif_name, time_vector_for_aif, params=aif_params)

            if aif_c_generated is not None:
                self.aif_time, self.aif_concentration = time_vector_for_aif, aif_c_generated
                if self._create_aif_interpolators():
                    self.aif_status_label.setText(f"AIF: Applied '{selected_aif_name}'. Points: {len(self.aif_time)}")
                    self.log_console.append(f"Population AIF '{selected_aif_name}' applied successfully.")
                    # Plot the generated AIF
                    self.plot_widget.clear()
                    self.plot_widget.plot(self.aif_time, self.aif_concentration, pen='r', name=f'{selected_aif_name} AIF')
                    self.plot_widget.autoRange()
                    self.save_plot_button.setEnabled(True)
                else:
                    # Failed to create interpolators
                    self.aif_time, self.aif_concentration = None, None
                    self.aif_status_label.setText(f"AIF: Error processing '{selected_aif_name}'.")
            else:
                # Failed to generate AIF curve
                self.aif_time, self.aif_concentration = None, None
                self.Cp_interp_func, self.integral_Cp_dt_interp_func = None, None
                self.aif_status_label.setText(f"AIF: Error applying '{selected_aif_name}'.")
                self.log_console.append(f"Failed to generate population AIF: {selected_aif_name}.")
        except Exception as e:
            # Catch-all for other errors during AIF application
            self.aif_time, self.aif_concentration = None, None
            self.Cp_interp_func, self.integral_Cp_dt_interp_func = None, None
            self.aif_status_label.setText(f"AIF: Error applying '{selected_aif_name}'.")
            self.log_console.append(f"Error applying population AIF '{selected_aif_name}': {e}\n{traceback.format_exc()}")
        self.update_aif_ui_state() # Update UI based on new AIF state

    def handle_save_derived_aif(self):
        """Handles saving the currently defined AIF (loaded, population, or ROI-derived) to a file."""
        if self.aif_time is None or self.aif_concentration is None:
            self.log_console.append("No derived AIF curve available to save.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Derived AIF Curve", "",
            "CSV files (*.csv);;Text files (*.txt)"
        )
        if filepath:
            try:
                aif.save_aif_curve(self.aif_time, self.aif_concentration, filepath)
                self.log_console.append(f"Derived AIF curve saved to {filepath}")
            except Exception as e:
                self.log_console.append(f"Error saving derived AIF curve: {e}\n{traceback.format_exc()}")

    def handle_draw_aif_roi_button(self):
        """
        Handles the 'Define/Redraw AIF ROI' button click.
        Creates or resets a draggable/resizable ROI item on the currently displayed image
        for defining an AIF. Connects ROI changes to `handle_aif_roi_processing`.
        """
        current_image_item = self.image_view.getImageItem()
        if current_image_item is None or current_image_item.image is None:
            self.log_console.append("No image displayed to draw AIF ROI on.")
            return

        # If an AIF ROI already exists, remove it before creating a new one
        if self.aif_roi_object and self.image_view.getView():
            if self.aif_roi_object in self.image_view.getView().items:
                self.image_view.removeItem(self.aif_roi_object)
            self.aif_roi_object = None

        # Get shape of the currently displayed slice for default ROI sizing/positioning
        # Note: image_view displays data transposed as (Z, Y, X) from original (X, Y, Z, T)
        # So, current_image_item.image[currentIndex] is a YX slice.
        current_display_data = current_image_item.image
        slice_shape_yx = current_display_data[self.image_view.currentIndex].shape # (rows, cols) -> (Y, X) in display

        # Define initial ROI position and size (e.g., centered, 1/4 of display dimensions)
        # These are in display coordinates (pixels of the current slice view)
        roi_y_disp = slice_shape_yx[0] // 4 # Y for display (rows)
        roi_x_disp = slice_shape_yx[1] // 4 # X for display (columns)
        roi_h_disp = slice_shape_yx[0] // 2 # Height for display
        roi_w_disp = slice_shape_yx[1] // 2 # Width for display

        self.aif_roi_object = pg.RectROI(
            pos=(roi_x_disp, roi_y_disp), size=(roi_w_disp, roi_h_disp),
            pen='r', movable=True, resizable=True, rotatable=False, hoverPen='m'
        )
        self.image_view.addItem(self.aif_roi_object)
        # Connect ROI modification signal to the processing handler
        self.aif_roi_object.sigRegionChangeFinished.connect(self.handle_aif_roi_processing)
        self.log_console.append("AIF ROI created/reset. Adjust it on the image. AIF will update on release.")
        self.handle_aif_roi_processing() # Initial processing with default ROI
        self.save_aif_roi_button.setEnabled(True) # Enable saving the ROI definition

    def handle_save_aif_roi_def(self):
        """Handles saving the current AIF ROI definition (position, size, slice) to a JSON file."""
        if self.aif_roi_object is None or not self.aif_roi_radio.isChecked():
            self.log_console.append("No active AIF ROI to save or ROI mode not selected.")
            return

        roi_state = self.aif_roi_object.getState()
        current_image_key = self.map_selector_combo.currentText() # Image on which ROI was drawn
        if not current_image_key or current_image_key not in self.displayable_volumes:
            self.log_console.append("Cannot determine reference image for AIF ROI. Ensure an image is selected.")
            return

        # ROI coordinates are relative to the displayed slice.
        # Slice index is the original Z index if display is Z-first.
        slice_idx_in_display = self.image_view.currentIndex

        roi_properties = {
            "slice_index": slice_idx_in_display, # This is the Z-index in the original volume if display is (Z,Y,X)
            "pos_x": roi_state['pos'].x(),     # X position in the displayed slice
            "pos_y": roi_state['pos'].y(),     # Y position in the displayed slice
            "size_w": roi_state['size'].x(),   # Width in the displayed slice
            "size_h": roi_state['size'].y(),   # Height in the displayed slice
            "image_ref_name": current_image_key # Name of the image the ROI was drawn on
        }

        filepath, _ = QFileDialog.getSaveFileName(self, "Save AIF ROI Definition", "", "JSON files (*.json)")
        if filepath:
            try:
                aif.save_aif_roi_definition(roi_properties, filepath)
                self.log_console.append(f"AIF ROI definition saved to {filepath}")
            except Exception as e:
                self.log_console.append(f"Error saving AIF ROI definition: {e}\n{traceback.format_exc()}")

    def handle_load_aif_roi_def(self):
        """Handles loading an AIF ROI definition from a JSON file and applies it."""
        if not self.aif_roi_radio.isChecked():
            self.aif_roi_radio.setChecked(True) # Switch to ROI mode
            QApplication.processEvents() # Ensure UI updates if mode change triggers other handlers

        filepath, _ = QFileDialog.getOpenFileName(self, "Load AIF ROI Definition", "", "JSON files (*.json)")
        if filepath:
            try:
                roi_props = aif.load_aif_roi_definition(filepath)
                if roi_props is None: # Should raise error in load_aif_roi_definition itself
                    self.log_console.append("Failed to load or parse AIF ROI definition file.")
                    return

                ref_img_name = roi_props.get("image_ref_name")
                # Check if the reference image for the ROI is loaded
                if ref_img_name not in self.displayable_volumes:
                    self.log_console.append(f"Required reference image '{ref_img_name}' for AIF ROI not currently loaded.")
                    return

                # Switch display to the reference image if not already selected
                if self.map_selector_combo.currentText() != ref_img_name:
                    idx = self.map_selector_combo.findText(ref_img_name)
                    if idx != -1:
                        self.map_selector_combo.setCurrentIndex(idx)
                        QApplication.processEvents() # Allow display to update
                    else:
                        self.log_console.append(f"Could not select reference image '{ref_img_name}' in map selector.")
                        return

                current_image_item = self.image_view.getImageItem()
                if current_image_item is None or current_image_item.image is None:
                    self.log_console.append(f"Reference image '{ref_img_name}' is selected but not displayed.")
                    return

                # Set the slice index
                slice_to_set = roi_props.get("slice_index", 0)
                if not (0 <= slice_to_set < self.image_view.image.shape[0]): # Check against current display's Z-dim
                    self.log_console.append(f"Warning: Saved slice index {slice_to_set} is out of bounds for the current display. Using current slice: {self.image_view.currentIndex}.")
                    slice_to_set = self.image_view.currentIndex
                self.image_view.setCurrentIndex(slice_to_set) # This will trigger handle_slice_changed
                self.slice_slider.setValue(slice_to_set) # Ensure slider reflects this
                QApplication.processEvents()

                # Remove existing AIF ROI if any
                if self.aif_roi_object and self.image_view.getView():
                    if self.aif_roi_object in self.image_view.getView().items:
                        self.image_view.removeItem(self.aif_roi_object)
                    self.aif_roi_object = None

                # Create and add the new ROI from loaded properties
                pos_roi = (roi_props["pos_x"], roi_props["pos_y"])
                size_roi = (roi_props["size_w"], roi_props["size_h"])
                self.aif_roi_object = pg.RectROI(
                    pos=pos_roi, size=size_roi, pen='r',
                    movable=True, resizable=True, rotatable=False, hoverPen='m'
                )
                self.image_view.addItem(self.aif_roi_object)
                self.aif_roi_object.sigRegionChangeFinished.connect(self.handle_aif_roi_processing)
                self.handle_aif_roi_processing() # Process the newly loaded ROI
                self.log_console.append(f"AIF ROI definition loaded from {filepath} and applied.")
                self.save_aif_roi_button.setEnabled(True)
            except Exception as e:
                self.log_console.append(f"Error loading AIF ROI definition: {e}\n{traceback.format_exc()}")
                if self.aif_roi_object and self.image_view.getView(): # Clean up partial ROI if error
                    if self.aif_roi_object in self.image_view.getView().items:
                         self.image_view.removeItem(self.aif_roi_object)
                    self.aif_roi_object = None
                self.save_aif_roi_button.setEnabled(False)

    # --- Data Loading and Initial Processing ---
+   def handle_aif_roi_processing(self):
+       """
+       Processes the AIF ROI when its region is changed or finalized.
+       Extracts the mean signal from the ROI in the DCE data, converts it to
+       concentration, and sets this as the current AIF.
+       """
        if self.aif_roi_object is None or self.dce_data is None:
            self.log_console.append("AIF ROI or DCE data not available for processing AIF from ROI.")
            return

        roi_state = self.aif_roi_object.getState()
        # ROI coordinates from pyqtgraph are in display coordinates (X_disp, Y_disp)
        # X_disp corresponds to original X, Y_disp corresponds to original Y for current Z slice.
        # The image_view might display data transposed, e.g. (Z, Y, X) from original (X, Y, Z, T)
        # So, roi_state['pos'].x() is effectively X in the displayed slice, roi_state['pos'].y() is Y.
        x_roi_disp = int(round(roi_state['pos'].x()))
        y_roi_disp = int(round(roi_state['pos'].y()))
        w_roi_disp = int(round(roi_state['size'].x()))
        h_roi_disp = int(round(roi_state['size'].y()))

        # The Z index is the current slice index in the image_view (which is original Z)
        z_orig_slice = self.image_view.currentIndex

        # Validate ROI coordinates against the *original* DCE data dimensions.
        # Assuming dce_data is (X, Y, Z, Time)
        # And display is (Z, Y, X) so x_roi_disp -> X_orig, y_roi_disp -> Y_orig
        # This part requires careful coordinate mapping if display is not direct.
        # For now, assume direct mapping for simplicity if display is Z,Y,X:
        # x_roi_disp is X_orig, y_roi_disp is Y_orig.
        # The dce_data.shape[0] is X_max, dce_data.shape[1] is Y_max.
        if not (0 <= x_roi_disp < self.dce_data.shape[0] and \
                0 <= y_roi_disp < self.dce_data.shape[1] and \
                x_roi_disp + w_roi_disp <= self.dce_data.shape[0] and \
                y_roi_disp + h_roi_disp <= self.dce_data.shape[1]):
            self.log_console.append(f"AIF ROI is outside original data boundaries. Adjust ROI.")
            return
        if w_roi_disp <= 0 or h_roi_disp <= 0:
            self.log_console.append("AIF ROI width or height is zero/negative. Adjust ROI.")
            return

        # Coordinates for core.aif.extract_aif_from_roi are (x_start, y_start, width, height) in original volume space
        roi_2d_coords_orig = (x_roi_disp, y_roi_disp, w_roi_disp, h_roi_disp)

        try:
            # Get parameters for AIF extraction from UI fields
            t10_b_str = self.aif_t10_blood_input.text()
            r1_b_str = self.aif_r1_blood_input.text()
            tr_val_str = self.tr_input.text() # TR from general settings
            aif_baseline_pts_str = self.aif_baseline_points_input.text()

            if not all([t10_b_str, r1_b_str, tr_val_str, aif_baseline_pts_str]):
                self.log_console.append("One or more AIF ROI parameters (T10 blood, r1 blood, TR, AIF baseline pts) are empty.")
                return

            t10_b = float(t10_b_str)
            r1_b = float(r1_b_str)
            tr_val = float(tr_val_str) # This should be self.tr_float if already validated
            aif_baseline_pts = int(aif_baseline_pts_str)

            if t10_b <= 0 or r1_b <= 0 or tr_val <= 0 or aif_baseline_pts <= 0:
                self.log_console.append("AIF ROI parameters (T10 blood, r1 blood, TR, AIF baseline pts) must be positive.")
                return

            self.log_console.append(f"Processing AIF from ROI: Slice Z={z_orig_slice}, Coords (orig X,Y)=({x_roi_disp},{y_roi_disp}), Size=({w_roi_disp},{h_roi_disp})")

            # Extract AIF using the core function
            self.aif_time, self.aif_concentration = aif.extract_aif_from_roi(
                self.dce_data, roi_2d_coords_orig, z_orig_slice,
                t10_b, r1_b, tr_val, aif_baseline_pts
            )

            if self._create_aif_interpolators():
                self.aif_status_label.setText(f"AIF from ROI (Z={z_orig_slice}) processed. Points: {len(self.aif_time)}")
                self.log_console.append(f"AIF extracted from ROI. Points: {len(self.aif_time)}. Plotting AIF.")
                # Plot the newly derived AIF
                self.plot_widget.clear()
                self.plot_widget.plot(self.aif_time, self.aif_concentration, pen='r', name='AIF (ROI)')
                self.plot_widget.autoRange()
                self.save_plot_button.setEnabled(True)
            else:
                self.aif_time, self.aif_concentration = None, None # Reset on failure
                self.aif_status_label.setText("AIF: Error processing ROI AIF.")
        except ValueError as ve:
            self.log_console.append(f"Invalid AIF ROI parameters: {ve}")
        except Exception as e:
            self.log_console.append(f"Error processing AIF from ROI: {e}\n{traceback.format_exc()}")
        self.update_aif_ui_state() # Update button states, etc.

    # --- Model Selection and Fitting Methods ---
    def handle_model_selection(self):
        """
        Handles changes in PK model selection via radio buttons.
        Updates `self.selected_model_name` and the state of map export buttons.
        """
        self.selected_model_name = None # Default to no model
        if self.model_standard_tofts_radio.isChecked():
            self.selected_model_name = "Standard Tofts"
        elif self.model_extended_tofts_radio.isChecked():
            self.selected_model_name = "Extended Tofts"
        elif self.model_patlak_radio.isChecked():
            self.selected_model_name = "Patlak"
        elif self.model_2cxm_radio.isChecked():
            self.selected_model_name = "2CXM"

        self.log_console.append(f"PK Model selected: {self.selected_model_name if self.selected_model_name else 'None'}")
        self.update_export_buttons_state()

    def run_analysis(self):
        """
        Executes the main analysis pipeline:
        1. Signal-to-Concentration (S->C) conversion if DCE data and T1 map are loaded.
        2. Voxel-wise fitting of the selected pharmacokinetic model if a model is chosen
           and S->C conversion was successful.
        Updates the display with generated concentration maps and parameter maps.
        """
        self.log_console.append("Run Analysis button clicked.")
        self.display_label.setText("Processing... See log for details.") # User feedback
        QApplication.processEvents() # Ensure UI updates

        # --- Validate Inputs ---
        if self.dce_data is None or self.t10_data is None:
            self.log_console.append("Error: DCE data and T1 map must be loaded before running analysis.")
            self.display_label.setText("Analysis failed: DCE or T1 data missing.")
            return
        if self.aif_time is None or self.aif_concentration is None:
            self.log_console.append("Error: AIF (Arterial Input Function) not defined or loaded.")
            self.display_label.setText("Analysis failed: AIF not defined.")
            return
        if not self._create_aif_interpolators(): # Ensure AIF interpolators are ready
            self.log_console.append("Error: Failed to create AIF interpolators. Cannot run analysis.")
            self.display_label.setText("Analysis failed: AIF interpolation error.")
            return

        try:
            # Get conversion parameters from UI
            r1_val = float(self.r1_input.text())
            # TR value for conversion should be the validated self.tr_float
            if self.tr_float is None or self.tr_float <=0:
                raise ValueError("TR value for conversion is not valid or not set. Please check TR input.")
            tr_val_run = self.tr_float
            baseline_pts = int(self.baseline_points_input.text())
            
            if r1_val <= 0 or baseline_pts <= 0: # tr_val_run already checked by self.tr_float logic
                raise ValueError("r1 relaxivity and baseline points must be positive.")
            if baseline_pts >= self.dce_data.shape[3]:
                raise ValueError("Number of baseline points exceeds total time points in DCE data.")
        except ValueError as e:
            self.log_console.append(f"Error: Invalid S->C conversion parameters: {e}")
            self.display_label.setText(f"Analysis failed: Invalid conversion parameters ({e}).")
            return

        # --- 1. Signal-to-Concentration Conversion ---
        self.log_console.append(f"Starting Signal-to-Concentration conversion with r1={r1_val}, TR={tr_val_run}, baseline_pts={baseline_pts}")
        QApplication.processEvents()
        try:
            self.Ct_data = conversion.signal_to_concentration(
                self.dce_data, self.t10_data, r1_val, tr_val_run, baseline_pts
            )
            self.log_console.append(f"S->C conversion successful. Ct_data shape: {self.Ct_data.shape}")
            self.display_label.setText(f"S->C conversion successful. Ct data shape: {self.Ct_data.shape}")
            # Display the mean concentration map
            self.update_displayable_volume("Ct (Concentration Mean)", np.mean(self.Ct_data, axis=3))
        except Exception as e:
            self.log_console.append(f"Error during S->C conversion: {e}\n{traceback.format_exc()}")
            self.display_label.setText("S->C Conversion failed. See log.")
            self.Ct_data = None # Ensure Ct_data is None if conversion fails
            return # Stop analysis if conversion fails
        QApplication.processEvents()

        # --- 2. Voxel-wise Model Fitting ---
        if not self.selected_model_name:
            self.log_console.append("No model selected. Skipping voxel-wise fitting.")
            self.display_label.setText("S->C conversion done. No model selected for fitting.")
            return
        if self.Ct_data is None: # Should be caught by previous error handling, but as a safeguard
            self.log_console.append("Ct_data not available (e.g. S->C failed). Skipping model fitting.")
            self.display_label.setText("Ct data not available. Model fitting skipped.")
            return

        num_cores_to_use = self.num_processes_input.value()
        self.log_console.append(f"Starting parallel voxel-wise {self.selected_model_name} model fitting using up to {num_cores_to_use} cores...")
        self.display_label.setText(f"Fitting {self.selected_model_name} voxel-wise (up to {num_cores_to_use} cores)... This may take a while.")
        QApplication.processEvents()

        t_tissue = self.dce_time_vector # Time vector for tissue curves
        if t_tissue is None:
             # This should ideally not happen if S->C used a valid TR that also set dce_time_vector
             if self.tr_float and self.tr_float > 0:
                 t_tissue = np.arange(self.Ct_data.shape[3]) * self.tr_float
                 self.log_console.append("Warning: dce_time_vector was None, re-created for fitting.")
             else:
                 self.log_console.append("Error: Cannot determine t_tissue for fitting (TR invalid or DCE time vector not set).")
                 self.display_label.setText("Model fitting failed: Tissue time vector unknown.")
                 return

        mask_to_use = self.mask_data if self.mask_data is not None else None # Use mask if loaded
        self.parameter_maps = {} # Reset or initialize parameter maps dictionary

        try:
            # Call the appropriate voxel-wise fitting function based on selected model
            if self.selected_model_name == "Standard Tofts":
                self.parameter_maps = modeling.fit_standard_tofts_voxelwise(self.Ct_data, t_tissue, self.aif_time, self.aif_concentration, mask=mask_to_use, num_processes=num_cores_to_use)
            elif self.selected_model_name == "Extended Tofts":
                self.parameter_maps = modeling.fit_extended_tofts_voxelwise(self.Ct_data, t_tissue, self.aif_time, self.aif_concentration, mask=mask_to_use, num_processes=num_cores_to_use)
            elif self.selected_model_name == "Patlak":
                self.parameter_maps = modeling.fit_patlak_model_voxelwise(self.Ct_data, t_tissue, self.aif_time, self.aif_concentration, mask=mask_to_use, num_processes=num_cores_to_use)
            elif self.selected_model_name == "2CXM":
                self.parameter_maps = modeling.fit_2cxm_model_voxelwise(self.Ct_data, t_tissue, self.aif_time, self.aif_concentration, mask=mask_to_use, num_processes=num_cores_to_use)

            self.log_console.append(f"Parallel voxel-wise {self.selected_model_name} fitting completed.")
            self.display_label.setText(f"{self.selected_model_name} fitting done. Maps generated: {', '.join(self.parameter_maps.keys())}")

            # Add generated parameter maps to displayable volumes
            for map_name, map_data in self.parameter_maps.items():
                self.update_displayable_volume(map_name, map_data)
            self.update_export_buttons_state() # Enable relevant export buttons
        except Exception as e:
            self.log_console.append(f"Error during voxel-wise {self.selected_model_name} fitting: {e}\n{traceback.format_exc()}")
            self.display_label.setText(f"Voxel-wise {self.selected_model_name} fitting failed. See log.")
        QApplication.processEvents() # Final UI update

    def update_export_buttons_state(self):
        """
        Updates the enabled/disabled state of parameter map export buttons
        based on the currently selected model and available parameter maps.
        """
        # Determine which model is active
        is_std_tofts = self.selected_model_name == "Standard Tofts"
        is_ext_tofts = self.selected_model_name == "Extended Tofts"
        is_patlak = self.selected_model_name == "Patlak"
        is_2cxm = self.selected_model_name == "2CXM"

        # Enable/disable Tofts related export buttons
        self.export_ktrans_button.setEnabled("Ktrans" in self.parameter_maps and (is_std_tofts or is_ext_tofts))
        self.export_ve_button.setEnabled("ve" in self.parameter_maps and (is_std_tofts or is_ext_tofts))
        self.export_vp_button.setEnabled("vp" in self.parameter_maps and is_ext_tofts) # vp only for Extended Tofts

        # Enable/disable Patlak related export buttons
        self.export_ktrans_patlak_button.setEnabled("Ktrans_patlak" in self.parameter_maps and is_patlak)
        self.export_vp_patlak_button.setEnabled("vp_patlak" in self.parameter_maps and is_patlak)

        # Enable/disable 2CXM related export buttons
        self.export_fp_2cxm_button.setEnabled("Fp_2cxm" in self.parameter_maps and is_2cxm)
        self.export_ps_2cxm_button.setEnabled("PS_2cxm" in self.parameter_maps and is_2cxm)
        self.export_vp_2cxm_button.setEnabled("vp_2cxm" in self.parameter_maps and is_2cxm)
        self.export_ve_2cxm_button.setEnabled("ve_2cxm" in self.parameter_maps and is_2cxm)

    def export_map(self, map_name: str):
        """
        Handles exporting a specified parameter map to a NIfTI file.
        Uses the loaded T1 map or DCE series as a reference for affine/header.

        Args:
            map_name (str): The name of the parameter map to export (e.g., "Ktrans", "ve").
        """
        self.log_console.append(f"Export map button clicked for: {map_name}")
        param_map_data = self.parameter_maps.get(map_name)
        if param_map_data is None:
            self.log_console.append(f"Error: Parameter map '{map_name}' not available for export.")
            return

        # Determine a reference NIfTI file for saving (to copy affine, header info)
        # Prioritize T1 map if loaded, otherwise use original DCE file.
        reference_nifti_path = self.t1_filepath if self.t1_filepath else self.dce_filepath
        if not reference_nifti_path:
            self.log_console.append("Error: No reference NIfTI file (T1 map or DCE series) loaded. Cannot save map with proper orientation.")
            return

        default_filename = f"{map_name}_map.nii.gz" # Default filename for saved map
        output_filepath, _ = QFileDialog.getSaveFileName(
            self, f"Save {map_name} Map", default_filename, "NIfTI files (*.nii *.nii.gz)"
        )
        if output_filepath:
            try:
                self.log_console.append(f"Saving {map_name} map to: {output_filepath} using reference NIfTI: {reference_nifti_path}")
                io.save_nifti_map(param_map_data, reference_nifti_path, output_filepath)
                self.log_console.append(f"Parameter map '{map_name}' saved successfully to {output_filepath}.")
                self.display_label.setText(f"{map_name} map saved to {os.path.basename(output_filepath)}")
            except Exception as e:
                self.log_console.append(f"Error saving {map_name} map: {e}\n{traceback.format_exc()}")
                self.display_label.setText(f"Error saving {map_name} map. See log.")

    # --- Display Update Methods ---
    def update_displayable_volume(self, name: str, data: np.ndarray):
        """
        Adds or updates a volume in `self.displayable_volumes` and refreshes the
        map selector combo boxes (for base image and overlay).

        If the added/updated volume is currently selected, the display is refreshed.

        Args:
            name (str): The name/key for the volume (e.g., "Ktrans", "Original DCE (Mean)").
            data (np.ndarray): The 3D or 4D NumPy array data of the volume.
        """
        if data is None:
            self.log_console.append(f"Attempted to update displayable volume '{name}' with None data.")
            return

        self.displayable_volumes[name] = data
        self.log_console.append(f"Volume '{name}' added/updated in displayable volumes. Shape: {data.shape}")

        # Store current selections to try and restore them after repopulating combo boxes
        current_base_selection = self.map_selector_combo.currentText()
        current_overlay_selection = self.overlay_map_selector_combo.currentText()

        # Block signals to prevent premature updates while modifying combo boxes
        self.map_selector_combo.blockSignals(True)
        self.overlay_map_selector_combo.blockSignals(True)

        # Repopulate base image selector
        self.map_selector_combo.clear()
        self.map_selector_combo.addItems(self.displayable_volumes.keys())
        idx_base = self.map_selector_combo.findText(name) # Try to select the newly added/updated map
        if idx_base != -1:
            self.map_selector_combo.setCurrentIndex(idx_base)
        elif current_base_selection and self.map_selector_combo.findText(current_base_selection) != -1:
            # Restore previous selection if the new map is not the one to be selected
            self.map_selector_combo.setCurrentText(current_base_selection)

        # Repopulate overlay image selector (only 3D maps suitable for overlay)
        self.overlay_map_selector_combo.clear()
        self.overlay_map_selector_combo.addItem("None") # Always have a "None" option
        for vol_name, vol_data in self.displayable_volumes.items():
            if vol_data.ndim == 3: # Only allow 3D volumes as overlays for simplicity
                self.overlay_map_selector_combo.addItem(vol_name)

        idx_overlay = self.overlay_map_selector_combo.findText(current_overlay_selection)
        if idx_overlay != -1:
            self.overlay_map_selector_combo.setCurrentIndex(idx_overlay)
        else: # Default to "None" if previous selection is no longer valid or was "None"
            self.overlay_map_selector_combo.setCurrentIndex(0)

        # Unblock signals
        self.map_selector_combo.blockSignals(False)
        self.overlay_map_selector_combo.blockSignals(False)

        # If the currently selected map was the one updated, refresh the display.
        # Otherwise, handle_map_selection_changed will be triggered if selection changes.
        if self.map_selector_combo.currentText() == name:
            self.handle_map_selection_changed() # This will also update overlay and stats
        elif self.overlay_map_selector_combo.currentText() == name : # if only overlay was updated
             self.handle_overlay_controls_changed()


    def handle_overlay_controls_changed(self):
        """
        Handles changes from overlay control widgets (map selector, alpha slider, cmap selector).
        Updates overlay properties and refreshes the overlay display.
        """
        self.current_overlay_map_key = self.overlay_map_selector_combo.currentText()
        self.overlay_alpha = self.overlay_alpha_slider.value() / 100.0 # Convert percentage to 0-1 range
        self.overlay_alpha_label.setText(f"{self.overlay_alpha*100:.0f}%") # Update alpha label
        self.overlay_cmap_name = self.overlay_cmap_combo.currentText()

        self.log_console.append(f"Overlay settings changed: Map='{self.current_overlay_map_key}', Alpha={self.overlay_alpha:.2f}, Cmap='{self.overlay_cmap_name}'")
        self.update_overlay_image_display()

    def update_overlay_image_display(self):
        """
        Updates the overlay image item based on current overlay settings
        (selected map, alpha, colormap) and the current slice of the base image.
        """
        if self.current_overlay_map_key is None or self.current_overlay_map_key == "None":
            self.overlay_image_item.clear()
            self.overlay_image_item.setVisible(False)
            return

        overlay_volume_data = self.displayable_volumes.get(self.current_overlay_map_key)
        base_image_item = self.image_view.getImageItem()

        if overlay_volume_data is None or base_image_item is None or base_image_item.image is None:
            self.overlay_image_item.clear()
            self.overlay_image_item.setVisible(False)
            return

        current_slice_idx_display = self.image_view.currentIndex # Z-index in the (Z,Y,X) displayed data

        # Ensure overlay data is 3D and get the correct slice
        if overlay_volume_data.ndim == 3:
            # Assuming overlay_volume_data is (X_orig, Y_orig, Z_orig)
            # And display is (Z_orig, Y_orig, X_orig)
            # So, we need to permute overlay to match display's Z-first convention for slicing.
            overlay_data_permuted_for_slicing = overlay_volume_data.transpose(2,1,0) # Now (Z_orig, Y_orig, X_orig)
        else: # Only 3D overlays are currently supported by this logic
            self.log_console.append(f"Overlay map '{self.current_overlay_map_key}' is not 3D, cannot display as overlay.")
            self.overlay_image_item.clear()
            self.overlay_image_item.setVisible(False)
            return

        if not (0 <= current_slice_idx_display < overlay_data_permuted_for_slicing.shape[0]):
            self.log_console.append(f"Current slice index {current_slice_idx_display} out of bounds for overlay map '{self.current_overlay_map_key}'.")
            self.overlay_image_item.clear()
            self.overlay_image_item.setVisible(False)
            return

        overlay_slice_to_display = overlay_data_permuted_for_slicing[current_slice_idx_display]

        self.overlay_image_item.setImage(overlay_slice_to_display, autoLevels=False) # autoLevels=False to use our own levels

        # Setup colormap and levels
        cmap = pg.colormap.get(self.overlay_cmap_name) # Get pyqtgraph colormap
        lut = cmap.getLookupTable(alpha=True) # Get lookup table with alpha channel

        min_val = np.nanmin(overlay_slice_to_display)
        max_val = np.nanmax(overlay_slice_to_display)

        # If slice is all NaN or flat, try using full volume stats for levels
        if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
            min_val_vol = np.nanmin(overlay_volume_data) # Use original orientation for volume stats
            max_val_vol = np.nanmax(overlay_volume_data)
            if np.isnan(min_val_vol) or np.isnan(max_val_vol) or min_val_vol == max_val_vol:
                # If volume is also flat/NaN, cannot set meaningful levels
                self.overlay_image_item.clear()
                self.overlay_image_item.setVisible(False)
                return
            min_val, max_val = min_val_vol, max_val_vol

        self.overlay_image_item.setLookupTable(lut)
        self.overlay_image_item.setLevels([min_val, max_val])
        self.overlay_image_item.setOpacity(self.overlay_alpha)
        self.overlay_image_item.setVisible(True)

    def handle_map_selection_changed(self):
        """
        Handles changes in the main map selector combo box.
        Updates the image view with the selected volume, configures the slice slider,
        and refreshes overlays and ROI statistics.
        """
        selected_key = self.map_selector_combo.currentText()
        if not selected_key: # No map selected (e.g., if combo is empty)
            self.image_view.clear()
            self.slice_slider.setEnabled(False)
            self.slice_slider_label.setText("Slice: 0/0")
            self.overlay_image_item.clear()
            self.overlay_image_item.setVisible(False)
            self.update_all_rois_stats_display() # Update stats (will show N/A)
            return

        self.current_display_key = selected_key
        volume_data = self.displayable_volumes.get(self.current_display_key)

        if volume_data is None: # Should not happen if combo box is populated correctly
            self.log_console.append(f"Error: Selected volume '{selected_key}' not found in displayable volumes.")
            self.image_view.clear()
            self.slice_slider.setEnabled(False)
            self.slice_slider_label.setText("Slice: 0/0")
            self.overlay_image_item.clear()
            self.overlay_image_item.setVisible(False)
            self.update_all_rois_stats_display()
            return

        # Prepare data for display: pyqtgraph.ImageView expects (Time/Z, Y, X)
        # Original data is (X, Y, Z) for 3D maps or (X, Y, Z, Time) for 4D DCE.
        display_data = None
        if volume_data.ndim == 3: # e.g., T1 map, Ktrans map
            display_data = volume_data.transpose(2, 1, 0) # Transpose to (Z, Y, X) for display
        elif volume_data.ndim == 4: # e.g., Original DCE, Ct_data
            # For 4D data, display the mean over the time dimension for now.
            # Could be changed to display a specific time point if needed.
            mean_over_time = np.mean(volume_data, axis=3)
            display_data = mean_over_time.transpose(2, 1, 0) # Transpose to (Z, Y, X)
        else:
            self.log_console.append(f"Volume '{selected_key}' has unsupported dimension: {volume_data.ndim}. Cannot display.")
            self.image_view.clear()
            self.slice_slider.setEnabled(False)
            self.slice_slider_label.setText("Slice: 0/0")
            self.overlay_image_item.clear()
            self.overlay_image_item.setVisible(False)
            self.update_all_rois_stats_display()
            return

        # Update the image view
        self.image_view.setImage(display_data, autoRange=True, autoLevels=True, autoHistogramRange=True)

        # Configure slice slider
        num_slices = display_data.shape[0] # Number of Z slices
        self.slice_slider.setEnabled(True)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(num_slices - 1)

        # Ensure current index is valid, then set it
        current_idx_display = self.image_view.currentIndex
        if not (0 <= current_idx_display < num_slices):
            current_idx_display = 0 # Default to first slice if current is invalid

        self.image_view.setCurrentIndex(current_idx_display) # Set current slice in view
        self.slice_slider.setValue(current_idx_display) # Sync slider
        self.slice_slider_label.setText(f"Slice: {current_idx_display + 1}/{num_slices}")

        # Update dependent displays
        self.update_overlay_image_display() # Refresh overlay based on new base image/slice
        self.update_all_rois_stats_display() # Refresh ROI stats for the new view

    def handle_slice_changed(self, value: int):
        """
        Handles changes from the slice slider. Updates the currently displayed slice
        in the image view and refreshes ROI statistics and overlays.

        Args:
            value (int): The new slice index from the slider.
        """
        self.current_slice_index = value # Store the new slice index
        if self.image_view.image is not None: # Check if an image is loaded
            num_slices = self.image_view.image.shape[0]
            # Ensure the value is within valid bounds for the current image
            safe_value = np.clip(value, 0, num_slices - 1)

            self.image_view.setCurrentIndex(safe_value) # Update image view to the new slice
            self.slice_slider_label.setText(f"Slice: {safe_value + 1}/{num_slices}")

            # If slider was out of sync due to programmatic change, sync it
            if value != safe_value and self.slice_slider.value() != safe_value :
                 self.slice_slider.setValue(safe_value)

            # Update displays that depend on the current slice
            self.update_all_rois_stats_display()
            self.update_overlay_image_display()

    def handle_voxel_clicked(self, mouse_click_event):
        """
        Handles double-click events on the image view.
        Maps the clicked display coordinates to original volume coordinates and
        triggers plotting of time-courses for that voxel (Ct_data, AIF, model fit).
        """
        if not mouse_click_event.double(): # Only process double-clicks
            return

        self.save_plot_button.setEnabled(False) # Disable save plot button initially

        image_item = self.image_view.getImageItem()
        if image_item is None or image_item.image is None: # No image loaded
            return

        # Map click position from scene coordinates to image coordinates
        scene_pos = mouse_click_event.scenePos()
        img_coords_float = image_item.mapFromScene(scene_pos)

        # Round to nearest integer for voxel indices
        # img_coords_float.y() is row index (Y in display), img_coords_float.x() is col index (X in display)
        y_in_slice_display = int(round(img_coords_float.y()))
        x_in_slice_display = int(round(img_coords_float.x()))
        current_z_index_display = self.image_view.currentIndex # This is the Z index in (Z,Y,X) display

        # Validate click is within current slice boundaries
        current_slice_shape_display = self.image_view.image[current_z_index_display].shape # (Y_disp, X_disp)
        if not (0 <= y_in_slice_display < current_slice_shape_display[0] and \
                0 <= x_in_slice_display < current_slice_shape_display[1]):
            self.log_console.append(f"Clicked outside current slice boundaries (display coords).")
            return

        # Convert display coordinates (Z_disp, Y_disp, X_disp) to original volume coordinates (X_orig, Y_orig, Z_orig)
        # Assuming display data is (Z, Y, X) and original is (X, Y, Z, Time) or (X,Y,Z)
        # Z_orig = Z_disp
        # Y_orig = Y_disp
        # X_orig = X_disp
        # This needs to be accurate based on how data was transposed for display in handle_map_selection_changed
        z_orig = current_z_index_display
        y_orig = y_in_slice_display
        x_orig = x_in_slice_display

        self.log_console.append(f"Image double-clicked at display (X_disp:{x_in_slice_display}, Y_disp:{y_in_slice_display}, Z_disp:{current_z_index_display}). Mapped to original (X:{x_orig}, Y:{y_orig}, Z:{z_orig})")

        # Check if Ct_data is available and click is within its bounds
        if self.Ct_data is None:
            self.log_console.append("Concentration data (Ct_data) not available for plotting.")
            return
        if not (0 <= x_orig < self.Ct_data.shape[0] and \
                0 <= y_orig < self.Ct_data.shape[1] and \
                0 <= z_orig < self.Ct_data.shape[2]):
            self.log_console.append(f"Clicked coordinates ({x_orig},{y_orig},{z_orig}) are outside Ct_data bounds ({self.Ct_data.shape[:3]}).")
            return

        self.plot_selected_voxel_curves(x_orig, y_orig, z_orig)

    def plot_selected_voxel_curves(self, x_idx: int, y_idx: int, z_idx: int):
        """
        Plots the tissue concentration curve (Ct_voxel), AIF, and fitted model curve
        for the specified voxel coordinates (original volume space).

        Args:
            x_idx (int): Original X-coordinate of the voxel.
            y_idx (int): Original Y-coordinate of the voxel.
            z_idx (int): Original Z-coordinate of the voxel.
        """
        self.plot_widget.clear() # Clear previous plot
        self.plot_widget.setTitle(f"Curves for Voxel (X_orig:{x_idx}, Y_orig:{y_idx}, Z_orig:{z_idx})")

        if self.Ct_data is None:
            self.log_console.append("Ct_data not available for plotting voxel curves.")
            self.save_plot_button.setEnabled(False)
            return

        Ct_voxel = self.Ct_data[x_idx, y_idx, z_idx, :] # Get voxel's time course
        t_values = self.dce_time_vector # Get time vector

        # Ensure time vector is available
        if t_values is None:
            # Fallback if dce_time_vector wasn't set (e.g. TR not entered before load)
            try:
                tr_val = float(self.tr_input.text())
                if tr_val <= 0: raise ValueError("TR must be positive.")
                num_time_points = self.Ct_data.shape[3]
                t_values = np.arange(num_time_points) * tr_val
            except ValueError:
                self.log_console.append("TR value invalid or not set. Cannot plot time axis correctly.")
                self.save_plot_button.setEnabled(False)
                return

        # Plot tissue concentration curve
        self.plot_widget.plot(t_values, Ct_voxel, pen=pg.mkPen('b', width=2), name='Tissue Conc. (Ct)')

        # Plot AIF if available
        if self.aif_time is not None and self.aif_concentration is not None:
            self.plot_widget.plot(self.aif_time, self.aif_concentration, pen='r', name='AIF (Cp)')

        # Plot fitted model curve if a model is selected and parameters are available
        if self.selected_model_name and self.parameter_maps and self.Cp_interp_func:
            model_params_for_voxel = {}
            valid_params_found = True
            fitted_curve = None
            param_log_str = "N/A"

            # Retrieve parameters for the selected model for the current voxel
            if self.selected_model_name == "Standard Tofts":
                if "Ktrans" in self.parameter_maps and "ve" in self.parameter_maps:
                    Ktrans_val = self.parameter_maps["Ktrans"][x_idx, y_idx, z_idx]
                    ve_val = self.parameter_maps["ve"][x_idx, y_idx, z_idx]
                    if np.isnan(Ktrans_val) or np.isnan(ve_val): valid_params_found = False
                    else: model_params_for_voxel = {'Ktrans': Ktrans_val, 've': ve_val}
                else: valid_params_found = False
                if valid_params_found:
                    fitted_curve = modeling.standard_tofts_model_conv(t_values, Ktrans_val, ve_val, self.Cp_interp_func)

            elif self.selected_model_name == "Extended Tofts":
                map_keys = ["Ktrans", "ve", "vp"]
                if all(key in self.parameter_maps for key in map_keys):
                    params = [self.parameter_maps[key][x_idx, y_idx, z_idx] for key in map_keys]
                    if any(np.isnan(p) for p in params): valid_params_found = False
                    else: model_params_for_voxel = dict(zip(map_keys, params))
                else: valid_params_found = False
                if valid_params_found:
                    fitted_curve = modeling.extended_tofts_model_conv(t_values, **model_params_for_voxel, Cp_t_interp_func=self.Cp_interp_func)

            elif self.selected_model_name == "Patlak":
                map_keys = ["Ktrans_patlak", "vp_patlak"]
                if all(key in self.parameter_maps for key in map_keys) and self.integral_Cp_dt_interp_func:
                    params = [self.parameter_maps[key][x_idx, y_idx, z_idx] for key in map_keys]
                    if any(np.isnan(p) for p in params): valid_params_found = False
                    else: model_params_for_voxel = dict(zip(map_keys, params))
                else: valid_params_found = False
                if valid_params_found:
                    fitted_curve = modeling.patlak_model(t_values, **model_params_for_voxel, Cp_t_interp_func=self.Cp_interp_func, integral_Cp_dt_interp_func=self.integral_Cp_dt_interp_func)

            elif self.selected_model_name == "2CXM":
                map_keys = ["Fp_2cxm", "PS_2cxm", "vp_2cxm", "ve_2cxm"]
                if all(key in self.parameter_maps for key in map_keys):
                    params = [self.parameter_maps[key][x_idx, y_idx, z_idx] for key in map_keys]
                    if any(np.isnan(p) for p in params): valid_params_found = False
                    else: model_params_for_voxel = dict(zip(map_keys, params))
                else: valid_params_found = False
                if valid_params_found:
                    t_aif_max_for_plot = self.aif_time[-1] if self.aif_time is not None and len(self.aif_time) > 0 else t_values[-1]
                    fitted_curve = modeling.solve_2cxm_ode_model(t_values, **model_params_for_voxel, Cp_aif_interp_func=self.Cp_interp_func, t_span_max=t_aif_max_for_plot)

            # Plot the fitted curve if successful
            if valid_params_found and fitted_curve is not None:
                self.plot_widget.plot(t_values, fitted_curve, pen='g', name=f'{self.selected_model_name} Fit')
                param_log_str = ", ".join([f"{k}={v:.4f}" for k, v in model_params_for_voxel.items()])
                self.log_console.append(f"Plotted model fit for voxel ({x_idx},{y_idx},{z_idx}). Parameters: {param_log_str}")
            elif valid_params_found and fitted_curve is None: # Should ideally not happen if params are valid
                self.log_console.append(f"Model parameters appear valid but curve generation failed for {self.selected_model_name} at ({x_idx},{y_idx},{z_idx}).")
            else:
                self.log_console.append(f"No valid pre-fitted parameters found for voxel ({x_idx},{y_idx},{z_idx}) for model {self.selected_model_name}.")

        self.plot_widget.autoRange() # Adjust plot axes
        # Enable save plot button only if there's something plotted
        self.save_plot_button.setEnabled(bool(self.plot_widget.getPlotItem().listDataItems()))

    def handle_save_plot(self):
        """Handles saving the current content of the plot widget to an image file."""
        if self.plot_widget.getPlotItem().listDataItems(): # Check if there's something plotted
            try:
                # Use pyqtgraph's ImageExporter for saving
                exporter = pg_exporters.ImageExporter(self.plot_widget.getPlotItem())
                # Open file dialog for user to choose save location and format
                filepath, selected_filter = QFileDialog.getSaveFileName(
                    self, "Save Plot", "",
                    "PNG files (*.png);;JPEG files (*.jpg *.jpeg);;SVG files (*.svg);;TIFF files (*.tif);;All files (*)"
                )
                if filepath: # If a file path was chosen
                    exporter.export(filepath)
                    self.log_console.append(f"Plot saved to {filepath}")
            except ImportError: # ImageExporter might not be available in all pyqtgraph installs
                self.log_console.append("Error: PyQtGraph ImageExporter not found. Please ensure pyqtgraph is fully installed, possibly with pyqtgraph[imageio] or similar.")
            except Exception as e:
                self.log_console.append(f"Error saving plot: {e}\n{traceback.format_exc()}")
        else:
            self.log_console.append("No plot available to save.")

    # --- Statistics ROI Handling ---
    def handle_add_stats_roi(self):
        """
        Handles the 'Add Stats ROI' button click.
        Creates a new rectangular ROI item and adds it to the image view and
        `self.stats_roi_list`. Each ROI is assigned a unique name and a distinct color.
        The ROI's statistics are updated when its region is changed.
        """
        current_img_item = self.image_view.getImageItem()
        if current_img_item is None or current_img_item.image is None:
            self.log_console.append("No image displayed to draw Statistics ROI on.")
            return

        self.stats_roi_counter += 1 # Increment for unique ROI name
        roi_name = f"StatsROI_{self.stats_roi_counter}"

        # Default ROI position and size (similar to AIF ROI)
        view_data_shape = current_img_item.image[self.image_view.currentIndex].shape # (Y_disp, X_disp)
        roi_y_disp = view_data_shape[0] // 4
        roi_x_disp = view_data_shape[1] // 4
        roi_h_disp = view_data_shape[0] // 2
        roi_w_disp = view_data_shape[1] // 2

        # Cycle through a list of colors for new ROIs
        roi_colors = ['#32CD32', '#FFFF00', '#00FFFF', '#FF00FF', '#FFA500', '#DA70D6'] # Lime, Yellow, Cyan, Magenta, Orange, Orchid
        pen_color = roi_colors[(len(self.stats_roi_list)) % len(roi_colors)]

        new_roi_item = pg.RectROI(
            pos=(roi_x_disp, roi_y_disp), size=(roi_w_disp, roi_h_disp),
            pen=pg.mkPen(pen_color, width=2),
            movable=True, resizable=True, hoverPen=pg.mkPen(pen_color, width=3), rotatable=False
        )
        self.image_view.addItem(new_roi_item)

        # Store ROI information along with its pyqtgraph item
        roi_entry = {
            'item': new_roi_item,
            'name': roi_name,
            'stats': None, # To be calculated
            'slice_idx': self.image_view.currentIndex, # Store current slice context
            'map_name': self.map_selector_combo.currentText() # And current map context
        }
        self.stats_roi_list.append(roi_entry)

        # Connect signal for when user finishes changing the ROI
        # Use functools.partial to pass the specific roi_entry to the handler
        new_roi_item.sigRegionChangeFinished.connect(
            functools.partial(self.handle_specific_stats_roi_updated, roi_entry)
        )

        self.handle_specific_stats_roi_updated(roi_entry) # Calculate initial stats
        self.update_all_rois_stats_display() # Refresh the text display
        self.save_stats_button.setEnabled(True) # Enable save button as there's now an ROI
        self.log_console.append(f"Added {roi_name} to slice {roi_entry['slice_idx']} of map '{roi_entry['map_name']}'.")

    def handle_specific_stats_roi_updated(self, roi_entry_to_update: dict):
        """
        Callback for when a specific statistics ROI is moved or resized.
        Recalculates and stores statistics for this ROI based on its new state and
        the currently displayed map slice.

        Args:
            roi_entry_to_update (dict): The dictionary entry from `self.stats_roi_list`
                                        corresponding to the ROI that was updated.
        """
        roi_item_changed = roi_entry_to_update['item']
        current_map_name_view = self.map_selector_combo.currentText()
        current_slice_idx_view = self.image_view.currentIndex # Z-index of displayed slice

        # Update the ROI's context if it has changed (e.g., user changed slice/map after drawing)
        roi_entry_to_update['map_name'] = current_map_name_view
        roi_entry_to_update['slice_idx'] = current_slice_idx_view

        img_item = self.image_view.getImageItem()
        if img_item is None or img_item.image is None: # No base image
            roi_entry_to_update['stats'] = None
            self.update_all_rois_stats_display()
            return

        # Get the original 3D volume data for the map this ROI is associated with
        original_volume_data = self.displayable_volumes.get(roi_entry_to_update['map_name'])
        if original_volume_data is None or original_volume_data.ndim != 3:
            # ROI is for a map not currently available or not 3D (e.g. mean of 4D)
            roi_entry_to_update['stats'] = None
            self.log_console.append(f"Stats for {roi_entry_to_update['name']}: Associated map '{roi_entry_to_update['map_name']}' not found or not 3D.")
            self.update_all_rois_stats_display()
            return

        # Extract the specific 2D slice from the original 3D volume
        # original_volume_data is (X,Y,Z), roi_entry_to_update['slice_idx'] is Z_orig
        current_slice_data_original_orientation = original_volume_data[:, :, roi_entry_to_update['slice_idx']]

        # Get ROI geometry (in display coordinates of the current slice)
        roi_state = roi_item_changed.getState()
        # These are X_disp, Y_disp coordinates on the slice
        x_start_disp = int(round(roi_state['pos'].x()))
        y_start_disp = int(round(roi_state['pos'].y()))
        w_disp = int(round(roi_state['size'].x()))
        h_disp = int(round(roi_state['size'].y()))

        # Create a 2D boolean mask for the ROI on the *original orientation* slice
        # Display is (Y_disp, X_disp), Original slice data is (X_orig, Y_orig)
        # So, x_start_disp -> X_orig, y_start_disp -> Y_orig
        slice_cols_orig, slice_rows_orig = current_slice_data_original_orientation.shape # (X_orig_max, Y_orig_max)

        roi_mask_on_slice = np.zeros_like(current_slice_data_original_orientation, dtype=bool)

        # Clip ROI coordinates to be within the slice boundaries (original orientation)
        x_start_clipped_orig = max(0, x_start_disp)
        y_start_clipped_orig = max(0, y_start_disp)
        x_end_clipped_orig = min(x_start_disp + w_disp, slice_cols_orig)
        y_end_clipped_orig = min(y_start_disp + h_disp, slice_rows_orig)

        if y_start_clipped_orig < y_end_clipped_orig and x_start_clipped_orig < x_end_clipped_orig:
            # Apply mask: Note that numpy array indexing is (row, col) -> (Y_orig, X_orig)
            roi_mask_on_slice[x_start_clipped_orig:x_end_clipped_orig, y_start_clipped_orig:y_end_clipped_orig] = True

        calculated_stats = reporting.calculate_roi_statistics(current_slice_data_original_orientation, roi_mask_on_slice)
        roi_entry_to_update['stats'] = calculated_stats

        # This function is often called by others, so delay full display update to avoid recursion if not needed.
        # self.update_all_rois_stats_display() # Can be called by the caller if needed.
        self.log_console.append(f"Updated stats for {roi_entry_to_update['name']} on map '{roi_entry_to_update['map_name']}' slice {roi_entry_to_update['slice_idx']}.")


    def update_all_rois_stats_display(self):
        """
        Updates the text display area with formatted statistics for all defined ROIs.
        It recalculates statistics for ROIs visible on the current map and slice.
        """
        full_stats_text = []
        current_map_name_view = self.map_selector_combo.currentText()
        current_slice_idx_view = self.image_view.currentIndex
        any_roi_has_valid_stats = False

        for roi_entry in self.stats_roi_list:
            # If ROI is on the currently viewed map and slice, ensure its stats are up-to-date
            # This is important if the underlying map data changed or ROI was just drawn
            if roi_entry['map_name'] == current_map_name_view and \
               roi_entry['slice_idx'] == current_slice_idx_view:
                self.handle_specific_stats_roi_updated(roi_entry) # Recalculate for this visible ROI

            if roi_entry['stats']: # If stats have been calculated
                formatted_str = reporting.format_roi_statistics_to_string(
                    roi_entry['stats'], roi_entry['map_name'], roi_entry['name']
                )
                full_stats_text.append(formatted_str)
                if roi_entry['stats'].get("N_valid", 0) > 0: # Check if there are valid data points
                    any_roi_has_valid_stats = True
            else:
                # For ROIs not on current view or with no stats
                full_stats_text.append(f"{roi_entry['name']}: Defined on map '{roi_entry['map_name']}' slice {roi_entry['slice_idx']}. (Stats not calculated or N/A for current view)")

        self.stats_results_display.setText("\n\n".join(full_stats_text) if full_stats_text else "No ROIs defined. Use 'Add Stats ROI' to create one.")
        self.save_stats_button.setEnabled(any_roi_has_valid_stats) # Enable save only if there's something to save

    def handle_clear_last_stats_roi(self):
        """Removes the most recently added statistics ROI from the list and the view."""
        if self.stats_roi_list:
            last_roi_entry = self.stats_roi_list.pop()
            # Remove from image view if the item still exists
            if last_roi_entry['item'] and self.image_view.getView() and last_roi_entry['item'] in self.image_view.getView().items:
                self.image_view.removeItem(last_roi_entry['item'])
            self.log_console.append(f"Removed {last_roi_entry['name']}.")
            self.update_all_rois_stats_display()
        else:
            self.log_console.append("No Stats ROIs to clear.")

        if not self.stats_roi_list: # Disable save button if no ROIs are left
            self.save_stats_button.setEnabled(False)

    def handle_clear_all_stats_rois(self):
        """Removes all statistics ROIs from the list and the view."""
        for roi_entry in self.stats_roi_list:
            if roi_entry['item'] and self.image_view.getView() and roi_entry['item'] in self.image_view.getView().items:
                self.image_view.removeItem(roi_entry['item'])
        self.stats_roi_list.clear()
        self.stats_roi_counter = 0 # Reset counter
        self.log_console.append("Cleared all Statistics ROIs.")
        self.update_all_rois_stats_display()
        self.save_stats_button.setEnabled(False) # No ROIs, so disable save

    def handle_save_all_roi_stats(self):
        """
        Saves statistics for all ROIs that have valid data to a single CSV file.
        """
        stats_to_save = []
        for roi_entry in self.stats_roi_list:
            # Only include ROIs that have successfully calculated statistics with valid data points
            if roi_entry['stats'] and roi_entry['stats'].get("N_valid", 0) > 0:
                stats_to_save.append((
                    roi_entry['map_name'],
                    roi_entry['slice_idx'],
                    roi_entry['name'],
                    roi_entry['stats']
                ))

        if not stats_to_save:
            self.log_console.append("No valid ROI statistics available from any view to save.")
            return

        # Suggest a filename based on the first ROI's map name
        first_map_name = stats_to_save[0][0].replace(' ','_').replace('(','').replace(')','') # Sanitize map name for filename
        default_filename = f"all_roi_stats_from_{first_map_name}.csv"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save All Valid ROI Statistics", default_filename, "CSV files (*.csv)"
        )
        if filepath:
            try:
                reporting.save_multiple_roi_statistics_csv(stats_to_save, filepath)
                self.log_console.append(f"All valid ROI statistics saved to {filepath}")
            except Exception as e:
                self.log_console.append(f"Error saving ROI statistics: {e}\n{traceback.format_exc()}")

    # --- Data Loading and File Handling ---
    def handle_tr_changed(self, text: str):
        """
        Handles changes in the TR (Repetition Time) input field.
        Validates the entered TR value, updates the `self.tr_float` attribute,
        and recalculates `self.dce_time_vector` if DCE data is already loaded.

        Args:
            text (str): The text from the TR input field.
        """
        try:
            tr_val = float(text)
            if tr_val <= 0:
                self.tr_float = None # Store invalid TR as None
                self.log_console.append("TR must be a positive number.")
                self.dce_time_vector = None # Invalidate time vector if TR is invalid
            else:
                self.tr_float = tr_val # Store valid TR
                self.log_console.append(f"TR set to: {self.tr_float} s")
                # If DCE data is already loaded, update its time vector
                if self.dce_data is not None:
                    self.dce_time_vector = np.arange(self.dce_data.shape[3]) * self.tr_float
                    self.log_console.append(f"DCE time vector updated based on new TR: {len(self.dce_time_vector)} points.")
        except ValueError:
            # Handle cases where text is not a valid float
            self.tr_float = None
            self.dce_time_vector = None
            # Optional: Log this, but can be noisy if user is typing
            # self.log_console.append("Invalid TR value entered. Please enter a number.")

    def load_dce_file(self):
        """
        Handles the 'Load DCE Series' button click.
        Opens a file dialog for selecting a NIfTI file, loads it as a 4D DCE series,
        updates relevant internal attributes (e.g., `self.dce_data`, `self.dce_time_vector`),
        and displays the mean of the series in the image view.
        """
        filepath, _ = QFileDialog.getOpenFileName(self, "Load DCE NIfTI File", "", "NIfTI Files (*.nii *.nii.gz);;All Files (*)")
        if filepath:
            self.dce_filepath = filepath # Store the path
            try:
                self.log_console.append(f"Loading DCE series: {filepath}")
                self.dce_data = io.load_dce_series(filepath) # Load using core.io
                self.dce_shape_for_validation = self.dce_data.shape # Store shape for validating other inputs
                self.dce_path_label.setText(os.path.basename(filepath)) # Update UI label
                self.log_console.append(f"DCE series loaded. Shape: {self.dce_data.shape}")

                # Add the mean of the DCE series (over time) as a displayable volume
                self.update_displayable_volume("Original DCE (Mean)", np.mean(self.dce_data, axis=3))

                # Define time vector if TR is valid
                if self.tr_float is not None and self.tr_float > 0:
                    self.dce_time_vector = np.arange(self.dce_data.shape[3]) * self.tr_float
                    self.log_console.append(f"DCE time vector defined: {len(self.dce_time_vector)} points, using TR={self.tr_float}s.")
                else:
                    self.dce_time_vector = None # Explicitly set to None if TR is not valid
                    self.log_console.append("TR not set or invalid at DCE load. DCE time vector not defined. Please set a valid TR.")
            except Exception as e:
                # Reset relevant attributes on error
                self.dce_data, self.dce_shape_for_validation, self.dce_time_vector, self.dce_filepath = None, None, None, None
                self.dce_path_label.setText("Error loading DCE file.")
                self.log_console.append(f"Error loading DCE series: {e}\n{traceback.format_exc()}")

    def load_t1_file(self):
        """
        Handles the 'Load T1 Map' button click.
        Opens a file dialog for selecting a NIfTI file, loads it as a 3D T1 map,
        validates its dimensions against the loaded DCE series, and updates
        `self.t10_data` and the display.
        """
        if self.dce_data is None:
            self.log_console.append("Please load a DCE series first to validate T1 map dimensions.")
            return

        filepath, _ = QFileDialog.getOpenFileName(self, "Load T1 Map NIfTI File", "", "NIfTI Files (*.nii *.nii.gz);;All Files (*)")
        if filepath:
            self.t1_filepath = filepath # Store path
            try:
                self.log_console.append(f"Loading T1 map: {filepath}")
                # Load T1 map, validating its spatial dimensions against the loaded DCE series
                self.t10_data = io.load_t1_map(filepath, dce_shape=self.dce_shape_for_validation)
                self.t1_path_label.setText(os.path.basename(filepath)) # Update UI label
                self.log_console.append(f"T1 map loaded. Shape: {self.t10_data.shape}")
                self.update_displayable_volume("T1 Map", self.t10_data) # Add T1 map for display
            except Exception as e:
                self.t10_data, self.t1_filepath = None, None # Reset on error
                self.t1_path_label.setText("Error loading T1 map.")
                self.log_console.append(f"Error loading T1 map: {e}\n{traceback.format_exc()}")

    def load_mask_file(self):
        """
        Handles the 'Load Mask' button click.
        Opens a file dialog for selecting a NIfTI file, loads it as a 3D boolean mask,
        validates its dimensions, and updates `self.mask_data` and the display.
        """
        if self.dce_data is None: self.log_console.append("Load DCE series first."); return
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Mask NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if filepath:
            try:
                self.log_console.append(f"Loading mask: {filepath}"); self.mask_data = io.load_mask(filepath, reference_shape=self.dce_shape_for_validation[:3])
                self.mask_path_label.setText(os.path.basename(filepath)); self.log_console.append(f"Mask loaded. Shape: {self.mask_data.shape}, Type: {self.mask_data.dtype}"); self.update_displayable_volume("Mask", self.mask_data.astype(np.uint8))
            except Exception as e: self.mask_data = None; self.mask_path_label.setText("Error loading file"); self.log_console.append(f"Error loading mask: {e}\n{traceback.format_exc()}")

    def run_analysis(self): # Unchanged
        self.log_console.append("Run Analysis button clicked."); self.display_label.setText("Processing... See log for details."); QApplication.processEvents() 
        if self.dce_data is None or self.t10_data is None: self.log_console.append("Error: DCE data and T1 map must be loaded."); self.display_label.setText("Analysis failed: DCE or T1 data missing."); return
        if self.aif_time is None or self.aif_concentration is None: self.log_console.append("Error: AIF not defined/loaded."); self.display_label.setText("Analysis failed: AIF not defined."); return
        if not self._create_aif_interpolators(): self.log_console.append("Error: Failed to create AIF interpolators. Cannot run analysis."); self.display_label.setText("Analysis failed: AIF interpolation error."); return
        try:
            r1_val = float(self.r1_input.text()); tr_val_run = float(self.tr_input.text()); baseline_pts = int(self.baseline_points_input.text()) 
            if r1_val <= 0 or tr_val_run <= 0 or baseline_pts <= 0: raise ValueError("Params must be positive.")
            if baseline_pts >= self.dce_data.shape[3]: raise ValueError("Baseline points exceed total time points.")
        except ValueError as e: self.log_console.append(f"Error: Invalid conversion parameters: {e}"); self.display_label.setText(f"Analysis failed: Invalid params ({e})."); return
        self.log_console.append(f"Starting S-to-C conversion: r1={r1_val}, TR={tr_val_run}, baseline={baseline_pts}"); QApplication.processEvents()
        try:
            self.Ct_data = conversion.signal_to_concentration(self.dce_data, self.t10_data, r1_val, tr_val_run, baseline_pts)
            self.log_console.append(f"S-to-C conversion successful. Ct_data shape: {self.Ct_data.shape}"); self.display_label.setText(f"Conversion successful. Ct shape: {self.Ct_data.shape}")
            self.update_displayable_volume("Ct (Concentration Mean)", np.mean(self.Ct_data, axis=3))
        except Exception as e: self.log_console.append(f"Error during S-to-C: {e}\n{traceback.format_exc()}"); self.display_label.setText("Conversion failed."); self.Ct_data = None; return
        QApplication.processEvents()
        if not self.selected_model_name: self.log_console.append("No model selected. Skipping voxel-wise fitting."); self.display_label.setText("Conversion done. No model selected for fitting."); return
        if self.Ct_data is None: self.log_console.append("Ct_data not available. Skipping model fitting."); self.display_label.setText("Ct data not available. Fitting skipped."); return
        num_cores_to_use = self.num_processes_input.value()
        self.log_console.append(f"Starting parallel voxel-wise {self.selected_model_name} model fitting using up to {num_cores_to_use} cores...")
        self.display_label.setText(f"Fitting {self.selected_model_name} voxel-wise (up to {num_cores_to_use} cores)... This may take a while."); QApplication.processEvents()
        t_tissue = self.dce_time_vector 
        if t_tissue is None: 
             if self.tr_float and self.tr_float > 0: t_tissue = np.arange(self.Ct_data.shape[3]) * self.tr_float
             else: self.log_console.append("Error: Cannot determine t_tissue for fitting (TR invalid)."); self.display_label.setText("Fitting failed: t_tissue unknown."); return
        mask_to_use = self.mask_data if self.mask_data is not None else None; self.parameter_maps = {} 
        try:
            if self.selected_model_name == "Standard Tofts": self.parameter_maps = modeling.fit_standard_tofts_voxelwise(self.Ct_data, t_tissue, self.aif_time, self.aif_concentration, mask=mask_to_use, num_processes=num_cores_to_use)
            elif self.selected_model_name == "Extended Tofts": self.parameter_maps = modeling.fit_extended_tofts_voxelwise(self.Ct_data, t_tissue, self.aif_time, self.aif_concentration, mask=mask_to_use, num_processes=num_cores_to_use)
            elif self.selected_model_name == "Patlak": self.parameter_maps = modeling.fit_patlak_model_voxelwise(self.Ct_data, t_tissue, self.aif_time, self.aif_concentration, mask=mask_to_use, num_processes=num_cores_to_use)
            elif self.selected_model_name == "2CXM": self.parameter_maps = modeling.fit_2cxm_model_voxelwise(self.Ct_data, t_tissue, self.aif_time, self.aif_concentration, mask=mask_to_use, num_processes=num_cores_to_use)
            self.log_console.append(f"Parallel voxel-wise {self.selected_model_name} fitting completed."); self.display_label.setText(f"{self.selected_model_name} fitting done. Maps generated: {', '.join(self.parameter_maps.keys())}")
            for map_name, map_data in self.parameter_maps.items(): self.update_displayable_volume(map_name, map_data)
            self.update_export_buttons_state() 
        except Exception as e: self.log_console.append(f"Error during voxel-wise fitting: {e}\n{traceback.format_exc()}"); self.display_label.setText(f"Voxel-wise fitting failed. See log.")
        QApplication.processEvents()

    def update_export_buttons_state(self): # Unchanged
        is_std_tofts = self.selected_model_name == "Standard Tofts"; is_ext_tofts = self.selected_model_name == "Extended Tofts"; is_patlak = self.selected_model_name == "Patlak"; is_2cxm = self.selected_model_name == "2CXM"
        self.export_ktrans_button.setEnabled("Ktrans" in self.parameter_maps and (is_std_tofts or is_ext_tofts))
        self.export_ve_button.setEnabled("ve" in self.parameter_maps and (is_std_tofts or is_ext_tofts))
        self.export_vp_button.setEnabled("vp" in self.parameter_maps and is_ext_tofts)
        self.export_ktrans_patlak_button.setEnabled("Ktrans_patlak" in self.parameter_maps and is_patlak)
        self.export_vp_patlak_button.setEnabled("vp_patlak" in self.parameter_maps and is_patlak)
        self.export_fp_2cxm_button.setEnabled("Fp_2cxm" in self.parameter_maps and is_2cxm)
        self.export_ps_2cxm_button.setEnabled("PS_2cxm" in self.parameter_maps and is_2cxm)
        self.export_vp_2cxm_button.setEnabled("vp_2cxm" in self.parameter_maps and is_2cxm)
        self.export_ve_2cxm_button.setEnabled("ve_2cxm" in self.parameter_maps and is_2cxm)

    def export_map(self, map_name: str): # Unchanged
        self.log_console.append(f"Export map button clicked for: {map_name}"); param_map_data = self.parameter_maps.get(map_name)
        if param_map_data is None: self.log_console.append(f"Error: {map_name} map not available for export."); return
        reference_nifti_path = self.t1_filepath if self.t1_filepath else self.dce_filepath
        if not reference_nifti_path: self.log_console.append("Error: No reference NIfTI (T1 or DCE) loaded for header info."); return
        default_filename = f"{map_name}_map.nii.gz"; output_filepath, _ = QFileDialog.getSaveFileName(self, f"Save {map_name} Map", default_filename, "NIfTI files (*.nii *.nii.gz)")
        if output_filepath:
            try:
                self.log_console.append(f"Saving {map_name} map to: {output_filepath} using ref: {reference_nifti_path}")
                io.save_nifti_map(param_map_data, reference_nifti_path, output_filepath)
                self.log_console.append(f"{map_name} map saved successfully."); self.display_label.setText(f"{map_name} map saved to {os.path.basename(output_filepath)}")
            except Exception as e: self.log_console.append(f"Error saving {map_name} map: {e}\n{traceback.format_exc()}"); self.display_label.setText(f"Error saving {map_name} map. See log.")

    def handle_add_stats_roi(self): # Unchanged
        current_img_item = self.image_view.getImageItem()
        if current_img_item is None or current_img_item.image is None: self.log_console.append("No image displayed to draw Stats ROI on."); return
        self.stats_roi_counter += 1; roi_name = f"StatsROI_{self.stats_roi_counter}"
        view_data_shape = current_img_item.image[self.image_view.currentIndex].shape 
        roi_y_disp = view_data_shape[0] // 4; roi_x_disp = view_data_shape[1] // 4; roi_h_disp = view_data_shape[0] // 2; roi_w_disp = view_data_shape[1] // 2
        roi_colors = ['#32CD32', '#FFFF00', '#00FFFF', '#FF00FF', '#FFA500', '#DA70D6']; pen_color = roi_colors[(len(self.stats_roi_list)) % len(roi_colors)] 
        new_roi_item = pg.RectROI(pos=(roi_x_disp, roi_y_disp), size=(roi_w_disp, roi_h_disp), pen=pg.mkPen(pen_color, width=2), movable=True, resizable=True, hoverPen=pg.mkPen(pen_color, width=3), rotatable=False)
        self.image_view.addItem(new_roi_item)
        roi_entry = {'item': new_roi_item, 'name': roi_name, 'stats': None, 'slice_idx': self.image_view.currentIndex, 'map_name': self.map_selector_combo.currentText()}
        self.stats_roi_list.append(roi_entry)
        new_roi_item.sigRegionChangeFinished.connect(functools.partial(self.handle_specific_stats_roi_updated, roi_entry))
        self.handle_specific_stats_roi_updated(roi_entry); self.update_all_rois_stats_display(); self.save_stats_button.setEnabled(True) 

    def handle_specific_stats_roi_updated(self, roi_entry_to_update): # Unchanged
        roi_item_changed = roi_entry_to_update['item']; current_map_name_view = self.map_selector_combo.currentText(); current_slice_idx_view = self.image_view.currentIndex
        roi_entry_to_update['map_name'] = current_map_name_view; roi_entry_to_update['slice_idx'] = current_slice_idx_view
        img_item = self.image_view.getImageItem()
        if img_item is None or img_item.image is None: roi_entry_to_update['stats'] = None; self.update_all_rois_stats_display(); return
        original_volume_data = self.displayable_volumes.get(roi_entry_to_update['map_name'])
        if original_volume_data is None or original_volume_data.ndim !=3 : roi_entry_to_update['stats'] = None; self.update_all_rois_stats_display(); return
        current_slice_data_original_orientation = original_volume_data[:, :, roi_entry_to_update['slice_idx']]
        roi_state = roi_item_changed.getState(); x_start_orig = int(round(roi_state['pos'].x())); y_start_orig = int(round(roi_state['pos'].y())); w_orig = int(round(roi_state['size'].x())); h_orig = int(round(roi_state['size'].y()))
        slice_cols_orig, slice_rows_orig = current_slice_data_original_orientation.shape 
        roi_mask_on_slice = np.zeros_like(current_slice_data_original_orientation, dtype=bool)
        x_start_clipped = max(0, x_start_orig); y_start_clipped = max(0, y_start_orig); x_end_clipped = min(x_start_orig + w_orig, slice_cols_orig); y_end_clipped = min(y_start_orig + h_orig, slice_rows_orig)
        if y_start_clipped < y_end_clipped and x_start_clipped < x_end_clipped: roi_mask_on_slice[x_start_clipped:x_end_clipped, y_start_clipped:y_end_clipped] = True
        calculated_stats = reporting.calculate_roi_statistics(current_slice_data_original_orientation, roi_mask_on_slice)
        roi_entry_to_update['stats'] = calculated_stats
        self.update_all_rois_stats_display()

    def update_all_rois_stats_display(self): # Unchanged
        full_stats_text = []; current_map_name_view = self.map_selector_combo.currentText(); current_slice_idx_view = self.image_view.currentIndex
        any_roi_has_valid_stats = False 
        for roi_entry in self.stats_roi_list:
            if roi_entry['map_name'] == current_map_name_view and roi_entry['slice_idx'] == current_slice_idx_view: self.handle_specific_stats_roi_updated(roi_entry) 
            if roi_entry['stats']:
                formatted_str = reporting.format_roi_statistics_to_string(roi_entry['stats'], roi_entry['map_name'], roi_entry['name'])
                full_stats_text.append(formatted_str)
                if roi_entry['stats'].get("N_valid",0) > 0: any_roi_has_valid_stats = True 
            else: full_stats_text.append(f"{roi_entry['name']}: Defined on {roi_entry['map_name']} slice {roi_entry['slice_idx']}. (Stats not yet calculated or N/A)")
        self.stats_results_display.setText("\n\n".join(full_stats_text) if full_stats_text else "No ROIs defined. Add ROIs using 'Add Stats ROI' button.")
        self.save_stats_button.setEnabled(any_roi_has_valid_stats) 

    def handle_clear_last_stats_roi(self): # Unchanged
        if self.stats_roi_list:
            last_roi_entry = self.stats_roi_list.pop()
            if last_roi_entry['item'] in self.image_view.getView().allChildren(): self.image_view.removeItem(last_roi_entry['item'])
            self.log_console.append(f"Removed {last_roi_entry['name']}."); self.update_all_rois_stats_display()
        if not self.stats_roi_list: self.save_stats_button.setEnabled(False)

    def handle_clear_all_stats_rois(self): # Unchanged
        for roi_entry in self.stats_roi_list:
            if roi_entry['item'] in self.image_view.getView().allChildren(): self.image_view.removeItem(roi_entry['item'])
        self.stats_roi_list.clear(); self.stats_roi_counter = 0
        self.log_console.append("Cleared all Stats ROIs."); self.update_all_rois_stats_display(); self.save_stats_button.setEnabled(False)

    def handle_save_all_roi_stats(self): # Unchanged
        stats_to_save = []
        for roi_entry in self.stats_roi_list:
            if roi_entry['stats'] and roi_entry['stats'].get("N_valid",0) > 0: stats_to_save.append((roi_entry['map_name'], roi_entry['slice_idx'], roi_entry['name'], roi_entry['stats']))
        if not stats_to_save: self.log_console.append("No valid ROI statistics from any view to save."); return
        first_map_name = stats_to_save[0][0].replace(' ','_').replace('(','').replace(')',''); default_filename = f"all_roi_stats_from_{first_map_name}.csv"
        filepath, _ = QFileDialog.getSaveFileName(self, "Save All ROI Statistics", default_filename, "CSV files (*.csv)")
        if filepath:
            try: reporting.save_multiple_roi_statistics_csv(stats_to_save, filepath); self.log_console.append(f"Multiple ROI statistics saved to {filepath}")
            except Exception as e: self.log_console.append(f"Error saving ROI stats: {e}\n{traceback.format_exc()}")
            
    def handle_save_plot(self): # New
        if self.plot_widget.getPlotItem().listDataItems(): # Check if there's something plotted
            try:
                exporter = pg_exporters.ImageExporter(self.plot_widget.getPlotItem())
                filepath, selected_filter = QFileDialog.getSaveFileName(
                    self, "Save Plot", "", 
                    "PNG files (*.png);;JPEG files (*.jpg *.jpeg);;SVG files (*.svg);;TIFF files (*.tif);;All files (*)"
                )
                if filepath:
                    exporter.export(filepath)
                    self.log_console.append(f"Plot saved to {filepath}")
            except ImportError:
                self.log_console.append("Error: PyQtGraph ImageExporter not found. Please ensure pyqtgraph is fully installed with exporter dependencies if needed.")
            except Exception as e:
                self.log_console.append(f"Error saving plot: {e}\n{traceback.format_exc()}")
        else:
            self.log_console.append("No plot available to save.")


if __name__ == '__main__':
    if sys.platform.startswith('win'): 
        import multiprocessing 
        multiprocessing.freeze_support() 
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
