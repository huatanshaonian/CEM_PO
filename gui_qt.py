import sys
import os
import time
import traceback
import numpy as np
import json
import csv
import pandas as pd

from PySide6.QtWidgets import (QApplication, QMainWindow, QDockWidget, QWidget,
                               QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QComboBox,
                               QCheckBox, QPushButton, QTextEdit, QLabel, QProgressBar,
                               QSplitter, QFrame, QGroupBox, QScrollArea, QFileDialog, QTabWidget,
                               QListWidget, QAbstractItemView, QListWidgetItem,
                               QDoubleSpinBox)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QAction, QIcon, QFont, QColor, QPalette

# 3D Visualization
import pyvista as pv
from pyvistaqt import QtInteractor

# 2D Plotting
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# Core Logic
from core.solver_bridge import SolverBridge
from geometry.factory import GeometryFactory
from solvers.api import AVAILABLE_ALGORITHMS
from physics.analytical_rcs import get_analytical_solution
from physics.analytical_rcs import get_analytical_solution, compute_error_stats

# UI modules
from ui.workers import CalculationWorker, MeshStatsWorker, LogBridge
from ui.styles import LIGHT_STYLE
from ui.comparison_panel import ComparisonManager
from ui.config_manager import save_config, load_config

class CEMPoQtWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CEM PO Solver - Professional Edition")
        self.resize(1600, 1000)
        
        self.bridge = SolverBridge()
        self.current_geo = None
        self.step_file_path = ""
        self.iges_file_path = ""
        self.last_result = None
        self.comparison_data = [] # List of dicts: {'name': str, 'data': DataFrame, 'path': str}
        self._surface_actors = []          # vtkActor per surface index
        self._actor_to_surface_idx = {}    # actor -> int
        self._highlighted_idx = -1         # currently highlighted surface
        self._picking_setup = False
        self._input_cache = {}             # Cache for dynamic widget values across geo_type switches
        self._comp_mgr = ComparisonManager(self)

        self.setup_ui()
        self.setup_menu()
        self.setStyleSheet(LIGHT_STYLE)

        # Load configuration
        load_config(self)

        self.log("Ready. Welcome to CEM PO Solver.")

    def showEvent(self, event):
        super().showEvent(event)
        if not hasattr(self, '_stdout_redirected'):
            try:
                self.log_bridge = LogBridge()
                self.log_bridge.new_log.connect(self.log)
                sys.stdout = self.log_bridge
                self._stdout_redirected = True
            except:
                pass

    def closeEvent(self, event):
        save_config(self)
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

    def setup_ui(self):
        # Layout: Main Splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.main_splitter)

        # --- Right Side (Tabs + Log) ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0,0,0,0)
        
        self.view_splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(self.view_splitter)

        # Tabs (3D View | RCS Plot)
        self.tabs = QTabWidget()
        self.view_splitter.addWidget(self.tabs)

        # Tab 1: 3D Plotter
        self.plotter_frame = QFrame()
        plotter_layout = QVBoxLayout(self.plotter_frame)
        plotter_layout.setContentsMargins(0,0,0,0)
        self.plotter = QtInteractor(self.plotter_frame)
        self.plotter.set_background("white") 
        self.plotter.add_axes(color='black')
        plotter_layout.addWidget(self.plotter.interactor)
        self.tabs.addTab(self.plotter_frame, "3D Model")

        # Tab 2: RCS Results
        self.rcs_frame = QWidget()
        rcs_layout = QVBoxLayout(self.rcs_frame)
        self.rcs_figure = Figure(figsize=(5, 4), dpi=100)
        self.rcs_canvas = FigureCanvas(self.rcs_figure)
        self.rcs_toolbar = NavigationToolbar(self.rcs_canvas, self.rcs_frame)
        rcs_layout.addWidget(self.rcs_toolbar)
        rcs_layout.addWidget(self.rcs_canvas)
        
        # Export Bar
        export_layout = QHBoxLayout()
        self.chk_analytical = QCheckBox("Show Analytical Solution")
        export_layout.addWidget(self.chk_analytical)
        export_layout.addWidget(QLabel("Unit:"))
        self.combo_rcs_unit_results = QComboBox()
        self.combo_rcs_unit_results.addItems(["dBsm", "m²"])
        self.combo_rcs_unit_results.currentTextChanged.connect(
            lambda: self.plot_results(self.last_result) if self.last_result else None
        )
        export_layout.addWidget(self.combo_rcs_unit_results)
        export_layout.addStretch()
        self.btn_export = QPushButton("Export Data (.csv)")
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_export.setStyleSheet("background-color: #4CAF50; color: white; border: 1px solid #388E3C;")
        export_layout.addWidget(self.btn_export)
        rcs_layout.addLayout(export_layout)
        
        self.tabs.addTab(self.rcs_frame, "RCS Results")

        # Tab 3: Data Comparison
        self.comp_frame = QWidget()
        comp_layout = QVBoxLayout(self.comp_frame)
        
        # Plot Area (Controls moved to Left Dock)
        self.comp_figure = Figure(figsize=(5, 4), dpi=100)
        self.comp_canvas = FigureCanvas(self.comp_figure)
        self.comp_toolbar = NavigationToolbar(self.comp_canvas, self.comp_frame)
        
        comp_layout.addWidget(self.comp_toolbar)
        comp_layout.addWidget(self.comp_canvas)
        
        self.tabs.addTab(self.comp_frame, "RCS Patterns")
        
        # Log Area
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        log_layout.addWidget(QLabel("SYSTEM LOG"))
        log_layout.addWidget(self.log_text)
        
        self.progress_bar = QProgressBar()
        log_layout.addWidget(self.progress_bar)
        
        self.view_splitter.addWidget(log_widget)
        self.view_splitter.setStretchFactor(0, 5) 
        self.view_splitter.setStretchFactor(1, 1) 

        self.main_splitter.addWidget(right_container)

        # --- Left Side (Params Tabs) ---
        self.dock = QDockWidget("CONFIGURATION", self)
        self.dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock.setFixedWidth(350)
        
        # We use a QTabWidget as the main widget of the Dock
        self.param_tabs = QTabWidget()
        self.dock.setWidget(self.param_tabs)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock)

        self.build_params()

    def build_params(self):
        # === Tab 1: Model (Geometry & Mesh) ===
        self.tab_model = QWidget()
        layout_model = QVBoxLayout(self.tab_model)
        
        # 1. Geometry Definition
        group_geo = QGroupBox("Geometry Definition")
        l_geo = QFormLayout()
        self.geo_type_combo = QComboBox()
        self.geo_type_combo.addItems(["Cylinder", "Plate", "Sphere", "Wedge", "Brick", "Infinite Wedge", "OCC Cylinder (NURBS)", "STEP File", "IGES File"])
        self.geo_type_combo.currentTextChanged.connect(self.update_geo_inputs)
        l_geo.addRow("Type:", self.geo_type_combo)
        
        self.geo_dynamic_container = QWidget()
        self.geo_dynamic_layout = QFormLayout(self.geo_dynamic_container)
        self.geo_dynamic_layout.setContentsMargins(0,0,0,0)
        l_geo.addRow(self.geo_dynamic_container)
        
        self.btn_build_geo = QPushButton("Build / Reload Geometry")
        self.btn_build_geo.clicked.connect(self.on_build_geometry)
        self.btn_build_geo.setStyleSheet("background-color: #E1E1E1; border: 1px solid #999;")
        l_geo.addRow(self.btn_build_geo)
        
        group_geo.setLayout(l_geo)
        layout_model.addWidget(group_geo)

        # 2. Surface Inspection
        group_surf = QGroupBox("Surface Inspection")
        l_surf = QVBoxLayout()
        self.surface_list = QListWidget()
        self.surface_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.surface_list.itemClicked.connect(self.on_surface_clicked)
        self.surface_list.itemDoubleClicked.connect(self.on_surface_selected)
        l_surf.addWidget(self.surface_list)
        
        h_ops = QHBoxLayout()
        self.chk_surf_invert = QCheckBox("Invert Normal")
        self.chk_surf_invert.toggled.connect(self.on_invert_surface_toggled)
        self.chk_surf_invert.setEnabled(False)
        h_ops.addWidget(self.chk_surf_invert)
        
        self.btn_show_all = QPushButton("Show All")
        self.btn_show_all.clicked.connect(self.on_show_all_surfaces)
        h_ops.addWidget(self.btn_show_all)
        l_surf.addLayout(h_ops)
        
        # View Controls
        h_view = QHBoxLayout()
        self.chk_show_normals = QCheckBox("Normals")
        self.chk_show_ptd = QCheckBox("PTD Edges")
        self.chk_show_wave = QCheckBox("Inc. Wave")
        
        # Connect signals to refresh view
        self.chk_show_normals.stateChanged.connect(lambda: self.on_preview() if not self.surface_list.selectedItems() else self.on_surface_selected(self.surface_list.selectedItems()[0]))
        self.chk_show_ptd.stateChanged.connect(lambda: self.on_preview() if not self.surface_list.selectedItems() else self.on_surface_selected(self.surface_list.selectedItems()[0]))
        self.chk_show_wave.stateChanged.connect(lambda: self.on_preview() if not self.surface_list.selectedItems() else self.on_surface_selected(self.surface_list.selectedItems()[0]))

        h_view.addWidget(self.chk_show_normals)
        h_view.addWidget(self.chk_show_ptd)
        h_view.addWidget(self.chk_show_wave)
        l_surf.addLayout(h_view)
        
        group_surf.setLayout(l_surf)
        layout_model.addWidget(group_surf)
        
        # 3. Meshing & Physics
        group_mesh = QGroupBox("Meshing & Physics")
        l_mesh = QFormLayout()
        self.freq_input = QLineEdit("3000.0")
        l_mesh.addRow("Freq (MHz):", self.freq_input)
        self.mesh_density = QLineEdit("10.0")
        l_mesh.addRow("Density (λ):", self.mesh_density)
        self.min_points = QLineEdit("18")
        l_mesh.addRow("Min Points/Edge:", self.min_points)
        self.degen_mesh = QCheckBox("Optimize for Degenerate")
        self.degen_mesh.setChecked(True)
        l_mesh.addRow(self.degen_mesh)
        
        # Visualization subsample rate
        self.vis_subsample = QLineEdit("1")
        self.vis_subsample.setToolTip("Subsample rate for mesh visualization (1=all, 10=every 10th)")
        l_mesh.addRow("Vis Subsample:", self.vis_subsample)

        self.btn_gen_stats = QPushButton("Generate Mesh (Stats)")
        self.btn_gen_stats.clicked.connect(self.on_generate_mesh_stats)
        self.btn_gen_stats.setStyleSheet("border: 1px solid #FF9800; color: #E65100;")
        l_mesh.addRow(self.btn_gen_stats)

        self.btn_view_mesh = QPushButton("VIEW MESH")
        self.btn_view_mesh.clicked.connect(self.on_preview_mesh)
        self.btn_view_mesh.setStyleSheet("border: 1px solid #009688; color: #00695C;")
        l_mesh.addRow(self.btn_view_mesh)
        
        group_mesh.setLayout(l_mesh)
        layout_model.addWidget(group_mesh)
        
        self.param_tabs.addTab(self.tab_model, "Model")

        # === Tab 2: Solver ===
        self.tab_solver = QWidget()
        layout_solver = QVBoxLayout(self.tab_solver)
        
        # 1. Algorithm
        group_algo = QGroupBox("Algorithm")
        l_algo = QFormLayout()
        self.algo_combo = QComboBox()
        for aid, meta in AVAILABLE_ALGORITHMS.items():
            self.algo_combo.addItem(meta['name'], aid)
        idx = self.algo_combo.findData('discrete_po_sinc_dual')
        if idx >= 0: self.algo_combo.setCurrentIndex(idx)
        l_algo.addRow(self.algo_combo)
        group_algo.setLayout(l_algo)
        layout_solver.addWidget(group_algo)
        
        # 2. Scan Range
        group_scan = QGroupBox("Scan Range")
        l_scan = QFormLayout()
        
        # Theta Helper
        self.theta_start = QLineEdit("-90")
        self.theta_end = QLineEdit("90")
        self.theta_n = QLineEdit("181")
        h_theta = QHBoxLayout()
        h_theta.addWidget(self.theta_start)
        h_theta.addWidget(self.theta_end)
        h_theta.addWidget(self.theta_n)
        l_scan.addRow("Theta (Start/End/N):", h_theta)

        # Phi Helper
        self.phi_start = QLineEdit("0")
        self.phi_end = QLineEdit("0")
        self.phi_n = QLineEdit("1")
        h_phi = QHBoxLayout()
        h_phi.addWidget(self.phi_start)
        h_phi.addWidget(self.phi_end)
        h_phi.addWidget(self.phi_n)
        l_scan.addRow("Phi (Start/End/N):", h_phi)
        
        group_scan.setLayout(l_scan)
        layout_solver.addWidget(group_scan)

        # 3. PTD Correction
        self.group_ptd = QGroupBox("PTD Correction")
        l_ptd = QFormLayout()

        self.chk_ptd_enabled = QCheckBox("Enable PTD")
        self.chk_ptd_enabled.setChecked(False)
        l_ptd.addRow(self.chk_ptd_enabled)

        self.ptd_edges = QLineEdit("")
        self.ptd_edges.setPlaceholderText("e.g. (0,1) or (0,1);(1,2)")
        l_ptd.addRow("Face Pairs:", self.ptd_edges)

        self.ptd_pol = QComboBox()
        self.ptd_pol.addItems(["VV", "HH"])
        l_ptd.addRow("Polarization:", self.ptd_pol)

        self.group_ptd.setLayout(l_ptd)
        layout_solver.addWidget(self.group_ptd)

        # 4. Performance
        group_compute = QGroupBox("Performance")
        l_compute = QFormLayout()
        
        self.use_gpu = QCheckBox("GPU Acceleration")
        l_compute.addRow(self.use_gpu)
        
        h_cpu = QHBoxLayout()
        self.use_parallel = QCheckBox("CPU Parallel")
        self.cpu_workers = QLineEdit("4")
        self.cpu_workers.setFixedWidth(50)
        h_cpu.addWidget(self.use_parallel)
        h_cpu.addWidget(QLabel("Workers:"))
        h_cpu.addWidget(self.cpu_workers)
        h_cpu.addStretch()
        l_compute.addRow(h_cpu)
        
        group_compute.setLayout(l_compute)
        layout_solver.addWidget(group_compute)
        
        layout_solver.addSpacing(10)
        h_btns = QHBoxLayout()
        
        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setObjectName("RunBtn")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self.on_run)
        h_btns.addWidget(self.btn_run, 2)
        
        self.btn_export_json = QPushButton("EXPORT JSON")
        self.btn_export_json.setMinimumHeight(40)
        self.btn_export_json.clicked.connect(self.export_batch_json)
        self.btn_export_json.setToolTip("Export current settings to a JSON file for batch processing.")
        h_btns.addWidget(self.btn_export_json, 1)
        
        layout_solver.addLayout(h_btns)
        layout_solver.addStretch()
        
        self.param_tabs.addTab(self.tab_solver, "Solver")

        # === Tab 3: Post-processing ===
        self.tab_comp = QWidget()
        layout_comp = QVBoxLayout(self.tab_comp)

        # --- Plot Mode ---
        group_mode = QGroupBox("Plot Mode")
        l_mode = QVBoxLayout()
        self.combo_postproc_mode = QComboBox()
        self.combo_postproc_mode.addItems(["1D Line", "Polar", "2D Heatmap"])
        self.combo_postproc_mode.currentTextChanged.connect(self._comp_mgr.update_comparison_plot)
        l_mode.addWidget(self.combo_postproc_mode)
        # Slice axis + angle — two rows to avoid crowding
        h_axis = QHBoxLayout()
        h_axis.addWidget(QLabel("Slice axis:"))
        self.combo_slice_axis = QComboBox()
        self.combo_slice_axis.addItems(["Phi", "Theta"])
        self.combo_slice_axis.currentTextChanged.connect(self._comp_mgr.update_comparison_plot)
        h_axis.addWidget(self.combo_slice_axis)
        l_mode.addLayout(h_axis)
        h_angle = QHBoxLayout()
        h_angle.addWidget(QLabel("Slice angle (°):"))
        self.spin_slice_angle = QDoubleSpinBox()
        self.spin_slice_angle.setRange(-180.0, 360.0)
        self.spin_slice_angle.setValue(0.0)
        self.spin_slice_angle.setSingleStep(5.0)
        self.spin_slice_angle.valueChanged.connect(self._comp_mgr.update_comparison_plot)
        h_angle.addWidget(self.spin_slice_angle)
        l_mode.addLayout(h_angle)
        group_mode.setLayout(l_mode)
        layout_comp.addWidget(group_mode)

        # --- Loaded Data ---
        group_comp = QGroupBox("Loaded Data")
        l_comp = QVBoxLayout()
        self.comp_files_list = QListWidget()
        self.comp_files_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.comp_files_list.setStyleSheet("border: 1px solid #CCC; border-radius: 2px;")
        l_comp.addWidget(self.comp_files_list)
        h_btn = QHBoxLayout()
        self.btn_add_csv = QPushButton("Import CSV")
        self.btn_add_csv.clicked.connect(self._comp_mgr.add_comparison_file)
        self.btn_rem_csv = QPushButton("Remove")
        self.btn_rem_csv.clicked.connect(self._comp_mgr.remove_comparison_file)
        h_btn.addWidget(self.btn_add_csv)
        h_btn.addWidget(self.btn_rem_csv)
        l_comp.addLayout(h_btn)
        group_comp.setLayout(l_comp)
        layout_comp.addWidget(group_comp)

        # --- Dataset Selection (2D Heatmap mode only) ---
        group_datasets = QGroupBox("Dataset Selection (2D Heatmap)")
        f_ds = QFormLayout()
        f_ds.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.combo_ds_a   = QComboBox(); self.combo_ds_a.addItem("(none)")
        self.combo_ds_b   = QComboBox(); self.combo_ds_b.addItem("(none)")
        self.combo_ds_ref = QComboBox(); self.combo_ds_ref.addItem("(none)")
        f_ds.addRow("A (primary):", self.combo_ds_a)
        f_ds.addRow("B (compare):", self.combo_ds_b)
        f_ds.addRow("Ref:",         self.combo_ds_ref)
        group_datasets.setLayout(f_ds)
        layout_comp.addWidget(group_datasets)

        # Wire RCS Results analytical checkbox (widget created there)
        self.chk_analytical.stateChanged.connect(
            lambda: self.plot_results(self.last_result) if self.last_result else None
        )

        # --- Plot Options ---
        group_opts = QGroupBox("Plot Elements")
        l_opts = QVBoxLayout()
        self.chk_show_total = QCheckBox("Total RCS")
        self.chk_show_total.setChecked(True)
        self.chk_show_total.stateChanged.connect(self._comp_mgr.update_comparison_plot)
        l_opts.addWidget(self.chk_show_total)
        self.chk_show_po = QCheckBox("PO")
        self.chk_show_po.setChecked(True)
        self.chk_show_po.stateChanged.connect(self._comp_mgr.update_comparison_plot)
        l_opts.addWidget(self.chk_show_po)
        self.chk_show_ptd = QCheckBox("PTD Fringe")
        self.chk_show_ptd.setChecked(True)
        self.chk_show_ptd.stateChanged.connect(self._comp_mgr.update_comparison_plot)
        l_opts.addWidget(self.chk_show_ptd)
        self.chk_analytical_comp = QCheckBox("Analytical Solution")
        self.chk_analytical_comp.stateChanged.connect(self._comp_mgr.update_comparison_plot)
        l_opts.addWidget(self.chk_analytical_comp)
        h_unit = QHBoxLayout()
        h_unit.addWidget(QLabel("Unit:"))
        self.combo_rcs_unit = QComboBox()
        self.combo_rcs_unit.addItems(["dBsm", "m²"])
        self.combo_rcs_unit.currentTextChanged.connect(self._comp_mgr.update_comparison_plot)
        h_unit.addWidget(self.combo_rcs_unit)
        l_opts.addLayout(h_unit)
        self.btn_refresh_comp = QPushButton("Refresh Plot")
        self.btn_refresh_comp.clicked.connect(self._comp_mgr.update_comparison_plot)
        l_opts.addWidget(self.btn_refresh_comp)
        group_opts.setLayout(l_opts)
        layout_comp.addWidget(group_opts)

        layout_comp.addStretch()
        self.param_tabs.addTab(self.tab_comp, "Post-processing")

        self.update_geo_inputs("Cylinder")

    def export_batch_json(self):
        """
        Generate a JSON task configuration compatible with main.py based on current UI settings.
        """
        try:
            gtype = self.geo_type_combo.currentText()
            geo_params = self.get_geo_params()
            
            # Construct standard task structure
            task = {
                "name": f"Exported_{gtype}_{time.strftime('%H%M%S')}",
                "description": f"Generated from GUI at {time.ctime()}",
                "geometry": {
                    "type": gtype,
                    "params": geo_params
                },
                "solver": {
                    "frequency_mhz": float(self.freq_input.text()),
                    "algorithm": self.algo_combo.currentData(),
                    "polarization": self.ptd_pol.currentText(),
                    "mesh_density": float(self.mesh_density.text()),
                    "min_points": int(self.min_points.text()),
                    "use_degenerate": self.degen_mesh.isChecked(),
                    "use_gpu": self.use_gpu.isChecked(),
                    "workers": int(self.cpu_workers.text()),
                    "ptd": {
                        "enabled": self.chk_ptd_enabled.isChecked(),
                        "edges": self.ptd_edges.text().strip(),
                    }
                },
                "scan": {
                    "theta": [float(self.theta_start.text()), float(self.theta_end.text()), int(self.theta_n.text())],
                    "phi": [float(self.phi_start.text()), float(self.phi_end.text()), int(self.phi_n.text())]
                }
            }

            # Wrap in a full batch config
            config = {
                "global_settings": {
                    "output_dir": "results/exported_run",
                    "log_dir": "results/exported_run/logs",
                    "save_plot": True,
                    "filename_format": "{task_name}"
                },
                "tasks": [task]
            }

            path, _ = QFileDialog.getSaveFileName(self, "Export Batch JSON", "task_config.json", "JSON Files (*.json)")
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                self.log(f"<font color='green'>Successfully exported batch config to {path}</font>")
                self.log("You can now run this using: <b>python main.py " + os.path.basename(path) + "</b>")

        except Exception as e:
            self.log(f"<font color='red'>Export failed: {e}</font>")
            traceback.print_exc()

    def _cache_dynamic_inputs(self):
        """Save current dynamic widget values before they are destroyed."""
        # Combo boxes
        for attr in ('step_unit_combo', 'iges_unit_combo', 'iges_mirror_plane_combo'):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    self._input_cache[attr] = w.currentText()
                except RuntimeError:
                    pass  # C++ object already deleted
        # Line edits
        for attr in ('invert_indices_input',
                     'iges_invert_indices_input', 'iges_delete_indices_input', 'iges_rotation_input'):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    self._input_cache[attr] = w.text()
                except RuntimeError:
                    pass

    def update_geo_inputs(self, gtype):
        # Cache current widget values before destroying them
        self._cache_dynamic_inputs()

        while self.geo_dynamic_layout.count():
            w = self.geo_dynamic_layout.takeAt(0).widget()
            if w: w.deleteLater()

        self.geo_inputs = {}
        if gtype == "Cylinder" or gtype == "OCC Cylinder (NURBS)":
            self.add_input("Radius (m):", "1.0", "radius")
            self.add_input("Height (m):", "2.0", "height")
        elif gtype == "Plate":
            self.add_input("Width (m):", "5.0", "width")
            self.add_input("Length (m):", "10.0", "length")
        elif gtype == "Sphere":
            self.add_input("Radius (m):", "1.0", "radius")
        elif gtype == "Wedge" or gtype == "Brick":
            self.add_input("Width:", "2.0", "width")
            self.add_input("Length:", "5.0", "length")
            self.add_input("Height:", "3.0", "height")
        elif gtype == "Infinite Wedge":
            self.add_input("Edge Length (m):", "5.0", "edge_length")
            self.add_input("Exterior Angle (°):", "270.0", "exterior_angle")
        elif gtype == "STEP File":
            btn = QPushButton("Select STEP...")
            btn.clicked.connect(self.pick_step_file)
            self.geo_dynamic_layout.addRow("File:", btn)
            self.lbl_step = QLabel("No file selected")
            self.lbl_step.setWordWrap(True)
            self.geo_dynamic_layout.addRow(self.lbl_step)

            self.step_unit_combo = QComboBox()
            self.step_unit_combo.addItems(["mm", "cm", "m"])
            self.geo_dynamic_layout.addRow("Unit:", self.step_unit_combo)

            self.invert_indices_input = QLineEdit("")
            self.invert_indices_input.setPlaceholderText("e.g. 0,1,3,5")
            self.geo_dynamic_layout.addRow("Invert Normals:", self.invert_indices_input)

            # Restore cached values
            if 'step_unit_combo' in self._input_cache:
                self.step_unit_combo.setCurrentText(self._input_cache['step_unit_combo'])
            if 'invert_indices_input' in self._input_cache:
                self.invert_indices_input.setText(self._input_cache['invert_indices_input'])
            if self.step_file_path and hasattr(self, 'lbl_step'):
                self.lbl_step.setText(os.path.basename(self.step_file_path))

        elif gtype == "IGES File":
            btn = QPushButton("Select IGES...")
            btn.clicked.connect(self.pick_iges_file)
            self.geo_dynamic_layout.addRow("File:", btn)
            self.lbl_iges = QLabel("No file selected")
            self.lbl_iges.setWordWrap(True)
            self.geo_dynamic_layout.addRow(self.lbl_iges)

            self.iges_unit_combo = QComboBox()
            self.iges_unit_combo.addItems(["mm", "cm", "m"])
            self.geo_dynamic_layout.addRow("Unit:", self.iges_unit_combo)

            self.iges_invert_indices_input = QLineEdit("")
            self.iges_invert_indices_input.setPlaceholderText("e.g. 0,1,3,5")
            self.geo_dynamic_layout.addRow("Invert Normals:", self.iges_invert_indices_input)

            self.iges_delete_indices_input = QLineEdit("")
            self.iges_delete_indices_input.setPlaceholderText("e.g. 1,3")
            self.geo_dynamic_layout.addRow("Delete Faces:", self.iges_delete_indices_input)

            self.iges_mirror_plane_combo = QComboBox()
            self.iges_mirror_plane_combo.addItems(["None", "X=0", "Y=0", "Z=0"])
            self.geo_dynamic_layout.addRow("Mirror Plane:", self.iges_mirror_plane_combo)

            self.iges_rotation_input = QLineEdit("")
            self.iges_rotation_input.setPlaceholderText("rx, ry, rz (deg), e.g. 90,0,0")
            self.geo_dynamic_layout.addRow("Rotation:", self.iges_rotation_input)

            btn_save = QPushButton("Save As IGES...")
            btn_save.clicked.connect(self.save_iges_as)
            self.geo_dynamic_layout.addRow("Export:", btn_save)

            # Restore cached values
            if 'iges_unit_combo' in self._input_cache:
                self.iges_unit_combo.setCurrentText(self._input_cache['iges_unit_combo'])
            if 'iges_invert_indices_input' in self._input_cache:
                self.iges_invert_indices_input.setText(self._input_cache['iges_invert_indices_input'])
            if 'iges_delete_indices_input' in self._input_cache:
                self.iges_delete_indices_input.setText(self._input_cache['iges_delete_indices_input'])
            if 'iges_mirror_plane_combo' in self._input_cache:
                self.iges_mirror_plane_combo.setCurrentText(self._input_cache['iges_mirror_plane_combo'])
            if 'iges_rotation_input' in self._input_cache:
                self.iges_rotation_input.setText(self._input_cache['iges_rotation_input'])
            if self.iges_file_path and hasattr(self, 'lbl_iges'):
                self.lbl_iges.setText(os.path.basename(self.iges_file_path))

    def add_input(self, label, default, key):
        le = QLineEdit(default)
        self.geo_dynamic_layout.addRow(label, le)
        self.geo_inputs[key] = le

    def pick_step_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open STEP", "", "STEP Files (*.step *.stp)")
        if path:
            self.step_file_path = path
            self.lbl_step.setText(os.path.basename(path))

    def pick_iges_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open IGES", "", "IGES Files (*.iges *.igs)")
        if path:
            self.iges_file_path = path
            self.lbl_iges.setText(os.path.basename(path))

    def save_iges_as(self):
        """将当前编辑参数应用后，另存为新的 IGES 文件。"""
        if not getattr(self, 'iges_file_path', ''):
            self.log("<font color='red'>Error: Please select an IGES source file first.</font>")
            return

        # 收集当前编辑参数（与 get_geo_params 逻辑一致）
        unit = self.iges_unit_combo.currentText() if hasattr(self, 'iges_unit_combo') else 'mm'
        unit_scale = {'mm': 1.0, 'cm': 0.01, 'm': 0.001}.get(unit, 1.0)

        def parse_indices(attr):
            text = getattr(self, attr, None)
            text = text.text().strip() if text else ""
            if not text:
                return []
            try:
                return [int(x.strip()) for x in text.split(',') if x.strip()]
            except ValueError:
                return []

        invert_indices = parse_indices('iges_invert_indices_input')
        delete_indices = parse_indices('iges_delete_indices_input')

        mp = self.iges_mirror_plane_combo.currentText() if hasattr(self, 'iges_mirror_plane_combo') else "None"
        mirror_plane = mp if mp != "None" else None

        rotation = None
        rot_str = self.iges_rotation_input.text().strip() if hasattr(self, 'iges_rotation_input') else ""
        if rot_str:
            try:
                parts = [float(x.strip()) for x in rot_str.split(',')]
                if len(parts) == 3:
                    rotation = tuple(parts)
            except ValueError:
                pass

        # 选择保存路径
        default_name = os.path.splitext(self.iges_file_path)[0] + "_edited.igs"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Edited IGES As", default_name, "IGES Files (*.igs *.iges)"
        )
        if not out_path:
            return

        self.log("Applying edits and saving IGES...")
        try:
            from geometry.step_loader import load_iges_file, save_iges_file
            surfaces = load_iges_file(
                self.iges_file_path,
                scale=unit_scale,
                invert_indices=invert_indices,
                delete_indices=delete_indices,
                mirror_plane=mirror_plane,
                rotation=rotation,
            )
            save_iges_file(surfaces, out_path)
            self.log(f"<font color='green'>Saved {len(surfaces)} faces to: {os.path.basename(out_path)}</font>")
        except Exception as e:
            self.log(f"<font color='red'>Save failed: {e}</font>")

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        view_menu = menubar.addMenu("View")
        
        reset_cam = QAction("Reset Camera", self)
        reset_cam.triggered.connect(lambda: self.plotter.reset_camera())
        view_menu.addAction(reset_cam)

    def log(self, msg):
        try:
            self.log_text.append(f"<b>[{time.strftime('%H:%M:%S')}]</b> {msg}")
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        except:
            # Fallback for thread safety issues or early calls
            print(msg)

    # --- Logic ---
    def get_geo_params(self):
        params = {}
        for k, v in self.geo_inputs.items():
            params[k] = v.text()
        if self.geo_type_combo.currentText() == "STEP File":
            params['file_path'] = self.step_file_path
            params['unit'] = self.step_unit_combo.currentText()
            # Parse invert indices
            invert_str = self.invert_indices_input.text().strip() if hasattr(self, 'invert_indices_input') else ""
            if invert_str:
                try:
                    params['invert_indices'] = [int(x.strip()) for x in invert_str.split(',') if x.strip()]
                except:
                    params['invert_indices'] = []
            else:
                params['invert_indices'] = []
        elif self.geo_type_combo.currentText() == "IGES File":
            params['file_path'] = getattr(self, 'iges_file_path', '')
            params['unit'] = self.iges_unit_combo.currentText() if hasattr(self, 'iges_unit_combo') else 'mm'
            # Parse index lists (invert / delete)
            for attr, key in [('iges_invert_indices_input', 'invert_indices'),
                              ('iges_delete_indices_input', 'delete_indices')]:
                text = getattr(self, attr, None)
                text = text.text().strip() if text else ""
                if text:
                    try:
                        params[key] = [int(x.strip()) for x in text.split(',') if x.strip()]
                    except ValueError:
                        params[key] = []
                else:
                    params[key] = []
            # Mirror plane
            mp = self.iges_mirror_plane_combo.currentText() if hasattr(self, 'iges_mirror_plane_combo') else "None"
            params['mirror_plane'] = mp if mp != "None" else None
            # Rotation
            rot_str = self.iges_rotation_input.text().strip() if hasattr(self, 'iges_rotation_input') else ""
            if rot_str:
                try:
                    parts = [float(x.strip()) for x in rot_str.split(',')]
                    if len(parts) == 3:
                        params['rotation'] = tuple(parts)
                except ValueError:
                    pass
        return params

    def tessellate_surface(self, surface, resolution=30):
        """
        Generate visualization mesh for any Surface object by sampling its UV domain.
        Returns: (points, faces) where faces are formatted for PyVista [3, i, j, k, ...]
        """
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain
        
        u = np.linspace(u_min, u_max, resolution)
        v = np.linspace(v_min, v_max, resolution)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Get 3D coordinates
        points_grid = surface.evaluate(u_grid, v_grid)
        rows, cols, _ = points_grid.shape
        points = points_grid.reshape(-1, 3)
        
        # Generate faces (Triangles)
        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Indices in the flattened array
                idx0 = i * cols + j
                idx1 = i * cols + (j + 1)
                idx2 = (i + 1) * cols + (j + 1)
                idx3 = (i + 1) * cols + j
                
                # Triangle 1 (0, 1, 3)
                faces.extend([3, idx0, idx1, idx3])
                # Triangle 2 (1, 2, 3)
                faces.extend([3, idx1, idx2, idx3])
                
        return points, np.array(faces)

    def on_preview(self):
        gtype = self.geo_type_combo.currentText()
        params = self.get_geo_params()
        
        # Validation for STEP/IGES
        if gtype == "STEP File" and not params.get('file_path'):
            self.log("<font color='red'>Error: Please select a STEP file first.</font>")
            return
        if gtype == "IGES File" and not params.get('file_path'):
            self.log("<font color='red'>Error: Please select an IGES file first.</font>")
            return

        self.log(f"Updating preview for {gtype}...")
        
        try:
            result = GeometryFactory.create_geometry(gtype, params)
            # Handle Wedge/Brick/InfiniteWedge which return (surfaces, ptd_id) tuple
            if isinstance(result, tuple):
                geo_list, ptd_id = result
                # Auto-fill PTD face pairs if available
                if ptd_id:
                    self.ptd_edges.setText(ptd_id)
                # Infinite Wedge: auto-enable PTD
                if gtype == "Infinite Wedge" and ptd_id:
                    self.chk_ptd_enabled.setChecked(True)
            else:
                geo_list = result

            if not geo_list:
                self.log("<font color='red'>Preview failed: GeometryFactory returned empty list.</font>")
                return
            
            # Global invert removed in favor of per-surface invert
            # if self.chk_invert_global.isChecked(): ...

            self.current_geo = geo_list
            self.plotter.clear()
            self._surface_actors = []
            self._actor_to_surface_idx = {}
            self._highlighted_idx = -1

            # 1. Visualize Surface
            for i, surface in enumerate(geo_list):
                points, faces = self.tessellate_surface(surface, resolution=30)
                mesh = pv.PolyData(points, faces)
                actor = self.plotter.add_mesh(mesh, color='lightblue', show_edges=False,
                                     opacity=0.6, label=f"Surface {i}")
                self._surface_actors.append(actor)
                self._actor_to_surface_idx[actor] = i
                
                # 2. Visualize Normals (inset 5% from edges, mag adaptive)
                if self.chk_show_normals.isChecked():
                    u0, u1 = surface.u_domain
                    v0, v1 = surface.v_domain
                    du, dv = (u1 - u0) * 0.05, (v1 - v0) * 0.05
                    u = np.linspace(u0 + du, u1 - du, 5)
                    v = np.linspace(v0 + dv, v1 - dv, 5)
                    ug, vg = np.meshgrid(u, v)
                    data = surface.get_data(ug, vg)
                    p, n, j = data[0], data[1], data[2]
                    p_flat = p.reshape(-1, 3)
                    diag = np.linalg.norm(p_flat.max(axis=0) - p_flat.min(axis=0))
                    arrow_mag = max(diag * 0.08, 0.01)
                    self.plotter.add_arrows(p_flat, n.reshape(-1, 3),
                                           mag=arrow_mag, color='red', opacity=0.6)

            # 3. Visualize PTD Edges (face-pair format)
            if self.chk_show_ptd.isChecked():
                raw_ptd = self.ptd_edges.text().strip()
                if raw_ptd:
                    try:
                        from solvers.ptd_edge_finder import find_shared_edge
                        import re as _re
                        pairs = _re.findall(r'(\d+)\s*,\s*(\d+)', raw_ptd)
                        for a_str, b_str in pairs:
                            a, b = int(a_str), int(b_str)
                            if a < len(geo_list) and b < len(geo_list):
                                try:
                                    edge_pts, _, _ = find_shared_edge(
                                        geo_list[a], geo_list[b], n_samples=50)
                                    line = pv.MultipleLines(points=edge_pts)
                                    self.plotter.add_mesh(line, color='yellow', line_width=5)
                                except Exception:
                                    pass
                    except Exception:
                        pass

            # 4. Visualize ALL Incident Wave Directions from scan range
            if self.chk_show_wave.isChecked():
                try:
                    theta_start = float(self.theta_start.text())
                    theta_end = float(self.theta_end.text())
                    theta_n = int(self.theta_n.text())
                    phi_start = float(self.phi_start.text())
                    phi_end = float(self.phi_end.text())
                    phi_n = int(self.phi_n.text())

                    # Generate ALL angles from scan range
                    theta_vals = np.linspace(theta_start, theta_end, theta_n)
                    phi_vals = np.linspace(phi_start, phi_end, phi_n) if phi_n > 1 else [phi_start]

                    start_points = []
                    directions = []
                    base_dist = 2.0  # distance from origin

                    for th_deg in theta_vals:
                        for ph_deg in phi_vals:
                            th = np.radians(th_deg)
                            ph = np.radians(ph_deg)
                            # Incident vector (pointing TO origin)
                            dx = -np.sin(th) * np.cos(ph)
                            dy = -np.sin(th) * np.sin(ph)
                            dz = -np.cos(th)
                            k_dir = np.array([dx, dy, dz])

                            start = -k_dir * base_dist
                            start_points.append(start)
                            directions.append(k_dir)

                    if start_points:
                        start_points = np.array(start_points)
                        directions = np.array(directions)
                        self.plotter.add_arrows(start_points, directions,
                                               mag=0.15, color='yellow')
                        self.log(f"  Showing {len(start_points)} incident directions")
                except Exception as e:
                    self.log(f"Wave arrow error: {e}")

            self.plotter.add_legend()
            self.plotter.reset_camera()
            self._setup_3d_picking()
            self.update_surface_list() # Sync list with new geometry
            self.log("Preview updated.")
            
        except Exception as e:
            self.log(f"<font color='red'>Preview Error: {str(e)}</font>")
            traceback.print_exc()

    def convert_faces(self, faces):
        pv_faces = []
        for f in faces:
            pv_faces.extend([3, f[0], f[1], f[2]])
        return np.array(pv_faces)

    def on_preview_mesh(self):
        if not self.current_geo:
            self.on_preview()
            if not self.current_geo: return

        self.log("Generating visualization mesh...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from solvers.api import get_integrator
            from core.mesh_data import detect_degenerate_edge

            freq = float(self.freq_input.text()) * 1e6
            samples = float(self.mesh_density.text())
            min_points = int(self.min_points.text())
            use_degen = self.degen_mesh.isChecked()
            subsample = max(1, int(self.vis_subsample.text()))
            wavelength = 299792458.0 / freq

            algo_id = self.algo_combo.currentData()
            # Only pass min_points to discrete_po algorithms
            if 'discrete_po' in algo_id:
                solver = get_integrator(algo_id, min_points=min_points)
            else:
                solver = get_integrator(algo_id)

            self.plotter.clear()
            total_cells = 0
            surfaces = self.current_geo if isinstance(self.current_geo, list) else [self.current_geo]

            # Collect all line segments for batch rendering
            all_points = []
            all_lines = []  # Each line is [start_idx, end_idx]
            point_offset = 0

            for i, surf in enumerate(surfaces):
                degen_edge = detect_degenerate_edge(surf) if use_degen else None
                has_triangle_mesh = hasattr(solver, 'get_triangle_mesh_cells')

                # Calculate actual mesh density based on wavelength (same as solver)
                if hasattr(solver, '_estimate_mesh_density'):
                    nu_actual, nv_actual = solver._estimate_mesh_density(surf, wavelength, samples)
                else:
                    nu_actual, nv_actual = int(samples * 2), int(samples * 2)

                # Apply subsample: visualization density = actual / subsample
                nu_vis = max(5, nu_actual // subsample)
                nv_vis = max(5, nv_actual // subsample)

                if use_degen and degen_edge and degen_edge not in ['degenerate', None] and has_triangle_mesh:
                    # Degenerate surface: use triangle mesh cells
                    vis_a = max(5, max(nu_actual, nv_actual) // subsample)
                    vis_b = vis_a
                    mesh_cells, a, b = solver.get_triangle_mesh_cells(surf, degen_edge, vis_a, vis_b)

                    if mesh_cells:
                        for cell in mesh_cells:
                            u_corners = np.array([c[0] for c in cell] + [cell[0][0]])
                            v_corners = np.array([c[1] for c in cell] + [cell[0][1]])
                            pts_3d = surf.evaluate(u_corners, v_corners)

                            # Add points and line segments
                            n_pts = len(pts_3d)
                            for pt in pts_3d:
                                all_points.append(pt)
                            for j in range(n_pts - 1):
                                all_lines.append([point_offset + j, point_offset + j + 1])
                            point_offset += n_pts

                        total_cells += len(mesh_cells)
                        self.log(f"  Surface {i}: {len(mesh_cells)} cells (a={a}, b={b})")
                else:
                    # Regular surface: use structured grid
                    u_min, u_max = surf.u_domain
                    v_min, v_max = surf.v_domain

                    u = np.linspace(u_min, u_max, nu_vis)
                    v = np.linspace(v_min, v_max, nv_vis)
                    uu, vv = np.meshgrid(u, v)
                    points = surf.evaluate(uu, vv)

                    # Add grid lines (horizontal and vertical)
                    for r in range(nv_vis):
                        for c in range(nu_vis):
                            all_points.append(points[r, c])
                            idx = point_offset + r * nu_vis + c
                            # Horizontal line
                            if c < nu_vis - 1:
                                all_lines.append([idx, idx + 1])
                            # Vertical line
                            if r < nv_vis - 1:
                                all_lines.append([idx, idx + nu_vis])
                    point_offset += nu_vis * nv_vis

                    total_cells += (nu_vis - 1) * (nv_vis - 1)
                    self.log(f"  Surface {i}: {nu_vis}x{nv_vis} (actual: {nu_actual}x{nv_actual})")

            # Create single PolyData with all lines and add once
            if all_points and all_lines:
                all_points = np.array(all_points)
                # PyVista line format: [n_pts, idx0, idx1, n_pts, idx0, idx1, ...]
                lines_pv = []
                for line in all_lines:
                    lines_pv.extend([2, line[0], line[1]])
                mesh = pv.PolyData(all_points, lines=np.array(lines_pv))
                self.plotter.add_mesh(mesh, color='blue', line_width=1)

            self.plotter.reset_camera()
            self.log(f"<b>Mesh visualization complete. Total cells: {total_cells}</b>")
            QApplication.restoreOverrideCursor()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.log(f"<font color='red'>Mesh Error: {e}</font>")
            traceback.print_exc()

    def on_generate_mesh_stats(self):
        """Generate mesh with statistics (similar to tk version's generate_mesh_stats)"""
        if not self.current_geo:
            self.on_preview()
            if not self.current_geo: return

        try:
            params = {
                'frequency': float(self.freq_input.text()) * 1e6,
                'algorithm': self.algo_combo.currentData(),
                'mesh': {
                    'density': float(self.mesh_density.text()),
                    'min_points': int(self.min_points.text()),
                    'use_degenerate': self.degen_mesh.isChecked()
                }
            }
        except Exception as e:
            self.log(f"Param Error: {e}")
            return

        self.btn_gen_stats.setEnabled(False)
        self.progress_bar.setValue(0)

        self.mesh_worker = MeshStatsWorker(self.bridge, self.current_geo, params)
        self.mesh_worker.progress_signal.connect(self._on_progress)
        self.mesh_worker.result_signal.connect(self._on_mesh_stats_finished)
        self.mesh_worker.error_signal.connect(lambda e: self.log(f"<font color='red'>Mesh Error: {e}</font>"))
        self.mesh_worker.finished.connect(lambda: self.btn_gen_stats.setEnabled(True))
        self.mesh_worker.start()

    def _on_mesh_stats_finished(self, result):
        """Handle mesh generation completion"""
        total_cells = result['total_cells']
        n_surfaces = result['n_surfaces']
        elapsed = result['elapsed']
        face_stats = result['face_stats']

        self.log(f"<b>Mesh Generation Complete</b>")
        self.log(f"  Surfaces: {n_surfaces}")
        self.log(f"  Total Cells: {total_cells:,}")
        self.log(f"  Time: {elapsed:.2f}s")
        self.log(f"  Speed: {total_cells/elapsed/1000:.1f} kPts/s")

        # Log per-face stats (first 10)
        for stat in face_stats[:10]:
            self.log(f"    Face {stat['index']}: {stat['nu']}x{stat['nv']} = {stat['cells']} cells")
        if len(face_stats) > 10:
            self.log(f"    ... and {len(face_stats) - 10} more")

        self.progress_bar.setValue(100)

    def on_run(self):
        if not self.current_geo:
            self.on_preview()
            if not self.current_geo: return

        try:
            # PTD face pairs (passed as raw string for downstream parsing)
            ptd_edges_str = self.ptd_edges.text().strip()

            params = {
                'frequency': float(self.freq_input.text()) * 1e6,
                'algorithm': self.algo_combo.currentData(),
                'angles': {
                    'theta_start': float(self.theta_start.text()),
                    'theta_end': float(self.theta_end.text()),
                    'n_theta': int(self.theta_n.text()),
                    'phi_start': float(self.phi_start.text()),
                    'phi_end': float(self.phi_end.text()),
                    'n_phi': int(self.phi_n.text())
                },
                'mesh': {
                    'density': float(self.mesh_density.text()),
                    'min_points': int(self.min_points.text()),
                    'use_degenerate': self.degen_mesh.isChecked()
                },
                'ptd': {
                    'enabled': self.chk_ptd_enabled.isChecked(),
                    'edges': ptd_edges_str,
                    'polarization': self.ptd_pol.currentText(),
                },
                'compute': {
                    'gpu': self.use_gpu.isChecked(),
                    'parallel': self.use_parallel.isChecked(),
                    'workers': int(self.cpu_workers.text())
                }
            }
        except Exception as e:
            self.log(f"Param Error: {e}")
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = CalculationWorker(self.bridge, self.current_geo, params)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.result_signal.connect(self._on_finished)
        self.worker.error_signal.connect(lambda e: self.log(f"ERROR: {e}"))
        self.worker.finished.connect(lambda: self.btn_run.setEnabled(True))
        self.worker.start()

    def _on_progress(self, val, msg):
        self.progress_bar.setValue(int(val))
        if msg: self.log(msg)

    def _on_finished(self, result):
        self.last_result = result
        self.log(f"Simulation SUCCESS in {result['elapsed_time']:.2f}s")
        rcs = result.get('rcs_total')
        
        # Plot Results
        self.plot_results(result)
        self._comp_mgr.update_comparison_plot()  # Update comparison tab
        self.tabs.setCurrentIndex(1) # Switch to Results Tab
        self.btn_run.setEnabled(True)

    def plot_results(self, result):
        self.rcs_figure.clear()
        ax = self.rcs_figure.add_subplot(111)

        mode = result.get('mode', '1d')
        freq_mhz = result.get('freq', 0) / 1e6
        use_db = self.combo_rcs_unit_results.currentText() == "dBsm"
        unit_label = "dBsm" if use_db else "m²"

        def _conv(rcs_db_arr):
            return rcs_db_arr if use_db else 10.0 ** (rcs_db_arr / 10.0)

        if mode == '2d':
            rcs_db = result['rcs_total']
            theta = result['theta_deg']
            phi = result['phi_deg']
            X, Y = np.meshgrid(phi, theta)
            Z = _conv(np.nan_to_num(rcs_db, nan=-200))
            c = ax.pcolormesh(X, Y, Z, cmap='jet', shading='auto')
            self.rcs_figure.colorbar(c, ax=ax, label=f'RCS ({unit_label})')
            ax.set_xlabel("Phi (deg)")
            ax.set_ylabel("Theta (deg)")
            phi_range = phi[-1] - phi[0] if len(phi) > 1 else 1
            theta_range = theta[-1] - theta[0] if len(theta) > 1 else 1
            ax.set_aspect(abs(phi_range / theta_range) if theta_range != 0 else 'equal', adjustable='box')
            ax.invert_yaxis()
            ax.set_title(f"RCS Pattern (2D Scan, f={freq_mhz:.1f} MHz)")

        else:
            angles = result['theta_deg']
            ax.plot(angles, _conv(result['rcs_total']),
                    label='Total RCS', linewidth=2, color='#007ACC')

            if result.get('rcs_po') is not None:
                ax.plot(angles, _conv(result['rcs_po']),
                        '--', label='PO', alpha=0.7, color='orange')

            gtype = self.geo_type_combo.currentText()
            if result.get('rcs_ptd') is not None and gtype == "Infinite Wedge":
                ax.plot(angles, _conv(result['rcs_ptd']),
                        ':', label='PTD Fringe (numerical)',
                        alpha=0.85, color='green', linewidth=1.5)

            if self.chk_analytical.isChecked():
                try:
                    geo_params = self.get_geo_params()
                    polarization = self.ptd_pol.currentText()
                    theta_rad = np.radians(angles)
                    rcs_analytic, label_analytic = get_analytical_solution(
                        gtype, geo_params, result['freq'], theta_rad, polarization
                    )
                    if rcs_analytic is not None:
                        ax.plot(angles, _conv(rcs_analytic), 'r:', linewidth=2,
                                label=label_analytic, alpha=0.9)
                except Exception:
                    pass

            ax.set_xlabel("Theta (deg)")
            ax.set_ylabel(f"RCS ({unit_label})")
            ax.set_title(f"RCS Pattern (f={freq_mhz:.1f} MHz)")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
        
        self.rcs_canvas.draw()

    def export_csv(self):
        if not self.last_result:
            self.log("<font color='orange'>No results to export.</font>")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if not path: return
        
        try:
            res = self.last_result
            mode = res.get('mode', '1d')
            
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write Header Info
                writer.writerow(["# CEM PO Solver Results"])
                writer.writerow(["# Frequency (Hz)", res.get('freq')])
                writer.writerow(["# Algorithm", res.get('params', {}).get('algorithm')])
                writer.writerow([])
                
                if mode == '2d':
                    # 2D Export: Theta, Phi, RCS
                    # Note: RCSAnalyzer returns dB values directly
                    writer.writerow(["Theta (deg)", "Phi (deg)", "RCS (dBsm)", "RCS (m^2)"])
                    rcs_db = res['rcs_total']  # Already in dB
                    theta = res['theta_deg']
                    phi = res['phi_deg']

                    for i, t in enumerate(theta):
                        for j, p in enumerate(phi):
                            val_db = rcs_db[i, j]
                            val_m2 = 10 ** (val_db / 10)  # Convert dB to linear m²
                            writer.writerow([t, p, val_db, val_m2])
                else:
                    # 1D Export
                    # Note: RCSAnalyzer returns dB values directly
                    header = ["Theta (deg)", "RCS Total (dBsm)", "RCS Total (m^2)"]
                    has_po = res.get('rcs_po') is not None
                    if has_po: header.extend(["RCS PO (dBsm)"])

                    writer.writerow(header)

                    theta = res['theta_deg']
                    rcs_db = res['rcs_total']  # Already in dB
                    rcs_po_db = res.get('rcs_po')  # Already in dB

                    for i, t in enumerate(theta):
                        val_db = rcs_db[i]
                        val_m2 = 10 ** (val_db / 10)  # Convert dB to linear m²
                        row = [t, val_db, val_m2]

                        if has_po:
                            po_db = rcs_po_db[i]
                            row.append(po_db)

                        writer.writerow(row)
                        
            self.log(f"Exported successfully to {path}")
            
        except Exception as e:
            self.log(f"<font color='red'>Export Failed: {e}</font>")

    def on_build_geometry(self):
        self.on_preview()

    def update_surface_list(self):
        self.surface_list.clear()
        if not self.current_geo: return
        for i, surf in enumerate(self.current_geo):
            item = QListWidgetItem(f"Surface {i}")
            self.surface_list.addItem(item)

    # ---- 3D pick-to-highlight ----

    def _setup_3d_picking(self):
        """Enable click-to-select in the 3D view (only registers once)."""
        if self._picking_setup:
            return
        try:
            self.plotter.track_click_position(
                callback=self._on_3d_click, side='left', viewport=True)
            self._picking_setup = True
        except Exception as e:
            print(f"  Warning: 3D picking not available: {e}")

    def _on_3d_click(self, click_pos):
        """Callback from 3D viewport click — pick the surface under cursor."""
        if not self._actor_to_surface_idx:
            return
        from vtkmodules.vtkRenderingCore import vtkCellPicker
        picker = vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer)
        picked_actor = picker.GetActor()
        if picked_actor in self._actor_to_surface_idx:
            idx = self._actor_to_surface_idx[picked_actor]
            self._highlight_surface_3d(idx)

    def _highlight_surface_3d(self, idx):
        """Highlight one surface in the all-surfaces view, sync list selection."""
        if idx < 0 or idx >= len(self._surface_actors):
            return

        # Update actor colours: selected = orange, others = lightblue
        for i, actor in enumerate(self._surface_actors):
            prop = actor.GetProperty()
            if i == idx:
                prop.SetColor(1.0, 0.65, 0.0)   # orange
                prop.SetOpacity(0.9)
            else:
                prop.SetColor(0.68, 0.85, 0.9)   # lightblue
                prop.SetOpacity(0.6)
        self.plotter.render()
        self._highlighted_idx = idx

        # Sync QListWidget (block signal to avoid loop)
        self.surface_list.blockSignals(True)
        self.surface_list.setCurrentRow(idx)
        self.surface_list.blockSignals(False)

        # Update invert checkbox
        if self.current_geo and idx < len(self.current_geo):
            surf = self.current_geo[idx]
            self.chk_surf_invert.blockSignals(True)
            self.chk_surf_invert.setEnabled(True)
            self.chk_surf_invert.setChecked(getattr(surf, 'invert_normal', False))
            self.chk_surf_invert.blockSignals(False)

    def on_surface_clicked(self, item):
        """Single click in list — highlight in 3D (if actors available)."""
        idx = self.surface_list.row(item)
        if self._surface_actors:
            self._highlight_surface_3d(idx)
        else:
            self.on_surface_selected(item)

    def on_surface_selected(self, item):
        idx = self.surface_list.row(item)
        if idx < 0 or idx >= len(self.current_geo): return
        surf = self.current_geo[idx]
        
        self.chk_surf_invert.blockSignals(True)
        self.chk_surf_invert.setEnabled(True)
        self.chk_surf_invert.setChecked(getattr(surf, 'invert_normal', False))
        self.chk_surf_invert.blockSignals(False)
        
        self.preview_single_surface(idx, surf)

    def preview_single_surface(self, idx, surf):
        self.plotter.clear()
        self._surface_actors = []
        self._actor_to_surface_idx = {}
        self._highlighted_idx = -1
        
        # Surface
        points, faces = self.tessellate_surface(surf, resolution=40)
        mesh = pv.PolyData(points, faces)
        self.plotter.add_mesh(mesh, color='orange', show_edges=False, opacity=0.9, label=f"Surface {idx}")
        
        # Normals (inset 5% from edges, mag adaptive)
        try:
            u0, u1 = surf.u_domain
            v0, v1 = surf.v_domain
            du, dv = (u1 - u0) * 0.05, (v1 - v0) * 0.05
            u = np.linspace(u0 + du, u1 - du, 5)
            v = np.linspace(v0 + dv, v1 - dv, 5)
            ug, vg = np.meshgrid(u, v)
            p, n, j, _, _ = surf.get_data(ug, vg)
            p_flat = p.reshape(-1, 3)
            diag = np.linalg.norm(p_flat.max(axis=0) - p_flat.min(axis=0))
            arrow_mag = max(diag * 0.08, 0.01)
            self.plotter.add_arrows(p_flat, n.reshape(-1, 3),
                                    mag=arrow_mag, color='red', opacity=0.6)
        except: pass

        # Edges & Labels
        if hasattr(surf, 'get_edges'):
            try:
                edges = surf.get_edges()
                for e_i, edge_data in enumerate(edges):
                    line = pv.MultipleLines(points=edge_data['points'])
                    self.plotter.add_mesh(line, color='red', line_width=4)
                    self.plotter.add_point_labels([edge_data['midpoint']], [f"E{e_i}"], 
                                                 point_size=16, text_color='white', shape_color='black', shape_opacity=0.5)
            except: pass
            
        self.plotter.add_legend()
        self.plotter.reset_camera()

    def on_invert_surface_toggled(self, checked):
        items = self.surface_list.selectedItems()
        if not items: return
        idx = self.surface_list.row(items[0])
        surf = self.current_geo[idx]
        surf.invert_normal = checked
        # Refresh: rebuild all-surfaces view if we have actors, else single view
        if self._surface_actors:
            self.on_preview()
            self._highlight_surface_3d(idx)
        else:
            self.preview_single_surface(idx, surf)

    def on_show_all_surfaces(self):
        self.surface_list.clearSelection()
        self.chk_surf_invert.setEnabled(False)
        self.on_preview()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CEMPoQtWindow()
    window.show()
    sys.exit(app.exec())
