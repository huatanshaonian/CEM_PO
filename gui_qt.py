import sys
import os
import time
import traceback
import numpy as np
import json
import csv
import pandas as pd

CONFIG_FILE = "cem_po_qt_config.json"

from PySide6.QtWidgets import (QApplication, QMainWindow, QDockWidget, QWidget, 
                               QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QComboBox, 
                               QCheckBox, QPushButton, QTextEdit, QLabel, QProgressBar,
                               QSplitter, QFrame, QGroupBox, QScrollArea, QFileDialog, QTabWidget,
                               QListWidget, QAbstractItemView, QListWidgetItem)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QAction, QIcon, QFont, QColor, QPalette

# 3D Visualization
import pyvista as pv
from pyvistaqt import QtInteractor

# 2D Plotting
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# Core Logic
from core.solver_bridge import SolverBridge
from geometry.factory import GeometryFactory
from solvers.api import AVAILABLE_ALGORITHMS
from physics.analytical_rcs import get_analytical_solution
from physics.analytical_rcs import get_analytical_solution, compute_error_stats

# --- Step 5: Multithreading (Worker) ---
class CalculationWorker(QThread):
    progress_signal = Signal(float, str)
    result_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, bridge, geo, params):
        super().__init__()
        self.bridge = bridge
        self.geo = geo
        self.params = params

    def run(self):
        def callback(current, total, msg=""):
            p = (current / total * 100) if total > 0 else 0
            self.progress_signal.emit(p, msg)

        try:
            result = self.bridge.run_simulation(self.geo, self.params, progress_callback=callback)
            self.result_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))

class LogBridge(QObject):
    new_log = Signal(str)
    def write(self, text):
        if text.strip(): self.new_log.emit(str(text))
    def flush(self): pass

class MeshStatsWorker(QThread):
    """Worker thread for generating mesh statistics"""
    progress_signal = Signal(float, str)
    result_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, bridge, geo, params):
        super().__init__()
        self.bridge = bridge
        self.geo = geo
        self.params = params

    def run(self):
        import time
        try:
            t_start = time.time()
            self.progress_signal.emit(0, "Generating mesh...")

            meshes = self.bridge.generate_mesh(self.geo, self.params)

            if meshes is None:
                self.error_signal.emit("Algorithm does not support mesh preview")
                return

            # Calculate statistics
            total_cells = 0
            face_stats = []

            for i, m in enumerate(meshes):
                pts = m.points
                if pts.ndim == 3:
                    nu, nv = pts.shape[1], pts.shape[0]
                    n_cells = nu * nv
                else:
                    n_cells = len(pts)
                    nu, nv = n_cells, 1
                total_cells += n_cells
                face_stats.append({'index': i, 'nu': nu, 'nv': nv, 'cells': n_cells})
                self.progress_signal.emit((i + 1) / len(meshes) * 100, f"Surface {i+1}/{len(meshes)}")

            elapsed = time.time() - t_start
            result = {
                'meshes': meshes,
                'total_cells': total_cells,
                'n_surfaces': len(meshes),
                'face_stats': face_stats,
                'elapsed': elapsed
            }
            self.result_signal.emit(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))

# --- UI Styles (Light Theme) ---
LIGHT_STYLE = """
    QMainWindow { background-color: #F8F9FA; color: #333; }
    
    QDockWidget { color: #333; font-weight: bold; }
    QDockWidget::title { 
        background-color: #E9ECEF; 
        padding: 8px; 
        border-bottom: 1px solid #DEE2E6; 
    }
    
    QGroupBox { 
        font-weight: bold; 
        border: 1px solid #DEE2E6; 
        border-radius: 4px;
        margin-top: 12px; 
        background-color: #FFFFFF;
        padding-top: 15px;
    }
    QGroupBox::title { 
        subcontrol-origin: margin; 
        left: 10px; 
        padding: 0 5px; 
        background-color: #FFFFFF;
        color: #007ACC;
    }
    
    QLabel { color: #495057; font-size: 13px; }
    
    QLineEdit { 
        background-color: #FFFFFF; 
        color: #212529; 
        border: 1px solid #CED4DA; 
        padding: 6px; 
        border-radius: 4px;
        selection-background-color: #007ACC;
    }
    QLineEdit:focus { border: 1px solid #007ACC; }
    
    QComboBox { 
        background-color: #FFFFFF; 
        color: #212529; 
        border: 1px solid #CED4DA; 
        padding: 6px; 
        border-radius: 4px;
        min-width: 6em;
    }
    QComboBox:hover { border: 1px solid #ADB5BD; }
    QComboBox::drop-down { 
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 0px;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
    }
    QComboBox::down-arrow { 
        width: 0; 
        height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid #555; /* Darker arrow */
        margin-right: 6px;
        margin-top: 2px;
    }
    
    QPushButton { 
        background-color: #E1E1E1; 
        color: #333; 
        border: 1px solid #999; /* Stronger border */
        padding: 8px 16px; 
        border-radius: 3px;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #D1D1D1; border-color: #666; }
    
    QTabWidget::pane { border: 1px solid #CCC; background: white; border-radius: 2px; }
    QTabBar::tab { background: #E1E1E1; color: #333; padding: 8px 15px; border: 1px solid #CCC; margin-bottom: -1px; }
    QTabBar::tab:selected { background: #FFF; border-bottom: 1px solid #FFF; font-weight: bold; }
"""

class CEMPoQtWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CEM PO Solver - Professional Edition")
        self.resize(1600, 1000)
        
        self.bridge = SolverBridge()
        self.current_geo = None
        self.step_file_path = ""
        self.last_result = None
        self.comparison_data = [] # List of dicts: {'name': str, 'data': DataFrame, 'path': str}

        self.setup_ui()
        self.setup_menu()
        self.setStyleSheet(LIGHT_STYLE)

        # Load configuration
        self.load_config()

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
        self.save_config()
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
        
        self.tabs.addTab(self.comp_frame, "Comparison")
        
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
        self.geo_type_combo.addItems(["Cylinder", "Plate", "Sphere", "Wedge", "Brick", "OCC Cylinder (NURBS)", "STEP File"])
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
        self.surface_list.itemClicked.connect(self.on_surface_selected)
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
        self.group_ptd.setCheckable(True)
        self.group_ptd.setChecked(False)
        l_ptd = QFormLayout()
        
        self.ptd_edges = QLineEdit("")
        self.ptd_edges.setPlaceholderText("Edge IDs (e.g. 1,4,5)")
        l_ptd.addRow("Edge IDs:", self.ptd_edges)
        
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
        
        layout_solver.addSpacing(20)
        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setObjectName("RunBtn")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self.on_run)
        layout_solver.addWidget(self.btn_run)
        layout_solver.addStretch()
        
        self.param_tabs.addTab(self.tab_solver, "Solver")

        # === Tab 3: Comparison ===
        self.tab_comp = QWidget()
        layout_comp = QVBoxLayout(self.tab_comp)
        
        group_comp = QGroupBox("Loaded Data")
        l_comp = QVBoxLayout()
        self.comp_files_list = QListWidget()
        self.comp_files_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.comp_files_list.setStyleSheet("border: 1px solid #CCC; border-radius: 2px;")
        l_comp.addWidget(self.comp_files_list)
        
        h_btn = QHBoxLayout()
        self.btn_add_csv = QPushButton("Import CSV")
        self.btn_add_csv.clicked.connect(self.add_comparison_file)
        self.btn_rem_csv = QPushButton("Remove")
        self.btn_rem_csv.clicked.connect(self.remove_comparison_file)
        h_btn.addWidget(self.btn_add_csv)
        h_btn.addWidget(self.btn_rem_csv)
        l_comp.addLayout(h_btn)
        group_comp.setLayout(l_comp)
        layout_comp.addWidget(group_comp)
        
        group_opts = QGroupBox("Plot Options")
        l_opts = QVBoxLayout()
        self.chk_analytical = QCheckBox("Show Analytical Solution")
        self.chk_analytical.stateChanged.connect(self.update_comparison_plot)
        l_opts.addWidget(self.chk_analytical)
        
        self.btn_refresh_comp = QPushButton("Update Plot")
        self.btn_refresh_comp.clicked.connect(self.update_comparison_plot)
        l_opts.addWidget(self.btn_refresh_comp)
        group_opts.setLayout(l_opts)
        layout_comp.addWidget(group_opts)
        
        layout_comp.addStretch()
        self.param_tabs.addTab(self.tab_comp, "Comparison")

        self.update_geo_inputs("Cylinder")

    def update_geo_inputs(self, gtype):
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

    def add_input(self, label, default, key):
        le = QLineEdit(default)
        self.geo_dynamic_layout.addRow(label, le)
        self.geo_inputs[key] = le

    def pick_step_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open STEP", "", "STEP Files (*.step *.stp)")
        if path:
            self.step_file_path = path
            self.lbl_step.setText(os.path.basename(path))

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
        
        # Validation for STEP
        if gtype == "STEP File" and not params.get('file_path'):
            self.log("<font color='red'>Error: Please select a STEP file first.</font>")
            return

        self.log(f"Updating preview for {gtype}...")
        
        try:
            result = GeometryFactory.create_geometry(gtype, params)
            # Handle Wedge/Brick which return (surfaces, ptd_id) tuple
            if isinstance(result, tuple):
                geo_list, ptd_id = result
                # Auto-fill PTD edges if available
                if ptd_id and self.group_ptd.isChecked():
                    self.ptd_edges.setText(ptd_id)
            else:
                geo_list = result

            if not geo_list:
                self.log("<font color='red'>Preview failed: GeometryFactory returned empty list.</font>")
                return
            
            # Global invert removed in favor of per-surface invert
            # if self.chk_invert_global.isChecked(): ...

            self.current_geo = geo_list
            self.plotter.clear()
            
            # 1. Visualize Surface
            for i, surface in enumerate(geo_list):
                points, faces = self.tessellate_surface(surface, resolution=30)
                mesh = pv.PolyData(points, faces)
                self.plotter.add_mesh(mesh, color='lightblue', show_edges=False, 
                                     opacity=0.6, label=f"Surface {i}")
                
                # 2. Visualize Normals
                if self.chk_show_normals.isChecked():
                    # Subsample for normals to avoid clutter
                    u = np.linspace(*surface.u_domain, 10)
                    v = np.linspace(*surface.v_domain, 10)
                    ug, vg = np.meshgrid(u, v)
                    p, n, j, _, _ = surface.get_data(ug, vg)
                    self.plotter.add_arrows(p.reshape(-1, 3), n.reshape(-1, 3), 
                                           mag=0.2, color='red', opacity=0.8)

                # 3. Visualize PTD Edges
                if self.chk_show_ptd.isChecked():
                    raw_ptd = self.ptd_edges.text().strip()
                    if raw_ptd:
                        try:
                            edge_ids = [int(s.strip()) for s in raw_ptd.split(',') if s.strip()]
                            for eid in edge_ids:
                                if hasattr(surface, 'get_edge_by_index'):
                                    try:
                                        edge_pts = surface.get_edge_by_index(eid, n_samples=50)
                                        # Draw as line
                                        line = pv.MultipleLines(points=edge_pts)
                                        self.plotter.add_mesh(line, color='yellow', line_width=5, 
                                                             label=f"PTD Edge {eid}")
                                    except: pass
                        except: pass

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
            # PTD Parsing
            ptd_edges_list = []
            raw_edges = self.ptd_edges.text().strip()
            if raw_edges:
                ptd_edges_list = [s.strip() for s in raw_edges.split(',') if s.strip()]

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
                    'enabled': self.group_ptd.isChecked(),
                    'edges': ptd_edges_list,
                    'polarization': self.ptd_pol.currentText()
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
        self.update_comparison_plot() # Update comparison tab
        self.tabs.setCurrentIndex(1) # Switch to Results Tab
        self.btn_run.setEnabled(True)

    def plot_results(self, result):
        self.rcs_figure.clear()
        ax = self.rcs_figure.add_subplot(111)
        
        mode = result.get('mode', '1d')
        freq_mhz = result.get('freq', 0) / 1e6
        
        if mode == '2d':
            # 2D Heatmap
            # Note: RCSAnalyzer already returns dB values, no need for log10 conversion
            rcs_db = result['rcs_total']
            theta = result['theta_deg']
            phi = result['phi_deg']

            # Use pcolormesh for better coordinate handling
            X, Y = np.meshgrid(phi, theta)
            # Handle NaN values (already in dB)
            Z = np.nan_to_num(rcs_db, nan=-200)

            c = ax.pcolormesh(X, Y, Z, cmap='jet', shading='auto')
            self.rcs_figure.colorbar(c, ax=ax, label='RCS (dBsm)')

            ax.set_xlabel("Phi (deg)")
            ax.set_ylabel("Theta (deg)")
            ax.set_title(f"RCS Pattern (2D Scan, f={freq_mhz:.1f} MHz)")

        else:
            # 1D Line Plot
            # Note: RCSAnalyzer already returns dB values, no need for log10 conversion
            angles = result['theta_deg']
            rcs_db = result['rcs_total']

            ax.plot(angles, rcs_db, label='Total RCS', linewidth=2, color='#007ACC')

            # PO component is also in dB from RCSAnalyzer
            if result.get('rcs_po') is not None:
                 ax.plot(angles, result['rcs_po'], '--', label='PO', alpha=0.7, color='orange')

            ax.set_xlabel("Theta (deg)")
            ax.set_ylabel("RCS (dBsm)")
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

    def save_config(self):
        try:
            # Gather Dynamic Geo Params
            geo_params_vals = {}
            for k, v in self.geo_inputs.items():
                geo_params_vals[k] = v.text()
            
            # Unit for STEP
            step_unit = ""
            if hasattr(self, 'step_unit_combo'):
                step_unit = self.step_unit_combo.currentText()

            # Invert indices for STEP
            invert_indices = ""
            if hasattr(self, 'invert_indices_input'):
                invert_indices = self.invert_indices_input.text()

            cfg = {
                "geo_type": self.geo_type_combo.currentText(),
                "geo_params": geo_params_vals,
                "step_file_path": self.step_file_path,
                "step_unit": step_unit,
                "invert_indices": invert_indices,

                "freq": self.freq_input.text(),
                "mesh_density": self.mesh_density.text(),
                "min_points": self.min_points.text(),
                "vis_subsample": self.vis_subsample.text(),
                "use_degen": self.degen_mesh.isChecked(),
                
                "algorithm": self.algo_combo.currentData(),
                "theta_start": self.theta_start.text(),
                "theta_end": self.theta_end.text(),
                "theta_n": self.theta_n.text(),
                "phi_start": self.phi_start.text(),
                "phi_end": self.phi_end.text(),
                "phi_n": self.phi_n.text(),
                
                "ptd_enabled": self.group_ptd.isChecked(),
                "ptd_edges": self.ptd_edges.text(),
                "ptd_pol": self.ptd_pol.currentText(),
                
                "use_gpu": self.use_gpu.isChecked(),
                "use_parallel": self.use_parallel.isChecked(),
                "cpu_workers": self.cpu_workers.text()
            }
            
            with open(CONFIG_FILE, 'w') as f:
                json.dump(cfg, f, indent=4)
            # print("Config saved.") 
        except Exception as e:
            print(f"Failed to save config: {e}")

    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            return
            
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            
            # Geometry
            gtype = cfg.get("geo_type", "Cylinder")
            self.geo_type_combo.setCurrentText(gtype)
            # update_geo_inputs is triggered by setCurrentText, but we might need to update inputs after
            
            saved_params = cfg.get("geo_params", {})
            for k, v in saved_params.items():
                if k in self.geo_inputs:
                    self.geo_inputs[k].setText(str(v))
            
            self.step_file_path = cfg.get("step_file_path", "")
            if self.step_file_path and hasattr(self, 'lbl_step'):
                self.lbl_step.setText(os.path.basename(self.step_file_path))
                
            if hasattr(self, 'step_unit_combo'):
                self.step_unit_combo.setCurrentText(cfg.get("step_unit", "mm"))

            if hasattr(self, 'invert_indices_input'):
                self.invert_indices_input.setText(cfg.get("invert_indices", ""))

            # Physics
            self.freq_input.setText(str(cfg.get("freq", "3000.0")))
            self.mesh_density.setText(str(cfg.get("mesh_density", "10.0")))
            self.min_points.setText(str(cfg.get("min_points", "18")))
            self.vis_subsample.setText(str(cfg.get("vis_subsample", "1")))
            self.degen_mesh.setChecked(cfg.get("use_degen", True))

            # Solver
            algo = cfg.get("algorithm")
            idx = self.algo_combo.findData(algo)
            if idx >= 0: self.algo_combo.setCurrentIndex(idx)
            
            self.theta_start.setText(str(cfg.get("theta_start", "-90")))
            self.theta_end.setText(str(cfg.get("theta_end", "90")))
            self.theta_n.setText(str(cfg.get("theta_n", "181")))
            
            self.phi_start.setText(str(cfg.get("phi_start", "0")))
            self.phi_end.setText(str(cfg.get("phi_end", "0")))
            self.phi_n.setText(str(cfg.get("phi_n", "1")))
            
            self.group_ptd.setChecked(cfg.get("ptd_enabled", False))
            self.ptd_edges.setText(cfg.get("ptd_edges", ""))
            self.ptd_pol.setCurrentText(cfg.get("ptd_pol", "VV"))
            
            self.use_gpu.setChecked(cfg.get("use_gpu", False))
            self.use_parallel.setChecked(cfg.get("use_parallel", False))
            self.cpu_workers.setText(str(cfg.get("cpu_workers", "4")))
            
            self.log("Configuration loaded.")
            
        except Exception as e:
            self.log(f"Failed to load config: {e}")

    def add_comparison_file(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select CSV Files", "", "CSV Files (*.csv)")
        for path in paths:
            try:
                # Read CSV using Pandas (flexible)
                # Assume columns like 'Theta', 'RCS', 'dBsm' etc.
                df = pd.read_csv(path, comment='#')
                name = os.path.basename(path)
                
                self.comparison_data.append({
                    'name': name,
                    'data': df,
                    'path': path
                })
                self.comp_files_list.addItem(name)
            except Exception as e:
                self.log(f"Error loading {path}: {e}")
        
        self.update_comparison_plot()

    def remove_comparison_file(self):
        selected_items = self.comp_files_list.selectedItems()
        if not selected_items: return
        
        # Remove from back to avoid index shift issues
        rows = sorted([self.comp_files_list.row(item) for item in selected_items], reverse=True)
        
        for row in rows:
            self.comp_files_list.takeItem(row)
            del self.comparison_data[row]
            
        self.update_comparison_plot()

    def update_comparison_plot(self):
        self.comp_figure.clear()
        ax = self.comp_figure.add_subplot(111)
        
        # 1. Plot Current Result
        # Note: RCSAnalyzer already returns dB values, no need for log10 conversion
        if self.last_result:
            mode = self.last_result.get('mode', '1d')
            if mode == '1d':
                theta = self.last_result['theta_deg']
                rcs_db = self.last_result['rcs_total']  # Already in dB
                ax.plot(theta, rcs_db, label='Current Sim', linewidth=2.5, color='blue', zorder=10)
        
        # 2. Plot Imported Files
        for item in self.comparison_data:
            df = item['data']
            # Heuristic to find Theta and RCS columns
            cols = df.columns
            # Case insensitive search
            theta_col = next((c for c in cols if 'theta' in c.lower()), None)
            rcs_col = next((c for c in cols if 'rcs' in c.lower() or 'dbsm' in c.lower()), None)
            
            if theta_col and rcs_col:
                try:
                    x = df[theta_col].values
                    y = df[rcs_col].values
                    # If column name doesn't imply dB, assume it is dB unless values are tiny?
                    # Most RCS exports are dB.
                    ax.plot(x, y, '--', label=item['name'], alpha=0.8)
                except:
                    pass

        # 3. Plot Analytical (Only if 1D and geometry valid)
        if self.chk_analytical.isChecked() and self.current_geo and self.last_result:
            try:
                freq = self.last_result['freq']
                phi_rad = np.radians(float(self.phi_start.text())) 
                theta_rad = np.radians(self.last_result['theta_deg'])
                
                wave = {'frequency': freq, 'phi': phi_rad}
                ana_rcs = get_analytical_solution(self.current_geo, wave, theta_rad)
                
                if ana_rcs is not None:
                     ana_safe = np.maximum(np.nan_to_num(ana_rcs, nan=1e-12), 1e-12)
                     ana_db = 10 * np.log10(ana_safe)
                     ax.plot(self.last_result['theta_deg'], ana_db, 'r:', label='Analytical', linewidth=2)
                else:
                    self.log("Analytical solution not available for this geometry.")
            except Exception as e:
                self.log(f"Analytical calc warning: {e}")

        ax.set_xlabel("Theta (deg)")
        ax.set_ylabel("RCS (dBsm)")
        ax.grid(True, linestyle='--', alpha=0.5)
        # Only show legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        self.comp_canvas.draw()

    def on_build_geometry(self):
        self.on_preview()

    def update_surface_list(self):
        self.surface_list.clear()
        if not self.current_geo: return
        for i, surf in enumerate(self.current_geo):
            item = QListWidgetItem(f"Surface {i}")
            self.surface_list.addItem(item)

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
        
        # Surface
        points, faces = self.tessellate_surface(surf, resolution=40)
        mesh = pv.PolyData(points, faces)
        self.plotter.add_mesh(mesh, color='orange', show_edges=True, opacity=0.9, label=f"Surface {idx}")
        
        # Normals
        try:
            u = np.linspace(*surf.u_domain, 8)
            v = np.linspace(*surf.v_domain, 8)
            ug, vg = np.meshgrid(u, v)
            p, n, j, _, _ = surf.get_data(ug, vg)
            self.plotter.add_arrows(p.reshape(-1, 3), n.reshape(-1, 3), mag=min(0.2, 1.0), color='red')
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
