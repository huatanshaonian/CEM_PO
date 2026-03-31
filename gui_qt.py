import sys
import os
import time
import traceback
import numpy as np
import json
import csv
import pandas as pd

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QComboBox,
                               QCheckBox, QPushButton, QTextEdit, QLabel, QProgressBar,
                               QSplitter, QFrame, QGroupBox, QScrollArea, QFileDialog, QTabWidget,
                               QListWidget, QAbstractItemView, QListWidgetItem,
                               QDoubleSpinBox, QSizePolicy, QTabBar)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QSize
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
from ui.workers import CalculationWorker, MeshStatsWorker, LogBridge, FreqSweepWorker
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
        self.last_freq_sweep_result = None
        self.imaging_datasets = []   # list of {'name': str, 'result': dict}
        self.comparison_data = [] # List of dicts: {'name': str, 'data': DataFrame, 'path': str}
        self._surface_actors = []          # vtkActor per surface index
        self._actor_to_surface_idx = {}    # actor -> int
        self._highlighted_indices = set()  # currently highlighted surface indices
        self._ptd_auto_fill = True         # allow auto-fill on next on_preview (set by Build btn)
        self._ptd_highlighted_pair = None  # 当前高亮的 PTD chip 文本，如 "(0,1)"
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
        # ── Central widget: top tab bars + content area ──
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Row 1: Config tab bar (Model / Solver / Post-processing / Imaging) ──
        self.config_tab_bar = QTabBar()
        self.config_tab_bar.setObjectName("ConfigTabBar")
        self.config_tab_bar.addTab("Model")
        self.config_tab_bar.addTab("Solver")
        self.config_tab_bar.addTab("Post-processing")
        self.config_tab_bar.addTab("Imaging")
        self.config_tab_bar.setExpanding(False)
        self.config_tab_bar.setDrawBase(False)
        main_layout.addWidget(self.config_tab_bar)

        # View tab bar will be placed above the view content area (not here)

        # ── Row 3: Main content area ──
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter, 1)  # stretch=1

        # --- Left Side (Config panels, hidden tab bar) ---
        self.param_tabs = QTabWidget()
        self.param_tabs.tabBar().setVisible(False)
        self.param_tabs.setFixedWidth(350)
        self.main_splitter.addWidget(self.param_tabs)

        # --- Right Side (View panels + Log) ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # View tab bar at top of right panel
        self.view_tab_bar = QTabBar()
        self.view_tab_bar.setObjectName("ViewTabBar")
        self.view_tab_bar.addTab("3D Model")
        self.view_tab_bar.addTab("RCS Results")
        self.view_tab_bar.addTab("RCS Patterns")
        self.view_tab_bar.addTab("Statistics")
        self.view_tab_bar.addTab("Radar Imaging")
        self.view_tab_bar.setExpanding(False)
        self.view_tab_bar.setDrawBase(False)
        right_layout.addWidget(self.view_tab_bar)

        self.view_splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(self.view_splitter, 1)

        # View panels (hidden tab bar)
        self.tabs = QTabWidget()
        self.tabs.tabBar().setVisible(False)
        self.view_splitter.addWidget(self.tabs)

        # Tab 1: 3D Plotter
        self.plotter_frame = QFrame()
        plotter_layout = QVBoxLayout(self.plotter_frame)
        plotter_layout.setContentsMargins(0, 0, 0, 0)
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

        self.comp_figure = Figure(figsize=(5, 4), dpi=100)
        self.comp_canvas = FigureCanvas(self.comp_figure)
        self.comp_toolbar = NavigationToolbar(self.comp_canvas, self.comp_frame)

        comp_layout.addWidget(self.comp_toolbar)
        comp_layout.addWidget(self.comp_canvas)

        self.tabs.addTab(self.comp_frame, "RCS Patterns")

        # Tab 4: Statistics
        self.stats_frame = QWidget()
        stats_layout = QVBoxLayout(self.stats_frame)
        stats_layout.setContentsMargins(4, 4, 4, 4)

        # Upper: PDF plot
        self.stats_figure = Figure(figsize=(5, 3), dpi=100)
        self.stats_canvas = FigureCanvas(self.stats_figure)
        self.stats_toolbar = NavigationToolbar(self.stats_canvas, self.stats_frame)
        stats_layout.addWidget(self.stats_toolbar)
        stats_layout.addWidget(self.stats_canvas, 3)

        # Lower: statistics tables (single + comparison, side by side)
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        _table_style = ("QTableWidget { font-size: 12px; }"
                        "QHeaderView::section { background: #E9ECEF; font-weight: bold; padding: 4px; }")

        def _make_stats_table():
            t = QTableWidget()
            t.setAlternatingRowColors(True)
            t.setStyleSheet(_table_style)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            t.horizontalHeader().setStretchLastSection(False)
            t.setEditTriggers(QTableWidget.NoEditTriggers)
            t.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
            return t

        self.stats_table = _make_stats_table()
        self.stats_comp_table = _make_stats_table()
        self.stats_comp_table.setVisible(False)

        stats_layout.addWidget(self.stats_table, 1)
        stats_layout.addWidget(self.stats_comp_table, 1)

        # Export button
        stats_btn_layout = QHBoxLayout()
        stats_btn_layout.addStretch()
        self.btn_export_stats = QPushButton("Export Statistics CSV")
        self.btn_export_stats.clicked.connect(self._export_statistics_csv)
        self.btn_export_stats.setStyleSheet("background-color: #4CAF50; color: white; border: 1px solid #388E3C;")
        stats_btn_layout.addWidget(self.btn_export_stats)
        stats_layout.addLayout(stats_btn_layout)

        self.tabs.addTab(self.stats_frame, "Statistics")

        # Tab 5: Radar Imaging
        self.imaging_frame = QWidget()
        imaging_layout = QVBoxLayout(self.imaging_frame)
        self.imaging_figure = Figure(figsize=(5, 4), dpi=100)
        self.imaging_canvas = FigureCanvas(self.imaging_figure)
        self.imaging_toolbar = NavigationToolbar(self.imaging_canvas, self.imaging_frame)
        imaging_layout.addWidget(self.imaging_toolbar)
        imaging_layout.addWidget(self.imaging_canvas)
        # Export buttons
        img_export_layout = QHBoxLayout()
        img_export_layout.addStretch()
        self.btn_export_rcs_csv = QPushButton("Export RCS CSV")
        self.btn_export_rcs_csv.clicked.connect(self.export_freq_sweep_rcs_csv)
        self.btn_export_rcs_csv.setStyleSheet("background-color: #2196F3; color: white; border: 1px solid #1565C0;")
        img_export_layout.addWidget(self.btn_export_rcs_csv)
        self.btn_export_range_csv = QPushButton("Export Range Profile CSV")
        self.btn_export_range_csv.clicked.connect(self.export_range_profile_csv)
        self.btn_export_range_csv.setStyleSheet("background-color: #4CAF50; color: white; border: 1px solid #388E3C;")
        img_export_layout.addWidget(self.btn_export_range_csv)
        imaging_layout.addLayout(img_export_layout)
        self.tabs.addTab(self.imaging_frame, "Radar Imaging")

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
        self.view_splitter.setStretchFactor(0, 3)
        self.view_splitter.setStretchFactor(1, 1)

        self.main_splitter.addWidget(right_container)
        self.main_splitter.setSizes([350, 1250])

        self.setCentralWidget(central)

        # ── Connect tab bars to panels ──
        self.config_tab_bar.currentChanged.connect(self.param_tabs.setCurrentIndex)
        self.view_tab_bar.currentChanged.connect(self.tabs.setCurrentIndex)
        # Sync back (when code calls setCurrentIndex directly)
        self.param_tabs.currentChanged.connect(self.config_tab_bar.setCurrentIndex)
        self.tabs.currentChanged.connect(self.view_tab_bar.setCurrentIndex)

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
        self.surface_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.surface_list.itemClicked.connect(self.on_surface_clicked)
        self.surface_list.itemDoubleClicked.connect(self.on_surface_selected)
        l_surf.addWidget(self.surface_list)

        h_ops1 = QHBoxLayout()
        self.chk_surf_invert = QCheckBox("Invert Normal")
        self.chk_surf_invert.toggled.connect(self.on_invert_surface_toggled)
        self.chk_surf_invert.setEnabled(False)
        h_ops1.addWidget(self.chk_surf_invert)
        self.btn_show_all = QPushButton("Show All")
        self.btn_show_all.clicked.connect(self.on_show_all_surfaces)
        h_ops1.addWidget(self.btn_show_all)
        l_surf.addLayout(h_ops1)

        h_ops2 = QHBoxLayout()
        self.btn_add_ptd_pair = QPushButton("Add to PTD Pairs")
        self.btn_add_ptd_pair.clicked.connect(self.on_add_ptd_pairs)
        self.btn_add_ptd_pair.setToolTip("Select 2+ surfaces, then click to add all adjacent pairs to PTD Face Pairs list")
        h_ops2.addWidget(self.btn_add_ptd_pair)
        l_surf.addLayout(h_ops2)
        
        # View Controls
        h_view = QHBoxLayout()
        self.chk_show_normals = QCheckBox("Normals")
        self.chk_show_ptd_edges = QCheckBox("PTD Edges")
        self.chk_show_wave = QCheckBox("Inc. Wave")

        # 勾选时始终刷新全模型视图，不触发单面模式
        self.chk_show_normals.stateChanged.connect(self.on_preview)
        self.chk_show_ptd_edges.stateChanged.connect(self.on_preview)
        self.chk_show_wave.stateChanged.connect(self.on_preview)

        h_view.addWidget(self.chk_show_normals)
        h_view.addWidget(self.chk_show_ptd_edges)
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

        self.ptd_seg_angle = QLineEdit("2.0")
        self.ptd_seg_angle.setToolTip(
            "PTD 边缘自适应分段的最大切线转角阈值（度）。\n"
            "越小段数越多、越精确，但计算量增加。直边不受影响。"
        )
        l_mesh.addRow("PTD Seg. Angle (°):", self.ptd_seg_angle)

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
        layout_solver.setSpacing(6)
        layout_solver.setContentsMargins(6, 6, 6, 6)
        
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

        self.chk_show_wave_solver = QCheckBox("Show Inc. Wave")
        self.chk_show_wave_solver.stateChanged.connect(self.on_preview)
        l_scan.addRow(self.chk_show_wave_solver)

        group_scan.setLayout(l_scan)
        layout_solver.addWidget(group_scan)

        # 3. PTD Correction
        self.group_ptd = QGroupBox("PTD Correction")
        l_ptd = QVBoxLayout()
        l_ptd.setSpacing(4)
        l_ptd.setContentsMargins(6, 4, 6, 4)

        h_ptd_top = QHBoxLayout()
        self.chk_ptd_enabled = QCheckBox("Enable PTD")
        self.chk_ptd_enabled.setChecked(False)
        h_ptd_top.addWidget(self.chk_ptd_enabled)
        self.chk_ptd_only = QCheckBox("PTD Only")
        self.chk_ptd_only.setChecked(False)
        self.chk_ptd_only.setToolTip(
            "仅计算 PTD，跳过 PO。\n"
            "需要先有一次完整计算的结果，PTD 将与已有 PO 叠加得到 Total。"
        )
        h_ptd_top.addWidget(self.chk_ptd_only)
        h_ptd_top.addStretch()
        lbl_pol = QLabel("Pol:")
        self.ptd_pol = QComboBox()
        self.ptd_pol.addItems(["VV", "HH"])
        self.ptd_pol.setFixedWidth(55)
        h_ptd_top.addWidget(lbl_pol)
        h_ptd_top.addWidget(self.ptd_pol)
        l_ptd.addLayout(h_ptd_top)

        # Face pairs displayed as wrapping chips
        self.ptd_pairs_list = QListWidget()
        self.ptd_pairs_list.setViewMode(QListWidget.IconMode)
        self.ptd_pairs_list.setFlow(QListWidget.LeftToRight)
        self.ptd_pairs_list.setWrapping(True)
        self.ptd_pairs_list.setResizeMode(QListWidget.Adjust)
        self.ptd_pairs_list.setSpacing(2)
        self.ptd_pairs_list.setMinimumHeight(50)
        self.ptd_pairs_list.setMaximumHeight(110)
        self.ptd_pairs_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.ptd_pairs_list.setDragEnabled(False)
        self.ptd_pairs_list.setToolTip("单击选中，双击高亮该边；Remove 删除选中项")
        self.ptd_pairs_list.itemDoubleClicked.connect(self._on_ptd_pair_dblclick)
        l_ptd.addWidget(self.ptd_pairs_list)

        h_ptd_btns = QHBoxLayout()
        h_ptd_btns.setSpacing(4)
        self.ptd_edges = QLineEdit("")
        self.ptd_edges.setPlaceholderText("(0,1);(1,2) + Enter")
        self.ptd_edges.setToolTip("Manually type pairs then press Enter to add")
        self.ptd_edges.returnPressed.connect(self._on_ptd_edges_manual_add)
        h_ptd_btns.addWidget(self.ptd_edges, 1)
        btn_ptd_add = QPushButton("Add")
        btn_ptd_add.setToolTip("将 3D 视图中选中的面的公共边添加到列表")
        btn_ptd_add.clicked.connect(self._on_ptd_pairs_add_from_selection)
        h_ptd_btns.addWidget(btn_ptd_add)
        self.btn_ptd_remove = QPushButton("Remove")
        self.btn_ptd_remove.setToolTip("删除列表中选中的条目；或删除 3D 视图多选面的公共边")
        self.btn_ptd_remove.clicked.connect(self._on_ptd_pairs_remove)
        h_ptd_btns.addWidget(self.btn_ptd_remove)
        btn_ptd_clear = QPushButton("Clear All")
        btn_ptd_clear.clicked.connect(self.ptd_pairs_list.clear)
        h_ptd_btns.addWidget(btn_ptd_clear)
        l_ptd.addLayout(h_ptd_btns)

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

        # 5. Frequency Sweep
        self.group_freq_sweep = QGroupBox("Frequency Sweep")
        l_fsweep = QVBoxLayout()
        l_fsweep.setSpacing(4)

        self.chk_freq_sweep_enabled = QCheckBox("Enable Frequency Sweep")
        self.chk_freq_sweep_enabled.setChecked(False)
        l_fsweep.addWidget(self.chk_freq_sweep_enabled)

        l_fsweep_form = QFormLayout()
        l_fsweep_form.setSpacing(4)

        # Row 1: Min ~ Max MHz
        h_minmax = QHBoxLayout()
        self.fsweep_start = QLineEdit("1000")
        self.fsweep_start.setFixedWidth(62)
        self.fsweep_end   = QLineEdit("5000")
        self.fsweep_end.setFixedWidth(62)
        h_minmax.addWidget(self.fsweep_start)
        h_minmax.addWidget(QLabel("~"))
        h_minmax.addWidget(self.fsweep_end)
        h_minmax.addWidget(QLabel("MHz"))
        h_minmax.addStretch()
        l_fsweep_form.addRow("Min / Max:", h_minmax)

        # Row 2: Step MHz
        h_step = QHBoxLayout()
        self.fsweep_step = QLineEdit("10")
        self.fsweep_step.setFixedWidth(62)
        h_step.addWidget(self.fsweep_step)
        h_step.addWidget(QLabel("MHz"))
        h_step.addStretch()
        l_fsweep_form.addRow("Step:", h_step)

        self.lbl_fsweep_info = QLabel("Resolution: -- cm | Max range: -- m")
        self.lbl_fsweep_info.setStyleSheet("color: #666; font-size: 10px;")
        l_fsweep_form.addRow(self.lbl_fsweep_info)

        self.lbl_model_size = QLabel("Model: --")
        self.lbl_model_size.setStyleSheet("color: #666; font-size: 10px;")
        l_fsweep_form.addRow(self.lbl_model_size)

        l_fsweep.addLayout(l_fsweep_form)
        self.group_freq_sweep.setLayout(l_fsweep)
        layout_solver.addWidget(self.group_freq_sweep)

        # Update resolution hint on any change
        self.fsweep_start.textChanged.connect(self._update_freq_sweep_info)
        self.fsweep_end.textChanged.connect(self._update_freq_sweep_info)
        self.fsweep_step.textChanged.connect(self._update_freq_sweep_info)
        self.chk_freq_sweep_enabled.toggled.connect(self._update_freq_sweep_info)

        layout_solver.addStretch()
        h_btns = QHBoxLayout()

        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setObjectName("RunBtn")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self.on_run)
        h_btns.addWidget(self.btn_run, 2)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setStyleSheet("QPushButton#StopBtn { background-color: #d9534f; color: white; font-weight: bold; }"
                                     "QPushButton#StopBtn:hover { background-color: #c9302c; }")
        self.btn_stop.clicked.connect(self._on_stop_simulation)
        self.btn_stop.setEnabled(False)
        h_btns.addWidget(self.btn_stop, 1)

        layout_solver.addLayout(h_btns)

        self.btn_export_json = QPushButton("EXPORT JSON")
        self.btn_export_json.setMinimumHeight(32)
        self.btn_export_json.clicked.connect(self.export_batch_json)
        self.btn_export_json.setToolTip("Export current settings to a JSON file for batch processing.")
        layout_solver.addWidget(self.btn_export_json)

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
        h_freq = QHBoxLayout()
        h_freq.addWidget(QLabel("Freq slice (MHz):"))
        self.combo_slice_freq = QComboBox()
        self.combo_slice_freq.setEnabled(False)
        self.combo_slice_freq.currentTextChanged.connect(self._comp_mgr.update_comparison_plot)
        h_freq.addWidget(self.combo_slice_freq)
        l_mode.addLayout(h_freq)
        group_mode.setLayout(l_mode)
        layout_comp.addWidget(group_mode)

        # --- Loaded Data ---
        group_comp = QGroupBox("Loaded Data")
        l_comp = QVBoxLayout()
        self.comp_files_list = QListWidget()
        self.comp_files_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.comp_files_list.setStyleSheet("border: 1px solid #CCC; border-radius: 2px;")
        self.comp_files_list.itemChanged.connect(lambda: self._comp_mgr.update_comparison_plot())
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
        self.combo_ds_a   = QComboBox(); self.combo_ds_a.addItems(["(none)", "Current Simulation"])
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
        # Statistics threshold
        h_thresh = QHBoxLayout()
        self.chk_stats_threshold = QCheckBox("Stats threshold:")
        self.chk_stats_threshold.setToolTip("低于阈值的点不计入 RMSE / MeanDiff（任一数据集 ≥ 阈值即有效）")
        self.chk_stats_threshold.stateChanged.connect(self._comp_mgr.update_comparison_plot)
        h_thresh.addWidget(self.chk_stats_threshold)
        self.stats_threshold_val = QLineEdit("-40")
        self.stats_threshold_val.setFixedWidth(50)
        self.stats_threshold_val.setPlaceholderText("dBsm")
        self.stats_threshold_val.editingFinished.connect(self._comp_mgr.update_comparison_plot)
        h_thresh.addWidget(self.stats_threshold_val)
        h_thresh.addWidget(QLabel("dBsm"))
        h_thresh.addStretch()
        l_opts.addLayout(h_thresh)
        # Colorbar manual range (2D heatmap)
        h_cbar = QHBoxLayout()
        self.chk_cbar_manual = QCheckBox("Colorbar:")
        self.chk_cbar_manual.setToolTip("勾选后使用手动设置的 colorbar 范围")
        self.chk_cbar_manual.stateChanged.connect(self._comp_mgr.update_comparison_plot)
        h_cbar.addWidget(self.chk_cbar_manual)
        self.cbar_vmin = QLineEdit("-40")
        self.cbar_vmin.setFixedWidth(46)
        self.cbar_vmin.setPlaceholderText("min")
        self.cbar_vmin.editingFinished.connect(self._comp_mgr.update_comparison_plot)
        h_cbar.addWidget(self.cbar_vmin)
        h_cbar.addWidget(QLabel("~"))
        self.cbar_vmax = QLineEdit("0")
        self.cbar_vmax.setFixedWidth(46)
        self.cbar_vmax.setPlaceholderText("max")
        self.cbar_vmax.editingFinished.connect(self._comp_mgr.update_comparison_plot)
        h_cbar.addWidget(self.cbar_vmax)
        l_opts.addLayout(h_cbar)

        self.btn_refresh_comp = QPushButton("Refresh Plot")
        self.btn_refresh_comp.clicked.connect(self._comp_mgr.update_comparison_plot)
        l_opts.addWidget(self.btn_refresh_comp)
        group_opts.setLayout(l_opts)
        layout_comp.addWidget(group_opts)

        # --- Statistics ---
        self.btn_calc_stats = QPushButton("Calc Statistics")
        self.btn_calc_stats.setMinimumHeight(34)
        self.btn_calc_stats.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border: 1px solid #1565C0;")
        self.btn_calc_stats.clicked.connect(self._calc_statistics)
        layout_comp.addWidget(self.btn_calc_stats)

        layout_comp.addStretch()
        self.param_tabs.addTab(self.tab_comp, "Post-processing")

        # === Tab 4: Imaging ===
        self.tab_imaging = QWidget()
        layout_imaging = QVBoxLayout(self.tab_imaging)
        layout_imaging.setSpacing(6)
        layout_imaging.setContentsMargins(6, 6, 6, 6)

        group_img = QGroupBox("Radar Imaging")
        l_img = QFormLayout()
        l_img.setSpacing(4)

        self.img_window = QComboBox()
        self.img_window.addItems(["hamming", "hanning", "blackman", "chebyshev", "taylor", "rectangular"])
        l_img.addRow("Window:", self.img_window)

        self.img_cheby_at = QLineEdit("40")
        self.img_cheby_at.setPlaceholderText("Chebyshev / Taylor SLL (dB)")
        l_img.addRow("Sidelobe (dB):", self.img_cheby_at)

        self.img_taylor_nbar = QLineEdit("4")
        self.img_taylor_nbar.setPlaceholderText("Taylor nbar (均匀旁瓣段数, 4~8)")
        l_img.addRow("Taylor nbar:", self.img_taylor_nbar)

        self.img_zeropad = QLineEdit("4")
        l_img.addRow("Zero Pad:", self.img_zeropad)

        h_db = QHBoxLayout()
        self.img_db_min = QLineEdit("-60")
        self.img_db_min.setFixedWidth(46)
        self.img_db_max = QLineEdit("5")
        self.img_db_max.setFixedWidth(46)
        h_db.addWidget(self.img_db_min)
        h_db.addWidget(QLabel("~"))
        h_db.addWidget(self.img_db_max)
        h_db.addWidget(QLabel("dB"))
        h_db.addStretch()
        l_img.addRow("dB Range:", h_db)

        self.img_range_limit = QLineEdit("")
        self.img_range_limit.setPlaceholderText("leave blank = auto")
        l_img.addRow("Display Range (m):", self.img_range_limit)

        btn_refresh = QPushButton("Refresh Plot")
        btn_refresh.clicked.connect(self._plot_selected_imaging)
        l_img.addRow("", btn_refresh)

        group_img.setLayout(l_img)
        layout_imaging.addWidget(group_img)

        # Dataset selector
        group_ds = QGroupBox("Dataset")
        group_ds.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        l_ds = QVBoxLayout()
        l_ds.setSpacing(4)
        l_ds.setContentsMargins(6, 4, 6, 6)
        self.imaging_ds_list = QListWidget()
        self.imaging_ds_list.setMaximumHeight(80)
        self.imaging_ds_list.setMinimumHeight(56)
        self.imaging_ds_list.addItem("(Current Simulation)")
        self.imaging_ds_list.setCurrentRow(0)
        l_ds.addWidget(self.imaging_ds_list)
        h_ds_btns = QHBoxLayout()
        btn_img_load = QPushButton("Load CSV")
        btn_img_load.clicked.connect(self._load_imaging_csv)
        btn_img_remove = QPushButton("Remove")
        btn_img_remove.clicked.connect(self._remove_imaging_csv)
        btn_img_plot = QPushButton("Plot Selected")
        btn_img_plot.clicked.connect(self._plot_selected_imaging)
        h_ds_btns.addWidget(btn_img_load)
        h_ds_btns.addWidget(btn_img_remove)
        h_ds_btns.addWidget(btn_img_plot)
        l_ds.addLayout(h_ds_btns)
        group_ds.setLayout(l_ds)
        layout_imaging.addWidget(group_ds)

        self.lbl_imaging_stats = QLabel("Last run: resolution -- cm, max range -- m")
        self.lbl_imaging_stats.setStyleSheet("color: #555; font-size: 10px;")
        self.lbl_imaging_stats.setWordWrap(True)
        layout_imaging.addWidget(self.lbl_imaging_stats)

        layout_imaging.addStretch()
        self.param_tabs.addTab(self.tab_imaging, "Imaging")

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
                        "edges": self._get_ptd_pairs_str(),
                        "seg_angle_deg": float(self.ptd_seg_angle.text() or '2.0'),
                        "use_parallel_ptd": self.ptd_parallel.isChecked() if hasattr(self, 'ptd_parallel') else False
                    }
                },
                "scan": {
                    "theta": [float(self.theta_start.text()), float(self.theta_end.text()), int(self.theta_n.text())],
                    "phi": [float(self.phi_start.text()), float(self.phi_end.text()), int(self.phi_n.text())]
                }
            }

            if self.chk_freq_sweep_enabled.isChecked():
                task["freq_sweep"] = {
                    "enabled": True,
                    "f_start": float(self.fsweep_start.text()),
                    "f_end": float(self.fsweep_end.text()),
                    "f_step": float(self.fsweep_step.text()),
                    "window": self.img_window.currentText(),
                    "zero_pad": int(self.img_zeropad.text() or "4"),
                    "cheby_at": float(self.img_cheby_at.text() or "40.0"),
                    "taylor_nbar": int(self.img_taylor_nbar.text() or "4"),
                    "taylor_sll": float(self.img_taylor_sll.text() or "30.0") if hasattr(self, 'img_taylor_sll') else 30.0,
                    "polarization": self.ptd_pol.currentText()
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

    # ── PTD Face Pairs helpers ──────────────────────────────────────────────

    def _ptd_existing_pairs(self):
        """返回已添加的面对集合，元素为 (min, max) tuple。"""
        pairs = set()
        for i in range(self.ptd_pairs_list.count()):
            text = self.ptd_pairs_list.item(i).text()   # "(a,b)"
            import re
            m = re.match(r'\((\d+),\s*(\d+)\)', text)
            if m:
                pairs.add((int(m.group(1)), int(m.group(2))))
        return pairs

    def _ptd_add_pairs(self, pairs):
        """将面对列表添加到 ptd_pairs_list，跳过重复项，返回新增数量。"""
        existing = self._ptd_existing_pairs()
        added = 0
        for a, b in pairs:
            key = (min(a, b), max(a, b))
            if key not in existing:
                item = QListWidgetItem(f"({key[0]},{key[1]})")
                item.setSizeHint(QSize(44, 22))
                item.setTextAlignment(Qt.AlignCenter)
                self.ptd_pairs_list.addItem(item)
                existing.add(key)
                added += 1
        return added

    def on_add_ptd_pairs(self):
        """将 surface_list 中选中的面的共享边面对添加到 PTD Face Pairs。"""
        selected_rows = sorted(
            self.surface_list.row(item)
            for item in self.surface_list.selectedItems()
        )
        if len(selected_rows) < 2:
            self.log("请在 Surface Inspection 列表中选中至少 2 个面。")
            return
        if not self.current_geo:
            return

        from itertools import combinations
        from solvers.ptd_edge_finder import find_shared_edge

        sharing, no_edge, duplicate = [], [], []
        for a, b in combinations(selected_rows, 2):
            if a >= len(self.current_geo) or b >= len(self.current_geo):
                continue
            key = (min(a, b), max(a, b))
            if key in self._ptd_existing_pairs():
                duplicate.append(key)
                continue
            try:
                find_shared_edge(self.current_geo[a], self.current_geo[b])
                sharing.append(key)
            except ValueError:
                no_edge.append(key)

        added = self._ptd_add_pairs(sharing)
        parts = [f"已添加 {added} 个面对"]
        if duplicate:
            parts.append(f"跳过 {len(duplicate)} 个重复")
        if no_edge:
            no_edge_str = ", ".join(f"({a},{b})" for a, b in no_edge)
            parts.append(f"无共享边跳过: {no_edge_str}")
        self.log("；".join(parts))

    def _on_ptd_edges_manual_add(self):
        """手动输入框按 Enter 时解析并添加面对。"""
        import re
        text = self.ptd_edges.text().strip()
        if not text:
            return
        raw_pairs = re.findall(r'(\d+)\s*,\s*(\d+)', text)
        if not raw_pairs:
            self.log("格式错误，请使用 (0,1);(1,2) 格式")
            return
        pairs = [(int(a), int(b)) for a, b in raw_pairs]
        added = self._ptd_add_pairs(pairs)
        skipped = len(pairs) - added
        msg = f"已添加 {added} 个面对"
        if skipped:
            msg += f"，跳过 {skipped} 个重复"
        self.log(msg)
        self.ptd_edges.clear()

    def _on_ptd_pairs_add_from_selection(self):
        """Add 按钮：将 3D 视图中高亮的面的公共边添加到 PTD 列表。"""
        selected = sorted(self._highlighted_indices)
        if len(selected) < 2:
            self.log("请先在 3D 视图或面列表中选中至少 2 个面。")
            return
        if not self.current_geo:
            return
        from itertools import combinations
        from solvers.ptd_edge_finder import find_shared_edge
        sharing, no_edge, duplicate = [], [], []
        for a, b in combinations(selected, 2):
            if a >= len(self.current_geo) or b >= len(self.current_geo):
                continue
            key = (min(a, b), max(a, b))
            if key in self._ptd_existing_pairs():
                duplicate.append(key)
                continue
            try:
                find_shared_edge(self.current_geo[a], self.current_geo[b])
                sharing.append(key)
            except ValueError:
                no_edge.append(key)
        added = self._ptd_add_pairs(sharing)
        parts = [f"已添加 {added} 个面对"]
        if duplicate:
            parts.append(f"跳过 {len(duplicate)} 个重复")
        if no_edge:
            parts.append(f"无共享边跳过: {', '.join(f'({a},{b})' for a,b in no_edge)}")
        self.log("；".join(parts))

    def _on_ptd_pairs_remove(self):
        """Remove 按钮：
        - 若 PTD 列表有选中条目，删除它们；
        - 否则，删除列表中所有属于当前 3D 多选面的公共边。
        """
        list_selected_rows = sorted(
            {self.ptd_pairs_list.row(item) for item in self.ptd_pairs_list.selectedItems()},
            reverse=True
        )
        if list_selected_rows:
            for row in list_selected_rows:
                self.ptd_pairs_list.takeItem(row)
            return

        # 无列表选中 → 按 3D 视图选中面计算公共边后删除
        selected = sorted(self._highlighted_indices)
        if len(selected) < 2 or not self.current_geo:
            return
        from itertools import combinations
        keys_to_remove = {
            (min(a, b), max(a, b))
            for a, b in combinations(selected, 2)
            if a < len(self.current_geo) and b < len(self.current_geo)
        }
        # 逆序删除匹配的行
        rows = []
        for i in range(self.ptd_pairs_list.count()):
            import re
            m = re.match(r'\((\d+),\s*(\d+)\)', self.ptd_pairs_list.item(i).text())
            if m:
                key = (min(int(m.group(1)), int(m.group(2))),
                       max(int(m.group(1)), int(m.group(2))))
                if key in keys_to_remove:
                    rows.append(i)
        for row in sorted(rows, reverse=True):
            self.ptd_pairs_list.takeItem(row)

    def _on_ptd_pair_dblclick(self, item):
        """双击 PTD 面对条目：高亮对应棱边（再次双击同一条目取消高亮）。"""
        import re
        text = item.text()
        # 切换逻辑：已高亮则取消
        if self._ptd_highlighted_pair == text:
            self._ptd_highlighted_pair = None
            self.on_preview()
            return
        m = re.match(r'\((\d+),\s*(\d+)\)', text)
        if not m or not self.current_geo:
            return
        a, b = int(m.group(1)), int(m.group(2))
        if a >= len(self.current_geo) or b >= len(self.current_geo):
            return
        try:
            from solvers.ptd_edge_finder import find_shared_edge
            edge_pts, normals_a, normals_b, ext_angle, _warn = find_shared_edge(
                self.current_geo[a], self.current_geo[b])
            self._ptd_highlighted_pair = text
            self._highlight_surfaces_3d({a, b})
            line = pv.MultipleLines(points=edge_pts)
            self.plotter.add_mesh(line, color='red', line_width=5)
            mid = edge_pts[len(edge_pts) // 2].reshape(1, 3)
            self.plotter.add_point_labels(
                pv.PolyData(mid),
                [f"({a},{b})  α={np.degrees(ext_angle):.0f}°"],
                font_size=9, text_color='red', always_visible=True, shape_opacity=0.0)
            self.plotter.render()
        except Exception as e:
            self.log(f"<font color='orange'>高亮棱边失败: {e}</font>")

    def _get_ptd_pairs_str(self):
        """将 ptd_pairs_list 内容拼接为 '(a,b);(c,d)' 字符串。"""
        return ";".join(
            self.ptd_pairs_list.item(i).text()
            for i in range(self.ptd_pairs_list.count())
        )

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
        for attr in ('invert_indices_input', 'step_max_param_input',
                     'iges_invert_indices_input', 'iges_delete_indices_input', 'iges_rotation_input'):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    self._input_cache[attr] = w.text()
                except RuntimeError:
                    pass

    def update_geo_inputs(self, gtype):
        self._ptd_auto_fill = True  # 切换几何类型视为重建，允许重新自动填充
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

            self.step_max_param_input = QLineEdit("1e9")
            self.step_max_param_input.setToolTip("参数域范围上限，超过此值的面会被跳过（用于过滤无限平面）。若读取失败/面被跳过，可调大此值。")
            self.geo_dynamic_layout.addRow("Max Param Range:", self.step_max_param_input)

            # Restore cached values
            if 'step_unit_combo' in self._input_cache:
                self.step_unit_combo.setCurrentText(self._input_cache['step_unit_combo'])
            if 'invert_indices_input' in self._input_cache:
                self.invert_indices_input.setText(self._input_cache['invert_indices_input'])
            if 'step_max_param_input' in self._input_cache:
                self.step_max_param_input.setText(self._input_cache['step_max_param_input'])
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
            # Max param range
            try:
                params['max_param_range'] = float(self.step_max_param_input.text().strip())
            except (ValueError, AttributeError):
                params['max_param_range'] = 1e9
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
                if self._ptd_auto_fill and ptd_id:
                    import re as _re
                    pairs = [(int(a), int(b)) for a, b in _re.findall(r'(\d+)\s*,\s*(\d+)', ptd_id)]
                    if gtype == "Infinite Wedge":
                        # Infinite Wedge 始终重置为固定的一对
                        self.chk_ptd_enabled.setChecked(True)
                        self.ptd_pairs_list.clear()
                    self._ptd_add_pairs(pairs)
                self._ptd_auto_fill = False  # 消耗后重置，避免 checkbox 刷新时重复填充
            else:
                geo_list = result

            if not geo_list:
                self.log("<font color='red'>Preview failed: GeometryFactory returned empty list.</font>")
                return
            
            # Global invert removed in favor of per-surface invert
            # if self.chk_invert_global.isChecked(): ...

            self.current_geo = geo_list
            self._update_model_size_label()
            self.plotter.clear()
            self._surface_actors = []
            self._actor_to_surface_idx = {}
            self._ptd_highlighted_pair = None  # 视图重建，重置 PTD 高亮状态
            self._highlighted_indices = set()

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
            if self.chk_show_ptd_edges.isChecked():
                raw_ptd = self._get_ptd_pairs_str()
                if raw_ptd:
                    try:
                        from solvers.ptd_edge_finder import find_shared_edge
                        from solvers.ptd_structures import PTDEdge
                        import re as _re
                        pairs = _re.findall(r'(\d+)\s*,\s*(\d+)', raw_ptd)
                        for a_str, b_str in pairs:
                            a, b = int(a_str), int(b_str)
                            if a < len(geo_list) and b < len(geo_list):
                                try:
                                    edge_pts, normals_a, normals_b, ext_angle, _warn = find_shared_edge(
                                        geo_list[a], geo_list[b])
                                    lit_normal = np.mean(normals_a, axis=0)
                                    edge = PTDEdge(
                                        f"({a},{b})", edge_pts, lit_normal,
                                        exterior_angle_rad=ext_angle,
                                        point_normals=normals_a,
                                        point_normals_b=normals_b)
                                    # Draw edge line
                                    line = pv.MultipleLines(points=edge_pts)
                                    self.plotter.add_mesh(line, color='yellow', line_width=5)
                                    # Draw segment boundary ticks
                                    n_segs = len(edge.segments)
                                    if n_segs > 1:
                                        bpts = np.array(
                                            [edge.segments[0].start] +
                                            [s.end for s in edge.segments])
                                        sphere_pts = pv.PolyData(bpts)
                                        self.plotter.add_mesh(sphere_pts, color='cyan',
                                                              point_size=10,
                                                              render_points_as_spheres=True)
                                    # Label at midpoint
                                    mid = edge_pts[len(edge_pts) // 2].reshape(1, 3)
                                    self.plotter.add_point_labels(
                                        pv.PolyData(mid),
                                        [f"{n_segs} seg(s)  α={np.degrees(ext_angle):.0f}°"],
                                        font_size=9, text_color='yellow',
                                        always_visible=True, shape_opacity=0.0)
                                except Exception:
                                    pass
                    except Exception:
                        pass

            # 4. Visualize ALL Incident Wave Directions from scan range
            if self.chk_show_wave.isChecked() or self.chk_show_wave_solver.isChecked():
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

        # 频扫模式分支
        if self.chk_freq_sweep_enabled.isChecked():
            self._run_freq_sweep()
            return

        try:
            # PTD face pairs (passed as raw string for downstream parsing)
            ptd_edges_str = self._get_ptd_pairs_str()

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
                    'ptd_only': self.chk_ptd_only.isChecked(),
                    'edges': ptd_edges_str,
                    'polarization': self.ptd_pol.currentText(),
                    'seg_angle_deg': float(self.ptd_seg_angle.text() or '2.0'),
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

        # PTD Only 模式校验
        ptd_only = params.get('ptd', {}).get('ptd_only', False)
        prev_result = None
        if ptd_only:
            if self.last_result is None or self.last_result.get('I_po') is None:
                self.log("<font color='red'>PTD Only 模式需要先有一次包含 PO 的计算结果</font>")
                return
            prev_result = self.last_result

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)

        self.worker = CalculationWorker(self.bridge, self.current_geo, params,
                                        prev_result=prev_result)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.result_signal.connect(self._on_finished)
        self.worker.error_signal.connect(lambda e: self.log(f"ERROR: {e}"))
        self.worker.finished.connect(self._on_simulation_ended)
        self.worker.start()

    def _on_stop_simulation(self):
        """立即终止当前仿真。"""
        stopped = False
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.request_stop()
            stopped = True
        if hasattr(self, 'fsweep_worker') and self.fsweep_worker and self.fsweep_worker.isRunning():
            self.fsweep_worker.request_stop()
            stopped = True
        if stopped:
            self.log("正在终止仿真...")
            self.btn_stop.setEnabled(False)

    def _on_simulation_ended(self):
        """仿真结束（正常完成或被终止）后恢复按钮状态。"""
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

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
            theta = result['theta_deg']
            phi = result['phi_deg']
            X, Y = np.meshgrid(phi, theta)
            has_ptd = result.get('ptd_enabled', False)

            if has_ptd and result.get('rcs_po') is not None:
                # 三图：PO / PTD / Total — 需要重建子图布局
                ax.remove()
                axes = self.rcs_figure.subplots(1, 3)
                plot_data = [
                    (axes[0], result['rcs_po'],    'PO'),
                    (axes[1], result['rcs_ptd'],   'PTD Fringe'),
                    (axes[2], result['rcs_total'], 'Total (PO+PTD)'),
                ]
            else:
                # 单图：仅 Total
                plot_data = [(ax, result['rcs_total'], 'RCS (PO)')]

            for ax_i, data, title_suffix in plot_data:
                Z = _conv(np.nan_to_num(data, nan=-200))
                c = ax_i.pcolormesh(X, Y, Z, cmap='jet', shading='auto')
                self.rcs_figure.colorbar(c, ax=ax_i, label=f'RCS ({unit_label})')
                ax_i.set_xlabel("Phi (deg)")
                ax_i.set_ylabel("Theta (deg)")
                phi_range = phi[-1] - phi[0] if len(phi) > 1 else 1
                theta_range = theta[-1] - theta[0] if len(theta) > 1 else 1
                ax_i.set_aspect(abs(phi_range / theta_range) if theta_range != 0 else 'equal', adjustable='box')
                ax_i.invert_yaxis()
                ax_i.set_title(f"{title_suffix}\n(f={freq_mhz:.1f} MHz)")

        elif mode == '1d_phi':
            angles = result['phi_deg']
            has_ptd = result.get('ptd_enabled', False)

            ax.plot(angles, _conv(result['rcs_total']),
                    label='Total (PO+PTD)' if has_ptd else 'RCS (PO)',
                    linewidth=2, color='#007ACC')

            if has_ptd and result.get('rcs_po') is not None:
                ax.plot(angles, _conv(result['rcs_po']),
                        '--', label='PO', alpha=0.7, color='orange')
                ax.plot(angles, _conv(result['rcs_ptd']),
                        ':', label='PTD Fringe',
                        alpha=0.85, color='green', linewidth=1.5)

            theta_fixed = result.get('theta_deg', 0)
            ax.set_xlabel("Phi (deg)")
            ax.set_ylabel(f"RCS ({unit_label})")
            ax.set_title(f"RCS Pattern (Phi scan, θ={theta_fixed:.1f}°, f={freq_mhz:.1f} MHz)")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        else:
            angles = result['theta_deg']
            has_ptd = result.get('ptd_enabled', False)

            ax.plot(angles, _conv(result['rcs_total']),
                    label='Total (PO+PTD)' if has_ptd else 'RCS (PO)',
                    linewidth=2, color='#007ACC')

            if has_ptd and result.get('rcs_po') is not None:
                ax.plot(angles, _conv(result['rcs_po']),
                        '--', label='PO', alpha=0.7, color='orange')
                ax.plot(angles, _conv(result['rcs_ptd']),
                        ':', label='PTD Fringe',
                        alpha=0.85, color='green', linewidth=1.5)

            gtype = self.geo_type_combo.currentText()

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
                p      = res.get('params', {})
                ang_p  = p.get('angles', {})
                ptd_p  = p.get('ptd', {})
                cmp_p  = p.get('compute', {})
                writer.writerow(["# CEM PO Solver Results"])
                writer.writerow(["# Mode",              mode])
                writer.writerow(["# Frequency (Hz)",    res.get('freq', '')])
                writer.writerow(["# Frequency (MHz)",   (res.get('freq') or 0) / 1e6])
                writer.writerow(["# Algorithm",         p.get('algorithm', '')])
                writer.writerow(["# Theta Start (deg)", ang_p.get('theta_start', '')])
                writer.writerow(["# Theta End (deg)",   ang_p.get('theta_end', '')])
                writer.writerow(["# N Theta",           ang_p.get('n_theta', '')])
                writer.writerow(["# Phi Start (deg)",   ang_p.get('phi_start', '')])
                writer.writerow(["# Phi End (deg)",     ang_p.get('phi_end', '')])
                writer.writerow(["# N Phi",             ang_p.get('n_phi', '')])
                writer.writerow(["# PTD Enabled",       ptd_p.get('enabled', False)])
                writer.writerow(["# PTD Edges",         ptd_p.get('edges', '')])
                writer.writerow(["# Polarization",      ptd_p.get('polarization', '')])
                writer.writerow(["# GPU",               cmp_p.get('gpu', False)])
                writer.writerow(["# Elapsed Time (s)",  f"{res.get('elapsed_time', 0):.3f}"])
                writer.writerow([])
                
                def _cplx_cols(key_c, idx=None):
                    """从 result 取复振幅，返回 [re, im] 或 [] (无数据时)"""
                    v = res.get(key_c)
                    if v is None:
                        return []
                    c = complex(v[idx] if idx is not None else v)
                    return [c.real, c.imag]

                if mode == '2d':
                    has_po  = res.get('rcs_po')  is not None
                    has_ptd = res.get('rcs_ptd') is not None
                    has_I   = res.get('I_total') is not None
                    header = ["Theta (deg)", "Phi (deg)", "RCS Total (dBsm)", "RCS Total (m^2)"]
                    if has_po:  header.append("RCS PO (dBsm)")
                    if has_ptd: header.append("RCS PTD (dBsm)")
                    if has_I:
                        header += ["I Total (Re)", "I Total (Im)",
                                   "I PO (Re)", "I PO (Im)",
                                   "I PTD (Re)", "I PTD (Im)"]
                    writer.writerow(header)

                    theta = res['theta_deg']
                    phi   = res['phi_deg']
                    for i, t in enumerate(theta):
                        for j, p in enumerate(phi):
                            val_db = res['rcs_total'][i, j]
                            row = [t, p, val_db, 10 ** (val_db / 10)]
                            if has_po:  row.append(res['rcs_po'][i, j])
                            if has_ptd: row.append(res['rcs_ptd'][i, j])
                            if has_I:
                                row += _cplx_cols('I_total', (i, j))
                                row += _cplx_cols('I_po',    (i, j))
                                row += _cplx_cols('I_ptd',   (i, j))
                            writer.writerow(row)

                elif mode == '1d_phi':
                    has_po  = res.get('rcs_po')  is not None
                    has_ptd = res.get('rcs_ptd') is not None
                    has_I   = res.get('I_total') is not None
                    header = ["Phi (deg)", "RCS Total (dBsm)", "RCS Total (m^2)"]
                    if has_po:  header.append("RCS PO (dBsm)")
                    if has_ptd: header.append("RCS PTD (dBsm)")
                    if has_I:
                        header += ["I Total (Re)", "I Total (Im)",
                                   "I PO (Re)", "I PO (Im)",
                                   "I PTD (Re)", "I PTD (Im)"]
                    writer.writerow(header)

                    for i, p in enumerate(res['phi_deg']):
                        val_db = res['rcs_total'][i]
                        row = [p, val_db, 10 ** (val_db / 10)]
                        if has_po:  row.append(res['rcs_po'][i])
                        if has_ptd: row.append(res['rcs_ptd'][i])
                        if has_I:
                            row += _cplx_cols('I_total', i)
                            row += _cplx_cols('I_po',    i)
                            row += _cplx_cols('I_ptd',   i)
                        writer.writerow(row)

                else:
                    has_po  = res.get('rcs_po')  is not None
                    has_ptd = res.get('rcs_ptd') is not None
                    has_I   = res.get('I_total') is not None
                    header = ["Theta (deg)", "RCS Total (dBsm)", "RCS Total (m^2)"]
                    if has_po:  header.append("RCS PO (dBsm)")
                    if has_ptd: header.append("RCS PTD (dBsm)")
                    if has_I:
                        header += ["I Total (Re)", "I Total (Im)",
                                   "I PO (Re)", "I PO (Im)",
                                   "I PTD (Re)", "I PTD (Im)"]
                    writer.writerow(header)

                    for i, t in enumerate(res['theta_deg']):
                        val_db = res['rcs_total'][i]
                        row = [t, val_db, 10 ** (val_db / 10)]
                        if has_po:  row.append(res['rcs_po'][i])
                        if has_ptd: row.append(res['rcs_ptd'][i])
                        if has_I:
                            row += _cplx_cols('I_total', i)
                            row += _cplx_cols('I_po',    i)
                            row += _cplx_cols('I_ptd',   i)
                        writer.writerow(row)
                        
            self.log(f"Exported successfully to {path}")
            
        except Exception as e:
            self.log(f"<font color='red'>Export Failed: {e}</font>")

    def on_build_geometry(self):
        self._ptd_auto_fill = True
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
        """Callback from 3D viewport click — pick the surface under cursor.
        Ctrl+click toggles the surface into/out of the current selection.
        """
        if not self._actor_to_surface_idx:
            return
        from vtkmodules.vtkRenderingCore import vtkCellPicker
        from PySide6.QtWidgets import QApplication
        picker = vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer)
        picked_actor = picker.GetActor()
        if picked_actor not in self._actor_to_surface_idx:
            return
        idx = self._actor_to_surface_idx[picked_actor]
        ctrl_held = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl_held:
            new_sel = set(self._highlighted_indices)
            if idx in new_sel:
                new_sel.discard(idx)
            else:
                new_sel.add(idx)
            self._highlight_surfaces_3d(new_sel)
        else:
            self._highlight_surfaces_3d({idx})

    def _highlight_surfaces_3d(self, indices):
        """Highlight a set of surfaces in the all-surfaces view, sync list selection."""
        indices = {i for i in indices if 0 <= i < len(self._surface_actors)}
        for i, actor in enumerate(self._surface_actors):
            prop = actor.GetProperty()
            if i in indices:
                prop.SetColor(1.0, 0.65, 0.0)   # orange
                prop.SetOpacity(0.9)
            else:
                prop.SetColor(0.68, 0.85, 0.9)   # lightblue
                prop.SetOpacity(0.6)
        self.plotter.render()
        self._highlighted_indices = indices

        # Sync QListWidget
        self.surface_list.blockSignals(True)
        self.surface_list.clearSelection()
        for i in indices:
            item = self.surface_list.item(i)
            if item:
                item.setSelected(True)
        self.surface_list.blockSignals(False)

        # Update invert checkbox (only when exactly one surface selected)
        if len(indices) == 1 and self.current_geo:
            idx = next(iter(indices))
            if idx < len(self.current_geo):
                surf = self.current_geo[idx]
                self.chk_surf_invert.blockSignals(True)
                self.chk_surf_invert.setEnabled(True)
                self.chk_surf_invert.setChecked(getattr(surf, 'invert_normal', False))
                self.chk_surf_invert.blockSignals(False)
        else:
            self.chk_surf_invert.setEnabled(False)

    def _highlight_surface_3d(self, idx):
        """Highlight a single surface (convenience wrapper)."""
        self._highlight_surfaces_3d({idx})

    def on_surface_clicked(self, item):
        """Single click in list — highlight all selected surfaces in 3D."""
        if self._surface_actors:
            selected = {self.surface_list.row(i) for i in self.surface_list.selectedItems()}
            self._highlight_surfaces_3d(selected)
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
        self._highlighted_indices = set()
        
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

    # ─────────────────────────── 频率扫描方法 ───────────────────────────

    def _run_freq_sweep(self):
        """读取参数，启动 FreqSweepWorker。"""
        try:
            ptd_edges_str = self._get_ptd_pairs_str()
            params = {
                'frequency': float(self.freq_input.text()) * 1e6,
                'algorithm': self.algo_combo.currentData(),
                'angles': {
                    'theta_start': float(self.theta_start.text()),
                    'theta_end':   float(self.theta_end.text()),
                    'n_theta':     int(self.theta_n.text()),
                    'phi_start':   float(self.phi_start.text()),
                    'phi_end':     float(self.phi_end.text()),
                    'n_phi':       int(self.phi_n.text()),
                },
                'mesh': {
                    'density':        float(self.mesh_density.text()),
                    'min_points':     int(self.min_points.text()),
                    'use_degenerate': self.degen_mesh.isChecked(),
                },
                'ptd': {
                    'enabled':       self.chk_ptd_enabled.isChecked(),
                    'ptd_only':      self.chk_ptd_only.isChecked(),
                    'edges':         ptd_edges_str,
                    'polarization':  self.ptd_pol.currentText(),
                    'seg_angle_deg': float(self.ptd_seg_angle.text() or '2.0'),
                },
                'compute': {
                    'gpu':      self.use_gpu.isChecked(),
                    'parallel': self.use_parallel.isChecked(),
                    'workers':  int(self.cpu_workers.text()),
                },
            }
            freq_sweep_params = {
                'f_start':      float(self.fsweep_start.text()),
                'f_end':        float(self.fsweep_end.text()),
                'f_step':       float(self.fsweep_step.text()),
                'window':       self.img_window.currentText(),
                'zero_pad':     int(self.img_zeropad.text()),
                'cheby_at':     float(self.img_cheby_at.text() or '40'),
                'taylor_nbar':  int(float(self.img_taylor_nbar.text() or '4')),
                'taylor_sll':   float(self.img_cheby_at.text() or '40'),
                'polarization': self.ptd_pol.currentText(),
            }
        except Exception as e:
            self.log(f"FreqSweep Param Error: {e}")
            return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)

        self.fsweep_worker = FreqSweepWorker(
            self.bridge, self.current_geo, params, freq_sweep_params
        )
        self.fsweep_worker.progress_signal.connect(self._on_progress)
        self.fsweep_worker.result_signal.connect(self._on_freq_sweep_finished)
        self.fsweep_worker.error_signal.connect(
            lambda e: (self.log(f"FreqSweep ERROR: {e}"), self._on_simulation_ended())
        )
        self.fsweep_worker.finished.connect(self._on_simulation_ended)
        self.fsweep_worker.start()

    def _on_freq_sweep_finished(self, result):
        self.last_freq_sweep_result = result
        stats = result.get('stats') or {}
        elapsed = result.get('elapsed_time', 0)
        Nf = len(result.get('frequencies', []))
        dr_cm = stats.get('range_resolution_m', 0) * 100
        r_max = stats.get('max_range_m', 0)
        bw    = stats.get('bandwidth_mhz', 0)

        self.log(f"FreqSweep SUCCESS: {Nf} pts, BW={bw:.1f} MHz, "
                 f"res={dr_cm:.1f} cm, Rmax={r_max:.1f} m, t={elapsed:.2f}s")
        self.lbl_imaging_stats.setText(
            f"Last run: resolution {dr_cm:.1f} cm, max range {r_max:.1f} m, BW={bw:.1f} MHz"
        )

        # 更新数据集列表中的 "(Current Simulation)" 条目
        self.imaging_ds_list.item(0).setText("(Current Simulation)")
        self.imaging_ds_list.setCurrentRow(0)

        self.plot_radar_imaging(result)
        self.tabs.setCurrentIndex(4)   # 切到 Radar Imaging tab

    def plot_radar_imaging(self, result):
        """绘制雷达成像结果（1D 距离像 或 2D 距离-角度图）。"""
        self.imaging_figure.clear()
        scan_mode    = result.get('scan_mode', '1d')
        frequencies  = result['frequencies']
        range_axis   = result['range_axis']
        stats        = result.get('stats') or {}

        try:
            db_min = float(self.img_db_min.text())
            db_max = float(self.img_db_max.text())
        except ValueError:
            db_min, db_max = -60.0, 5.0

        range_limit_txt = self.img_range_limit.text().strip()
        range_limit = None
        try:
            if range_limit_txt:
                range_limit = float(range_limit_txt)
        except ValueError:
            pass

        f_mhz = frequencies / 1e6

        from_csv = result.get('source') == 'csv'

        if scan_mode == '1d':
            rcs_1d     = np.asarray(result['rcs_matrix'])      # (Nf,)
            profile_1d = np.asarray(result['profile_matrix'])  # (N_pad,) or (N_half,) from CSV

            ax_freq  = self.imaging_figure.add_subplot(2, 1, 1)
            ax_range = self.imaging_figure.add_subplot(2, 1, 2)

            ax_freq.plot(f_mhz, rcs_1d, color='#007ACC', linewidth=1.5)
            ax_freq.set_xlabel("Frequency (MHz)")
            ax_freq.set_ylabel("RCS (dBsm)")
            ax_freq.set_title("RCS vs Frequency")
            ax_freq.grid(True, linestyle='--', alpha=0.5)

            if from_csv:
                r_plot = range_axis           # already positive-range half
                p_plot = profile_1d
            else:
                N_half = len(profile_1d) // 2
                r_plot = range_axis[:N_half]
                p_plot = profile_1d[:N_half]

            if range_limit is not None:
                mask = r_plot <= range_limit
                r_plot = r_plot[mask]
                p_plot = p_plot[mask]

            ax_range.plot(r_plot, p_plot, color='#E65100', linewidth=1.5)
            dr_m = stats.get('range_resolution_m', 0)
            ax_range.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax_range.set_xlabel("Range (m)")
            ax_range.set_ylabel("Amplitude (dB)")
            ax_range.set_title(f"1D Range Profile  res={dr_m*100:.1f} cm")
            ax_range.set_ylim(db_min, db_max)
            ax_range.grid(True, linestyle='--', alpha=0.5)

        else:  # 2d_angle_freq
            rcs_mat     = np.asarray(result['rcs_matrix'])      # (N_angles, Nf)
            profile_mat = np.asarray(result['profile_matrix'])  # (N_angles, N_pad)
            theta_deg   = np.asarray(result['theta_deg'])
            phi_deg     = np.asarray(result['phi_deg'])

            angle_list = [(th, ph) for th in theta_deg for ph in phi_deg]
            angle_idx  = np.arange(len(angle_list))

            ax_top = self.imaging_figure.add_subplot(2, 1, 1)
            ax_bot = self.imaging_figure.add_subplot(2, 1, 2)

            c1 = ax_top.pcolormesh(
                f_mhz, angle_idx, rcs_mat,
                cmap='jet', shading='auto', vmin=db_min, vmax=db_max
            )
            self.imaging_figure.colorbar(c1, ax=ax_top, label='RCS (dBsm)')
            ax_top.set_xlabel("Frequency (MHz)")
            ax_top.set_ylabel("Angle index")
            ax_top.set_title("RCS vs Frequency vs Angle")

            if from_csv:
                r_half = range_axis
                p_half = profile_mat
            else:
                N_half = len(range_axis) // 2
                r_half = range_axis[:N_half]
                p_half = profile_mat[:, :N_half]

            if range_limit is not None:
                mask = r_half <= range_limit
                r_half = r_half[mask]
                p_half = p_half[:, mask]

            c2 = ax_bot.pcolormesh(
                r_half, angle_idx, p_half,
                cmap='jet', shading='auto', vmin=db_min, vmax=db_max
            )
            self.imaging_figure.colorbar(c2, ax=ax_bot, label='Amplitude (dB)')
            ax_bot.set_xlabel("Range (m)")
            ax_bot.set_ylabel("Angle index")
            dr_m = stats.get('range_resolution_m', 0)
            ax_bot.set_title(f"2D Range-Angle Profile  res={dr_m*100:.1f} cm")

        self.imaging_figure.tight_layout()
        self.imaging_canvas.draw()

    def _update_freq_sweep_info(self):
        """实时更新频扫分辨率/最大距离标签。"""
        try:
            from physics.constants import C0 as _C0
            f_start = float(self.fsweep_start.text()) * 1e6
            f_end   = float(self.fsweep_end.text())   * 1e6
            f_step  = float(self.fsweep_step.text())  * 1e6
            if f_end <= f_start or f_step <= 0:
                raise ValueError
            bw    = f_end - f_start
            dr_m  = _C0 / (2.0 * bw)
            r_max = _C0 / (2.0 * f_step)
            Nf    = int(round(bw / f_step)) + 1
            self.lbl_fsweep_info.setText(
                f"Resolution: {dr_m*100:.1f} cm | Max range: {r_max:.1f} m | {Nf} pts"
            )
        except Exception:
            self.lbl_fsweep_info.setText("Resolution: -- cm | Max range: -- m")

    def _update_model_size_label(self):
        """采样几何包围盒，更新 lbl_model_size。"""
        if not hasattr(self, 'lbl_model_size'):
            return
        if not self.current_geo:
            self.lbl_model_size.setText("Model: --")
            return
        try:
            pts_all = []
            for surf in self.current_geo:
                u0, u1 = surf.u_domain
                v0, v1 = surf.v_domain
                us = np.linspace(u0, u1, 8)
                vs = np.linspace(v0, v1, 8)
                ug, vg = np.meshgrid(us, vs)
                data = surf.get_data(ug, vg)
                pts_all.append(data[0].reshape(-1, 3))
            pts = np.concatenate(pts_all, axis=0)
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            dims = maxs - mins
            self.lbl_model_size.setText(
                f"Model: {dims[0]:.2f}×{dims[1]:.2f}×{dims[2]:.2f} m"
            )
        except Exception:
            self.lbl_model_size.setText("Model: --")

    def _load_imaging_csv(self):
        """从 CSV 文件加载频扫结果到 Imaging 数据集列表。"""
        from core.freq_sweep import load_freq_sweep_csv
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Freq Sweep CSV", "", "CSV Files (*.csv)"
        )
        for path in paths:
            try:
                result = load_freq_sweep_csv(path)
                import os
                name = os.path.basename(path)
                self.imaging_datasets.append({'name': name, 'result': result})
                self.imaging_ds_list.addItem(name)
                self.log(f"Imaging: loaded {name}  "
                         f"({result['stats']['bandwidth_mhz']:.0f} MHz BW, "
                         f"pol={result['freq_sweep_params']['polarization']})")
            except Exception as e:
                self.log(f"<font color='red'>Imaging load error ({path}): {e}</font>")

    def _remove_imaging_csv(self):
        """从列表中移除选中的 CSV 条目（不能移除第 0 行 Current Simulation）。"""
        row = self.imaging_ds_list.currentRow()
        if row <= 0:
            self.log("Cannot remove '(Current Simulation)'.")
            return
        self.imaging_ds_list.takeItem(row)
        del self.imaging_datasets[row - 1]   # offset by 1 (row 0 = current sim)

    def _plot_selected_imaging(self):
        """绘制数据集列表中当前选中的结果。"""
        row = self.imaging_ds_list.currentRow()
        if row < 0:
            return
        if row == 0:
            if self.last_freq_sweep_result is None:
                self.log("<font color='orange'>No simulation result yet. Run a freq sweep first.</font>")
                return
            result = self.last_freq_sweep_result
        else:
            result = self.imaging_datasets[row - 1]['result']
        self.plot_radar_imaging(result)
        self.tabs.setCurrentIndex(4)   # Radar Imaging tab

    def export_freq_sweep_rcs_csv(self):
        """导出频扫 RCS 为长格式 CSV（列名与角扫 CSV 兼容，末列为 Frequency (MHz)）。"""
        if not self.last_freq_sweep_result:
            self.log("<font color='orange'>No freq sweep results to export.</font>")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Freq Sweep RCS CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            res       = self.last_freq_sweep_result
            freqs     = res['frequencies']           # (Nf,) Hz
            theta_deg = res['theta_deg']             # (n_theta,)
            phi_deg   = res['phi_deg']               # (n_phi,)
            scan_mode = res.get('scan_mode', '1d')
            fsp       = res.get('freq_sweep_params') or {}
            params    = res.get('params') or {}
            ptd_p     = params.get('ptd', {})
            ang_p     = params.get('angles', {})

            rcs_mat   = np.atleast_2d(res['rcs_matrix'])        # (N_angles, Nf)
            I_total   = np.atleast_2d(res['I_total_matrix'])    # (N_angles, Nf)
            I_po_raw  = res.get('I_po_matrix')
            I_ptd_raw = res.get('I_ptd_matrix')
            has_po    = I_po_raw is not None
            has_ptd   = I_ptd_raw is not None

            rcs_po_mat = rcs_ptd_mat = None
            I_po_mat = I_ptd_mat = None
            k_arr = 2.0 * np.pi * freqs / 299792458.0   # (Nf,)
            if has_po:
                I_po_mat = np.atleast_2d(I_po_raw)
                sigma_po = (k_arr[np.newaxis, :] ** 2 / np.pi) * np.abs(I_po_mat) ** 2
                rcs_po_mat = 10.0 * np.log10(np.maximum(sigma_po, 1e-30))
            if has_ptd:
                I_ptd_mat = np.atleast_2d(I_ptd_raw)
                sigma_ptd = (k_arr[np.newaxis, :] ** 2 / np.pi) * np.abs(I_ptd_mat) ** 2
                rcs_ptd_mat = 10.0 * np.log10(np.maximum(sigma_ptd, 1e-30))

            # angle_list mirrors run_freq_sweep: outer theta, inner phi
            angle_list = [(th, ph) for th in theta_deg for ph in phi_deg]

            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["# CEM PO Solver – Frequency Sweep RCS Results"])
                writer.writerow(["# Algorithm",          params.get('algorithm', '')])
                writer.writerow(["# Polarization",       fsp.get('polarization', '')])
                writer.writerow(["# PTD Enabled",        ptd_p.get('enabled', False)])
                writer.writerow(["# PTD Edges",          ptd_p.get('edges', '')])
                writer.writerow(["# Freq Start (MHz)",   fsp.get('f_start', '')])
                writer.writerow(["# Freq End (MHz)",     fsp.get('f_end', '')])
                writer.writerow(["# Freq Step (MHz)",    fsp.get('f_step', '')])
                writer.writerow(["# Theta Start (deg)",  ang_p.get('theta_start', '')])
                writer.writerow(["# Theta End (deg)",    ang_p.get('theta_end', '')])
                writer.writerow(["# N Theta",            ang_p.get('n_theta', '')])
                writer.writerow(["# Phi Start (deg)",    ang_p.get('phi_start', '')])
                writer.writerow(["# Phi End (deg)",      ang_p.get('phi_end', '')])
                writer.writerow(["# N Phi",              ang_p.get('n_phi', '')])
                writer.writerow(["# Scan Mode",          scan_mode])
                writer.writerow(["# Elapsed Time (s)",   f"{res.get('elapsed_time', 0):.3f}"])
                writer.writerow([])

                has_I_po  = I_po_raw  is not None
                has_I_ptd = I_ptd_raw is not None
                header = ["Theta (deg)", "Phi (deg)", "RCS Total (dBsm)"]
                if has_po:  header.append("RCS PO (dBsm)")
                if has_ptd: header.append("RCS PTD (dBsm)")
                header += ["I Total (Re)", "I Total (Im)"]
                if has_I_po:  header += ["I PO (Re)", "I PO (Im)"]
                if has_I_ptd: header += ["I PTD (Re)", "I PTD (Im)"]
                header.append("Frequency (MHz)")
                writer.writerow(header)

                for i, (th, ph) in enumerate(angle_list):
                    for j, freq_hz in enumerate(freqs):
                        row = [th, ph, rcs_mat[i, j]]
                        if has_po:  row.append(rcs_po_mat[i, j])
                        if has_ptd: row.append(rcs_ptd_mat[i, j])
                        row += [I_total[i, j].real, I_total[i, j].imag]
                        if has_I_po:  row += [I_po_mat[i, j].real,  I_po_mat[i, j].imag]
                        if has_I_ptd: row += [I_ptd_mat[i, j].real, I_ptd_mat[i, j].imag]
                        row.append(freq_hz / 1e6)
                        writer.writerow(row)

            self.log(f"Freq sweep RCS exported: {path}")
        except Exception as e:
            self.log(f"Export error: {e}")

    def export_range_profile_csv(self):
        """导出距离像数据到 CSV（不含频域原始数据）。"""
        if not self.last_freq_sweep_result:
            self.log("<font color='orange'>No freq sweep results to export.</font>")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Range Profile CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            res         = self.last_freq_sweep_result
            range_axis  = res['range_axis']
            profile_mat = np.atleast_2d(res['profile_matrix'])
            stats       = res.get('stats') or {}
            fsp         = res.get('freq_sweep_params') or {}
            params      = res.get('params') or {}
            ang_p       = params.get('angles', {})
            scan_mode   = res.get('scan_mode', '1d')

            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["# CEM PO Solver – Range Profile Results"])
                writer.writerow(["# Window",               fsp.get('window', '')])
                writer.writerow(["# Sidelobe (dB)",        fsp.get('cheby_at', '')])
                writer.writerow(["# Taylor nbar",          fsp.get('taylor_nbar', '')])
                writer.writerow(["# Zero Pad",             fsp.get('zero_pad', '')])
                writer.writerow(["# Range Resolution (m)", stats.get('range_resolution_m', '')])
                writer.writerow(["# Max Range (m)",        stats.get('max_range_m', '')])
                writer.writerow(["# Bandwidth (MHz)",      stats.get('bandwidth_mhz', '')])
                writer.writerow(["# Freq Start (MHz)",     fsp.get('f_start', '')])
                writer.writerow(["# Freq End (MHz)",       fsp.get('f_end', '')])
                writer.writerow(["# Theta Start (deg)",    ang_p.get('theta_start', '')])
                writer.writerow(["# Theta End (deg)",      ang_p.get('theta_end', '')])
                writer.writerow(["# N Theta",              ang_p.get('n_theta', '')])
                writer.writerow(["# Phi Start (deg)",      ang_p.get('phi_start', '')])
                writer.writerow(["# Phi End (deg)",        ang_p.get('phi_end', '')])
                writer.writerow(["# N Phi",                ang_p.get('n_phi', '')])
                writer.writerow(["# Scan Mode",            scan_mode])
                writer.writerow([])

                N_half = len(range_axis) // 2
                if scan_mode == '1d':
                    writer.writerow(["Range (m)", "Profile (dB)"])
                    for j in range(N_half):
                        writer.writerow([range_axis[j], profile_mat[0, j]])
                else:
                    hdr = ["Range (m)"] + [f"Profile_{i} (dB)" for i in range(len(profile_mat))]
                    writer.writerow(hdr)
                    for j in range(N_half):
                        row = [range_axis[j]] + [profile_mat[i, j] for i in range(len(profile_mat))]
                        writer.writerow(row)

            self.log(f"Range profile exported: {path}")
        except Exception as e:
            self.log(f"Export error: {e}")

    # ─────────────────────────── Statistics ───────────────────────────

    def _gather_stats_datasets(self):
        """
        收集统计数据集：当前仿真 + loaded data 中被勾选的项。
        返回: list of (name, rcs_db_array)
        """
        datasets = []

        # 1. Current simulation result (Total RCS only)
        r = self.last_result
        if r and r.get('rcs_total') is not None:
            rcs = np.asarray(r['rcs_total']).ravel()
            datasets.append(('Simulation', rcs))

        # 2. Loaded CSV files — 仅勾选项
        for idx, item in enumerate(self.comparison_data):
            list_item = self.comp_files_list.item(idx)
            if list_item and list_item.checkState() != Qt.Checked:
                continue
            filtered = self._comp_mgr._apply_freq_filter(item)
            res = self._comp_mgr._parse_csv_1d(filtered)
            if res:
                angle, rcs_total, name, rcs_po, rcs_ptd = res
                datasets.append((name, rcs_total))
            else:
                # fallback: 尝试找 RCS 列
                df = filtered['data']
                name = item['name']
                for col in df.columns:
                    cl = col.lower()
                    if 'total' in cl or 'rcs' in cl or 'dbsm' in cl:
                        datasets.append((name, np.asarray(df[col], dtype=float)))
                        break

        return datasets

    def _calc_statistics(self):
        """计算统计数据并更新 Statistics tab。"""
        from PySide6.QtWidgets import QTableWidgetItem
        from ui.statistics import (compute_statistics, compute_comparison_statistics,
                                   SINGLE_STAT_ROWS, COMPARE_STAT_ROWS)

        datasets = self._gather_stats_datasets()
        if not datasets:
            self.log("No data available for statistics.")
            return

        n_ds = len(datasets)
        stat_rows_single = SINGLE_STAT_ROWS

        # Compute single stats for each dataset
        all_single = []
        for name, rcs_db in datasets:
            s = compute_statistics(rcs_db)
            all_single.append((name, s))

        # Compute comparisons (each vs ref = first dataset)
        comparisons = []
        has_compare = (n_ds >= 2)
        if has_compare:
            ref_name, ref_rcs = datasets[0]
            for i in range(1, n_ds):
                other_name, other_rcs = datasets[i]
                min_len = min(len(ref_rcs), len(other_rcs))
                cs = compute_comparison_statistics(ref_rcs[:min_len], other_rcs[:min_len])
                comparisons.append((other_name, cs))

        # ── Single stats table ──
        def _fmt_val(val, fmt, key):
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                return 'N/A'
            if isinstance(val, int) or key == 'N':
                return str(int(val))
            return fmt.format(val)

        self.stats_table.setRowCount(len(stat_rows_single))
        self.stats_table.setColumnCount(1 + n_ds)
        headers = ['Statistic'] + [name for name, _ in all_single]
        self.stats_table.setHorizontalHeaderLabels(headers)

        for row_idx, (_, label, fmt, key) in enumerate(stat_rows_single):
            self.stats_table.setItem(row_idx, 0, QTableWidgetItem(label))
            for col_idx, (_, s) in enumerate(all_single):
                val_str = 'N/A' if s is None else _fmt_val(s.get(key), fmt, key)
                self.stats_table.setItem(row_idx, 1 + col_idx, QTableWidgetItem(val_str))

        # ── Comparison table (vs ref) ──
        if comparisons:
            ref_label = all_single[0][0]
            self.stats_comp_table.setVisible(True)
            self.stats_comp_table.setRowCount(len(COMPARE_STAT_ROWS))
            self.stats_comp_table.setColumnCount(1 + len(comparisons))
            comp_headers = ['Statistic'] + [f"{name} vs Ref" for name, _ in comparisons]
            self.stats_comp_table.setHorizontalHeaderLabels(comp_headers)

            for row_idx, (_, label, fmt, key) in enumerate(COMPARE_STAT_ROWS):
                self.stats_comp_table.setItem(row_idx, 0, QTableWidgetItem(label))
                for col_idx, (_, cs) in enumerate(comparisons):
                    val_str = 'N/A' if cs is None else _fmt_val(cs.get(key), fmt, key)
                    self.stats_comp_table.setItem(row_idx, 1 + col_idx, QTableWidgetItem(val_str))
        else:
            self.stats_comp_table.setVisible(False)

        self.stats_table.resizeColumnsToContents()
        if self.stats_comp_table.isVisible():
            self.stats_comp_table.resizeColumnsToContents()

        # ── Plot PDF ──
        self.stats_figure.clear()
        ax = self.stats_figure.add_subplot(111)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, (name, rcs_db) in enumerate(datasets):
            valid = np.isfinite(rcs_db) & (rcs_db > -200)
            if not valid.any():
                continue
            data = rcs_db[valid]
            color = colors[i % len(colors)]

            # Histogram as density
            bins = min(max(int(np.sqrt(len(data))), 10), 80)
            ax.hist(data, bins=bins, density=True, alpha=0.35, color=color, edgecolor=color, linewidth=0.5)

            # KDE curve
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min() - 3, data.max() + 3, 300)
                ax.plot(x_range, kde(x_range), color=color, linewidth=1.5, label=name)
            except Exception:
                pass

        ax.set_xlabel('RCS (dBsm)')
        ax.set_ylabel('Probability Density')
        ax.set_title('RCS Distribution')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        self.stats_figure.tight_layout()
        self.stats_canvas.draw()

        # Cache for export
        self._stats_cache = {
            'single': all_single,
            'comparisons': comparisons,
            'single_rows': stat_rows_single,
            'compare_rows': COMPARE_STAT_ROWS,
        }

        # Switch to Statistics tab
        self.tabs.setCurrentIndex(3)
        self.log(f"Statistics calculated for {n_ds} dataset(s).")

    def _export_statistics_csv(self):
        """导出统计数据到 CSV 文件。"""
        if not hasattr(self, '_stats_cache') or not self._stats_cache:
            self.log("No statistics to export. Run Calc Statistics first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Statistics CSV", "statistics.csv", "CSV Files (*.csv)")
        if not path:
            return

        try:
            cache = self._stats_cache
            all_single = cache['single']
            comparisons = cache['comparisons']
            stat_rows = cache['single_rows']
            comp_rows = cache['compare_rows']

            def _fmt_csv(val, fmt, key):
                if val is None or (isinstance(val, float) and not np.isfinite(val)):
                    return ''
                if isinstance(val, int) or key == 'N':
                    return int(val)
                return fmt.format(val)

            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)

                # Single stats
                headers = ['Statistic'] + [name for name, _ in all_single]
                writer.writerow(headers)
                for _, label, fmt, key in stat_rows:
                    row = [label]
                    for _, s in all_single:
                        row.append('' if s is None else _fmt_csv(s.get(key), fmt, key))
                    writer.writerow(row)

                # Comparison stats
                if comparisons:
                    writer.writerow([])
                    comp_headers = ['Comparison vs Ref'] + [f"{name} vs Ref" for name, _ in comparisons]
                    writer.writerow(comp_headers)
                    for _, label, fmt, key in comp_rows:
                        row = [label]
                        for _, cs in comparisons:
                            row.append('' if cs is None else _fmt_csv(cs.get(key), fmt, key))
                        writer.writerow(row)

            self.log(f"Statistics exported: {path}")
        except Exception as e:
            self.log(f"Export error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CEMPoQtWindow()
    window.show()
    sys.exit(app.exec())
