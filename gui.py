import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os
import json
import time
import numpy as np
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置matplotlib中文字体 (Windows)
# 必须在导入pyplot后立即设置
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 确保能导入项目模块
sys.path.append(os.path.dirname(__file__))

from geometry.plate import AnalyticPlate
from geometry.sphere import AnalyticSphere
from geometry.cylinder import AnalyticCylinder
from geometry.occ_surface import OCCSurface
from geometry.step_loader import load_step_file
from geometry.wedge import create_analytic_wedge
from geometry.brick import create_analytic_brick
from physics.wave import IncidentWave
from physics.analytical_rcs import get_analytical_solution, compute_error_stats
from solver.ribbon_solver import RibbonIntegrator, RCSAnalyzer, get_integrator, list_algorithms, AVAILABLE_ALGORITHMS
from tools.visualize_mesh import create_occ_cylinder
from ui.plotting import VisualizationManager

CONFIG_FILE = "cem_po_config.json"

class CEMPoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CEM PO Solver")
        self.root.geometry("1900x1000")
        self.root.minsize(1600, 900)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- 初始化所有状态变量 ---
        self.viz_manager = None
        self.last_result = None
        self.current_geometry = None
        self.step_file_path = None
        self.step_unit_var = tk.StringVar(value="mm")
        self.invert_indices_var = tk.StringVar(value="0,1,3,5")

        # 几何参数缓存
        self.geo_params_cache = {
            "radius": 1.0,
            "height": 2.0,
            "width": 5.0,
            "length": 10.0
        }

        # 网格缓存
        self.cached_mesh_data = None
        self.cached_mesh_params = {}
        self.current_geometry = None
        self.current_geo_params = {}

        # 配置 TTK 样式
        self.colors = {
            "bg_main": "#FAFAFA",
            "bg_panel": "#FFFFFF",
            "fg_text": "#333333",
            "accent": "#007ACC",
            "border": "#E0E0E0",
            "button_bg": "#F0F0F0",
        }
        self.root.configure(bg=self.colors["bg_main"])

        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background=self.colors["bg_main"], foreground=self.colors["fg_text"], font=("Microsoft YaHei UI", 9))
        style.configure("TFrame", background=self.colors["bg_main"])
        style.configure("Card.TFrame", background=self.colors["bg_panel"], relief="flat")
        style.configure("TLabelframe", background=self.colors["bg_panel"], bordercolor=self.colors["border"], relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=self.colors["bg_panel"], foreground="#555555", font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("TLabel", background=self.colors["bg_panel"], foreground=self.colors["fg_text"])
        style.configure("Main.TLabel", background=self.colors["bg_main"])
        style.configure("TButton", background=self.colors["button_bg"], borderwidth=1, relief="solid", padding=5)
        style.map("TButton", background=[("active", "#E5E5E5")], relief=[("pressed", "sunken")])
        style.configure("TEntry", fieldbackground="#FFFFFF", bordercolor=self.colors["border"], padding=5)
        style.configure("TCombobox", fieldbackground="#FFFFFF", arrowcolor=self.colors["fg_text"])

        # --- 界面布局 ---
        main_frame = ttk.Frame(root, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        left_panel = ttk.LabelFrame(main_frame, text=" 配置与几何 (Configuration) ", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), ipadx=5)

        right_panel = ttk.Frame(main_frame, style="TFrame")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_config_widgets(left_panel)
        self.create_geometry_widgets(left_panel)
        self.create_action_widgets(left_panel)
        self.create_log_widgets(right_panel)

        self.load_config()

    def create_config_widgets(self, parent):
        ttk.Label(parent, text="频率 Frequency (MHz):").pack(anchor=tk.W, pady=(0, 5))
        self.freq_var = tk.DoubleVar(value=300.0)
        ttk.Entry(parent, textvariable=self.freq_var).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(parent, text="网格密度 Grid Density (Samples/Lambda):").pack(anchor=tk.W, pady=(0, 5))
        self.density_var = tk.IntVar(value=10)
        ttk.Entry(parent, textvariable=self.density_var).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(parent, text="Theta 范围 (Start, End, Points):").pack(anchor=tk.W, pady=(0, 5))
        theta_frame = ttk.Frame(parent, style="Card.TFrame")
        theta_frame.pack(fill=tk.X, pady=(0, 10))
        self.theta_start = tk.DoubleVar(value=0.0)
        self.theta_end = tk.DoubleVar(value=180.0)
        self.theta_n = tk.IntVar(value=91)
        ttk.Entry(theta_frame, textvariable=self.theta_start, width=6).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(theta_frame, textvariable=self.theta_end, width=6).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(theta_frame, textvariable=self.theta_n, width=6).pack(side=tk.LEFT)

        ttk.Label(parent, text="Phi 范围 (Start, End, Points):").pack(anchor=tk.W, pady=(0, 5))
        phi_frame = ttk.Frame(parent, style="Card.TFrame")
        phi_frame.pack(fill=tk.X, pady=(0, 10))
        self.phi_start = tk.DoubleVar(value=-45.0)
        self.phi_end = tk.DoubleVar(value=45.0)
        self.phi_n = tk.IntVar(value=1)
        ttk.Entry(phi_frame, textvariable=self.phi_start, width=6).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(phi_frame, textvariable=self.phi_end, width=6).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(phi_frame, textvariable=self.phi_n, width=6).pack(side=tk.LEFT)

        ttk.Label(parent, text="(Phi点数=1为1D扫描，>1为2D扫描)", foreground="#888888", font=("Microsoft YaHei UI", 8)).pack(anchor=tk.W)
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

    def create_geometry_widgets(self, parent):
        ttk.Label(parent, text="几何类型 Geometry Type:").pack(anchor=tk.W, pady=(0, 5))
        self.geo_type_var = tk.StringVar(value="STEP File")
        types = ["Analytic Cylinder", "Analytic Plate", "Analytic Sphere", "Standard Wedge (PTD Test)", "Standard Brick (PTD Test)", "OCC Cylinder (NURBS)", "STEP File"]
        combo = ttk.Combobox(parent, textvariable=self.geo_type_var, values=types, state="readonly")
        combo.pack(fill=tk.X, pady=(0, 10))
        combo.bind("<<ComboboxSelected>>", self.update_geo_inputs)

        self.geo_params_frame = ttk.Frame(parent, style="Card.TFrame")
        self.geo_params_frame.pack(fill=tk.X, pady=5)
        self.update_geo_inputs()

    def save_geo_params(self):
        for key in ["radius", "height", "width", "length"]:
            var_name = f"geo_{key}"
            if hasattr(self, var_name):
                try:
                    val = getattr(self, var_name).get()
                    self.geo_params_cache[key] = val
                except:
                    pass

    def update_geo_inputs(self, event=None):
        self.save_geo_params()
        for widget in self.geo_params_frame.winfo_children():
            widget.destroy()

        geo_type = self.geo_type_var.get()

        if geo_type == "Analytic Cylinder" or geo_type == "OCC Cylinder (NURBS)":
            self.add_param_input("半径 Radius (m):", "radius", 1.0)
            self.add_param_input("高度 Height (m):", "height", 2.0)
        elif geo_type == "Analytic Plate":
            self.add_param_input("宽度 Width (m):", "width", 5.0)
            self.add_param_input("高度 Height (m):", "height", 5.0)
        elif geo_type == "Analytic Sphere":
            self.add_param_input("半径 Radius (m):", "radius", 1.0)
        elif geo_type == "Standard Wedge (PTD Test)":
            self.add_param_input("长度 Length (m):", "length", 10.0)
            self.add_param_input("宽度 Width (m):", "width", 5.0)
            self.add_param_input("高度 Height (m):", "height", 5.0)
            # Auto-set PTD
            self.ptd_enabled_var.set(True)
            self.ptd_edges_var.set("F0E2")
            if hasattr(self, 'ptd_entry'):
                self.ptd_entry.configure(state='normal')
        elif geo_type == "Standard Brick (PTD Test)":
            self.add_param_input("长度 Length (m):", "length", 10.0)
            self.add_param_input("宽度 Width (m):", "width", 5.0)
            self.add_param_input("高度 Height (m):", "height", 3.0)
            # Auto-set PTD for Brick (12 edges)
            self.ptd_enabled_var.set(True)
            try:
                l = self.geo_params_cache.get("length", 10.0)
                w = self.geo_params_cache.get("width", 5.0)
                h = self.geo_params_cache.get("height", 3.0)
                _, ptd_ids = create_analytic_brick(l, w, h)
                self.ptd_edges_var.set(ptd_ids)
            except:
                pass

            if hasattr(self, 'ptd_entry'):
                self.ptd_entry.configure(state='normal')

        elif geo_type == "STEP File":
            btn = ttk.Button(self.geo_params_frame, text="📂 选择 STEP 文件...", command=self.browse_step)
            btn.pack(fill=tk.X, pady=5)
            self.step_label = ttk.Label(self.geo_params_frame, text="未选择文件", foreground="#888888", wraplength=200)
            if self.step_file_path:
                self.step_label.config(text=os.path.basename(self.step_file_path))
            self.step_label.pack(fill=tk.X)

            unit_frame = ttk.Frame(self.geo_params_frame, style="Card.TFrame")
            unit_frame.pack(fill=tk.X, pady=(5, 0))
            ttk.Label(unit_frame, text="STEP 单位:").pack(side=tk.LEFT)
            if not hasattr(self, 'step_unit_var'):
                self.step_unit_var = tk.StringVar(value="mm")
            ttk.Radiobutton(unit_frame, text="mm", variable=self.step_unit_var, value="mm").pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(unit_frame, text="m", variable=self.step_unit_var, value="m").pack(side=tk.LEFT)

            face_frame = ttk.Frame(self.geo_params_frame, style="Card.TFrame")
            face_frame.pack(fill=tk.X, pady=(5, 0))
            ttk.Label(face_frame, text="面索引:").pack(side=tk.LEFT)
            self.step_face_idx = tk.IntVar(value=0)
            ttk.Entry(face_frame, textvariable=self.step_face_idx, width=5).pack(side=tk.LEFT, padx=5)
            ttk.Button(face_frame, text="👁 预览单面", command=self.preview_step_single).pack(side=tk.LEFT)

            invert_frame = ttk.Frame(self.geo_params_frame, style="Card.TFrame")
            invert_frame.pack(fill=tk.X, pady=(5, 0))
            ttk.Label(invert_frame, text="翻转法线索引:").pack(side=tk.LEFT)
            if not hasattr(self, 'invert_indices_var'):
                self.invert_indices_var = tk.StringVar(value="0,1,3,5")
            ttk.Entry(invert_frame, textvariable=self.invert_indices_var, width=15).pack(side=tk.LEFT, padx=5)
            ttk.Label(invert_frame, text="(逗号分隔)", foreground="#888888", font=("", 8)).pack(side=tk.LEFT)

        # === 公共按钮区域 (无论选什么几何体都显示) ===
        ttk.Separator(self.geo_params_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        btn_preview = ttk.Button(self.geo_params_frame, text="👁 预览全部 (Preview Geometry)", command=self.preview_geometry)
        btn_preview.pack(fill=tk.X, pady=(5, 0))

    def add_param_input(self, label, var_name, default):
        frame = ttk.Frame(self.geo_params_frame, style="Card.TFrame")
        frame.pack(fill=tk.X, pady=3)
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        current_val = self.geo_params_cache.get(var_name, default)
        var = tk.DoubleVar(value=current_val)
        ttk.Entry(frame, textvariable=var, width=10).pack(side=tk.RIGHT)
        setattr(self, f"geo_{var_name}", var)

    def browse_step(self):
        filename = filedialog.askopenfilename(filetypes=[("STEP Files", "*.stp;*.step")])
        if filename:
            self.step_file_path = filename
            self.step_label.config(text=os.path.basename(filename))
            self.log(f"Selected STEP file: {filename}")

    def preview_step_single(self):
        if not self.step_file_path:
            messagebox.showwarning("Warning", "请先选择 STEP 文件")
            return
        try:
            from solver.ribbon_solver import detect_degenerate_edge, RibbonIntegrator
            surfaces = load_step_file(self.step_file_path)
            face_idx = self.step_face_idx.get()
            if face_idx < 0 or face_idx >= len(surfaces):
                messagebox.showwarning("Warning", f"面索引超出范围 (0-{len(surfaces)-1})")
                return
            surf = surfaces[face_idx]
            degen_edge = detect_degenerate_edge(surf)
            edges_data = surf.get_edges(n_samples=30)
            solver = RibbonIntegrator()
            self.viz_manager.show_single_face_preview(surf, face_idx, degen_edge, edges_data, solver)
        except Exception as e:
            self.log(f"Single face preview error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def preview_geometry(self):
        geo = self.build_geometry()
        if not geo: return
        self.log(f"Loading geometry for preview...")
        surfaces = geo if isinstance(geo, list) else [geo]
        try:
            real_freq = self.freq_var.get() * 1e6
            real_samples = self.density_var.get()
        except:
            real_freq = 300e6
            real_samples = 10
        self.log(f"Previewing with Freq={real_freq/1e6:.1f}MHz, Density={real_samples} pts/lambda")
        try:
            scan_params = {'theta_start': self.theta_start.get(), 'theta_end': self.theta_end.get(), 'theta_n': self.theta_n.get(), 'phi_start': self.phi_start.get(), 'phi_end': self.phi_end.get(), 'phi_n': self.phi_n.get()}
        except:
            scan_params = {'theta_start': 0.0, 'theta_end': 0.0, 'theta_n': 1, 'phi_start': 0.0, 'phi_end': 0.0, 'phi_n': 1}
        enable_ptd = self.ptd_enabled_var.get()
        ptd_identifiers = []
        if enable_ptd:
            raw = self.ptd_edges_var.get()
            ptd_identifiers = [s.strip() for s in raw.split(',') if s.strip()]
        threading.Thread(target=self._preview_geometry_thread, args=(surfaces, real_freq, real_samples, scan_params, enable_ptd, ptd_identifiers), daemon=True).start()

    def _preview_geometry_thread(self, surfaces, freq, samples, scan_params, enable_ptd=False, ptd_identifiers=None):
        """后台线程：计算预览网格数据 (快速预览模式)"""
        try:
            mesh_data_list = []
            total_points = 0

            # 使用固定网格密度进行快速预览，忽略物理频率设置
            fixed_nu, fixed_nv = 30, 30

            for i, surf in enumerate(surfaces):
                try:
                    u_min, u_max = surf.u_domain
                    v_min, v_max = surf.v_domain

                    u = np.linspace(u_min, u_max, fixed_nu)
                    v = np.linspace(v_min, v_max, fixed_nv)
                    uu, vv = np.meshgrid(u, v)

                    points, normals, jacobians = surf.get_data(uu, vv)

                    step_id = getattr(surf, 'step_id', -1)
                    mesh_data_list.append({
                        'points': points,
                        'normals': normals,
                        'jacobians': jacobians,
                        'nu': fixed_nu,
                        'nv': fixed_nv,
                        'step_id': step_id,
                        'local_idx': i
                    })
                    total_points += fixed_nu * fixed_nv
                except Exception as e:
                    self.root.after(0, lambda e=e, i=i: self.log(f"Surface {i} preview error: {e}"))

            ptd_edges_data = []
            if enable_ptd and ptd_identifiers:
                try:
                    from solver.manual_edges import extract_manual_edges
                    ptd_edges = extract_manual_edges(surfaces, ptd_identifiers)
                    for edge in ptd_edges:
                        if hasattr(edge, 'points'):
                            ptd_edges_data.append({'name': edge.name, 'points': edge.points})
                        else:
                            ptd_edges_data.append({'name': edge.name, 'start': edge.start, 'end': edge.end})
                except Exception as e:
                    self.root.after(0, lambda e=e: self.log(f"PTD Preview Error: {e}"))

            self.root.after(0, lambda: self.viz_manager.show_step_preview(mesh_data_list, total_points, scan_params, ptd_edges_data))

        except Exception as e:
            self.root.after(0, lambda: self.log(f"Preview Error: {e}"))

    def create_action_widgets(self, parent):
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)

        algo_frame = ttk.LabelFrame(parent, text="算法 Algorithm", padding=(10, 5))
        algo_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(algo_frame, text="积分方法:").pack(anchor=tk.W)
        default_algo = 'discrete_po_sinc_dual'
        algo_choices = [(info['name'], key) for key, info in AVAILABLE_ALGORITHMS.items()]
        self._algo_name_to_id = {name: key for name, key in algo_choices}
        self._algo_id_to_name = {key: name for name, key in algo_choices}

        self.algorithm_var = tk.StringVar(value=self._algo_id_to_name.get(default_algo, algo_choices[0][0]))
        algo_combo = ttk.Combobox(
            algo_frame,
            textvariable=self.algorithm_var,
            values=[name for name, _ in algo_choices],
            state="readonly",
            width=30
        )
        algo_combo.pack(fill=tk.X, pady=(0, 5))

        self.algo_desc_label = ttk.Label(
            algo_frame,
            text=AVAILABLE_ALGORITHMS[default_algo]['description'],
            foreground="#666666",
            font=("Microsoft YaHei UI", 8),
            wraplength=250
        )
        self.algo_desc_label.pack(anchor=tk.W)

        def on_algo_change(event=None):
            display_name = self.algorithm_var.get()
            algo_id = self._algo_name_to_id.get(display_name, default_algo)
            desc = AVAILABLE_ALGORITHMS.get(algo_id, {}).get('description', '')
            self.algo_desc_label.config(text=desc)

        algo_combo.bind("<<ComboboxSelected>>", on_algo_change)

        # === PTD 设置 (Edge Diffraction) ===
        ptd_frame = ttk.LabelFrame(parent, text="PTD 设置 (Edge Diffraction)", padding=(10, 5))
        ptd_frame.pack(fill=tk.X, pady=(0, 10))

        self.ptd_enabled_var = tk.BooleanVar(value=False)
        def toggle_ptd():
            if self.ptd_enabled_var.get():
                self.ptd_entry.configure(state='normal')
            else:
                self.ptd_entry.configure(state='disabled')

        ttk.Checkbutton(
            ptd_frame,
            text="启用 PTD (Enable PTD)",
            variable=self.ptd_enabled_var,
            command=toggle_ptd
        ).pack(anchor=tk.W)

        ttk.Label(ptd_frame, text="边缘列表 (Edges, e.g. F0E1,F1E2):").pack(anchor=tk.W, pady=(5, 0))
        self.ptd_edges_var = tk.StringVar(value="F0E1,F1E2,F2E3,F5E1,F6E3,F7E3")
        self.ptd_entry = ttk.Entry(ptd_frame, textvariable=self.ptd_edges_var, state='disabled')
        self.ptd_entry.pack(fill=tk.X, pady=(0, 5))

        # 极化选择
        pol_frame = ttk.Frame(ptd_frame)
        pol_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(pol_frame, text="极化 (Polarization):").pack(side=tk.LEFT)
        self.pol_var = tk.StringVar(value="VV")
        self.pol_combo = ttk.Combobox(pol_frame, textvariable=self.pol_var, values=["VV", "HH"], state="readonly", width=5)
        self.pol_combo.pack(side=tk.LEFT, padx=5)

        # 对比解析解
        self.compare_analytical_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ptd_frame,
            text="显示参考/解析解 (Show Ref/GTD)",
            variable=self.compare_analytical_var
        ).pack(anchor=tk.W, pady=(5, 0))

        # === 并行计算设置 ===
        parallel_frame = ttk.LabelFrame(parent, text="性能 Performance", padding=(10, 5))
        parallel_frame.pack(fill=tk.X, pady=(0, 10))

        import os
        max_cpu = os.cpu_count() or 4
        self.parallel_var = tk.BooleanVar(value=False)
        self.workers_var = tk.IntVar(value=max_cpu)

        def toggle_workers():
            if self.parallel_var.get():
                spin_workers.configure(state='normal')
            else:
                spin_workers.configure(state='disabled')

        chk_parallel = ttk.Checkbutton(
            parallel_frame,
            text="启用并行计算 (Parallel)",
            variable=self.parallel_var,
            command=toggle_workers
        )
        chk_parallel.pack(anchor=tk.W)

        workers_frame = ttk.Frame(parallel_frame)
        workers_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(workers_frame, text="CPU 核心数:").pack(side=tk.LEFT)
        spin_workers = ttk.Spinbox(
            workers_frame,
            from_=1,
            to=32,
            textvariable=self.workers_var,
            width=5,
            state='disabled'
        )
        spin_workers.pack(side=tk.LEFT, padx=5)

        btn_gen_mesh = ttk.Button(parent, text="📊 生成网格 (Generate Mesh)", command=self.generate_mesh_stats)
        btn_gen_mesh.pack(fill=tk.X, pady=(0, 8))

        btn_mesh = ttk.Button(parent, text="🧊 可视化网格 (Visualize Mesh)", command=self.visualize_mesh)
        btn_mesh.pack(fill=tk.X, pady=(0, 8))

        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=(0, 10))

        btn_calc = ttk.Button(action_frame, text="🚀 计算 RCS (Calculate)", command=self.run_calculation)
        btn_calc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        btn_export = ttk.Button(action_frame, text="💾 导出 (Export)", command=self.export_to_csv)
        btn_export.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        ttk.Label(parent, text="计算进度:").pack(anchor=tk.W, pady=(10, 2))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            parent,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        self.progress_label = ttk.Label(parent, text="就绪", foreground="#888888")
        self.progress_label.pack(anchor=tk.W)

    def create_log_widgets(self, parent):
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        log_container = ttk.Frame(paned, width=420)
        paned.add(log_container, weight=0)

        ttk.Label(log_container, text="系统日志 System Log:", style="Main.TLabel",
                  font=("Microsoft YaHei UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))

        log_frame = ttk.Frame(log_container)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame,
            height=20,
            width=50,
            state='disabled',
            bg="#FFFFFF",
            fg="#444444",
            font=("Microsoft YaHei UI", 9),
            relief="flat",
            padx=10, pady=10,
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.colors["border"]
        )
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        plot_container = ttk.Frame(paned)
        paned.add(plot_container, weight=1)

        self.viz_notebook = ttk.Notebook(plot_container)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)

        self.preview_tab = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.preview_tab, text="  几何预览 Preview  ")

        self.rcs_tab = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.rcs_tab, text="  RCS 结果 Results  ")

        self.compare_tab = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.compare_tab, text="  对比 Comparison  ")

        self.preview_placeholder = ttk.Label(self.preview_tab,
            text="点击 '预览全部' 或 '可视化网格' 查看几何图形",
            foreground="#888888", font=("Microsoft YaHei UI", 10))
        self.preview_placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.rcs_placeholder = ttk.Label(self.rcs_tab,
            text="计算完成后将在此显示 RCS 结果",
            foreground="#888888", font=("Microsoft YaHei UI", 10))
        self.rcs_placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.create_comparison_widgets(self.compare_tab)

        self.log("CEM PO Solver GUI Ready.")
        self.log("Based on PythonOCC and Ribbon Method.")

        self.viz_manager = VisualizationManager(
            self.root, self.log, self.colors,
            preview_frame=self.preview_tab,
            rcs_frame=self.rcs_tab,
            compare_frame=self.compare_plot_frame,
            notebook=self.viz_notebook
        )

    def create_comparison_widgets(self, parent):
        control_frame = ttk.Frame(parent, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        calc_group = ttk.LabelFrame(control_frame, text="计算数据源 (Calculated Data)", padding=5)
        calc_group.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(calc_group, text="数据源 A (Primary):", font=("", 9, "bold")).pack(anchor=tk.W)
        self.calc_source_var = tk.StringVar(value="current")

        rb1 = ttk.Radiobutton(calc_group, text="当前计算结果 (Current Session)", variable=self.calc_source_var, value="current")
        rb1.pack(anchor=tk.W)

        frame_csv = ttk.Frame(calc_group)
        frame_csv.pack(anchor=tk.W, fill=tk.X)
        rb2 = ttk.Radiobutton(frame_csv, text="加载 CSV:", variable=self.calc_source_var, value="csv")
        rb2.pack(side=tk.LEFT)

        self.btn_load_calc = ttk.Button(frame_csv, text="浏览...", width=6, command=lambda: self.load_csv_slot(1))
        self.btn_load_calc.pack(side=tk.LEFT, padx=5)

        self.lbl_calc_file = ttk.Label(calc_group, text="(未选择文件)", foreground="#888888", font=("", 8))
        self.lbl_calc_file.pack(anchor=tk.W, padx=20)

        ttk.Separator(calc_group, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Label(calc_group, text="数据源 B (Optional Compare):", font=("", 9, "bold")).pack(anchor=tk.W)
        frame_csv2 = ttk.Frame(calc_group)
        frame_csv2.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(frame_csv2, text="对比 CSV:").pack(side=tk.LEFT)
        self.btn_load_calc2 = ttk.Button(frame_csv2, text="浏览...", width=6, command=lambda: self.load_csv_slot(2))
        self.btn_load_calc2.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_csv2, text="清除", width=4, command=self.clear_csv2).pack(side=tk.LEFT, padx=2)
        self.lbl_calc_file_2 = ttk.Label(calc_group, text="(未选择 - 仅单对比)", foreground="#888888", font=("", 8))
        self.lbl_calc_file_2.pack(anchor=tk.W, padx=20)

        self.loaded_calc_result = None
        self.loaded_calc_result_2 = None

        ref_group = ttk.LabelFrame(control_frame, text="参考数据 (Reference Data)", padding=5)
        ref_group.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.ref_data_dir = tk.StringVar(value=r"F:\data\parameter\csv_output")
        ttk.Label(ref_group, text="目录:").pack(side=tk.LEFT)
        ttk.Entry(ref_group, textvariable=self.ref_data_dir, width=25).pack(side=tk.LEFT, padx=5)
        ttk.Button(ref_group, text="...", width=3, command=lambda: self.ref_data_dir.set(filedialog.askdirectory() or self.ref_data_dir.get())).pack(side=tk.LEFT, padx=(0, 10))

        frame_ref_params = ttk.Frame(ref_group)
        frame_ref_params.pack(side=tk.LEFT, fill=tk.X, pady=5)
        ttk.Label(frame_ref_params, text="模型ID:").pack(side=tk.LEFT)
        self.comp_model_id = tk.StringVar(value="001")
        ttk.Entry(frame_ref_params, textvariable=self.comp_model_id, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame_ref_params, text="频率:").pack(side=tk.LEFT)
        self.comp_freq_suffix = tk.StringVar(value="1.5G")
        combo_freq = ttk.Combobox(frame_ref_params, textvariable=self.comp_freq_suffix, values=["1.5G", "3G"], width=6)
        combo_freq.pack(side=tk.LEFT, padx=5)

        action_group = ttk.Frame(control_frame, padding=5)
        action_group.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(action_group, text="风格:").pack(side=tk.TOP, anchor=tk.W)
        self.plot_style_var = tk.StringVar(value="pixel")
        ttk.Combobox(action_group, textvariable=self.plot_style_var, values=["pixel", "contour"], width=8, state="readonly").pack(side=tk.TOP, pady=2)
        ttk.Button(action_group, text="执行对比\nRun Comparison", command=self.run_comparison).pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.compare_plot_frame = ttk.Frame(parent)
        self.compare_plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(self.compare_plot_frame, text="请设置参数并点击 '执行对比'\n支持: [计算结果 A vs 参考] 或 [结果 A vs 结果 B vs 参考]", foreground="#888888", justify=tk.CENTER).place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def load_csv_slot(self, slot_id):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")], title=f"选择计算结果 CSV {slot_id}")
        if not file_path: return
        result = self._parse_csv_result(file_path)
        if result:
            filename = os.path.basename(file_path)
            if slot_id == 1:
                self.loaded_calc_result = result
                self.lbl_calc_file.config(text=filename)
                self.calc_source_var.set("csv")
            else:
                self.loaded_calc_result_2 = result
                self.lbl_calc_file_2.config(text=filename, foreground="#000000")
            self.log(f"CSV {slot_id} loaded: {filename}")

    def clear_csv2(self):
        self.loaded_calc_result_2 = None
        self.lbl_calc_file_2.config(text="(未选择 - 仅单对比)", foreground="#888888")

    def _parse_csv_result(self, file_path):
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            required_cols_2d = {'Theta', 'Phi', 'RCS(dBsm)'}
            cols = set(df.columns)
            if required_cols_2d.issubset(cols):
                pivot = df.pivot(index='Theta', columns='Phi', values='RCS(dBsm)')
                result_data = {
                    'mode': '2d',
                    'theta_deg': pivot.index.values,
                    'phi_deg': pivot.columns.values,
                    'rcs_2d': pivot.values,
                    'geo_type': 'Loaded CSV',
                    'freq': 0
                }
                return result_data
            else:
                 messagebox.showerror("格式错误", f"CSV 缺少必要列: {required_cols_2d}\n检测到: {cols}")
                 return None
        except Exception as e:
            self.log(f"Error parsing CSV: {e}")
            messagebox.showerror("CSV 解析失败", str(e))
            return None

    def run_comparison(self):
        source = self.calc_source_var.get()
        calc_result_1 = None
        if source == "current":
            calc_result_1 = self.last_result
            if calc_result_1 is None:
                messagebox.showwarning("警告", "没有当前计算结果！")
                return
        elif source == "csv":
            calc_result_1 = self.loaded_calc_result
            if calc_result_1 is None:
                 messagebox.showwarning("警告", "尚未加载 CSV 1 文件！")
                 return
        calc_result_2 = self.loaded_calc_result_2
        data_dir = self.ref_data_dir.get()
        model_id = self.comp_model_id.get()
        freq_suffix = self.comp_freq_suffix.get()
        style = self.plot_style_var.get()
        if not os.path.exists(data_dir):
             messagebox.showerror("错误", f"数据目录不存在:\n{data_dir}")
             return
        msg = f"Starting comparison: A vs Ref"
        if calc_result_2: msg += " vs B"
        self.log(msg + f" (Model={model_id})")
        threading.Thread(target=self._comparison_thread, args=(data_dir, model_id, freq_suffix, style, calc_result_1, calc_result_2), daemon=True).start()

    def _comparison_thread(self, data_dir, model_id, freq_suffix, style, calc_result_1, calc_result_2=None):
        try:
            import sys
            base_dir = os.path.dirname(os.path.abspath(__file__))
            read_compare_path = os.path.join(base_dir, 'read-compare')
            if read_compare_path not in sys.path: sys.path.append(read_compare_path)
            try: from rcs_data_reader import get_adaptive_rcs_matrix
            except ImportError as ie:
                self.root.after(0, lambda: messagebox.showerror("模块导入错误", f"无法导入 rcs_data_reader。\n详细错误: {ie}"))
                return
            self.root.after(0, lambda: self.log(f"Processing Data A..."))
            if calc_result_1['mode'] != '2d':
                self.root.after(0, lambda: messagebox.showwarning("模式错误", "数据 A 必须是 2D 扫描结果。" ))
                return
            rcs_a = calc_result_1.get('rcs_2d')
            if np.nanmax(rcs_a) < 0.001 and np.nanmin(rcs_a) >= 0:
                 rcs_a = 10 * np.log10(np.maximum(rcs_a, 1e-15))
            theta_a, phi_a = calc_result_1['theta_deg'], calc_result_1['phi_deg']
            rcs_b = None
            if calc_result_2:
                self.root.after(0, lambda: self.log(f"Processing Data B..."))
                if calc_result_2['mode'] != '2d':
                    self.root.after(0, lambda: messagebox.showwarning("模式错误", "数据 B 必须是 2D 扫描结果。" ))
                    return
                rcs_b = calc_result_2.get('rcs_2d')
                if np.nanmax(rcs_b) < 0.001 and np.nanmin(rcs_b) >= 0:
                     rcs_b = 10 * np.log10(np.maximum(rcs_b, 1e-15))
                if rcs_a.shape != rcs_b.shape:
                    self.root.after(0, lambda: messagebox.showwarning("网格不匹配", "数据 A 和 B 的网格尺寸不一致，无法同台对比。" ))
                    return
            self.root.after(0, lambda: self.log(f"Loading Reference {model_id}..."))
            try: ref_data_pkg = get_adaptive_rcs_matrix(model_id, freq_suffix, data_dir, verbose=False)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("参考数据读取失败", str(e)))
                return
            rcs_ref = ref_data_pkg['rcs_db']
            if rcs_a.shape != rcs_ref.shape:
                msg = f"网格不匹配!\nA: {rcs_a.shape}, Ref: {rcs_ref.shape}"
                self.root.after(0, lambda: messagebox.showerror("网格不匹配", msg))
                return
            def calc_metrics(diff):
                mse = np.nanmean(diff**2)
                return {'rmse': np.sqrt(mse), 'mean_error': np.nanmean(diff)}
            diff_a = rcs_a - rcs_ref
            metrics_a = calc_metrics(diff_a)
            metrics_b = None
            diff_b = None
            if rcs_b is not None:
                diff_b = rcs_b - rcs_ref
                metrics_b = calc_metrics(diff_b)
            self.root.after(0, lambda: self.log("Rendering plots..."))
            if rcs_b is not None:
                self.root.after(0, lambda: self.viz_manager.show_comparison_dual_2d(rcs_a, rcs_b, rcs_ref, diff_a, diff_b, theta_a, phi_a, metrics_a, metrics_b, style))
            else:
                self.root.after(0, lambda: self.viz_manager.show_comparison_2d(rcs_a, rcs_ref, diff_a, theta_a, phi_a, metrics_a, style))
            self.root.after(0, lambda: self.log("Comparison complete."))
        except Exception as e:
             import traceback
             traceback.print_exc()
             self.root.after(0, lambda: messagebox.showerror("错误", str(e)))

    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, ">> " + msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update_idletasks()

    def build_geometry(self):
        geo_type = self.geo_type_var.get()

        # 1. 构建议当前的参数指纹 (用于对比是否需要重新加载)
        current_params = {'type': geo_type}

        try:
            # 收集关键几何参数
            if geo_type in ["Analytic Cylinder", "OCC Cylinder (NURBS)"]:
                current_params.update({'r': self.geo_radius.get(), 'h': self.geo_height.get()})
            elif geo_type == "Analytic Plate":
                current_params.update({'w': self.geo_width.get(), 'h': self.geo_height.get()})
            elif geo_type == "Analytic Sphere":
                current_params.update({'r': self.geo_radius.get()})
            elif geo_type == "Standard Wedge (PTD Test)":
                current_params.update({'l': self.geo_length.get(), 'w': self.geo_width.get(), 'h': self.geo_height.get()})
            elif geo_type == "Standard Brick (PTD Test)":
                current_params.update({'l': self.geo_length.get(), 'w': self.geo_width.get(), 'h': self.geo_height.get()})
            elif geo_type == "STEP File":
                if not self.step_file_path: return None
                # STEP 特有参数
                step_unit = self.step_unit_var.get() if hasattr(self, 'step_unit_var') else 'mm'
                invert_idx = self.invert_indices_var.get() if hasattr(self, 'invert_indices_var') else ''
                current_params.update({
                    'path': self.step_file_path,
                    'unit': step_unit,
                    'invert': invert_idx
                })

            # 2. 检查缓存：如果参数完全一致且几何对象存在，直接返回缓存对象 (跳过 IO)
            if (self.current_geometry is not None and
                self.current_geo_params == current_params):
                # Log only if it's a heavy STEP file to confirm cache usage
                if geo_type == "STEP File":
                    pass # self.log("Using cached geometry (Skip IO).")
                return self.current_geometry

        except Exception as e:
            print(f"Param build error: {e}")
            pass

        # 3. 缓存未命中，开始重新构建/加载
        try:
            geom = None
            if geo_type == "Analytic Cylinder":
                r = self.geo_radius.get()
                h = self.geo_height.get()
                geom = AnalyticCylinder(r, h)
            elif geo_type == "Analytic Plate":
                w = self.geo_width.get()
                h = self.geo_height.get()
                geom = AnalyticPlate(w, h)
            elif geo_type == "Analytic Sphere":
                r = self.geo_radius.get()
                geom = AnalyticSphere(r)
            elif geo_type == "Standard Wedge (PTD Test)":
                l = self.geo_length.get()
                w = self.geo_width.get()
                h = self.geo_height.get()
                surfaces, ptd_id = create_analytic_wedge(l, w, h)
                self.ptd_edges_var.set(ptd_id)
                self.log(f"Created Analytic Wedge: L={l}, W={w}, H={h}")
                geom = surfaces
            elif geo_type == "Standard Brick (PTD Test)":
                l = self.geo_length.get()
                w = self.geo_width.get()
                h = self.geo_height.get()
                surfaces, ptd_id = create_analytic_brick(l, w, h)
                self.ptd_edges_var.set(ptd_id)
                self.log(f"Created Analytic Brick: L={l}, W={w}, H={h}")
                geom = surfaces
            elif geo_type == "OCC Cylinder (NURBS)":
                r = self.geo_radius.get()
                h = self.geo_height.get()
                occ_geom = create_occ_cylinder(r, h)
                geom = OCCSurface(occ_geom)
            elif geo_type == "STEP File":
                if not self.step_file_path: raise ValueError("请先选择 STEP 文件")
                unit = getattr(self, 'step_unit_var', None)
                scale = 0.001 if (unit and unit.get() == "mm") else 1.0
                invert_indices = []
                try:
                    idx_str = self.invert_indices_var.get()
                    if idx_str.strip(): invert_indices = [int(x.strip()) for x in idx_str.split(',') if x.strip()]
                except: pass
                self.log(f"Loading STEP file: {self.step_file_path} (scale: {scale}, invert: {invert_indices})...")
                surfaces = load_step_file(self.step_file_path, scale=scale, invert_indices=invert_indices)
                self.log(f"Loaded {len(surfaces)} surfaces.")
                geom = surfaces

            # 4. 更新全局几何缓存
            if geom:
                self.current_geometry = geom
                self.current_geo_params = current_params
                # 几何变了，之前的网格缓存必须作废
                self.cached_mesh_data = None
                self.cached_mesh_params = {}
            return geom
        except Exception as e:
            self.log(f"Error building geometry: {str(e)}")
            messagebox.showerror("Geometry Error", str(e))
            return None

    def generate_mesh_stats(self):
        geo = self.build_geometry()
        if not geo: return
        freq = self.freq_var.get() * 1e6
        samples = self.density_var.get()
        surfaces = geo if isinstance(geo, list) else [geo]
        self.log("Generating mesh statistics...")
        self.progress_var.set(0)
        self.progress_label.config(text="正在计算网格统计...")
        threading.Thread(target=self._compute_mesh_stats, args=(surfaces, freq, samples), daemon=True).start()

    def _compute_mesh_stats(self, surfaces, freq, samples):
        try:
            from solver.ribbon_solver import RibbonIntegrator
            from physics.wave import IncidentWave
            solver = RibbonIntegrator()
            wave = IncidentWave(freq, 0, 0)
            total_cells = 0
            total_vertices = 0
            n_total = len(surfaces)
            face_stats = []
            all_min = np.array([np.inf, np.inf, np.inf])
            all_max = np.array([-np.inf, -np.inf, -np.inf])
            for i, surf in enumerate(surfaces):
                nu, nv = solver.get_mesh_size(surf, wave, samples)
                n_cells = nu * nv
                n_vertices = (nu + 1) * (nv + 1)
                total_cells += n_cells
                total_vertices += n_vertices
                u_min, u_max = surf.u_domain
                v_min, v_max = surf.v_domain
                corners_uv = [(u_min, v_min), (u_max, v_min), (u_min, v_max), (u_max, v_max)]
                for u, v in corners_uv:
                    pt = surf.evaluate(np.array([u]), np.array([v]))[0]
                    all_min = np.minimum(all_min, pt)
                    all_max = np.maximum(all_max, pt)
                step_id = getattr(surf, 'step_id', i)
                face_stats.append({'index': i, 'step_id': step_id, 'nu': nu, 'nv': nv, 'cells': n_cells})
                if n_total < 100 or i % 10 == 0 or i == n_total - 1:
                    self._update_progress(i + 1, n_total, f"计算网格: {i+1}/{n_total}")
            model_size = all_max - all_min
            self.root.after(0, lambda: self._show_mesh_stats(n_total, total_cells, total_vertices, wave.wavelength, face_stats, model_size))
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda msg=err_msg: self.log(f"Mesh Stats Error: {msg}"))

    def _show_mesh_stats(self, n_surfaces, total_cells, total_vertices, wavelength, face_stats, model_size=None):
        self.progress_var.set(100)
        self.progress_label.config(text="网格统计完成")
        est_mem = total_vertices * 48 / (1024*1024)
        stats_msg = (f"═══════════════════════════════════════\n          网格统计信息 (Mesh Statistics)\n═══════════════════════════════════════\n  曲面数量 (Surfaces):     {n_surfaces}\n  总网格数 (Total Cells):  {total_cells:,}\n  总顶点数 (Total Vertices): {total_vertices:,}\n  波长 (Wavelength):       {wavelength*1000:.2f} mm\n  预估内存 (Est. Memory):  {est_mem:.1f} MB\n═══════════════════════════════════════\n")
        self.log(stats_msg)
        if len(face_stats) <= 20:
            self.log("各面网格详情:")
            for fs in face_stats: self.log(f"  Face {fs['index']} (#{fs['step_id']}): {fs['nu']}×{fs['nv']} = {fs['cells']:,} cells")
        else:
            self.log(f"前10个面的网格详情:")
            for fs in face_stats[:10]: self.log(f"  Face {fs['index']} (#{fs['step_id']}): {fs['nu']}×{fs['nv']} = {fs['cells']:,} cells")
            self.log(f"  ... 还有 {len(face_stats)-10} 个面")

    def visualize_mesh(self):
        geo = self.build_geometry()
        if not geo: return
        freq = self.freq_var.get() * 1e6
        samples = self.density_var.get()
        surfaces = geo if isinstance(geo, list) else [geo]
        if len(surfaces) > 20:
             if not messagebox.askyesno("Warning", f"该模型包含 {len(surfaces)} 个面，可视化可能较慢。是否继续？"): return
        self.log("Generating mesh for visualization...")
        self.progress_var.set(0)
        self.progress_label.config(text="正在生成网格数据...")
        try: threading.Thread(target=self.plot_multi_surface_mesh, args=(surfaces, freq, samples), daemon=True).start()
        except Exception as e: self.log(f"Visualization Error: {e}")

    def plot_multi_surface_mesh(self, surfaces, freq, samples):
        """
        后台线程：生成全量网格数据。
        优化：
        1. 使用 'precompute_mesh' 而不是 'get_mesh_data'，这样生成的数据不仅能画图，还能直接给 Solver 用。
        2. 将生成的 CachedMeshData 对象存入全局缓存 'self.cached_mesh_data'。
        """
        try:
            from physics.constants import C0
            solver = RibbonIntegrator()
            # wave 仅用于提供波长，precompute 不依赖角度
            wave = IncidentWave(freq, 0, 0)
            wavelength = C0 / freq

            mesh_data_list = []      # 用于画图的轻量数据 (dict)
            cached_surfaces = []     # 用于计算的重型数据 (CachedMeshData objects)

            total_points = 0
            n_total = len(surfaces)

            # 记录当前计算的参数签名，用于后续校验缓存有效性
            current_params = {
                'freq': freq,
                'samples': samples,
                'surfaces_id': id(surfaces) # 绑定到当前的几何对象实例
            }

            for i, surf in enumerate(surfaces):
                # 关键修改：调用 precompute_mesh 生成全量物理数据 (points, normals, jacobians, derivatives)
                cached_data = solver.precompute_mesh(surf, wavelength, samples)
                cached_surfaces.append(cached_data)

                # 提取画图所需数据
                points = cached_data.points
                normals = cached_data.normals
                nu, nv = points.shape[1], points.shape[0]

                total_points += nu * nv
                mesh_data_list.append({'points': points, 'normals': normals, 'nu': nu, 'nv': nv})

                # 汇报进度 (降低刷新频率以减少开销)
                if n_total < 100 or i % 5 == 0 or i == n_total - 1:
                    self._update_progress(i + 1, n_total, f"生成网格数据: {i+1}/{n_total}")

            # 关键修改：将全量数据回写到 GUI 主类中进行持久化缓存
            def update_cache():
                self.cached_mesh_data = cached_surfaces
                self.cached_mesh_params = current_params

            self.root.after(0, update_cache)

            # 数据计算完成，通知主线程开始绘图
            self.root.after(0, lambda: self._do_update_progress(100, "正在渲染 3D 图形 (这可能需要几秒钟)..."))
            self.root.after(0, lambda: self.viz_manager.show_mesh_visualization(mesh_data_list, total_points, len(surfaces), wave.wavelength))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.log(f"Vis Error: {e}"))

    def run_calculation(self):
        # 1. 获取几何对象 (如果之前加载过且参数没变，build_geometry 会直接返回缓存，秒开)
        geo = self.build_geometry()
        if not geo: return
        freq = self.freq_var.get() * 1e6
        samples = self.density_var.get()
        is_parallel = self.parallel_var.get()
        n_workers = self.workers_var.get()
        theta_start = self.theta_start.get()
        theta_end = self.theta_end.get()
        n_theta = self.theta_n.get()
        phi_start = self.phi_start.get()
        phi_end = self.phi_end.get()
        n_phi = self.phi_n.get()

        theta_deg = np.linspace(theta_start, theta_end, n_theta)
        theta_rad = np.radians(theta_deg)
        phi_deg = np.linspace(phi_start, phi_end, n_phi)
        phi_rad = np.radians(phi_deg)

        is_2d = n_phi > 1

        enable_ptd = self.ptd_enabled_var.get()
        ptd_edge_identifiers = []
        ptd_pol = 'VV'
        if enable_ptd:
            raw_edges = self.ptd_edges_var.get()
            ptd_edge_identifiers = [s.strip() for s in raw_edges.split(',') if s.strip()]
            ptd_pol = self.pol_var.get()

        geo_type = self.geo_type_var.get()
        geo_params = self._get_geometry_params()

        # 2. 检查是否有现成的网格缓存
        cached_mesh = None
        current_algo_id = self._get_selected_algorithm_id()

        # 只有离散类算法支持复用网格缓存
        if 'discrete_po' in current_algo_id and self.cached_mesh_data is not None:
            # 校验缓存是否匹配当前参数
            check_params = {
                'freq': freq,
                'samples': samples,
                'surfaces_id': id(geo) # 确保是同一个几何对象实例
            }
            if check_params == self.cached_mesh_params:
                cached_mesh = self.cached_mesh_data
                self.log("🚀 Optimization: Using pre-calculated mesh from visualization cache.")
            else:
                pass # self.log("Cache mismatch, re-calculating mesh...")

        # 重置进度条
        self.progress_var.set(0)
        self.progress_label.config(text="准备计算...")
        if is_2d:
            self.log(f"Starting 2D scan: {n_theta}×{n_phi} angles, {freq/1e6} MHz...")
            if is_parallel: self.log(f"Parallel mode enabled: {n_workers} workers")
            if enable_ptd: self.log(f"PTD Enabled: {len(ptd_edge_identifiers)} edges ({ptd_pol})")
            threading.Thread(target=self._calc_thread_2d, args=(geo, freq, theta_rad, theta_deg, phi_rad, phi_deg, samples, geo_type, geo_params, is_parallel, n_workers, enable_ptd, ptd_edge_identifiers, cached_mesh, ptd_pol), daemon=True).start()
        else:
            self.log(f"Starting 1D scan: {n_theta} angles, {freq/1e6} MHz...")
            if is_parallel: self.log(f"Parallel mode enabled: {n_workers} workers")
            if enable_ptd: self.log(f"PTD Enabled: {len(ptd_edge_identifiers)} edges ({ptd_pol})")
            threading.Thread(target=self._calc_thread, args=(geo, freq, theta_rad, theta_deg, samples, geo_type, geo_params, phi_rad[0], is_parallel, n_workers, enable_ptd, ptd_edge_identifiers, cached_mesh, ptd_pol), daemon=True).start()

    def _get_geometry_params(self):
        geo_type = self.geo_type_var.get()
        params = {}
        try:
            if "Cylinder" in geo_type:
                params['radius'] = self.geo_radius.get()
                params['height'] = self.geo_height.get()
            elif "Plate" in geo_type:
                params['width'] = self.geo_width.get()
                params['height'] = self.geo_height.get()
            elif "Sphere" in geo_type:
                params['radius'] = self.geo_radius.get()
        except:
            pass
        return params

    def _update_progress(self, current, total, message):
        """进度回调函数（在后台线程中调用）"""
        progress = (current / total * 100) if total > 0 else 0
        self.root.after(0, lambda: self._do_update_progress(progress, message))

    def _do_update_progress(self, progress, message):
        """在主线程中更新进度条"""
        self.progress_var.set(progress)
        self.progress_label.config(text=message)
        self.log(message)

    def _get_selected_algorithm_id(self):
        display_name = self.algorithm_var.get()
        return self._algo_name_to_id.get(display_name, 'discrete_po_sinc_dual')

    def _calc_thread(self, geo, freq, angles_rad, angles_deg, samples, geo_type, geo_params, phi_rad=0.0, parallel=False, n_workers=None, enable_ptd=False, ptd_edge_identifiers=None, cached_mesh_data=None, polarization='VV'):
        """1D扫描线程"""
        try:
            start_time = time.time()
            algo_id = self._get_selected_algorithm_id()
            solver = get_integrator(algo_id)
            analyzer = RCSAnalyzer(solver)
            self.root.after(0, lambda: self.log(f"Using algorithm: {AVAILABLE_ALGORITHMS[algo_id]['name']}"))
            rcs_result = analyzer.compute_monostatic_rcs(geo, {'frequency': freq, 'phi': phi_rad}, angles_rad, samples_per_lambda=samples, parallel=parallel, n_workers=n_workers, show_progress=False, progress_callback=self._update_progress, enable_ptd=enable_ptd, ptd_edge_identifiers=ptd_edge_identifiers, cached_mesh_data=cached_mesh_data, polarization=polarization)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if isinstance(rcs_result, dict):
                rcs_total = rcs_result['total']
                rcs_po = rcs_result.get('po')
                rcs_ptd = rcs_result.get('ptd')
            else:
                rcs_total = rcs_result
                rcs_po = None
                rcs_ptd = None
            result_data = {
                'mode': '1d',
                'angles_deg': angles_deg,
                'angles_rad': angles_rad,
                'phi_deg': np.degrees(phi_rad),
                'rcs': rcs_total,
                'rcs_po': rcs_po,
                'rcs_ptd': rcs_ptd,
                'freq': freq,
                'geo_type': geo_type,
                'geo_params': geo_params,
                'elapsed_time': elapsed_time,
                'polarization': polarization
            }
            self.root.after(0, lambda: self.show_results(result_data))
            self.root.after(0, lambda: self.log(f"Calculation finished. Time elapsed: {elapsed_time:.2f} s"))
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Calculation Error: {e}"))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.progress_label.config(text="计算失败"))

    def _calc_thread_2d(self, geo, freq, theta_rad, theta_deg, phi_rad, phi_deg, samples, geo_type, geo_params, parallel=False, n_workers=None, enable_ptd=False, ptd_edge_identifiers=None, cached_mesh_data=None, polarization='VV'):
        """2D扫描线程"""
        try:
            start_time = time.time()
            algo_id = self._get_selected_algorithm_id()
            solver = get_integrator(algo_id)
            analyzer = RCSAnalyzer(solver)
            self.root.after(0, lambda: self.log(f"Using algorithm: {AVAILABLE_ALGORITHMS[algo_id]['name']}"))
            rcs_2d_result = analyzer.compute_monostatic_rcs_2d(geo, freq, theta_rad, phi_rad, samples_per_lambda=samples, parallel=parallel, n_workers=n_workers, show_progress=False, progress_callback=self._update_progress, enable_ptd=enable_ptd, ptd_edge_identifiers=ptd_edge_identifiers, cached_mesh_data=cached_mesh_data, polarization=polarization)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if isinstance(rcs_2d_result, dict):
                rcs_2d_total = rcs_2d_result['total']
                rcs_2d_po = rcs_2d_result.get('po')
                rcs_2d_ptd = rcs_2d_result.get('ptd')
            else:
                rcs_2d_total = rcs_2d_result
                rcs_2d_po = None
                rcs_2d_ptd = None
            result_data = {'mode': '2d', 'theta_deg': theta_deg, 'theta_rad': theta_rad, 'phi_deg': phi_deg, 'phi_rad': phi_rad, 'rcs_2d': rcs_2d_total, 'rcs_2d_po': rcs_2d_po, 'rcs_2d_ptd': rcs_2d_ptd, 'freq': freq, 'geo_type': geo_type, 'geo_params': geo_params, 'elapsed_time': elapsed_time}
            self.root.after(0, lambda: self.show_results(result_data))
            self.root.after(0, lambda: self.log(f"2D Calculation finished. Time elapsed: {elapsed_time:.2f} s"))
        except Exception as e:
            self.root.after(0, lambda: self.log(f"2D Calculation Error: {e}"))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.progress_label.config(text="计算失败"))

    def show_results(self, result_data):
        self.last_result = result_data
        mode = result_data.get('mode', '1d')
        if mode == '2d':
            rcs_2d = result_data['rcs_2d']
            shape = rcs_2d.shape
            if shape[0] == 1 or shape[1] == 1:
                if shape[0] == 1:
                    x_values = result_data['phi_deg']
                    x_label = 'Phi (deg)'
                    flat_rcs = rcs_2d.flatten()
                    flat_po = result_data.get('rcs_2d_po').flatten() if result_data.get('rcs_2d_po') is not None else None
                    flat_ptd = result_data.get('rcs_2d_ptd').flatten() if result_data.get('rcs_2d_ptd') is not None else None
                else:
                    x_values = result_data['theta_deg']
                    x_label = 'Theta (deg)'
                    flat_rcs = rcs_2d.flatten()
                    flat_po = result_data.get('rcs_2d_po').flatten() if result_data.get('rcs_2d_po') is not None else None
                    flat_ptd = result_data.get('rcs_2d_ptd').flatten() if result_data.get('rcs_2d_ptd') is not None else None
                result_1d = result_data.copy()
                result_1d['mode'] = '1d'
                result_1d['rcs'] = flat_rcs
                result_1d['x_values'] = x_values
                result_1d['x_label'] = x_label
                if flat_po is not None: result_1d['rcs_po'] = flat_po
                if flat_ptd is not None: result_1d['rcs_ptd'] = flat_ptd
                self.viz_manager.show_1d_results(result_1d, self.compare_analytical_var.get())
            else:
                self.viz_manager.show_2d_results(result_data)
        else:
            self.viz_manager.show_1d_results(result_data, self.compare_analytical_var.get())

    def export_to_csv(self):
        """将最后一次计算结果导出为 CSV 文件"""
        if self.last_result is None:
            messagebox.showwarning("警告", "没有可导出的计算结果，请先进行计算。")
            return

        mode = self.last_result.get('mode', '1d')
        freq_mhz = self.last_result.get('freq', 0) / 1e6
        geo_type = self.last_result.get('geo_type', 'unknown')

        # 如果是 STEP 模式，尝试使用文件名作为标识
        if geo_type.lower() == 'step' and hasattr(self, 'step_file_path') and self.step_file_path:
            geo_label = os.path.splitext(os.path.basename(self.step_file_path))[0]
        else:
            geo_label = geo_type

        default_filename = f"rcs_{mode}_{geo_label}_{freq_mhz:.1f}MHz.csv"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile=default_filename,
            title="选择导出位置"
        )

        if not file_path:
            return

        try:
            import pandas as pd
            if mode == '2d':
                theta_deg = self.last_result['theta_deg']
                phi_deg = self.last_result['phi_deg']
                rcs_2d = self.last_result['rcs_2d']

                rows = []
                for i, t in enumerate(theta_deg):
                    for j, p in enumerate(phi_deg):
                        rows.append({'Theta': t, 'Phi': p, 'RCS(dBsm)': rcs_2d[i, j]})
                df = pd.DataFrame(rows)
            else:
                df = pd.DataFrame({
                    'Theta': self.last_result['angles_deg'],
                    'RCS(dBsm)': self.last_result['rcs']
                })

            df.to_csv(file_path, index=False)
            self.log(f"Successfully exported to {file_path}")
            messagebox.showinfo("成功", f"数据已导出至:\n{file_path}")
        except Exception as e:
            self.log(f"Export Error: {e}")
            messagebox.showerror("导出失败", str(e))

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.freq_var.set(config.get('freq', 300.0))
                    self.density_var.set(config.get('density', 10))
                    self.theta_start.set(config.get('theta_start', 0.0))
                    self.theta_end.set(config.get('theta_end', 180.0))
                    self.theta_n.set(config.get('theta_n', 91))
                    self.phi_start.set(config.get('phi_start', 0.0))
                    self.phi_end.set(config.get('phi_end', 0.0))
                    self.phi_n.set(config.get('phi_n', 1))
                    self.step_file_path = config.get('step_file_path', None)
                    if self.step_file_path and os.path.exists(self.step_file_path):
                        if hasattr(self, 'step_label'):
                            self.step_label.config(text=os.path.basename(self.step_file_path))
            except:
                pass

    def save_config(self):
        config = {
            'freq': self.freq_var.get(),
            'density': self.density_var.get(),
            'theta_start': self.theta_start.get(),
            'theta_end': self.theta_end.get(),
            'theta_n': self.theta_n.get(),
            'phi_start': self.phi_start.get(),
            'phi_end': self.phi_end.get(),
            'phi_n': self.phi_n.get(),
            'step_file_path': self.step_file_path
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f)
        except:
            pass

    def on_closing(self):
        self.save_config()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CEMPoGUI(root)
    root.mainloop()
