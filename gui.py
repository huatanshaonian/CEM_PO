import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os
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
from physics.wave import IncidentWave
from physics.analytical_rcs import get_analytical_solution, compute_error_stats
from solver.ribbon_solver import RibbonIntegrator, RCSAnalyzer
from tools.visualize_mesh import create_occ_cylinder

class GeminiPOGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemini PO Solver")
        self.root.geometry("950x650")
        
        # --- 现代配色与样式设置 ---
        # 定义颜色变量
        self.colors = {
            "bg_main": "#FAFAFA",        # 极淡的灰白，作为主背景
            "bg_panel": "#FFFFFF",       # 纯白，用于内容区域
            "fg_text": "#333333",        # 深灰字体，比纯黑柔和
            "accent": "#007ACC",         # 提亮色 (虽然ttk很难深度定制，但用于部分高亮)
            "border": "#E0E0E0",         # 淡灰色边框
            "button_bg": "#F0F0F0",      # 按钮背景
        }

        # 设置根窗口背景
        self.root.configure(bg=self.colors["bg_main"])

        # 配置 TTK 样式
        style = ttk.Style()
        style.theme_use('clam') # 基于 clam 主题修改，因为它支持较多自定义

        # 全局样式 (使用支持中文的字体)
        style.configure(".",
            background=self.colors["bg_main"],
            foreground=self.colors["fg_text"],
            font=("Microsoft YaHei UI", 9)
        )

        # Frame 样式
        style.configure("TFrame", background=self.colors["bg_main"])
        style.configure("Card.TFrame", background=self.colors["bg_panel"], relief="flat")

        # LabelFrame 样式 (卡片式)
        style.configure("TLabelframe", 
            background=self.colors["bg_panel"], 
            bordercolor=self.colors["border"],
            relief="solid", 
            borderwidth=1
        )
        style.configure("TLabelframe.Label",
            background=self.colors["bg_panel"],
            foreground="#555555",
            font=("Microsoft YaHei UI", 9, "bold")
        )

        # Label 样式
        style.configure("TLabel", background=self.colors["bg_panel"], foreground=self.colors["fg_text"])
        style.configure("Main.TLabel", background=self.colors["bg_main"])

        # Button 样式 (扁平化)
        style.configure("TButton", 
            background=self.colors["button_bg"], 
            borderwidth=1, 
            relief="solid",
            padding=5
        )
        style.map("TButton",
            background=[("active", "#E5E5E5")],
            relief=[("pressed", "sunken")]
        )
        
        # Entry 样式
        style.configure("TEntry", 
            fieldbackground="#FFFFFF",
            bordercolor=self.colors["border"],
            padding=5
        )

        # Combobox 样式
        style.configure("TCombobox", 
            fieldbackground="#FFFFFF",
            arrowcolor=self.colors["fg_text"]
        )
        
        # --- 界面布局 ---

        # 主容器 (增加外边距)
        main_frame = ttk.Frame(root, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 左侧面板 (参数设置 - 卡片式设计)
        left_panel = ttk.LabelFrame(main_frame, text=" 配置与几何 (Configuration) ", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), ipadx=5)
        
        # 右侧面板 (日志与操作 - 透明背景容器)
        right_panel = ttk.Frame(main_frame, style="TFrame")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_config_widgets(left_panel)
        self.create_geometry_widgets(left_panel)
        self.create_action_widgets(left_panel)
        self.create_log_widgets(right_panel)
        
        # 状态变量
        self.current_geometry = None
        self.step_file_path = None

    def create_config_widgets(self, parent):
        # 频率
        ttk.Label(parent, text="频率 Frequency (MHz):").pack(anchor=tk.W, pady=(0, 5))
        self.freq_var = tk.DoubleVar(value=300.0)
        ttk.Entry(parent, textvariable=self.freq_var).pack(fill=tk.X, pady=(0, 10))
        
        # 采样密度
        ttk.Label(parent, text="网格密度 Grid Density (Samples/Lambda):").pack(anchor=tk.W, pady=(0, 5))
        self.density_var = tk.IntVar(value=10)
        ttk.Entry(parent, textvariable=self.density_var).pack(fill=tk.X, pady=(0, 10))

        # Theta 扫描范围
        ttk.Label(parent, text="Theta 范围 (Start, End, Points):").pack(anchor=tk.W, pady=(0, 5))
        theta_frame = ttk.Frame(parent, style="Card.TFrame")
        theta_frame.pack(fill=tk.X, pady=(0, 10))

        self.theta_start = tk.DoubleVar(value=0.0)
        self.theta_end = tk.DoubleVar(value=180.0)
        self.theta_n = tk.IntVar(value=91)

        ttk.Entry(theta_frame, textvariable=self.theta_start, width=6).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(theta_frame, textvariable=self.theta_end, width=6).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(theta_frame, textvariable=self.theta_n, width=6).pack(side=tk.LEFT)

        # Phi 扫描范围 (2D扫描)
        ttk.Label(parent, text="Phi 范围 (Start, End, Points):").pack(anchor=tk.W, pady=(0, 5))
        phi_frame = ttk.Frame(parent, style="Card.TFrame")
        phi_frame.pack(fill=tk.X, pady=(0, 10))

        self.phi_start = tk.DoubleVar(value=-45.0)
        self.phi_end = tk.DoubleVar(value=45.0)
        self.phi_n = tk.IntVar(value=1)  # 默认1点=1D扫描

        ttk.Entry(phi_frame, textvariable=self.phi_start, width=6).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(phi_frame, textvariable=self.phi_end, width=6).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(phi_frame, textvariable=self.phi_n, width=6).pack(side=tk.LEFT)

        # 提示
        ttk.Label(parent, text="(Phi点数=1为1D扫描，>1为2D扫描)",
                  foreground="#888888", font=("Microsoft YaHei UI", 8)).pack(anchor=tk.W)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

    def create_geometry_widgets(self, parent):
        ttk.Label(parent, text="几何类型 Geometry Type:").pack(anchor=tk.W, pady=(0, 5))
        self.geo_type_var = tk.StringVar(value="Analytic Cylinder")
        types = ["Analytic Cylinder", "Analytic Plate", "Analytic Sphere", "OCC Cylinder (NURBS)", "STEP File"]
        combo = ttk.Combobox(parent, textvariable=self.geo_type_var, values=types, state="readonly")
        combo.pack(fill=tk.X, pady=(0, 10))
        combo.bind("<<ComboboxSelected>>", self.update_geo_inputs)
        
        # 动态参数区域 (保持白色背景)
        self.geo_params_frame = ttk.Frame(parent, style="Card.TFrame")
        self.geo_params_frame.pack(fill=tk.X, pady=5)
        
        # 初始刷新
        self.update_geo_inputs()

    def update_geo_inputs(self, event=None):
        # 清空旧控件
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
        elif geo_type == "STEP File":
            btn = ttk.Button(self.geo_params_frame, text="📂 选择 STEP 文件...", command=self.browse_step)
            btn.pack(fill=tk.X, pady=5)
            self.step_label = ttk.Label(self.geo_params_frame, text="未选择文件", foreground="#888888", wraplength=200)
            self.step_label.pack(fill=tk.X)

    def add_param_input(self, label, var_name, default):
        frame = ttk.Frame(self.geo_params_frame, style="Card.TFrame")
        frame.pack(fill=tk.X, pady=3)
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        var = tk.DoubleVar(value=default)
        ttk.Entry(frame, textvariable=var, width=10).pack(side=tk.RIGHT)
        setattr(self, f"geo_{var_name}", var)

    def browse_step(self):
        filename = filedialog.askopenfilename(filetypes=[("STEP Files", "*.stp;*.step")])
        if filename:
            self.step_file_path = filename
            self.step_label.config(text=os.path.basename(filename))
            self.log(f"Selected STEP file: {filename}")

    def create_action_widgets(self, parent):
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)

        # 解析解对比选项
        self.compare_analytical_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent,
            text="对比解析解 (Compare with Analytical)",
            variable=self.compare_analytical_var
        ).pack(anchor=tk.W, pady=(0, 10))

        # 使用自定义样式的按钮
        btn_mesh = ttk.Button(parent, text="🧊 可视化网格 (Visualize Mesh)", command=self.visualize_mesh)
        btn_mesh.pack(fill=tk.X, pady=(0, 8))

        btn_calc = ttk.Button(parent, text="🚀 计算 RCS (Calculate)", command=self.run_calculation)
        btn_calc.pack(fill=tk.X, pady=(0, 8))

        # 进度条
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
        # 头部标签
        ttk.Label(parent, text="系统日志 System Log:", style="Main.TLabel", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # 带有边框的容器
        log_frame = ttk.Frame(parent)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # 日志文本框 (自定义背景色，使用支持中文的字体)
        self.log_text = tk.Text(log_frame,
            height=20,
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
        
        self.log("Gemini PO Solver GUI Ready.")
        self.log("Based on PythonOCC and Ribbon Method.")

    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, ">> " + msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update_idletasks()

    def build_geometry(self):
        """根据当前配置构建几何对象"""
        geo_type = self.geo_type_var.get()
        
        try:
            if geo_type == "Analytic Cylinder":
                r = self.geo_radius.get()
                h = self.geo_height.get()
                return AnalyticCylinder(r, h)
            
            elif geo_type == "Analytic Plate":
                w = self.geo_width.get()
                h = self.geo_height.get()
                return AnalyticPlate(w, h)
            
            elif geo_type == "Analytic Sphere":
                r = self.geo_radius.get()
                return AnalyticSphere(r)
            
            elif geo_type == "OCC Cylinder (NURBS)":
                r = self.geo_radius.get()
                h = self.geo_height.get()
                occ_geom = create_occ_cylinder(r, h)
                return OCCSurface(occ_geom)
            
            elif geo_type == "STEP File":
                if not self.step_file_path:
                    raise ValueError("请先选择 STEP 文件")
                self.log(f"Loading STEP file: {self.step_file_path}...")
                surfaces = load_step_file(self.step_file_path)
                self.log(f"Loaded {len(surfaces)} surfaces.")
                return surfaces
                
        except Exception as e:
            self.log(f"Error building geometry: {str(e)}")
            messagebox.showerror("Geometry Error", str(e))
            return None

    def visualize_mesh(self):
        geo = self.build_geometry()
        if not geo:
            return
            
        freq = self.freq_var.get() * 1e6
        samples = self.density_var.get()
        
        surfaces = geo if isinstance(geo, list) else [geo]
        
        if len(surfaces) > 20:
             if not messagebox.askyesno("Warning", f"该模型包含 {len(surfaces)} 个面，可视化可能较慢。是否继续？"):
                 return

        self.log("Generating mesh for visualization...")
        
        try:
            # 启动线程避免界面冻结
            threading.Thread(target=self.plot_multi_surface_mesh, args=(surfaces, freq, samples)).start()
        except Exception as e:
            self.log(f"Visualization Error: {e}")

    def plot_multi_surface_mesh(self, surfaces, freq, samples):
        """后台线程：只计算数据，不创建matplotlib对象"""
        try:
            solver = RibbonIntegrator()
            wave = IncidentWave(freq, 0, 0)

            # 收集所有曲面的网格数据
            mesh_data_list = []
            total_points = 0

            for i, surf in enumerate(surfaces):
                points, normals, (nu, nv) = solver.get_mesh_data(surf, wave, samples)
                total_points += nu * nv
                mesh_data_list.append({
                    'points': points,
                    'normals': normals,
                    'nu': nu,
                    'nv': nv
                })

            # 将绘图操作调度到主线程
            self.root.after(0, lambda: self._do_mesh_plot(
                mesh_data_list, total_points, len(surfaces), wave.wavelength
            ))

        except Exception as e:
            self.root.after(0, lambda: self.log(f"Vis Error: {e}"))

    def _do_mesh_plot(self, mesh_data_list, total_points, n_surfaces, wavelength):
        """主线程：创建matplotlib图形"""
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            for i, data in enumerate(mesh_data_list):
                points = data['points']
                normals = data['normals']
                nu, nv = data['nu'], data['nv']

                X = points[..., 0]
                Y = points[..., 1]
                Z = points[..., 2]

                stride_u = max(1, nu // 30)
                stride_v = max(1, nv // 30)

                ax.plot_wireframe(X, Y, Z, color='#007ACC', linewidth=0.5,
                                  rstride=stride_v, cstride=stride_u, alpha=0.4)

                # 只给少量面画法线
                if i == 0 or n_surfaces < 5:
                    skip = max(1, min(nu, nv) // 8)
                    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], Z[::skip, ::skip],
                              normals[::skip, ::skip, 0], normals[::skip, ::skip, 1],
                              normals[::skip, ::skip, 2],
                              length=wavelength/8, color='#FF5555', alpha=0.6)

            ax.set_title(f"Mesh Visualization ({n_surfaces} surfaces)")
            self.log(f"Visualization complete. Total vertices: {total_points}")
            plt.show()

        except Exception as e:
            self.log(f"Plot Error: {e}")

    def run_calculation(self):
        geo = self.build_geometry()
        if not geo:
            return

        freq = self.freq_var.get() * 1e6
        samples = self.density_var.get()

        # Theta 参数
        theta_start = self.theta_start.get()
        theta_end = self.theta_end.get()
        n_theta = self.theta_n.get()
        theta_deg = np.linspace(theta_start, theta_end, n_theta)
        theta_rad = np.radians(theta_deg)

        # Phi 参数
        phi_start = self.phi_start.get()
        phi_end = self.phi_end.get()
        n_phi = self.phi_n.get()
        phi_deg = np.linspace(phi_start, phi_end, max(1, n_phi))
        phi_rad = np.radians(phi_deg)

        # 判断1D还是2D扫描
        is_2d = n_phi > 1

        # 获取几何类型和参数（用于解析解）
        geo_type = self.geo_type_var.get()
        geo_params = self._get_geometry_params()

        # 重置进度条
        self.progress_var.set(0)
        self.progress_label.config(text="准备计算...")

        if is_2d:
            self.log(f"Starting 2D scan: {n_theta}×{n_phi} angles, {freq/1e6} MHz...")
            threading.Thread(
                target=self._calc_thread_2d,
                args=(geo, freq, theta_rad, theta_deg, phi_rad, phi_deg, samples, geo_type, geo_params)
            ).start()
        else:
            self.log(f"Starting 1D scan: {n_theta} angles, {freq/1e6} MHz...")
            threading.Thread(
                target=self._calc_thread,
                args=(geo, freq, theta_rad, theta_deg, samples, geo_type, geo_params, phi_rad[0])
            ).start()

    def _get_geometry_params(self):
        """获取当前几何参数"""
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
        # 使用 after 调度到主线程
        self.root.after(0, lambda: self._do_update_progress(progress, message))

    def _do_update_progress(self, progress, message):
        """在主线程中更新进度条"""
        self.progress_var.set(progress)
        self.progress_label.config(text=message)
        self.log(message)

    def _calc_thread(self, geo, freq, angles_rad, angles_deg, samples, geo_type, geo_params, phi_rad=0.0):
        """1D扫描线程"""
        try:
            solver = RibbonIntegrator()
            analyzer = RCSAnalyzer(solver)

            # 使用进度回调
            rcs = analyzer.compute_monostatic_rcs(
                geo,
                {'frequency': freq, 'phi': phi_rad},
                angles_rad,
                samples_per_lambda=samples,
                parallel=False,
                show_progress=False,
                progress_callback=self._update_progress
            )

            # 准备结果数据
            result_data = {
                'mode': '1d',
                'angles_deg': angles_deg,
                'angles_rad': angles_rad,
                'phi_deg': np.degrees(phi_rad),
                'rcs': rcs,
                'freq': freq,
                'geo_type': geo_type,
                'geo_params': geo_params
            }

            self.root.after(0, lambda: self.show_results(result_data))
            self.root.after(0, lambda: self.log("Calculation finished."))

        except Exception as e:
            self.root.after(0, lambda: self.log(f"Calculation Error: {e}"))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.progress_label.config(text="计算失败"))

    def _calc_thread_2d(self, geo, freq, theta_rad, theta_deg, phi_rad, phi_deg, samples, geo_type, geo_params):
        """2D扫描线程"""
        try:
            solver = RibbonIntegrator()
            analyzer = RCSAnalyzer(solver)

            # 2D扫描
            rcs_2d = analyzer.compute_monostatic_rcs_2d(
                geo,
                freq,
                theta_rad,
                phi_rad,
                samples_per_lambda=samples,
                show_progress=False,
                progress_callback=self._update_progress
            )

            # 准备结果数据
            result_data = {
                'mode': '2d',
                'theta_deg': theta_deg,
                'theta_rad': theta_rad,
                'phi_deg': phi_deg,
                'phi_rad': phi_rad,
                'rcs_2d': rcs_2d,
                'freq': freq,
                'geo_type': geo_type,
                'geo_params': geo_params
            }

            self.root.after(0, lambda: self.show_results(result_data))
            self.root.after(0, lambda: self.log("2D Calculation finished."))

        except Exception as e:
            self.root.after(0, lambda: self.log(f"2D Calculation Error: {e}"))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.progress_label.config(text="计算失败"))

    def show_results(self, result_data):
        """显示计算结果，支持1D线图和2D热图"""
        mode = result_data.get('mode', '1d')
        freq = result_data['freq']
        geo_type = result_data['geo_type']
        geo_params = result_data['geo_params']

        if mode == '2d':
            self._show_results_2d(result_data)
        else:
            self._show_results_1d(result_data)

    def _show_results_2d(self, result_data):
        """显示2D扫描结果热图"""
        theta_deg = result_data['theta_deg']
        phi_deg = result_data['phi_deg']
        rcs_2d = result_data['rcs_2d']
        freq = result_data['freq']
        geo_type = result_data['geo_type']

        # 创建网格
        Theta, Phi = np.meshgrid(theta_deg, phi_deg, indexing='ij')

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=self.colors["bg_main"])

        # 绘制热图
        levels = np.linspace(np.nanmin(rcs_2d), np.nanmax(rcs_2d), 50)
        contour = ax.contourf(Theta, Phi, rcs_2d, levels=levels, cmap='jet')

        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax, shrink=0.9, aspect=20)
        cbar.set_label('RCS (dBsm)', fontsize=11)

        # 添加等高线
        contour_lines = ax.contour(Theta, Phi, rcs_2d, levels=15, colors='k',
                                    linewidths=0.3, alpha=0.5)

        ax.set_xlabel('Theta (deg)', fontsize=11)
        ax.set_ylabel('Phi (deg)', fontsize=11)
        ax.set_title(f'2D Monostatic RCS - {geo_type} @ {freq/1e6:.1f} MHz', fontsize=12)

        # 显示统计信息
        rcs_max = np.nanmax(rcs_2d)
        rcs_min = np.nanmin(rcs_2d)
        rcs_mean = np.nanmean(rcs_2d)
        stats_text = (f"RCS 统计:\n"
                      f"  最大: {rcs_max:.2f} dBsm\n"
                      f"  最小: {rcs_min:.2f} dBsm\n"
                      f"  平均: {rcs_mean:.2f} dBsm")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontfamily='Microsoft YaHei')

        self.log(f"2D RCS - 最大: {rcs_max:.2f}dBsm, 最小: {rcs_min:.2f}dBsm, 平均: {rcs_mean:.2f}dBsm")

        plt.tight_layout()
        plt.show()

    def _show_results_1d(self, result_data):
        """显示1D扫描结果线图"""
        angles_deg = result_data['angles_deg']
        angles_rad = result_data['angles_rad']
        rcs = result_data['rcs']
        freq = result_data['freq']
        geo_type = result_data['geo_type']
        geo_params = result_data['geo_params']

        # 创建图形
        fig, ax = plt.subplots(figsize=(11, 6), facecolor=self.colors["bg_main"])

        # 绘制数值解
        ax.plot(angles_deg, rcs, color=self.colors["accent"], linewidth=2,
                label='Ribbon PO (数值解)')

        # 解析解对比
        rcs_analytical = None
        if self.compare_analytical_var.get() and geo_params:
            # 映射几何类型
            analytical_type = None
            if "Cylinder" in geo_type:
                analytical_type = 'cylinder'
            elif "Plate" in geo_type:
                analytical_type = 'plate'
            elif "Sphere" in geo_type:
                analytical_type = 'sphere'

            if analytical_type:
                rcs_analytical, label = get_analytical_solution(
                    analytical_type, geo_params, freq, angles_rad
                )

                if rcs_analytical is not None:
                    ax.plot(angles_deg, rcs_analytical, 'r--', linewidth=2,
                            label=label)

                    # 计算误差统计
                    stats = compute_error_stats(rcs, rcs_analytical)
                    error_text = (f"误差统计:\n"
                                  f"  最大: {stats['max_error']:.2f} dB\n"
                                  f"  平均: {stats['mean_error']:.2f} dB\n"
                                  f"  RMS: {stats['rms_error']:.2f} dB")

                    # 在图上添加误差信息
                    ax.text(0.02, 0.98, error_text, transform=ax.transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                            fontfamily='Microsoft YaHei')

                    # 记录到日志
                    self.log(f"误差统计 - 最大: {stats['max_error']:.2f}dB, "
                             f"平均: {stats['mean_error']:.2f}dB, "
                             f"RMS: {stats['rms_error']:.2f}dB")

        ax.set_xlabel('Theta (deg)', fontsize=11)
        ax.set_ylabel('RCS (dBsm)', fontsize=11)
        ax.set_title(f'Monostatic RCS - {geo_type} @ {freq/1e6:.1f} MHz', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = GeminiPOGUI(root)
    root.mainloop()