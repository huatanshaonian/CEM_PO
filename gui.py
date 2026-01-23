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

class CEMPoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CEM PO Solver")
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
            # STEP 单位选择
            unit_frame = ttk.Frame(self.geo_params_frame, style="Card.TFrame")
            unit_frame.pack(fill=tk.X, pady=(5, 0))
            ttk.Label(unit_frame, text="STEP 单位:").pack(side=tk.LEFT)
            self.step_unit_var = tk.StringVar(value="mm")
            ttk.Radiobutton(unit_frame, text="mm", variable=self.step_unit_var, value="mm").pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(unit_frame, text="m", variable=self.step_unit_var, value="m").pack(side=tk.LEFT)
            # STEP 预览按钮
            btn_preview = ttk.Button(self.geo_params_frame, text="👁 预览全部", command=self.preview_step)
            btn_preview.pack(fill=tk.X, pady=(5, 0))
            # 单面预览
            face_frame = ttk.Frame(self.geo_params_frame, style="Card.TFrame")
            face_frame.pack(fill=tk.X, pady=(5, 0))
            ttk.Label(face_frame, text="面索引:").pack(side=tk.LEFT)
            self.step_face_idx = tk.IntVar(value=0)
            ttk.Entry(face_frame, textvariable=self.step_face_idx, width=5).pack(side=tk.LEFT, padx=5)
            ttk.Button(face_frame, text="👁 预览单面", command=self.preview_step_single).pack(side=tk.LEFT)

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

    def preview_step_single(self):
        """预览 STEP 模型的单个面，显示网格、边和编号"""
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

            face_type = "四边形" if degen_edge is None else f"三角形({degen_edge}退化)" if degen_edge != 'degenerate' else "完全退化"
            step_id = getattr(surf, 'step_id', -1)
            n_edges_attr = getattr(surf, 'n_edges', 0)
            self.log(f"Preview face [idx={face_idx}] STEP#{step_id}: {face_type}, {n_edges_attr} edges")
            self.log(f"  u=[{surf.u_min:.2f},{surf.u_max:.2f}], v=[{surf.v_min:.2f},{surf.v_max:.2f}]")

            # 获取边界边
            edges_data = surf.get_edges(n_samples=30)
            n_edges = len(edges_data)
            edge_colors = plt.cm.Set1(np.linspace(0, 1, max(n_edges, 1)))

            # 绘图
            fig = plt.figure(figsize=(14, 5))

            # 左图：3D 网格 + 边
            ax1 = fig.add_subplot(121, projection='3d')

            # 根据面类型显示不同网格
            if degen_edge is not None and degen_edge != 'degenerate':
                # 三角形面：使用条带状网格
                solver = RibbonIntegrator()

                mesh_cells, a, b = solver.get_triangle_mesh_cells(
                    surf, degen_edge=degen_edge, preview_a=15, preview_b=15
                )
                self.log(f"  Triangle mesh: a={a} strips, b={b} subdivs, {len(mesh_cells)} cells")

                if mesh_cells:
                    # 绘制网格单元边界线
                    for cell in mesh_cells:
                        # cell = [(u0,v0), (u1,v1), (u2,v2), (u3,v3)]
                        u_corners = [c[0] for c in cell] + [cell[0][0]]  # 闭合
                        v_corners = [c[1] for c in cell] + [cell[0][1]]
                        pts_3d = surf.evaluate(np.array(u_corners), np.array(v_corners))
                        ax1.plot(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
                                color='#007ACC', linewidth=0.3, alpha=0.6)

                    ax1.plot([], [], [], color='#007ACC', linewidth=1, label=f'{len(mesh_cells)} cells')
                    ax2_data = {'cells': mesh_cells, 'a': a, 'b': b, 'type': 'triangle'}
                else:
                    ax2_data = {'type': 'empty'}
            else:
                # 四边形面：使用矩形网格
                nu, nv = 20, 20
                u = np.linspace(surf.u_min, surf.u_max, nu)
                v = np.linspace(surf.v_min, surf.v_max, nv)
                uu, vv = np.meshgrid(u, v)
                points, normals, jacobians = surf.get_data(uu, vv)

                X, Y, Z = points[..., 0], points[..., 1], points[..., 2]
                ax1.plot_wireframe(X, Y, Z, color='#007ACC', linewidth=0.3, rstride=2, cstride=2, alpha=0.5)

                ax2_data = {'U': uu, 'V': vv, 'jac': jacobians, 'type': 'quad'}

            # 绘制边并标注局部索引
            for edge_idx, edge in enumerate(edges_data):
                ep = edge['points']
                color = edge_colors[edge_idx % len(edge_colors)]
                ax1.plot(ep[:, 0], ep[:, 1], ep[:, 2], color=color, linewidth=2.5)
                mp = edge['midpoint']
                ax1.text(mp[0], mp[1], mp[2], f' E{edge_idx}', fontsize=9, fontweight='bold',
                         color='white', backgroundcolor=color)

            ax1.set_title(f'Face [idx={face_idx}] STEP#{step_id} ({face_type}) - {n_edges} edges')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # 仅在有标签时显示图例
            handles, labels = ax1.get_legend_handles_labels()
            if labels:
                ax1.legend(loc='upper left')

            # 设置坐标轴比例一致
            all_pts_list = []
            if 'U' in ax2_data: # Quad mesh points
                all_pts_list.append(points.reshape(-1, 3))
            
            # Add edge points
            for edge in edges_data:
                all_pts_list.append(edge['points'])
            
            if all_pts_list:
                all_pts = np.vstack(all_pts_list)
                self._set_axes_equal(ax1, all_pts)

            self._add_scroll_zoom(ax1, fig)

            # 右图：参数域
            ax2 = fig.add_subplot(122)

            if ax2_data['type'] == 'triangle':
                # 三角形网格单元在参数域
                for cell in ax2_data['cells']:
                    u_corners = [c[0] for c in cell] + [cell[0][0]]
                    v_corners = [c[1] for c in cell] + [cell[0][1]]
                    ax2.plot(u_corners, v_corners, color='#007ACC', linewidth=0.5, alpha=0.7)
                ax2.set_xlim(surf.u_min, surf.u_max)
                ax2.set_ylim(surf.v_min, surf.v_max)
                ax2.set_title(f'参数域 - 条带网格 (a={ax2_data["a"]}, b={ax2_data["b"]})')
            elif ax2_data['type'] == 'quad':
                c = ax2.pcolormesh(ax2_data['U'], ax2_data['V'], ax2_data['jac'],
                                   cmap='viridis', shading='auto')
                plt.colorbar(c, ax=ax2, label='Jacobian')
                ax2.set_title(f'参数域 (u,v) - {face_type}')
            else:
                ax2.text(0.5, 0.5, 'No mesh data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('参数域')

            ax2.set_xlabel('u')
            ax2.set_ylabel('v')
            ax2.set_aspect('equal')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.log(f"Single face preview error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def preview_step(self):
        """预览 STEP 模型（使用参数曲面采样，非三角化）"""
        if not self.step_file_path:
            messagebox.showwarning("Warning", "请先选择 STEP 文件")
            return

        self.log(f"Loading STEP file for preview...")

        try:
            surfaces = load_step_file(self.step_file_path)
            self.log(f"Loaded {len(surfaces)} valid surfaces")

            if len(surfaces) == 0:
                messagebox.showwarning("Warning", "STEP 文件中没有有效的曲面")
                return

            # 使用 GUI 设置的实际参数进行预览
            try:
                real_freq = self.freq_var.get() * 1e6
                real_samples = self.density_var.get()
            except:
                real_freq = 300e6
                real_samples = 10
            
            self.log(f"Previewing with Freq={real_freq/1e6:.1f}MHz, Density={real_samples} pts/lambda")

            # 获取扫描参数
            try:
                scan_params = {
                    'theta_start': self.theta_start.get(),
                    'theta_end': self.theta_end.get(),
                    'theta_n': self.theta_n.get(),
                    'phi_start': self.phi_start.get(),
                    'phi_end': self.phi_end.get(),
                    'phi_n': self.phi_n.get()
                }
            except:
                scan_params = {
                    'theta_start': 0.0, 'theta_end': 0.0, 'theta_n': 1,
                    'phi_start': 0.0, 'phi_end': 0.0, 'phi_n': 1
                }

            # 启动后台线程
            t = threading.Thread(
                target=self._preview_step_thread,
                args=(surfaces, real_freq, real_samples, scan_params),
                daemon=True
            )
            t.start()

        except Exception as e:
            self.log(f"STEP Load Error: {e}")
            messagebox.showerror("Error", str(e))

    def _preview_step_thread(self, surfaces, freq, samples, scan_params):
        """后台线程：计算 STEP 预览网格数据"""
        try:
            from physics.wave import IncidentWave
            solver = RibbonIntegrator()
            wave = IncidentWave(freq, 0, 0)

            mesh_data_list = []
            total_points = 0

            for i, surf in enumerate(surfaces):
                try:
                    # 对每个曲面使用固定采样数（预览用）
                    nu, nv = 30, 30  # 固定网格密度用于预览
                    u_min, u_max = surf.u_domain
                    v_min, v_max = surf.v_domain

                    u = np.linspace(u_min, u_max, nu)
                    v = np.linspace(v_min, v_max, nv)
                    uu, vv = np.meshgrid(u, v)

                    points, normals, jacobians = surf.get_data(uu, vv)
                    total_points += nu * nv

                    # 获取 STEP ID
                    step_id = getattr(surf, 'step_id', -1)

                    mesh_data_list.append({
                        'points': points,
                        'normals': normals,
                        'jacobians': jacobians,
                        'nu': nu,
                        'nv': nv,
                        'step_id': step_id,
                        'local_idx': i
                    })
                except Exception as e:
                    self.root.after(0, lambda e=e, i=i: self.log(f"Surface {i} error: {e}"))

            self.root.after(0, lambda: self._do_step_preview_plot(mesh_data_list, total_points, scan_params))

        except Exception as e:
            self.root.after(0, lambda: self.log(f"Preview Error: {e}"))

    def _set_axes_equal(self, ax, points):
        """设置 3D 坐标轴为等比例显示"""
        if points.ndim > 2:
            points = points.reshape(-1, 3)
            
        x_limits = [points[:, 0].min(), points[:, 0].max()]
        y_limits = [points[:, 1].min(), points[:, 1].max()]
        z_limits = [points[:, 2].min(), points[:, 2].max()]

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])
        if plot_radius == 0: plot_radius = 1.0

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        ax.set_box_aspect((1, 1, 1))

    def _create_plot_window(self, title):
        """创建一个嵌入 Matplotlib 的 Toplevel 窗口，避免 plt.show() 的冲突"""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        new_win = tk.Toplevel(self.root)
        new_win.title(title)
        new_win.geometry("1000x800")
        
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=new_win)
        
        def on_scroll(event):
            ax = event.inaxes
            if ax is None: return
            scale = 1.15 if event.button == 'down' else 1/1.15
            xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
            xmid, ymid, zmid = (xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2, (zlim[0]+zlim[1])/2
            xh, yh, zh = (xlim[1]-xlim[0])/2 * scale, (ylim[1]-ylim[0])/2 * scale, (zlim[1]-zlim[0])/2 * scale
            ax.set_xlim(xmid-xh, xmid+xh)
            ax.set_ylim(ymid-yh, ymid+yh)
            ax.set_zlim(zmid-zh, zmid+zh)
            canvas.draw_idle()
        
        canvas.mpl_connect('scroll_event', on_scroll)
        
        toolbar = NavigationToolbar2Tk(canvas, new_win)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        return fig, canvas, new_win

    def _add_scroll_zoom(self, ax, fig):
        """为 3D 图添加滚轮缩放功能"""
        def on_scroll(event):
            if event.inaxes != ax:
                return
            scale = 1.15 if event.button == 'down' else 1/1.15
            # 获取当前范围
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            # 计算中心和新范围
            xmid, ymid, zmid = (xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2, (zlim[0]+zlim[1])/2
            xhalf = (xlim[1]-xlim[0])/2 * scale
            yhalf = (ylim[1]-ylim[0])/2 * scale
            zhalf = (zlim[1]-zlim[0])/2 * scale
            ax.set_xlim(xmid - xhalf, xmid + xhalf)
            ax.set_ylim(ymid - yhalf, ymid + yhalf)
            ax.set_zlim(zmid - zhalf, zmid + zhalf)
            fig.canvas.draw_idle()
        fig.canvas.mpl_connect('scroll_event', on_scroll)

    def _do_step_preview_plot(self, mesh_data_list, total_points, scan_params=None):
        """主线程：在嵌入式窗口中绘制 STEP 预览"""
        try:
            from physics.wave import IncidentWave  # Ensure import
            if scan_params is None:
                scan_params = {}

            fig, canvas, win = self._create_plot_window(f"STEP Preview ({len(mesh_data_list)} surfaces)")
            ax = fig.add_subplot(111, projection='3d')

            colors = plt.cm.tab10(np.linspace(0, 1, min(len(mesh_data_list), 10)))

            all_points_list = [] # For bounding box

            for idx, data in enumerate(mesh_data_list):
                points = data['points']
                all_points_list.append(points.reshape(-1, 3))
                
                nu, nv = data['nu'], data['nv']
                step_id = data.get('step_id', -1)
                local_idx = data.get('local_idx', idx)

                X = points[..., 0]
                Y = points[..., 1]
                Z = points[..., 2]

                # 同样应用动态步长
                stride_u = max(1, nu // 30)
                stride_v = max(1, nv // 30)

                color = colors[idx % len(colors)]
                ax.plot_wireframe(X, Y, Z, color=color, linewidth=0.5,
                                  rstride=stride_v, cstride=stride_u, alpha=0.7)

                # 在面中心添加编号标注
                center_i, center_j = nv // 2, nu // 2
                cx, cy, cz = X[center_i, center_j], Y[center_i, center_j], Z[center_i, center_j]
                # 仅在非密集时显示部分标签
                if len(mesh_data_list) < 50 or idx % 5 == 0:
                     label = f' #{step_id}'
                     ax.text(cx, cy, cz, label, fontsize=8, fontweight='bold',
                             color='black', alpha=0.6)

            # --- 绘制所有扫描方向箭头 ---
            if all_points_list and scan_params:
                all_points = np.vstack(all_points_list)
                
                # 计算包围盒
                min_xyz = np.min(all_points, axis=0)
                max_xyz = np.max(all_points, axis=0)
                center = (min_xyz + max_xyz) / 2.0
                diag = np.linalg.norm(max_xyz - min_xyz)
                radius = diag / 2.0 if diag > 0 else 1.0

                # 生成扫描角度
                theta_s = scan_params.get('theta_start', 0)
                theta_e = scan_params.get('theta_end', 0)
                theta_n = scan_params.get('theta_n', 1)
                
                phi_s = scan_params.get('phi_start', 0)
                phi_e = scan_params.get('phi_end', 0)
                phi_n = scan_params.get('phi_n', 1)
                
                thetas = np.linspace(theta_s, theta_e, theta_n)
                phis = np.linspace(phi_s, phi_e, max(1, phi_n))
                
                scan_directions = []
                
                if phi_n > 1: # 2D 扫描: Meshgrid
                    T, P = np.meshgrid(thetas, phis)
                    scan_directions = list(zip(T.flatten(), P.flatten()))
                    scan_mode = "2D Scan"
                else: # 1D 扫描: 遍历 Theta, Phi 固定
                    # 注意：如果 phi_n=1, phis 只有一个值
                    for t in thetas:
                        for p in phis:
                            scan_directions.append((t, p))
                    scan_mode = "1D Scan"

                # 限制最大显示数量，避免卡顿
                max_arrows = 500
                total_dirs = len(scan_directions)
                step = 1
                if total_dirs > max_arrows:
                    step = total_dirs // max_arrows + 1
                    self.log(f"Showing {total_dirs} directions (subsampled by {step})")
                
                # 准备 quiver 数据
                Xq, Yq, Zq, Uq, Vq, Wq = [], [], [], [], [], []
                
                arrow_len = radius * 0.15  # 保持精致的小尺寸
                dist_from_center = radius * 2.5 # 移到更远的位置，不干扰模型观察
                
                for i in range(0, total_dirs, step):
                    t_deg, p_deg = scan_directions[i]
                    wave = IncidentWave(1e9, np.radians(t_deg), np.radians(p_deg))
                    k = wave.k_dir # 传播方向
                    
                    # 箭头起点
                    start = center - k * dist_from_center
                    
                    Xq.append(start[0])
                    Yq.append(start[1])
                    Zq.append(start[2])
                    Uq.append(k[0] * arrow_len)
                    Vq.append(k[1] * arrow_len)
                    Wq.append(k[2] * arrow_len)

                # 批量绘制箭头 (更细、更短、比例更协调)
                ax.quiver(Xq, Yq, Zq, Uq, Vq, Wq, color='#FF8C00', linewidth=0.6, 
                          arrow_length_ratio=0.3, alpha=0.8)
                
                # 添加总体标签
                ax.text2D(0.05, 0.95, f"{scan_mode}\nDirs: {total_dirs}\nRange: T[{theta_s:.0f}:{theta_e:.0f}], P[{phi_s:.0f}:{phi_e:.0f}]", 
                          transform=ax.transAxes, color='#FF8C00', fontsize=10, fontweight='bold',
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

            # 自动设置坐标轴范围 (Equal Aspect Ratio)
            self._set_axes_equal(ax, all_points)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            canvas.draw()
            self.log(f"STEP Preview: {len(mesh_data_list)} surfaces, {total_points} points")

        except Exception as e:
            self.log(f"Plot Error: {e}")

            # 自动设置坐标轴范围 (Equal Aspect Ratio)
            # 使用包含箭头的数据重新计算可能更好，或者保持仅物体居中
            # 这里保持仅物体居中，箭头可能会延伸出视图，但用户可以缩放
            self._set_axes_equal(ax, all_points)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            canvas.draw()
            self.log(f"STEP Preview: {len(mesh_data_list)} surfaces, {total_points} points")

        except Exception as e:
            self.log(f"Plot Error: {e}")

    def create_action_widgets(self, parent):
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)

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
        # ==================

        # 解析解对比选项
        self.compare_analytical_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent,
            text="对比解析解 (Compare with Analytical)",
            variable=self.compare_analytical_var
        ).pack(anchor=tk.W, pady=(0, 10))

        # 使用自定义样式的按钮
        btn_gen_mesh = ttk.Button(parent, text="📊 生成网格 (Generate Mesh)", command=self.generate_mesh_stats)
        btn_gen_mesh.pack(fill=tk.X, pady=(0, 8))

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
        
        self.log("CEM PO Solver GUI Ready.")
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
                # 获取单位缩放系数
                unit = getattr(self, 'step_unit_var', None)
                scale = 0.001 if (unit and unit.get() == "mm") else 1.0
                self.log(f"Loading STEP file: {self.step_file_path} (unit: {unit.get() if unit else 'm'}, scale: {scale})...")
                surfaces = load_step_file(self.step_file_path, scale=scale)
                self.log(f"Loaded {len(surfaces)} surfaces.")
                return surfaces
                
        except Exception as e:
            self.log(f"Error building geometry: {str(e)}")
            messagebox.showerror("Geometry Error", str(e))
            return None

    def generate_mesh_stats(self):
        """生成网格并显示统计信息，不进行3D可视化（节省内存）"""
        geo = self.build_geometry()
        if not geo:
            return

        freq = self.freq_var.get() * 1e6
        samples = self.density_var.get()

        surfaces = geo if isinstance(geo, list) else [geo]

        self.log("Generating mesh statistics...")
        self.progress_var.set(0)
        self.progress_label.config(text="正在计算网格统计...")

        # 启动后台线程
        t = threading.Thread(target=self._compute_mesh_stats, args=(surfaces, freq, samples), daemon=True)
        t.start()

    def _compute_mesh_stats(self, surfaces, freq, samples):
        """后台线程：计算网格统计信息"""
        try:
            from solver.ribbon_solver import RibbonIntegrator
            from physics.wave import IncidentWave
            solver = RibbonIntegrator()
            wave = IncidentWave(freq, 0, 0)

            total_cells = 0
            total_vertices = 0
            n_total = len(surfaces)
            face_stats = []

            # 用于计算包围盒
            all_min = np.array([np.inf, np.inf, np.inf])
            all_max = np.array([-np.inf, -np.inf, -np.inf])

            for i, surf in enumerate(surfaces):
                # 只计算网格尺寸，不存储实际数据
                nu, nv = solver.get_mesh_size(surf, wave, samples)
                n_cells = nu * nv
                n_vertices = (nu + 1) * (nv + 1)
                total_cells += n_cells
                total_vertices += n_vertices

                # 采样角点来估算包围盒
                u_min, u_max = surf.u_domain
                v_min, v_max = surf.v_domain
                corners_uv = [(u_min, v_min), (u_max, v_min), (u_min, v_max), (u_max, v_max)]
                for u, v in corners_uv:
                    pt = surf.evaluate(np.array([u]), np.array([v]))[0]
                    all_min = np.minimum(all_min, pt)
                    all_max = np.maximum(all_max, pt)

                # 获取面信息
                step_id = getattr(surf, 'step_id', i)
                face_stats.append({
                    'index': i,
                    'step_id': step_id,
                    'nu': nu,
                    'nv': nv,
                    'cells': n_cells
                })

                # 更新进度
                if n_total < 100 or i % 10 == 0 or i == n_total - 1:
                    self._update_progress(i + 1, n_total, f"计算网格: {i+1}/{n_total}")

            # 计算模型尺寸
            model_size = all_max - all_min

            # 在主线程显示结果
            self.root.after(0, lambda: self._show_mesh_stats(
                n_total, total_cells, total_vertices, wave.wavelength, face_stats, model_size
            ))

        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda msg=err_msg: self.log(f"Mesh Stats Error: {msg}"))

    def _show_mesh_stats(self, n_surfaces, total_cells, total_vertices, wavelength, face_stats):
        """显示网格统计信息"""
        self.progress_var.set(100)
        self.progress_label.config(text="网格统计完成")

        # 估算内存占用 (每个顶点约 3*8=24 bytes for coordinates + 3*8=24 for normals)
        estimated_memory_mb = total_vertices * 48 / (1024 * 1024)

        stats_msg = (
            f"═══════════════════════════════════════\n"
            f"          网格统计信息 (Mesh Statistics)\n"
            f"═══════════════════════════════════════\n"
            f"  曲面数量 (Surfaces):     {n_surfaces}\n"
            f"  总网格数 (Total Cells):  {total_cells:,}\n"
            f"  总顶点数 (Total Vertices): {total_vertices:,}\n"
            f"  波长 (Wavelength):       {wavelength*1000:.2f} mm\n"
            f"  预估内存 (Est. Memory):  {estimated_memory_mb:.1f} MB\n"
            f"═══════════════════════════════════════\n"
        )

        self.log(stats_msg)

        # 显示前10个面的详细信息
        if len(face_stats) <= 20:
            self.log("各面网格详情:")
            for fs in face_stats:
                self.log(f"  Face {fs['index']} (#{fs['step_id']}): {fs['nu']}×{fs['nv']} = {fs['cells']:,} cells")
        else:
            self.log(f"前10个面的网格详情:")
            for fs in face_stats[:10]:
                self.log(f"  Face {fs['index']} (#{fs['step_id']}): {fs['nu']}×{fs['nv']} = {fs['cells']:,} cells")
            self.log(f"  ... 还有 {len(face_stats)-10} 个面")

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
        
        # 重置进度条
        self.progress_var.set(0)
        self.progress_label.config(text="正在生成网格数据...")
        
        try:
            # 启动守护线程避免界面冻结
            t = threading.Thread(target=self.plot_multi_surface_mesh, args=(surfaces, freq, samples), daemon=True)
            t.start()
        except Exception as e:
            self.log(f"Visualization Error: {e}")

    def plot_multi_surface_mesh(self, surfaces, freq, samples):
        """后台线程：计算数据并汇报进度"""
        try:
            solver = RibbonIntegrator()
            wave = IncidentWave(freq, 0, 0)

            # 收集所有曲面的网格数据
            mesh_data_list = []
            total_points = 0
            n_total = len(surfaces)

            for i, surf in enumerate(surfaces):
                points, normals, (nu, nv) = solver.get_mesh_data(surf, wave, samples)
                total_points += nu * nv
                mesh_data_list.append({
                    'points': points,
                    'normals': normals,
                    'nu': nu,
                    'nv': nv
                })
                
                # 汇报进度 (降低刷新频率以减少开销)
                if n_total < 100 or i % 5 == 0 or i == n_total - 1:
                    self._update_progress(i + 1, n_total, f"生成网格数据: {i+1}/{n_total}")

            # 数据计算完成，通知主线程开始绘图
            self.root.after(0, lambda: self._do_update_progress(100, "正在渲染 3D 图形 (这可能需要几秒钟)..."))
            self.root.after(0, lambda: self._do_mesh_plot(
                mesh_data_list, total_points, len(surfaces), wave.wavelength
            ))

        except Exception as e:
            self.root.after(0, lambda: self.log(f"Vis Error: {e}"))

    def _do_mesh_plot(self, mesh_data_list, total_points, n_surfaces, wavelength):
        """主线程：在嵌入式窗口中绘制网格数据（使用 Line3DCollection 优化内存）"""
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        try:
            fig, canvas, win = self._create_plot_window(f"Mesh Visualization ({n_surfaces} surfaces)")
            ax = fig.add_subplot(111, projection='3d')

            all_lines = []  # 收集所有线段

            for i, data in enumerate(mesh_data_list):
                points = data['points']
                normals = data['normals']
                nu, nv = data['nu'], data['nv']

                # 动态计算步长以保证高频网格下的流畅度
                stride_u = max(1, nu // 40)
                stride_v = max(1, nv // 40)

                # 采样后的网格
                X = points[::stride_v, ::stride_u, 0]
                Y = points[::stride_v, ::stride_u, 1]
                Z = points[::stride_v, ::stride_u, 2]
                rows, cols = X.shape

                # 生成 u 方向线段 (横向)
                for r in range(rows):
                    for c in range(cols - 1):
                        all_lines.append([
                            (X[r, c], Y[r, c], Z[r, c]),
                            (X[r, c+1], Y[r, c+1], Z[r, c+1])
                        ])

                # 生成 v 方向线段 (纵向)
                for r in range(rows - 1):
                    for c in range(cols):
                        all_lines.append([
                            (X[r, c], Y[r, c], Z[r, c]),
                            (X[r+1, c], Y[r+1, c], Z[r+1, c])
                        ])

                # 绘制法线（仅在面数较少时）
                if i == 0 or n_surfaces < 5:
                    skip = max(1, min(nu, nv) // 8)
                    ax.quiver(points[::skip, ::skip, 0], points[::skip, ::skip, 1], points[::skip, ::skip, 2],
                              normals[::skip, ::skip, 0], normals[::skip, ::skip, 1],
                              normals[::skip, ::skip, 2],
                              length=wavelength/8, color='#FF5555', alpha=0.6)

            # 一次性添加所有线段为单个 Collection 对象
            if all_lines:
                line_collection = Line3DCollection(all_lines, colors='#007ACC', linewidths=0.5, alpha=0.4)
                ax.add_collection3d(line_collection)

            # 设置坐标轴比例一致
            all_points = np.vstack([d['points'].reshape(-1, 3) for d in mesh_data_list])
            self._set_axes_equal(ax, all_points)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            canvas.draw()
            n_lines = len(all_lines)
            self.log(f"Visualization complete. {n_lines} line segments in 1 collection object.")

        except Exception as e:
            self.log(f"Plot Error: {e}")

    def run_calculation(self):
        geo = self.build_geometry()
        if not geo:
            return

        freq = self.freq_var.get() * 1e6
        samples = self.density_var.get()

        # 并行参数
        is_parallel = self.parallel_var.get()
        n_workers = self.workers_var.get()

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
            if is_parallel:
                self.log(f"Parallel mode enabled: {n_workers} workers")
            
            t = threading.Thread(
                target=self._calc_thread_2d,
                args=(geo, freq, theta_rad, theta_deg, phi_rad, phi_deg, samples, geo_type, geo_params, is_parallel, n_workers),
                daemon=True
            )
            t.start()
        else:
            self.log(f"Starting 1D scan: {n_theta} angles, {freq/1e6} MHz...")
            if is_parallel:
                self.log(f"Parallel mode enabled: {n_workers} workers")

            t = threading.Thread(
                target=self._calc_thread,
                args=(geo, freq, theta_rad, theta_deg, samples, geo_type, geo_params, phi_rad[0], is_parallel, n_workers),
                daemon=True
            )
            t.start()

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

    def _calc_thread(self, geo, freq, angles_rad, angles_deg, samples, geo_type, geo_params, phi_rad=0.0, parallel=False, n_workers=None):
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
                parallel=parallel,
                n_workers=n_workers,
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

    def _calc_thread_2d(self, geo, freq, theta_rad, theta_deg, phi_rad, phi_deg, samples, geo_type, geo_params, parallel=False, n_workers=None):
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
                parallel=parallel,
                n_workers=n_workers,
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
    try:
        root = tk.Tk()
        app = CEMPoGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("\n程序已停止 (User Interrupted)")
        try:
            root.destroy()
        except:
            pass
        sys.exit(0)