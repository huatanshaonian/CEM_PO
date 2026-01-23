import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from physics.wave import IncidentWave
from physics.analytical_rcs import get_analytical_solution, compute_error_stats

# 设置 matplotlib 后端和字体
matplotlib.use('TkAgg')
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


class VisualizationManager:
    """
    负责所有 Matplotlib 相关的绘图和可视化操作。
    支持双 Tab 嵌入式绘图：预览 Tab 和 RCS 结果 Tab。
    """
    def __init__(self, root, log_callback=None, theme_colors=None,
                 preview_frame=None, rcs_frame=None, compare_frame=None, notebook=None):
        self.root = root
        self.log_callback = log_callback if log_callback else print
        self.colors = theme_colors if theme_colors else {
            "bg_main": "#FAFAFA",
            "accent": "#007ACC"
        }
        # 多 Tab 框架
        self.preview_frame = preview_frame  # 几何预览 Tab
        self.rcs_frame = rcs_frame          # RCS 结果 Tab
        self.compare_frame = compare_frame  # 对比 Tab (新增)
        self.notebook = notebook            # Notebook 控件

        # 当前嵌入的 canvas 和 toolbar
        self.preview_canvas = None
        self.preview_toolbar = None
        self.rcs_canvas = None
        self.rcs_toolbar = None
        self.compare_canvas = None
        self.compare_toolbar = None

    def log(self, msg):
        self.log_callback(msg)

    def _clear_frame(self, frame):
        """清除指定框架中的所有控件"""
        if frame:
            for widget in frame.winfo_children():
                widget.destroy()

    def _switch_to_preview_tab(self):
        """切换到预览 Tab (Index 0)"""
        if self.notebook:
            self.notebook.select(0)

    def _switch_to_rcs_tab(self):
        """切换到 RCS 结果 Tab (Index 1)"""
        if self.notebook:
            self.notebook.select(1)
            
    def _switch_to_compare_tab(self):
        """切换到对比 Tab (Index 2)"""
        if self.notebook and self.notebook.index("end") > 2:
            self.notebook.select(2)

    def _add_scroll_zoom(self, ax, fig, canvas=None):
        """为 3D 坐标轴添加滚轮缩放功能"""
        def on_scroll(event):
            if event.inaxes != ax:
                return
            scale = 1.15 if event.button == 'down' else 1/1.15
            xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
            xmid = (xlim[0] + xlim[1]) / 2
            ymid = (ylim[0] + ylim[1]) / 2
            zmid = (zlim[0] + zlim[1]) / 2
            xh = (xlim[1] - xlim[0]) / 2 * scale
            yh = (ylim[1] - ylim[0]) / 2 * scale
            zh = (zlim[1] - zlim[0]) / 2 * scale
            ax.set_xlim(xmid - xh, xmid + xh)
            ax.set_ylim(ymid - yh, ymid + yh)
            ax.set_zlim(zmid - zh, zmid + zh)
            if canvas:
                canvas.draw_idle()
            else:
                fig.canvas.draw_idle()
        fig.canvas.mpl_connect('scroll_event', on_scroll)

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
        if plot_radius == 0:
            plot_radius = 1.0

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        ax.set_box_aspect((1, 1, 1))

    def show_step_preview(self, mesh_data_list, total_points, scan_params=None):
        """显示 STEP 模型预览，带交互式面选择侧边栏（嵌入式）"""
        try:
            if scan_params is None:
                scan_params = {}

            # 清除旧内容并切换到预览 Tab
            self._clear_frame(self.preview_frame)
            self._switch_to_preview_tab()

            if not self.preview_frame:
                self.log("Error: No preview frame available")
                return

            # --- 布局：左侧绘图，右侧控制 ---
            plot_frame = ttk.Frame(self.preview_frame)
            plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            control_frame = ttk.Frame(self.preview_frame, width=200, padding=5)
            control_frame.pack(side=tk.RIGHT, fill=tk.Y)

            ttk.Label(control_frame, text="面可见性控制", font=("Microsoft YaHei UI", 10, "bold")).pack(pady=(0, 5))

            # 全选/全不选按钮
            btn_frame = ttk.Frame(control_frame)
            btn_frame.pack(fill=tk.X, pady=5)

            # 滚动区域容器
            canvas_container = ttk.Frame(control_frame)
            canvas_container.pack(fill=tk.BOTH, expand=True)

            # 滚动条和 Canvas
            scrollbar = ttk.Scrollbar(canvas_container, orient="vertical")
            scroll_canvas = tk.Canvas(canvas_container, yscrollcommand=scrollbar.set, width=180)

            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=scroll_canvas.yview)

            # 复选框容器 Frame
            chk_frame = ttk.Frame(scroll_canvas)
            scroll_canvas.create_window((0, 0), window=chk_frame, anchor="nw")

            def on_frame_configure(event):
                scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
            chk_frame.bind("<Configure>", on_frame_configure)

            # --- Matplotlib 设置 ---
            fig = plt.Figure(figsize=(6, 6), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            self.preview_canvas = canvas

            ax = fig.add_subplot(111, projection='3d')
            self._add_scroll_zoom(ax, fig, canvas)

            # 工具栏框架
            toolbar_frame = ttk.Frame(plot_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            self.preview_toolbar = toolbar

            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # --- 预计算 ---
            if mesh_data_list:
                all_pts_flat = np.vstack([d['points'].reshape(-1, 3) for d in mesh_data_list])
                min_p = np.min(all_pts_flat, axis=0)
                max_p = np.max(all_pts_flat, axis=0)
                center = (min_p + max_p) / 2.0
                model_diag = np.linalg.norm(max_p - min_p)
                radius = model_diag / 2.0 if model_diag > 0 else 1.0
                normal_len = model_diag / 25.0 if model_diag > 0 else 1.0
            else:
                center = np.array([0, 0, 0])
                radius = 1.0
                normal_len = 0.1

            colors = plt.cm.tab10(np.linspace(0, 1, min(len(mesh_data_list), 10)))

            # 存储每个面的绘图对象和控制变量
            surface_artists = {}
            check_vars = []

            # 切换可见性回调
            def toggle_visibility(idx, var):
                is_visible = var.get()
                if idx in surface_artists:
                    for artist in surface_artists[idx]:
                        if artist:
                            artist.set_visible(is_visible)
                    canvas.draw_idle()

            for idx, data in enumerate(mesh_data_list):
                points = data['points']
                normals = data['normals']
                nu, nv = data['nu'], data['nv']
                step_id = data.get('step_id', -1)

                X = points[..., 0]
                Y = points[..., 1]
                Z = points[..., 2]

                stride_u = max(1, nu // 30)
                stride_v = max(1, nv // 30)

                color = colors[idx % len(colors)]

                # 绘制网格
                w_artist = ax.plot_wireframe(X, Y, Z, color=color, linewidth=0.5,
                                             rstride=stride_v, cstride=stride_u, alpha=0.7)

                # 绘制法向量
                target_arrows = 6
                skip_u = max(1, nu // target_arrows)
                skip_v = max(1, nv // target_arrows)

                q_artist = ax.quiver(points[::skip_v, ::skip_u, 0], points[::skip_v, ::skip_u, 1], points[::skip_v, ::skip_u, 2],
                                     normals[::skip_v, ::skip_u, 0], normals[::skip_v, ::skip_u, 1], normals[::skip_v, ::skip_u, 2],
                                     length=normal_len, color='#FF5555', alpha=0.7, linewidth=0.8)

                # 绘制标签
                center_i, center_j = nv // 2, nu // 2
                cx, cy, cz = X[center_i, center_j], Y[center_i, center_j], Z[center_i, center_j]

                t_artist = None
                if len(mesh_data_list) < 50 or idx % 5 == 0:
                    label = f' [{idx}] #{step_id}'
                    t_artist = ax.text(cx, cy, cz, label, fontsize=8, fontweight='bold',
                                       color='black', alpha=0.6)

                # 收集对象
                surface_artists[idx] = [w_artist, q_artist, t_artist]

                # 添加复选框
                var = tk.BooleanVar(value=True)
                check_vars.append(var)

                cb = ttk.Checkbutton(chk_frame, text=f"Face {idx} (#{step_id})", variable=var,
                                     command=lambda i=idx, v=var: toggle_visibility(i, v))
                cb.pack(anchor="w", padx=5, pady=2)

            # 全选/全不选功能
            def select_all(state):
                for v in check_vars:
                    v.set(state)
                for idx, artists in surface_artists.items():
                    for artist in artists:
                        if artist:
                            artist.set_visible(state)
                canvas.draw_idle()

            ttk.Button(btn_frame, text="全选", command=lambda: select_all(True), width=6).pack(side=tk.LEFT, padx=2)
            ttk.Button(btn_frame, text="全不选", command=lambda: select_all(False), width=8).pack(side=tk.LEFT, padx=2)

            # 绘制入射波方向
            if mesh_data_list and scan_params:
                theta_s = scan_params.get('theta_start', 0)
                theta_e = scan_params.get('theta_end', 0)
                theta_n = scan_params.get('theta_n', 1)
                phi_s = scan_params.get('phi_start', 0)
                phi_e = scan_params.get('phi_end', 0)
                phi_n = scan_params.get('phi_n', 1)

                thetas = np.linspace(theta_s, theta_e, theta_n)
                phis = np.linspace(phi_s, phi_e, max(1, phi_n))
                scan_directions = []

                if phi_n > 1:
                    T, P = np.meshgrid(thetas, phis)
                    scan_directions = list(zip(T.flatten(), P.flatten()))
                    scan_mode = "2D Scan"
                else:
                    for t in thetas:
                        for p in phis:
                            scan_directions.append((t, p))
                    scan_mode = "1D Scan"

                max_arrows = 500
                total_dirs = len(scan_directions)
                step = 1
                if total_dirs > max_arrows:
                    step = total_dirs // max_arrows + 1

                Xq, Yq, Zq, Uq, Vq, Wq = [], [], [], [], [], []
                arrow_len = normal_len * 4
                dist_from_center = radius * 2.5

                for i in range(0, total_dirs, step):
                    t_deg, p_deg = scan_directions[i]
                    wave = IncidentWave(1e9, np.radians(t_deg), np.radians(p_deg))
                    k = wave.k_dir
                    start = center - k * dist_from_center
                    Xq.append(start[0])
                    Yq.append(start[1])
                    Zq.append(start[2])
                    Uq.append(k[0] * arrow_len)
                    Vq.append(k[1] * arrow_len)
                    Wq.append(k[2] * arrow_len)

                ax.quiver(Xq, Yq, Zq, Uq, Vq, Wq, color='#FF8C00', linewidth=0.6,
                          arrow_length_ratio=0.3, alpha=0.8)

                ax.text2D(0.05, 0.95, f"{scan_mode}\nDirs: {total_dirs}\nRange: T[{theta_s:.0f}:{theta_e:.0f}], P[{phi_s:.0f}:{phi_e:.0f}]",
                          transform=ax.transAxes, color='#FF8C00', fontsize=10, fontweight='bold',
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

            self._set_axes_equal(ax, all_pts_flat if mesh_data_list else np.array([[0, 0, 0]]))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            canvas.draw()
            self.log(f"STEP Preview: {len(mesh_data_list)} surfaces, {total_points} points")

        except Exception as e:
            self.log(f"Plot Error: {e}")
            import traceback
            traceback.print_exc()

    def show_mesh_visualization(self, mesh_data_list, total_points, n_surfaces, wavelength):
        """显示 PO 网格可视化（嵌入式）"""
        try:
            # 清除旧内容并切换到预览 Tab
            self._clear_frame(self.preview_frame)
            self._switch_to_preview_tab()

            if not self.preview_frame:
                self.log("Error: No preview frame available")
                return

            # 嵌入式布局
            fig = plt.Figure(figsize=(8, 6), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
            ax = fig.add_subplot(111, projection='3d')
            self.preview_canvas = canvas

            self._add_scroll_zoom(ax, fig, canvas)

            # 工具栏框架
            toolbar_frame = ttk.Frame(self.preview_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            self.preview_toolbar = toolbar

            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            all_lines = []

            for i, data in enumerate(mesh_data_list):
                points = data['points']
                normals = data['normals']
                nu, nv = data['nu'], data['nv']

                stride_u = max(1, nu // 40)
                stride_v = max(1, nv // 40)

                X = points[::stride_v, ::stride_u, 0]
                Y = points[::stride_v, ::stride_u, 1]
                Z = points[::stride_v, ::stride_u, 2]
                rows, cols = X.shape

                for r in range(rows):
                    for c in range(cols - 1):
                        all_lines.append([(X[r, c], Y[r, c], Z[r, c]), (X[r, c + 1], Y[r, c + 1], Z[r, c + 1])])

                for r in range(rows - 1):
                    for c in range(cols):
                        all_lines.append([(X[r, c], Y[r, c], Z[r, c]), (X[r + 1, c], Y[r + 1, c], Z[r + 1, c])])

                # 绘制法线
                target_arrows = 6
                skip_u = max(1, nu // target_arrows)
                skip_v = max(1, nv // target_arrows)

                ax.quiver(points[::skip_v, ::skip_u, 0], points[::skip_v, ::skip_u, 1], points[::skip_v, ::skip_u, 2],
                          normals[::skip_v, ::skip_u, 0], normals[::skip_v, ::skip_u, 1],
                          normals[::skip_v, ::skip_u, 2],
                          length=wavelength / 6, color='#FF5555', alpha=0.8, linewidth=0.8)

            if all_lines:
                line_collection = Line3DCollection(all_lines, colors='#007ACC', linewidths=0.5, alpha=0.4)
                ax.add_collection3d(line_collection)

            all_points = np.vstack([d['points'].reshape(-1, 3) for d in mesh_data_list])
            self._set_axes_equal(ax, all_points)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            canvas.draw()
            self.log(f"Mesh Visualization: {n_surfaces} surfaces, {len(all_lines)} segments")

        except Exception as e:
            self.log(f"Plot Error: {e}")

    def show_single_face_preview(self, surf, face_idx, degen_edge, edges_data, solver=None):
        """显示单个面的详细预览（嵌入式，带参数域视图）"""
        try:
            # 清除旧内容并切换到预览 Tab
            self._clear_frame(self.preview_frame)
            self._switch_to_preview_tab()

            if not self.preview_frame:
                self.log("Error: No preview frame available")
                return

            face_type = "四边形" if degen_edge is None else f"三角形({degen_edge}退化)" if degen_edge != 'degenerate' else "完全退化"
            step_id = getattr(surf, 'step_id', -1)
            n_edges = len(edges_data)

            self.log(f"Preview face [idx={face_idx}] STEP#{step_id}: {face_type}, {n_edges} edges")

            # 创建 Figure
            fig = plt.Figure(figsize=(12, 5), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
            self.preview_canvas = canvas

            # 左图：3D 网格 + 边
            ax1 = fig.add_subplot(121, projection='3d')

            # 获取边界边颜色
            edge_colors = plt.cm.Set1(np.linspace(0, 1, max(n_edges, 1)))

            # 根据面类型显示不同网格
            if degen_edge is not None and degen_edge != 'degenerate' and solver:
                # 三角形面：使用条带状网格
                mesh_cells, a, b = solver.get_triangle_mesh_cells(
                    surf, degen_edge=degen_edge, preview_a=15, preview_b=15
                )
                self.log(f"  Triangle mesh: a={a} strips, b={b} subdivs, {len(mesh_cells)} cells")

                if mesh_cells:
                    for cell in mesh_cells:
                        u_corners = [c[0] for c in cell] + [cell[0][0]]
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

                ax2_data = {'U': uu, 'V': vv, 'jac': jacobians, 'type': 'quad', 'points': points}

            # 绘制边并标注局部索引
            all_edge_pts = []
            for edge_idx, edge in enumerate(edges_data):
                ep = edge['points']
                all_edge_pts.append(ep)
                color = edge_colors[edge_idx % len(edge_colors)]
                ax1.plot(ep[:, 0], ep[:, 1], ep[:, 2], color=color, linewidth=2.5)
                mp = edge['midpoint']
                ax1.text(mp[0], mp[1], mp[2], f' E{edge_idx}', fontsize=9, fontweight='bold',
                         color='white', backgroundcolor=color)

            ax1.set_title(f'Face [idx={face_idx}] STEP#{step_id} ({face_type}) - {n_edges} edges')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')

            handles, labels = ax1.get_legend_handles_labels()
            if labels:
                ax1.legend(loc='upper left')

            # 设置坐标轴比例
            if all_edge_pts:
                all_pts = np.vstack(all_edge_pts)
                if ax2_data.get('type') == 'quad' and 'points' in ax2_data:
                    all_pts = np.vstack([all_pts, ax2_data['points'].reshape(-1, 3)])
                self._set_axes_equal(ax1, all_pts)

            self._add_scroll_zoom(ax1, fig, canvas)

            # 右图：参数域
            ax2 = fig.add_subplot(122)

            if ax2_data['type'] == 'triangle':
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
                fig.colorbar(c, ax=ax2, label='Jacobian')
                ax2.set_title(f'参数域 (u,v) - {face_type}')
            else:
                ax2.text(0.5, 0.5, 'No mesh data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('参数域')

            ax2.set_xlabel('u')
            ax2.set_ylabel('v')
            ax2.set_aspect('equal')

            # 工具栏
            toolbar_frame = ttk.Frame(self.preview_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            self.preview_toolbar = toolbar

            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            fig.tight_layout()
            canvas.draw()

        except Exception as e:
            self.log(f"Single face preview error: {e}")
            import traceback
            traceback.print_exc()

    def show_1d_results(self, result_data, compare_analytical=True):
        """显示 1D RCS 结果（嵌入式）"""
        try:
            # 清除旧内容并切换到 RCS Tab
            self._clear_frame(self.rcs_frame)
            self._switch_to_rcs_tab()

            if not self.rcs_frame:
                self.log("Error: No RCS frame available")
                return

            angles_deg = result_data['angles_deg']
            rcs = result_data['rcs']
            freq = result_data['freq']
            geo_type = result_data['geo_type']
            geo_params = result_data['geo_params']
            angles_rad = result_data['angles_rad']

            fig = plt.Figure(figsize=(10, 6), dpi=100, facecolor=self.colors["bg_main"])
            canvas = FigureCanvasTkAgg(fig, master=self.rcs_frame)
            self.rcs_canvas = canvas

            ax = fig.add_subplot(111)
            ax.plot(angles_deg, rcs, color=self.colors["accent"], linewidth=2, label='Ribbon PO (Numerical)')

            if compare_analytical and geo_params:
                analytical_type = None
                if "Cylinder" in geo_type:
                    analytical_type = 'cylinder'
                elif "Plate" in geo_type:
                    analytical_type = 'plate'
                elif "Sphere" in geo_type:
                    analytical_type = 'sphere'

                if analytical_type:
                    rcs_analytical, label = get_analytical_solution(analytical_type, geo_params, freq, angles_rad)
                    if rcs_analytical is not None:
                        ax.plot(angles_deg, rcs_analytical, 'r--', linewidth=2, label=label)
                        stats = compute_error_stats(rcs, rcs_analytical)
                        error_text = (f"Error Stats:\n"
                                      f"  Max: {stats['max_error']:.2f} dB\n"
                                      f"  Mean: {stats['mean_error']:.2f} dB\n"
                                      f"  RMS: {stats['rms_error']:.2f} dB")
                        ax.text(0.02, 0.98, error_text, transform=ax.transAxes, fontsize=9, va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                        self.log(f"Error Stats - Max: {stats['max_error']:.2f}dB, RMS: {stats['rms_error']:.2f}dB")

            ax.set_xlabel('Theta (deg)', fontsize=11)
            ax.set_ylabel('RCS (dBsm)', fontsize=11)
            ax.set_title(f'Monostatic RCS - {geo_type} @ {freq / 1e6:.1f} MHz', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='best')

            # 工具栏
            toolbar_frame = ttk.Frame(self.rcs_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            self.rcs_toolbar = toolbar

            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            fig.tight_layout()
            canvas.draw()
            self.log(f"1D RCS result displayed: {len(angles_deg)} angles")

        except Exception as e:
            self.log(f"1D RCS Plot Error: {e}")

    def show_2d_results(self, result_data, style='contour'):
        """显示 2D RCS 结果（嵌入式）"""
        try:
            # 清除旧内容并切换到 RCS Tab
            self._clear_frame(self.rcs_frame)
            self._switch_to_rcs_tab()

            if not self.rcs_frame:
                self.log("Error: No RCS frame available")
                return

            theta_deg = result_data['theta_deg']
            phi_deg = result_data['phi_deg']
            rcs_2d = result_data['rcs_2d']
            freq = result_data['freq']
            geo_type = result_data['geo_type']

            fig = plt.Figure(figsize=(10, 7), dpi=100, facecolor=self.colors["bg_main"])
            canvas = FigureCanvasTkAgg(fig, master=self.rcs_frame)
            self.rcs_canvas = canvas

            ax = fig.add_subplot(111)
            
            # 统计信息
            rcs_max = np.nanmax(rcs_2d)
            rcs_min = np.nanmin(rcs_2d)
            rcs_mean = np.nanmean(rcs_2d)
            
            if style == 'pixel':
                # 使用 imshow (像素模式)
                # 注意：imshow 的 extent 顺序是 [xmin, xmax, ymax, ymin] (注意Y轴通常是反的，但在RCS图中Theta从0到180向下)
                # 这里我们希望 Theta 0 在上，180 在下
                im = ax.imshow(rcs_2d, 
                               extent=[phi_deg.min(), phi_deg.max(), theta_deg.max(), theta_deg.min()],
                               aspect='equal', origin='upper', cmap='jet')
                cbar = fig.colorbar(im, ax=ax, shrink=0.9, aspect=20)
                plot_title = f'2D RCS (Pixel) - {geo_type}'
            else:
                # 使用 contourf (平滑模式) - 默认
                Theta, Phi = np.meshgrid(theta_deg, phi_deg, indexing='ij')
                levels = np.linspace(rcs_min, rcs_max, 50)
                contour = ax.contourf(Phi, Theta, rcs_2d, levels=levels, cmap='jet')
                cbar = fig.colorbar(contour, ax=ax, shrink=0.9, aspect=20)
                # 叠加等高线
                ax.contour(Phi, Theta, rcs_2d, levels=15, colors='k', linewidths=0.3, alpha=0.5)
                ax.invert_yaxis() # 确保 Theta 0 在上方
                ax.set_aspect('equal')
                plot_title = f'2D RCS (Contour) - {geo_type}'

            cbar.set_label('RCS (dBsm)', fontsize=11)
            ax.set_xlabel('Phi (deg)', fontsize=11)
            ax.set_ylabel('Theta (deg)', fontsize=11)
            ax.set_title(f'{plot_title} @ {freq / 1e6:.1f} MHz', fontsize=12)

            stats_text = f"Stats:\nMax: {rcs_max:.2f} dBsm\nMin: {rcs_min:.2f} dBsm\nMean: {rcs_mean:.2f} dBsm"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # 工具栏
            toolbar_frame = ttk.Frame(self.rcs_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            self.rcs_toolbar = toolbar

            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            fig.tight_layout()
            canvas.draw()
            self.log(f"2D RCS displayed ({style}) - Max: {rcs_max:.2f}dBsm")

        except Exception as e:
            self.log(f"2D RCS Plot Error: {e}")
            import traceback
            traceback.print_exc()

    def show_comparison_2d(self, calc_data, ref_data, diff_data, theta, phi, metrics, style='pixel'):
        """显示 2D 对比结果：计算值 vs 参考值 vs 误差"""
        try:
            # 清除旧内容并切换到对比 Tab
            self._clear_frame(self.compare_frame)
            self._switch_to_compare_tab()
            
            if not self.compare_frame:
                self.log("Error: No Comparison frame available")
                return

            target_frame = self.compare_frame

            # 使用 constrained_layout 替代 tight_layout，处理多子图和 colorbar 更稳健
            fig = plt.Figure(figsize=(12, 5), dpi=100, facecolor=self.colors["bg_main"], constrained_layout=True)
            canvas = FigureCanvasTkAgg(fig, master=target_frame)
            self.compare_canvas = canvas
            
            # 创建 1行3列 的子图
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            
            axes = [ax1, ax2, ax3]
            titles = ["Calculated RCS", "Reference RCS", "Difference (Calc - Ref)"]
            datas = [calc_data, ref_data, diff_data]
            
            # 统一计算/参考的颜色范围
            vmin = min(np.nanmin(calc_data), np.nanmin(ref_data))
            vmax = max(np.nanmax(calc_data), np.nanmax(ref_data))
            
            # 误差的颜色范围 (以0为中心)
            diff_max = np.nanmax(np.abs(diff_data))
            diff_vmin, diff_vmax = -diff_max, diff_max

            extent = [phi.min(), phi.max(), theta.max(), theta.min()]

            for i, ax in enumerate(axes):
                data = datas[i]
                
                # 绘图逻辑
                if i < 2: # 计算值和参考值
                    if style == 'pixel':
                        im = ax.imshow(data, extent=extent, aspect='auto', origin='upper', 
                                     cmap='jet', vmin=vmin, vmax=vmax)
                    else:
                        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
                        im = ax.contourf(Phi, Theta, data, levels=50, cmap='jet', vmin=vmin, vmax=vmax)
                        ax.invert_yaxis()
                    
                    # 只在第二个图(参考值)右侧加 colorbar，代表前两个图的标尺
                    if i == 1: 
                         cbar = fig.colorbar(im, ax=ax, shrink=0.9, aspect=20)
                         cbar.set_label('RCS (dBsm)')
                
                else: # 误差图
                    if style == 'pixel':
                        im = ax.imshow(data, extent=extent, aspect='auto', origin='upper', 
                                     cmap='seismic', vmin=diff_vmin, vmax=diff_vmax)
                    else:
                        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
                        im = ax.contourf(Phi, Theta, data, levels=50, cmap='seismic', vmin=diff_vmin, vmax=diff_vmax)
                        ax.invert_yaxis()
                    
                    cbar = fig.colorbar(im, ax=ax, shrink=0.9, aspect=20)
                    cbar.set_label('Error (dB)')

                ax.set_title(titles[i], fontsize=10)
                ax.set_xlabel('Phi')
                if i == 0: ax.set_ylabel('Theta')
            
            # 添加整体统计信息
            fig.suptitle(f"Comparison: Mean Error = {metrics['mean_error']:.2f} dB, RMSE = {metrics['rmse']:.2f} dB", fontsize=12)

            # 工具栏
            toolbar_frame = ttk.Frame(target_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            self.compare_toolbar = toolbar
            
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
            # constrained_layout 会自动处理布局，不需要 tight_layout
            canvas.draw()
            self.log(f"Comparison displayed: RMSE={metrics['rmse']:.2f}dB")
            
        except Exception as e:
            self.log(f"Comparison Plot Error: {e}")
            import traceback
            traceback.print_exc()
