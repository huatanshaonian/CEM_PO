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

        # 可视化状态标志
        self.show_normals = False
        self.show_incidence = False
        self.normal_quivers = [] # 存储法向箭头对象列表 (每个 subplot 或 patch 一个)
        self.incidence_quivers = [] # 存储入射波箭头对象

    def set_normals_visible(self, visible):
        self.show_normals = visible
        # 尝试直接更新现有图形
        for q in self.normal_quivers:
            try:
                # Quiver 不支持 set_visible 的 bug workaround: 设置 alpha
                alpha = 0.6 if visible else 0.0
                q.set_alpha(alpha)
            except:
                pass
        
        # 触发重绘
        if self.preview_canvas:
            self.preview_canvas.draw()

    def set_incidence_visible(self, visible):
        self.show_incidence = visible
        for q in self.incidence_quivers:
            try:
                alpha = 1.0 if visible else 0.0
                q.set_alpha(alpha)
            except:
                pass

        if self.preview_canvas:
            self.preview_canvas.draw()

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

    def show_step_preview(self, mesh_data_list, total_points, scan_params, ptd_edges_data=None, incident_dir=None):
        """显示整体几何预览（所有面）"""
        try:
            self._clear_frame(self.preview_frame)
            self._switch_to_preview_tab()

            # 重置箭头
            self.normal_quivers = []
            self.incidence_quivers = []

            fig = plt.Figure(figsize=(10, 6), dpi=100, facecolor=self.colors["bg_main"])
            canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
            self.preview_canvas = canvas

            ax = fig.add_subplot(111, projection='3d')
            self._add_scroll_zoom(ax, fig, canvas)

            # 工具栏框架
            toolbar_frame = ttk.Frame(self.preview_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            self.preview_toolbar = toolbar

            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # --- 右侧控制面板 ---
            right_panel = ttk.Frame(self.preview_frame) # 不指定宽度，靠内容撑开或 pack 属性
            # 注意：之前 canvas 是 pack(side=TOP)，这会占据整个宽度。
            # 如果想要右侧面板，canvas 应该 pack(side=LEFT, expand=1)。
            # 但为了不破坏现有布局（可能是上下结构？），我先把 canvas 改为 LEFT，或者 right_panel 改为 BOTTOM？
            # 看原代码逻辑，preview_frame 是主容器。
            # 让我们把 canvas 改为 LEFT，right_panel 改为 RIGHT。
            canvas.get_tk_widget().pack_forget() # 先忘记之前可能 pack 的
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

            # 按钮区
            btn_frame = ttk.Frame(right_panel)
            btn_frame.pack(fill=tk.X, pady=5)
            
            # 滚动列表区
            list_frame = ttk.Frame(right_panel)
            list_frame.pack(fill=tk.BOTH, expand=True)
            
            chk_canvas = tk.Canvas(list_frame, width=200) # 给个默认宽度
            scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=chk_canvas.yview)
            chk_frame = ttk.Frame(chk_canvas)

            chk_frame.bind(
                "<Configure>",
                lambda e: chk_canvas.configure(scrollregion=chk_canvas.bbox("all"))
            )

            chk_canvas.create_window((0, 0), window=chk_frame, anchor="nw")
            chk_canvas.configure(yscrollcommand=scrollbar.set)

            chk_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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

                init_alpha = 0.7 if self.show_normals else 0.0
                q_artist = ax.quiver(points[::skip_v, ::skip_u, 0], points[::skip_v, ::skip_u, 1], points[::skip_v, ::skip_u, 2],
                                     normals[::skip_v, ::skip_u, 0], normals[::skip_v, ::skip_u, 1], normals[::skip_v, ::skip_u, 2],
                                     length=normal_len, color='#FF5555', alpha=init_alpha, linewidth=0.8)
                self.normal_quivers.append(q_artist)

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
            
            # --- 绘制 PTD 边缘 ---
            if ptd_edges_data:
                self.log(f"Highlighting {len(ptd_edges_data)} PTD edges")
                for edge in ptd_edges_data:
                    # edge 包含 'points': (N, 3) 数组
                    pts = edge['points']
                    
                    # 绘制多段线
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
                            color='#FFFF00', linewidth=3.0, alpha=1.0, zorder=100)
                    
                    # 标注边缘名称 (在中点位置)
                    mid_idx = len(pts) // 2
                    mid = pts[mid_idx]
                    ax.text(mid[0], mid[1], mid[2], edge['name'], 
                           color='blue', fontsize=9, fontweight='bold', zorder=101)

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

            # --- 全局显示控制 ---
            # 放在 right_panel 的最上面 (before btn_frame) 或者下面
            # 由于 btn_frame 已经 pack 了，我们再 pack 一个 ctrl_frame 到 right_panel
            ctrl_frame = ttk.Frame(right_panel)
            ctrl_frame.pack(side=tk.TOP, fill=tk.X, pady=5, before=btn_frame)
            
            # 显示法向
            var_norm = tk.BooleanVar(value=self.show_normals)
            def toggle_norm():
                self.set_normals_visible(var_norm.get())
            cb_norm = ttk.Checkbutton(ctrl_frame, text="显示法向", variable=var_norm, command=toggle_norm)
            cb_norm.pack(anchor="w", padx=5)

            # 显示入射波
            var_inc = tk.BooleanVar(value=self.show_incidence)
            def toggle_inc():
                self.set_incidence_visible(var_inc.get())
            cb_inc = ttk.Checkbutton(ctrl_frame, text="显示入射波", variable=var_inc, command=toggle_inc)
            cb_inc.pack(anchor="w", padx=5)

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

                init_inc_alpha = 0.8 if self.show_incidence else 0.0
                q_inc = ax.quiver(Xq, Yq, Zq, Uq, Vq, Wq, color='#FF8C00', linewidth=0.6,
                          arrow_length_ratio=0.3, alpha=init_inc_alpha)
                self.incidence_quivers.append(q_inc)

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

    def show_single_face_preview(self, surf, face_idx, degen_edge, edges_data, solver, incident_dir=None):
        """显示单面详细预览（带参数域和拓扑检查）"""
        try:
            self._clear_frame(self.preview_frame)
            self._switch_to_preview_tab()

            # 重置箭头列表
            self.normal_quivers = []
            self.incidence_quivers = []

            fig = plt.Figure(figsize=(12, 6), dpi=100, facecolor=self.colors["bg_main"])
            canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
            self.preview_canvas = canvas

            # 左图：3D 几何
            ax1 = fig.add_subplot(121, projection='3d')
            # ...

            # 根据面类型显示不同网格
            u_min, u_max = surf.u_domain
            v_min, v_max = surf.v_domain
            face_type = "Quad"
            step_id = getattr(surf, 'step_id', -1)
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
                u = np.linspace(u_min, u_max, nu)
                v = np.linspace(v_min, v_max, nv)
                uu, vv = np.meshgrid(u, v)

                data_res = surf.get_data(uu, vv)
                if len(data_res) == 5:
                    points, normals, jacobians, _, _ = data_res
                else:
                    points, normals, jacobians = data_res

                X, Y, Z = points[..., 0], points[..., 1], points[..., 2]
                ax1.plot_wireframe(X, Y, Z, color='#007ACC', linewidth=0.3, rstride=2, cstride=2, alpha=0.5)

                ax2_data = {'U': uu, 'V': vv, 'jac': jacobians, 'type': 'quad', 'points': points}

            # --- 绘制法向和入射波 ---
            if 'points' in locals() and points is not None and 'normals' in locals() and normals is not None:
                # 降采样
                try:
                    step_u = max(1, points.shape[1] // 8)
                    step_v = max(1, points.shape[0] // 8)
                    pts_sub = points[::step_v, ::step_u].reshape(-1, 3)
                    nrm_sub = normals[::step_v, ::step_u].reshape(-1, 3)
                    
                    # 法向 (Red)
                    alpha_n = 0.6 if self.show_normals else 0.0
                    q_norm = ax1.quiver(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2],
                                        nrm_sub[:, 0], nrm_sub[:, 1], nrm_sub[:, 2],
                                        length=0.2 * np.max(np.ptp(points.reshape(-1,3), axis=0)), 
                                        color='red', pivot='tail', alpha=alpha_n)
                    self.normal_quivers.append(q_norm)
                    
                    # 入射波 (Green)
                    if incident_dir is not None:
                        alpha_i = 1.0 if self.show_incidence else 0.0
                        k_sub = np.tile(incident_dir, (len(pts_sub), 1))
                        # 让箭头指向点：起点 = 点 - dir * length
                        length = 0.2 * np.max(np.ptp(points.reshape(-1,3), axis=0))
                        starts = pts_sub - k_sub * length
                        q_inc = ax1.quiver(starts[:, 0], starts[:, 1], starts[:, 2],
                                           k_sub[:, 0], k_sub[:, 1], k_sub[:, 2],
                                           length=length, color='green', pivot='tail', alpha=alpha_i)
                        self.incidence_quivers.append(q_inc)
                except Exception as e:
                    self.log(f"Vector drawing error: {e}")

            # 绘制边并标注局部索引
            n_edges = len(edges_data) if edges_data else 0
            edge_colors = plt.cm.Set1(np.linspace(0, 1, max(n_edges, 1))) if n_edges > 0 else []
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

            # 获取 X 轴数据和标签 (默认 Theta)
            angles_deg = result_data.get('x_values', result_data.get('angles_deg'))
            x_label = result_data.get('x_label', 'Theta (deg)')
            
            rcs = result_data['rcs']
            
            # 检查是否有分量
            rcs_po = result_data.get('rcs_po')
            rcs_ptd = result_data.get('rcs_ptd')

            # 只有当开启了 PTD 且两个分量都有数据时，才显示分量图
            # 如果 PTD 未开启，即使后端返回了全零或空的分量，我们也不显示它们
            has_components = (rcs_po is not None) and (rcs_ptd is not None)

            # 额外检查：如果 PTD 分量全部为 0 (通常意味着未计算 PTD)，则不显示分量图
            if has_components:
                # 检查 PTD 是否有意义的数据（在 dB 域，如果全是 -200 左右或 10*log10(1e-20)，说明没算）
                if np.all(rcs_ptd <= -190):
                    has_components = False

            freq = result_data['freq']
            geo_type = result_data['geo_type']
            geo_params = result_data['geo_params']
            angles_rad = result_data.get('angles_rad') # 仅用于解析解计算
            polarization = result_data.get('polarization', 'VV')

            # 如果有分量，创建 3 个子图；否则 1 个
            figsize = (10, 8) if has_components else (10, 6)
            
            fig = plt.Figure(figsize=figsize, dpi=100, facecolor=self.colors["bg_main"], constrained_layout=True)
            canvas = FigureCanvasTkAgg(fig, master=self.rcs_frame)
            self.rcs_canvas = canvas

            axes = []
            if has_components:
                gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
                ax_total = fig.add_subplot(gs[0])
                ax_po = fig.add_subplot(gs[1], sharex=ax_total)
                ax_ptd = fig.add_subplot(gs[2], sharex=ax_total)
                axes = [ax_total, ax_po, ax_ptd]
                
                # 绘制分量
                ax_po.plot(angles_deg, rcs_po, color='green', linewidth=1.5, label='PO Component')
                ax_po.set_ylabel('RCS (dBsm)')
                ax_po.legend(loc='upper right')
                ax_po.grid(True, linestyle='--', alpha=0.6)
                
                ax_ptd.plot(angles_deg, rcs_ptd, color='orange', linewidth=1.5, label=f'PTD Component ({polarization})')
                ax_ptd.set_ylabel('RCS (dBsm)')
                ax_ptd.set_xlabel(x_label) # 只有最下面显示标签
                ax_ptd.legend(loc='upper right')
                ax_ptd.grid(True, linestyle='--', alpha=0.6)
                
                # 隐藏中间轴标签
                plt.setp(ax_po.get_xticklabels(), visible=False)
                plt.setp(ax_total.get_xticklabels(), visible=False)
                
                main_ax = ax_total
            else:
                main_ax = fig.add_subplot(111)
                main_ax.set_xlabel(x_label, fontsize=11)

            # 绘制 Total (或唯一结果)
            main_ax.plot(angles_deg, rcs, color=self.colors["accent"], linewidth=2, label=f'Total RCS (PO+PTD, {polarization})')

            # 仅当 X 轴为 Theta 时才尝试绘制解析解
            if compare_analytical and geo_params and 'Theta' in x_label and angles_rad is not None:
                rcs_analytical, label_analytical = get_analytical_solution(
                    geo_type, geo_params, freq, angles_rad, polarization=polarization
                )
                
                if rcs_analytical is not None:
                    # 1. 在 Total 图上绘制
                    main_ax.plot(angles_deg, rcs_analytical, 'r--', linewidth=1.5, label=label_analytical)
                    
                    # 2. 如果是 GTD 参考解，且存在 PTD 子图，也在 PTD 子图上绘制以便对比
                    if "GTD" in label_analytical and has_components:
                        ax_ptd.plot(angles_deg, rcs_analytical, 'r--', linewidth=1.0, alpha=0.7, label='Ref GTD (Singular)')
                        # 更新图例以包含新加的线
                        ax_ptd.legend(loc='upper right')

                    # 如果不是 GTD，计算误差统计
                    if "GTD" not in label_analytical:
                        stats = compute_error_stats(rcs, rcs_analytical)
                        error_text = (f"Error Stats:\n"
                                      f"  Max: {stats['max_error']:.2f} dB\n"
                                      f"  Mean: {stats['mean_error']:.2f} dB\n"
                                      f"  RMS: {stats['rms_error']:.2f} dB")
                        main_ax.text(0.02, 0.98, error_text, transform=main_ax.transAxes, fontsize=9, va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                        self.log(f"Analytical Comparison ({polarization}):")
                        self.log(f"  Max Error: {stats['max_error']:.2f} dB")
                        self.log(f"  RMS Error: {stats['rms_error']:.2f} dB")

            main_ax.set_ylabel('RCS (dBsm)', fontsize=11)
            main_ax.set_title(f'Monostatic RCS - {geo_type} @ {freq / 1e6:.1f} MHz', fontsize=12)
            main_ax.grid(True, linestyle='--', alpha=0.6)
            main_ax.legend(loc='upper right')

            # 工具栏
            toolbar_frame = ttk.Frame(self.rcs_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            self.rcs_toolbar = toolbar

            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # fig.tight_layout() # constrained_layout handled it
            canvas.draw()
            self.log(f"1D RCS result displayed: {len(angles_deg)} angles")

        except Exception as e:
            self.log(f"1D RCS Plot Error: {e}")
            import traceback
            traceback.print_exc()

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
            
            rcs_po = result_data.get('rcs_2d_po')
            rcs_ptd = result_data.get('rcs_2d_ptd')

            # 只有当开启了 PTD 且两个分量都有数据时，才显示分量图
            has_components = (rcs_po is not None) and (rcs_ptd is not None)

            # 额外检查：如果 PTD 分量无意义（未计算），则降级为单图模式
            if has_components:
                if np.all(rcs_ptd <= -190):
                    has_components = False

            freq = result_data['freq']
            geo_type = result_data['geo_type']

            # 如果有分量，显示 3 个图；否则 1 个
            figsize = (14, 6) if has_components else (10, 7)

            fig = plt.Figure(figsize=figsize, dpi=100, facecolor=self.colors["bg_main"], constrained_layout=True)
            canvas = FigureCanvasTkAgg(fig, master=self.rcs_frame)
            self.rcs_canvas = canvas

            axes = []
            if has_components:
                gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
                ax_total = fig.add_subplot(gs[0])
                ax_po = fig.add_subplot(gs[1])
                ax_ptd = fig.add_subplot(gs[2])
                axes = [(ax_total, rcs_2d, "Total RCS"), (ax_po, rcs_po, "PO Component"), (ax_ptd, rcs_ptd, "PTD Component")]
            else:
                ax = fig.add_subplot(111)
                axes = [(ax, rcs_2d, f"2D RCS - {geo_type}")]

            extent = [phi_deg.min(), phi_deg.max(), theta_deg.max(), theta_deg.min()]
            Theta, Phi = np.meshgrid(theta_deg, phi_deg, indexing='ij')

            for ax, data, title in axes:
                d_min, d_max = np.nanmin(data), np.nanmax(data)
                if np.isclose(d_min, d_max):
                    d_max = d_min + 1.0 # Ensure range for constant data
                
                # Check shape for contourf suitability
                if data.shape[0] < 2 or data.shape[1] < 2:
                    current_style = 'pixel'
                else:
                    current_style = style

                if current_style == 'pixel':
                    im = ax.imshow(data, extent=extent, aspect='equal', origin='upper', cmap='jet')
                else:
                    levels = np.linspace(d_min, d_max, 50)
                    im = ax.contourf(Phi, Theta, data, levels=levels, cmap='jet')
                    ax.contour(Phi, Theta, data, levels=15, colors='k', linewidths=0.3, alpha=0.3)
                    ax.invert_yaxis()
                    ax.set_aspect('equal')

                cbar = fig.colorbar(im, ax=ax, shrink=0.7, location='bottom', pad=0.1)
                cbar.set_label('dBsm')
                
                ax.set_title(f"{title}\nMax: {d_max:.1f} dBsm", fontsize=10)
                ax.set_xlabel('Phi')
                ax.set_ylabel('Theta')

            fig.suptitle(f'Monostatic RCS @ {freq / 1e6:.1f} MHz', fontsize=12)

            # 工具栏
            toolbar_frame = ttk.Frame(self.rcs_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            self.rcs_toolbar = toolbar

            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            canvas.draw()
            self.log(f"2D RCS displayed ({style})")

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
            titles = ["Calculated RCS", "Reference RCS", f"Difference (Calc - Ref)\nRMSE={metrics['rmse']:.2f}, Mean={metrics['mean_error']:.2f}"]
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
                        im = ax.imshow(data, extent=extent, aspect='equal', origin='upper', 
                                     cmap='jet', vmin=vmin, vmax=vmax)
                    else:
                        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
                        im = ax.contourf(Phi, Theta, data, levels=50, cmap='jet', vmin=vmin, vmax=vmax)
                        ax.invert_yaxis()
                        ax.set_aspect('equal')
                    
                    # 只在第二个图(参考值)右侧加 colorbar，代表前两个图的标尺
                    if i == 1: 
                         cbar = fig.colorbar(im, ax=ax, shrink=0.9, aspect=20)
                         cbar.set_label('RCS (dBsm)')
                
                else: # 误差图
                    if style == 'pixel':
                        im = ax.imshow(data, extent=extent, aspect='equal', origin='upper', 
                                     cmap='seismic', vmin=diff_vmin, vmax=diff_vmax)
                    else:
                        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
                        im = ax.contourf(Phi, Theta, data, levels=50, cmap='seismic', vmin=diff_vmin, vmax=diff_vmax)
                        ax.invert_yaxis()
                        ax.set_aspect('equal')
                    
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

    def show_comparison_dual_2d(self, rcs_a, rcs_b, rcs_ref, diff_a, diff_b, theta, phi, metrics_a, metrics_b, style='pixel'):
        """显示双数据对比结果 (A vs B vs Ref)"""
        try:
            # 清除旧内容并切换到对比 Tab
            self._clear_frame(self.compare_frame)
            self._switch_to_compare_tab()
            
            if not self.compare_frame:
                self.log("Error: No Comparison frame available")
                return

            target_frame = self.compare_frame

            # 使用 constrained_layout
            fig = plt.Figure(figsize=(14, 8), dpi=100, facecolor=self.colors["bg_main"], constrained_layout=True)
            canvas = FigureCanvasTkAgg(fig, master=target_frame)
            self.compare_canvas = canvas
            
            # 创建 2行3列 的子图
            # Row 1: Data A, Data B, Reference
            # Row 2: Diff A-Ref, Diff B-Ref, Diff A-B
            gs = fig.add_gridspec(2, 3)
            
            ax_a = fig.add_subplot(gs[0, 0])
            ax_b = fig.add_subplot(gs[0, 1])
            ax_ref = fig.add_subplot(gs[0, 2])
            
            ax_diff_a = fig.add_subplot(gs[1, 0])
            ax_diff_b = fig.add_subplot(gs[1, 1])
            ax_diff_ab = fig.add_subplot(gs[1, 2])
            
            axes_row1 = [ax_a, ax_b, ax_ref]
            axes_row2 = [ax_diff_a, ax_diff_b, ax_diff_ab]
            
            titles_row1 = ["Data A (Primary)", "Data B (Compare)", "Reference"]
            datas_row1 = [rcs_a, rcs_b, rcs_ref]
            
            # 计算 A-B 差异
            diff_ab = rcs_a - rcs_b
            rmse_ab = np.sqrt(np.nanmean(diff_ab**2))
            mean_ab = np.nanmean(diff_ab)
            
            # 更新标题包含 Mean Error
            titles_row2 = [f"Error A (A-Ref)\nRMSE={metrics_a['rmse']:.2f}, Mean={metrics_a['mean_error']:.2f}", 
                           f"Error B (B-Ref)\nRMSE={metrics_b['rmse']:.2f}, Mean={metrics_b['mean_error']:.2f}", 
                           f"Diff A-B\nRMSE={rmse_ab:.2f}, Mean={mean_ab:.2f}"]
            datas_row2 = [diff_a, diff_b, diff_ab]

            # 统一 row1 的颜色范围 (RCS)
            vmin = min(np.nanmin(rcs_a), np.nanmin(rcs_b), np.nanmin(rcs_ref))
            vmax = max(np.nanmax(rcs_a), np.nanmax(rcs_b), np.nanmax(rcs_ref))
            
            # 统一 row2 的颜色范围 (Error)
            err_max = max(np.nanmax(np.abs(diff_a)), np.nanmax(np.abs(diff_b)), np.nanmax(np.abs(diff_ab)))
            diff_vmin, diff_vmax = -err_max, err_max

            extent = [phi.min(), phi.max(), theta.max(), theta.min()]
            Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

            # 绘制 Row 1 (RCS Values)
            for i, ax in enumerate(axes_row1):
                data = datas_row1[i]
                if style == 'pixel':
                    im = ax.imshow(data, extent=extent, aspect='equal', origin='upper', 
                                 cmap='jet', vmin=vmin, vmax=vmax)
                else:
                    im = ax.contourf(Phi, Theta, data, levels=50, cmap='jet', vmin=vmin, vmax=vmax)
                    ax.invert_yaxis()
                    ax.set_aspect('equal') # 强制正方形

                ax.set_title(titles_row1[i], fontsize=10)
                if i == 0: ax.set_ylabel('Theta')
                # 只在最后一个图右侧加 colorbar
                if i == 2:
                    cbar = fig.colorbar(im, ax=axes_row1, shrink=0.9, aspect=30, location='right')
                    cbar.set_label('RCS (dBsm)')

            # 绘制 Row 2 (Differences)
            for i, ax in enumerate(axes_row2):
                data = datas_row2[i]
                if style == 'pixel':
                    im = ax.imshow(data, extent=extent, aspect='equal', origin='upper', 
                                 cmap='seismic', vmin=diff_vmin, vmax=diff_vmax)
                else:
                    im = ax.contourf(Phi, Theta, data, levels=50, cmap='seismic', vmin=diff_vmin, vmax=diff_vmax)
                    ax.invert_yaxis()
                    ax.set_aspect('equal') # 强制正方形
                
                ax.set_title(titles_row2[i], fontsize=10)
                ax.set_xlabel('Phi')
                if i == 0: ax.set_ylabel('Theta')
                # 只在最后一个图右侧加 colorbar
                if i == 2:
                    cbar = fig.colorbar(im, ax=axes_row2, shrink=0.9, aspect=30, location='right')
                    cbar.set_label('Difference (dB)')

            fig.suptitle(f"Dual Model Comparison", fontsize=14)

            # 工具栏
            toolbar_frame = ttk.Frame(target_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            self.compare_toolbar = toolbar
            
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            canvas.draw()
            self.log(f"Dual comparison displayed.")

        except Exception as e:
            self.log(f"Comparison Plot Error: {e}")
            import traceback
            traceback.print_exc()
