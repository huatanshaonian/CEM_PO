import numpy as np
import sys

from core.env import HAS_GPU, cp
from core.mesh_data import CachedMeshData, MergedMeshData, detect_degenerate_edge
from .po_kernel import po_integrate

class DiscretePOIntegrator:
    """
    离散物理光学 (PO) 积分器，支持多种 sinc 校正模式
    """

    def __init__(self, nu=None, nv=None, samples_per_lambda=10, sinc_mode='dual',
                 min_points=18, precision='double'):
        self.nu_manual = nu
        self.nv_manual = nv
        self.default_samples_per_lambda = samples_per_lambda
        self.min_points = min_points
        if sinc_mode not in ('none', 'u_only', 'dual'):
            raise ValueError(f"sinc_mode 必须是 'none', 'u_only' 或 'dual'，收到: {sinc_mode}")
        self.sinc_mode = sinc_mode
        if precision not in ('double', 'mixed', 'single'):
            raise ValueError(f"precision 必须是 'double'/'mixed'/'single'，收到: {precision}")
        self.precision = precision

    def _estimate_mesh_density(self, surface, wavelength, samples_per_lambda):
        if self.nu_manual is not None and self.nv_manual is not None:
            return self.nu_manual, self.nv_manual
            
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain
        
        u_samples = np.linspace(u_min, u_max, 10)
        v_samples = np.linspace(v_min, v_max, 10)
        
        u_mid = (u_min + u_max) / 2
        p_v = surface.evaluate(u_mid, v_samples)
        dist_v = np.sum(np.sqrt(np.sum(np.diff(p_v, axis=0)**2, axis=-1)))
        
        v_mid = (v_min + v_max) / 2
        p_u = surface.evaluate(u_samples, v_mid)
        dist_u = np.sum(np.sqrt(np.sum(np.diff(p_u, axis=0)**2, axis=-1)))

        # 3. 根据波长和采样率计算网格数
        nv = int(max(self.min_points, (dist_v / wavelength) * samples_per_lambda))
        nu = int(max(self.min_points, (dist_u / wavelength) * samples_per_lambda))
        
        return nu, nv

    def precompute_mesh(self, surface, wavelength, samples_per_lambda=None, use_degenerate_mesh=False):
        """
        预计算网格数据，返回 CachedMeshData 对象。

        参数：
            use_degenerate_mesh: 是否对退化面使用条带状网格（自动检测退化边）
        """
        spl = samples_per_lambda if samples_per_lambda is not None else self.default_samples_per_lambda
        nu, nv = self._estimate_mesh_density(surface, wavelength, spl)

        # 检测退化边
        degen_edge = None
        if use_degenerate_mesh:
            degen_edge = detect_degenerate_edge(surface)

        # 退化面：使用条带状网格
        if degen_edge is not None and degen_edge != 'degenerate':
            return self._precompute_degenerate_mesh(surface, degen_edge, nu, nv)

        # 普通面：使用规则矩形网格
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        du = (u_max - u_min) / nu
        dv = (v_max - v_min) / nv

        u_centers = np.linspace(u_min + du/2, u_max - du/2, nu)
        v_centers = np.linspace(v_min + dv/2, v_max - dv/2, nv)

        uu, vv = np.meshgrid(u_centers, v_centers)

        data = surface.get_data(uu, vv)

        direct_derivatives = False
        if len(data) == 5:
            # 标记使用了优化路径
            direct_derivatives = True
            points, normals, jacobians, dP_du, dP_dv = data
        else:
            points, normals, jacobians = data

            # 2. 回退方案：使用有限差分计算 dP/du (用于 alpha 计算)
            eps_u = du * 1e-4
            p_plus_u = surface.evaluate(uu + eps_u, vv)
            p_minus_u = surface.evaluate(uu - eps_u, vv)
            dP_du = (p_plus_u - p_minus_u) / (2 * eps_u)

            # 3. 回退方案：使用有限差分计算 dP/dv (用于 beta 计算)
            eps_v = dv * 1e-4
            p_plus_v = surface.evaluate(uu, vv + eps_v)
            p_minus_v = surface.evaluate(uu, vv - eps_v)
            dP_dv = (p_plus_v - p_minus_v) / (2 * eps_v)

        res = CachedMeshData(points, normals, jacobians, dP_du, dP_dv, du, dv)
        res.direct_derivatives = direct_derivatives
        return res

    def _precompute_degenerate_mesh(self, surface, degen_edge, a, b):
        """
        预计算退化面的网格数据（条带状网格）。
        使用与预览相同的网格拓扑。
        """
        # 获取单元中心点和面积
        uu, vv, cell_areas, layer_sizes = self.get_degenerate_mesh_direct(surface, degen_edge, a, b)

        if len(uu) == 0:
            # 回退到规则网格
            return None

        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        # 获取几何数据
        data = surface.get_data(uu, vv)

        direct_derivatives = False
        if len(data) == 5:
            direct_derivatives = True
            points, normals, jacobians, dP_du, dP_dv = data
        else:
            points, normals, jacobians = data

            # 有限差分计算导数
            avg_du = (u_max - u_min) / a
            avg_dv = (v_max - v_min) / b
            eps_u = avg_du * 1e-4
            eps_v = avg_dv * 1e-4

            p_plus_u = surface.evaluate(uu + eps_u, vv)
            p_minus_u = surface.evaluate(uu - eps_u, vv)
            dP_du = (p_plus_u - p_minus_u) / (2 * eps_u)

            p_plus_v = surface.evaluate(uu, vv + eps_v)
            p_minus_v = surface.evaluate(uu, vv - eps_v)
            dP_dv = (p_plus_v - p_minus_v) / (2 * eps_v)

        # 使用 cell_areas 作为 du，dv=1，这样 du*dv = cell_areas
        # sinc 校正使用逐单元的真实步长
        avg_du = (u_max - u_min) / a
        avg_dv = (v_max - v_min) / b

        # 计算逐单元的 sinc 校正步长
        sinc_du_list = []
        sinc_dv_list = []
        for layer_idx, n_cells in enumerate(layer_sizes):
            if degen_edge in ['u_min', 'u_max']:
                # u 方向退化：u 步长固定，v 步长随层变化
                n_segments = max(1, b - layer_idx)
                du_layer = avg_du
                dv_layer = (v_max - v_min) / n_segments
            elif degen_edge in ['v_min', 'v_max']:
                # v 方向退化：v 步长固定，u 步长随层变化
                n_segments = max(1, b - layer_idx)
                du_layer = (u_max - u_min) / n_segments
                dv_layer = avg_dv
            else:  # u_both, v_both
                # 双边退化：使用平均值（简化处理）
                du_layer = avg_du
                dv_layer = avg_dv
            for _ in range(n_cells):
                sinc_du_list.append(du_layer)
                sinc_dv_list.append(dv_layer)

        res = CachedMeshData(points, normals, jacobians, dP_du, dP_dv,
                             cell_areas, np.ones_like(cell_areas))
        res.direct_derivatives = direct_derivatives
        res.sinc_du = np.array(sinc_du_list)  # 逐单元 sinc 校正步长
        res.sinc_dv = np.array(sinc_dv_list)
        res.avg_du = avg_du  # 保留用于兼容
        res.avg_dv = avg_dv
        res.layer_sizes = layer_sizes  # 用于可视化
        return res

    def integrate_cached(self, cached_data, wave, precision=None):
        """统一 PO 积分入口, 调 po_kernel.po_integrate.

        参数:
            cached_data: 单个 CachedMeshData 或 list[CachedMeshData]
                         (传 MergedMeshData 会 raise TypeError, 该类已不再生产)
            wave:        IncidentWave 实例
            precision:   None → 用 self.precision; 否则覆盖

        返回:
            complex —— 该 mesh(列表) 在此入射方向下的 PO 散射积分.
        """
        if isinstance(cached_data, MergedMeshData):
            raise TypeError(
                "MergedMeshData 已不再被 PO kernel 接受; "
                "请直接传 list[CachedMeshData] (上层 SolverBridge 已停止合并). "
                "如果还需要合并优化, 在 po_kernel 内实现 segmented reduction.")

        prec = precision if precision is not None else self.precision
        mesh_list = cached_data if isinstance(cached_data, list) else [cached_data]
        k_mags = np.array([2.0 * np.pi / wave.wavelength])
        I = po_integrate(mesh_list, wave.k_dir, k_mags,
                         sinc_mode=self.sinc_mode, precision=prec)
        return complex(I[0])

    def integrate_surface(self, surface, wave, samples_per_lambda=None,
                          precision=None):
        """(旧接口) 直接计算, 不缓存. 现在统一走 kernel."""
        mesh_data = self.precompute_mesh(surface, wave.wavelength, samples_per_lambda)
        return self.integrate_cached(mesh_data, wave, precision=precision)

    def get_mesh_data(self, surface, wave, samples_per_lambda=None):
        spl = samples_per_lambda if samples_per_lambda is not None else self.default_samples_per_lambda
        nu, nv = self._estimate_mesh_density(surface, wave.wavelength, spl)
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain
        du = (u_max - u_min) / nu
        dv = (v_max - v_min) / nv
        u_centers = np.linspace(u_min + du/2, u_max - du/2, nu)
        v_centers = np.linspace(v_min + dv/2, v_max - dv/2, nv)
        uu, vv = np.meshgrid(u_centers, v_centers)
        data = surface.get_data(uu, vv)
        if len(data) == 5:
            points, normals, jacobians, _, _ = data
        else:
            points, normals, jacobians = data
        return points, normals, (nu, nv)

    def get_mesh_size(self, surface, wave, samples_per_lambda=None):
        spl = samples_per_lambda if samples_per_lambda is not None else self.default_samples_per_lambda
        nu, nv = self._estimate_mesh_density(surface, wave.wavelength, spl)
        return nu, nv

    def get_triangle_mesh_cells(self, surface, degen_edge=None, preview_a=15, preview_b=15):
        if degen_edge is None:
            degen_edge = detect_degenerate_edge(surface)
        if degen_edge is None or degen_edge == 'degenerate':
            return [], 0, 0
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain
        a, b = preview_a, preview_b
        mesh_cells = []

        def get_layer_nodes(layer_idx, total_layers, n_subdivs_base, type_edge):
            nodes = []
            n_segs = max(1, n_subdivs_base - layer_idx)
            r_ratio = layer_idx / total_layers
            if type_edge == 'u_min':
                u_curr = u_max - r_ratio * (u_max - u_min)
                for k in range(n_segs + 1):
                    v_curr = v_min + (k / n_segs) * (v_max - v_min)
                    nodes.append((u_curr, v_curr))
            elif type_edge == 'u_max':
                u_curr = u_min + r_ratio * (u_max - u_min)
                for k in range(n_segs + 1):
                    v_curr = v_min + (k / n_segs) * (v_max - v_min)
                    nodes.append((u_curr, v_curr))
            elif type_edge == 'v_min':
                v_curr = v_max - r_ratio * (v_max - v_min)
                for k in range(n_segs + 1):
                    u_curr = u_min + (k / n_segs) * (u_max - u_min)
                    nodes.append((u_curr, v_curr))
            elif type_edge == 'v_max':
                v_curr = v_min + r_ratio * (v_max - v_min)
                for k in range(n_segs + 1):
                    u_curr = u_min + (k / n_segs) * (u_max - u_min)
                    nodes.append((u_curr, v_curr))
            return nodes

        layers_nodes = []
        if degen_edge in ['u_both', 'v_both']:
            mid_layer = a // 2
            for i in range(a + 1):
                dist = abs(i - mid_layer)
                n_sub = max(1, int(b * (1 - dist / (mid_layer + 1))))
                nodes = []
                n_segs = n_sub
                r_ratio = i / a
                if degen_edge == 'u_both':
                    u_curr = u_min + r_ratio * (u_max - u_min)
                    for k in range(n_segs + 1):
                        v_curr = v_min + (k / n_segs) * (v_max - v_min)
                        nodes.append((u_curr, v_curr))
                elif degen_edge == 'v_both':
                    v_curr = v_min + r_ratio * (v_max - v_min)
                    for k in range(n_segs + 1):
                        u_curr = u_min + (k / n_segs) * (u_max - u_min)
                        nodes.append((u_curr, v_curr))
                layers_nodes.append(nodes)
        else:
            for i in range(a + 1):
                layers_nodes.append(get_layer_nodes(i, a, b, degen_edge))

        for i in range(a):
            current_nodes = layers_nodes[i]
            next_nodes = layers_nodes[i + 1]
            n_curr = len(current_nodes) - 1
            n_next = len(next_nodes) - 1
            if n_next < n_curr:
                tri_corners = [current_nodes[0], current_nodes[1], next_nodes[0]]
                mesh_cells.append(tri_corners)
                for k in range(1, n_curr):
                    corners = [current_nodes[k], current_nodes[k + 1],
                               next_nodes[min(k, n_next)], next_nodes[max(0, k - 1)]]
                    mesh_cells.append(corners)
            elif n_next > n_curr:
                tri_corners = [current_nodes[0], next_nodes[1], next_nodes[0]]
                mesh_cells.append(tri_corners)
                for k in range(1, n_next):
                    corners = [current_nodes[max(0, k - 1)], current_nodes[min(k, n_curr)],
                               next_nodes[k + 1], next_nodes[k]]
                    mesh_cells.append(corners)
            else:
                for k in range(n_curr):
                    corners = [current_nodes[k], current_nodes[k + 1],
                               next_nodes[k + 1], next_nodes[k]]
                    mesh_cells.append(corners)
        return mesh_cells, a, b

    def get_degenerate_mesh_direct(self, surface, degen_edge, a, b):
        """
        基于 get_triangle_mesh_cells 的拓扑，计算每个单元的中心点和面积。
        这保证了计算网格和预览网格拓扑一致。

        返回：
            u_centers: (N,) 每个单元中心的 u 坐标
            v_centers: (N,) 每个单元中心的 v 坐标
            cell_areas: (N,) 每个单元在参数空间的面积
            layer_sizes: list, 每层的单元数（用于可视化时连接中心点）
        """
        # 获取与预览相同的单元拓扑
        mesh_cells, _, _ = self.get_triangle_mesh_cells(surface, degen_edge, a, b)

        if not mesh_cells:
            return np.array([]), np.array([]), np.array([]), []

        u_centers_list = []
        v_centers_list = []
        cell_areas_list = []

        for cell in mesh_cells:
            n = len(cell)
            # 计算重心
            u_c = sum(c[0] for c in cell) / n
            v_c = sum(c[1] for c in cell) / n

            # 计算参数空间面积
            if n == 3:
                # 三角形面积：0.5 * |cross product|
                area = 0.5 * abs(
                    (cell[1][0] - cell[0][0]) * (cell[2][1] - cell[0][1]) -
                    (cell[2][0] - cell[0][0]) * (cell[1][1] - cell[0][1])
                )
            else:
                # 四边形面积：Shoelace 公式
                area = 0.5 * abs(
                    (cell[0][0] * cell[1][1] - cell[1][0] * cell[0][1]) +
                    (cell[1][0] * cell[2][1] - cell[2][0] * cell[1][1]) +
                    (cell[2][0] * cell[3][1] - cell[3][0] * cell[2][1]) +
                    (cell[3][0] * cell[0][1] - cell[0][0] * cell[3][1])
                )

            u_centers_list.append(u_c)
            v_centers_list.append(v_c)
            cell_areas_list.append(area)

        # 计算每层的单元数（用于可视化）
        # 基于 get_triangle_mesh_cells 的层结构
        layer_sizes = []
        if degen_edge in ['u_min', 'u_max', 'v_min', 'v_max']:
            for i in range(a):
                n_curr = max(1, b - i)
                n_next = max(1, b - i - 1)
                # 层间单元数 = 三角形 + 四边形
                if n_next < n_curr:
                    layer_sizes.append(1 + (n_curr - 1))  # 1个三角形 + (n_curr-1)个四边形
                elif n_next > n_curr:
                    layer_sizes.append(1 + (n_next - 1))
                else:
                    layer_sizes.append(n_curr)
        elif degen_edge in ['u_both', 'v_both']:
            mid = a // 2
            for i in range(a):
                dist_curr = abs(i - mid)
                dist_next = abs(i + 1 - mid)
                n_curr = max(1, int(b * (1 - dist_curr / (mid + 1))))
                n_next = max(1, int(b * (1 - dist_next / (mid + 1))))
                if n_next < n_curr:
                    layer_sizes.append(1 + (n_curr - 1))
                elif n_next > n_curr:
                    layer_sizes.append(1 + (n_next - 1))
                else:
                    layer_sizes.append(n_curr)

        return (np.array(u_centers_list), np.array(v_centers_list),
                np.array(cell_areas_list), layer_sizes)
