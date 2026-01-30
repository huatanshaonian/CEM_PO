import numpy as np
import sys

from core.env import HAS_GPU, cp
from core.mesh_data import CachedMeshData, MergedMeshData, detect_degenerate_edge

class DiscretePOIntegrator:
    """
    离散物理光学 (PO) 积分器，支持多种 sinc 校正模式
    """

    def __init__(self, nu=None, nv=None, samples_per_lambda=10, sinc_mode='dual', min_points=18):
        self.nu_manual = nu
        self.nv_manual = nv
        self.default_samples_per_lambda = samples_per_lambda
        self.min_points = min_points
        if sinc_mode not in ('none', 'u_only', 'dual'):
            raise ValueError(f"sinc_mode 必须是 'none', 'u_only' 或 'dual'，收到: {sinc_mode}")
        self.sinc_mode = sinc_mode

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

    def integrate_cached(self, cached_data, wave):
        """
        使用预计算的 CachedMeshData 或 MergedMeshData 进行积分。
        """
        # --- 分支 1: 处理合并的大批次网格 (GPU 极速模式) ---
        if isinstance(cached_data, MergedMeshData):
            xp = cp.get_array_module(cached_data.points) if HAS_GPU else np
            
            points = cached_data.points
            normals = cached_data.normals
            weights = cached_data.weights # 包含了 jac * du * dv
            
            k_vec = wave.k_vector
            k_dir = wave.k_dir

            # 确保波矢量在正确的设备上
            if xp is not np:
                k_vec_xp = cp.asarray(k_vec)
                k_dir_xp = cp.asarray(k_dir)
            else:
                k_vec_xp = k_vec
                k_dir_xp = k_dir

            ref_point = points[0]
            phase_ref = 2.0 * xp.sum(ref_point * k_vec_xp)
            phase_local = 2.0 * xp.sum((points - ref_point) * k_vec_xp, axis=-1)

            # 照射检测
            n_dot_k = xp.sum(normals * k_dir_xp, axis=-1)
            illumination_factor = xp.maximum(-n_dot_k, 0.0)

            # Sinc 校正
            sinc_factor = 1.0
            if self.sinc_mode != 'none' and cached_data.has_derivs:
                dP_du = cached_data.dP_du
                du = cached_data.du
                alpha = 2.0 * xp.sum(dP_du * k_vec_xp, axis=-1)
                
                if self.sinc_mode == 'u_only':
                    sinc_factor = xp.sinc(alpha * du / (2.0 * xp.pi))
                else: # dual
                    dP_dv = cached_data.dP_dv
                    dv = cached_data.dv
                    beta = 2.0 * xp.sum(dP_dv * k_vec_xp, axis=-1)
                    sinc_u = xp.sinc(alpha * du / (2.0 * xp.pi))
                    sinc_v = xp.sinc(beta * dv / (2.0 * xp.pi))
                    sinc_factor = sinc_u * sinc_v

            lit_indices = illumination_factor > 1e-6
            
            if xp.any(lit_indices):
                # 只计算亮区
                lit_illum = illumination_factor[lit_indices]
                lit_weights = weights[lit_indices]
                lit_phase = phase_local[lit_indices]
                
                if isinstance(sinc_factor, float):
                    lit_sinc = sinc_factor
                else:
                    lit_sinc = sinc_factor[lit_indices]

                contributions = (lit_illum * lit_weights * 
                                xp.exp(1j * lit_phase) * 
                                lit_sinc)
                
                result = xp.sum(contributions) * xp.exp(1j * phase_ref)
            else:
                result = 0.0 + 0.0j

            if hasattr(result, 'get'):
                return complex(result.get())
            return complex(result)

        # --- 分支 2: 原始逐个曲面处理逻辑 ---
        points = cached_data.points
        normals = cached_data.normals
        jacobians = cached_data.jacobians
        dP_du = cached_data.dP_du
        dP_dv = cached_data.dP_dv
        du = cached_data.du
        dv = cached_data.dv

        # 自动选择 numpy 或 cupy
        xp = np
        if HAS_GPU:
            xp = cp.get_array_module(points)

        k_vec = wave.k_vector
        k_dir = wave.k_dir

        # 判断是否为退化网格（一维数组）
        is_degenerate = (points.ndim == 2)  # (N, 3) vs (nv, nu, 3)

        # 相位稳定化
        if is_degenerate:
            ref_point = points[len(points) // 2]
        else:
            nv, nu = points.shape[:2]
            ref_point = points[nv // 2, nu // 2]

        if xp is not np:
            k_vec_xp = cp.asarray(k_vec)
            k_dir_xp = cp.asarray(k_dir)
        else:
            k_vec_xp = k_vec
            k_dir_xp = k_dir

        phase_ref = 2.0 * xp.sum(ref_point * k_vec_xp)
        phase_local = 2.0 * xp.sum((points - ref_point) * k_vec_xp, axis=-1)

        # 照射检测
        n_dot_k = xp.sum(normals * k_dir_xp, axis=-1)
        lit_mask = n_dot_k < 0
        illumination_factor = -n_dot_k

        # sinc 校正步长：优先使用逐单元步长，其次平均步长
        if is_degenerate and hasattr(cached_data, 'sinc_du'):
            # 逐单元 sinc 步长（精度更高）
            sinc_du = cached_data.sinc_du
            sinc_dv = cached_data.sinc_dv
        elif is_degenerate and hasattr(cached_data, 'avg_du'):
            # 兼容旧版：平均步长
            sinc_du = cached_data.avg_du
            sinc_dv = cached_data.avg_dv
        else:
            sinc_du = du if np.isscalar(du) else np.mean(du)
            sinc_dv = dv if np.isscalar(dv) else np.mean(dv)

        if self.sinc_mode == 'none':
            sinc_factor = 1.0
        elif self.sinc_mode == 'u_only':
            alpha = 2.0 * xp.sum(dP_du * k_vec_xp, axis=-1)
            sinc_factor = xp.sinc(alpha * sinc_du / (2.0 * xp.pi))
        else:  # 'dual'
            alpha = 2.0 * xp.sum(dP_du * k_vec_xp, axis=-1)
            beta = 2.0 * xp.sum(dP_dv * k_vec_xp, axis=-1)
            sinc_u = xp.sinc(alpha * sinc_du / (2.0 * xp.pi))
            sinc_v = xp.sinc(beta * sinc_dv / (2.0 * xp.pi))
            sinc_factor = sinc_u * sinc_v

        # du * dv 可能是标量或数组，numpy 会自动广播
        contributions = (illumination_factor * jacobians *
                        xp.exp(1j * phase_local) *
                        sinc_factor *
                        du * dv)

        result = xp.sum(contributions[lit_mask]) * xp.exp(1j * phase_ref)

        if hasattr(result, 'get'):
            return complex(result.get())
        return complex(result)

    def integrate_surface(self, surface, wave, samples_per_lambda=None):
        """(旧接口) 直接计算，不缓存"""
        mesh_data = self.precompute_mesh(surface, wave.wavelength, samples_per_lambda)
        return self.integrate_cached(mesh_data, wave)

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
