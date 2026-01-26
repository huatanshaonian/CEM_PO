import numpy as np
from physics.constants import ETA0, C0


def detect_degenerate_edge(surface, threshold_ratio=0.01):
    """
    检测曲面的退化边（三角形面或纺锤形面）
    """
    u_min, u_max = surface.u_domain
    v_min, v_max = surface.v_domain

    # 检查四个角的 Jacobian
    corners = [
        (u_min, v_min),  # (0,0)
        (u_max, v_min),  # (1,0)
        (u_min, v_max),  # (0,1)
        (u_max, v_max),  # (1,1)
    ]

    jac_values = []
    for u, v in corners:
        _, _, jac = surface.get_data(np.array([[u]]), np.array([[v]]))
        jac_values.append(jac[0, 0])

    max_jac = max(jac_values)
    if max_jac < 1e-10:
        return 'degenerate'

    threshold = max_jac * threshold_ratio
    is_degenerate = [j < threshold for j in jac_values]

    # 判断哪条边退化
    u_min_deg = is_degenerate[0] and is_degenerate[2]  # (0,0)-(0,1)
    u_max_deg = is_degenerate[1] and is_degenerate[3]  # (1,0)-(1,1)
    v_min_deg = is_degenerate[0] and is_degenerate[1]  # (0,0)-(1,0)
    v_max_deg = is_degenerate[2] and is_degenerate[3]  # (0,1)-(1,1)

    if u_min_deg and u_max_deg: return 'u_both'
    if v_min_deg and v_max_deg: return 'v_both'
    if u_min_deg: return 'u_min'
    if u_max_deg: return 'u_max'
    if v_min_deg: return 'v_min'
    if v_max_deg: return 'v_max'

    return None

class CachedMeshData:
    """
    预计算的网格数据，用于加速计算和并行传输。
    包含积分所需的所有几何信息（纯 NumPy 数组）。
    """
    def __init__(self, points, normals, jacobians, dP_du, dP_dv, du, dv):
        self.points = points
        self.normals = normals
        self.jacobians = jacobians
        self.dP_du = dP_du  # Partial derivative dP/du for alpha calc
        self.dP_dv = dP_dv  # Partial derivative dP/dv for beta calc
        self.du = du
        self.dv = dv

class DiscretePOIntegrator:
    """
    离散物理光学 (PO) 积分器，支持多种 sinc 校正模式
    """

    def __init__(self, nu=None, nv=None, samples_per_lambda=10, sinc_mode='dual'):
        self.nu_manual = nu
        self.nv_manual = nv
        self.default_samples_per_lambda = samples_per_lambda
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

        nv = int(max(20, (dist_v / wavelength) * samples_per_lambda))
        nu = int(max(20, (dist_u / wavelength) * samples_per_lambda))
        
        return nu, nv

    def precompute_mesh(self, surface, wavelength, samples_per_lambda=None):
        """
        预计算网格数据，返回 CachedMeshData 对象。
        """
        spl = samples_per_lambda if samples_per_lambda is not None else self.default_samples_per_lambda
        nu, nv = self._estimate_mesh_density(surface, wavelength, spl)
        
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain
        
        du = (u_max - u_min) / nu
        dv = (v_max - v_min) / nv
        
        u_centers = np.linspace(u_min + du/2, u_max - du/2, nu)
        v_centers = np.linspace(v_min + dv/2, v_max - dv/2, nv)
        
        uu, vv = np.meshgrid(u_centers, v_centers)
        
        # 1. 基础几何数据
        points, normals, jacobians = surface.get_data(uu, vv)
        
        # 2. 计算 dP/du (用于 alpha 计算)
        eps_u = du * 1e-4
        p_plus_u = surface.evaluate(uu + eps_u, vv)
        p_minus_u = surface.evaluate(uu - eps_u, vv)
        dP_du = (p_plus_u - p_minus_u) / (2 * eps_u)

        # 3. 计算 dP/dv (用于 beta 计算)
        eps_v = dv * 1e-4
        p_plus_v = surface.evaluate(uu, vv + eps_v)
        p_minus_v = surface.evaluate(uu, vv - eps_v)
        dP_dv = (p_plus_v - p_minus_v) / (2 * eps_v)

        return CachedMeshData(points, normals, jacobians, dP_du, dP_dv, du, dv)

    def integrate_cached(self, cached_data, wave):
        """
        使用预计算的 CachedMeshData 进行积分。
        """
        points = cached_data.points
        normals = cached_data.normals
        jacobians = cached_data.jacobians
        dP_du = cached_data.dP_du
        dP_dv = cached_data.dP_dv
        du = cached_data.du
        dv = cached_data.dv

        k_vec = wave.k_vector
        k_dir = wave.k_dir

        # 相位稳定化
        nv, nu = points.shape[:2]
        ref_point = points[nv // 2, nu // 2]
        phase_ref = 2.0 * np.sum(ref_point * k_vec)
        phase_local = 2.0 * np.sum((points - ref_point) * k_vec, axis=-1)

        # 照射检测
        n_dot_k = np.sum(normals * k_dir, axis=-1)
        lit_mask = n_dot_k < 0
        illumination_factor = -n_dot_k

        if self.sinc_mode == 'none':
            sinc_factor = 1.0
        elif self.sinc_mode == 'u_only':
            alpha = 2.0 * np.sum(dP_du * k_vec, axis=-1)
            sinc_factor = np.sinc(alpha * du / (2.0 * np.pi))
        else:  # 'dual'
            alpha = 2.0 * np.sum(dP_du * k_vec, axis=-1)
            beta = 2.0 * np.sum(dP_dv * k_vec, axis=-1)
            sinc_u = np.sinc(alpha * du / (2.0 * np.pi))
            sinc_v = np.sinc(beta * dv / (2.0 * np.pi))
            sinc_factor = sinc_u * sinc_v

        contributions = (illumination_factor * jacobians *
                        np.exp(1j * phase_local) *
                        sinc_factor *
                        du * dv)

        return np.sum(contributions[lit_mask]) * np.exp(1j * phase_ref)

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
        points, normals, jacobians = surface.get_data(uu, vv)
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


class RCSAnalyzer:
    def __init__(self, solver):
        self.solver = solver

    def _compute_single_angle_cached(self, args):
        """
        并行计算任务函数 (使用 CachedMeshData)
        """
        # 兼容性处理：支持 4, 5, 6 参数
        ptd_edges = None
        polarization = 'VV'
        
        if len(args) == 6:
            theta, cached_surfaces, wave_params, k_mag, ptd_edges, polarization = args
        elif len(args) == 5:
            theta, cached_surfaces, wave_params, k_mag, ptd_edges = args
        elif len(args) == 4:
            theta, cached_surfaces, wave_params, k_mag = args
        else:
            raise ValueError(f"Invalid args length: {len(args)}, expected 4, 5 or 6")
        
        # wave_params = {'frequency': f, 'phi': phi}
        wave = self._make_wave(wave_params['frequency'], theta, wave_params['phi'])

        total_I_po = 0j
        for mesh_data in cached_surfaces:
            total_I_po += self.solver.integrate_cached(mesh_data, wave)

        total_I_ptd = 0j
        # 添加 PTD 贡献
        if ptd_edges:
            from physics.ptd_core import compute_ptd_contribution
            for edge in ptd_edges:
                total_I_ptd += compute_ptd_contribution(edge, wave, polarization=polarization)

        total_I = total_I_po + total_I_ptd

        def to_rcs(val):
            sigma = (k_mag**2 / np.pi) * np.abs(val)**2
            return 10.0 * np.log10(max(sigma, 1e-20))

        return {
            'total': to_rcs(total_I),
            'po': to_rcs(total_I_po),
            'ptd': to_rcs(total_I_ptd)
        }

    def _make_wave(self, freq, theta, phi):
        from physics.wave import IncidentWave
        return IncidentWave(freq, theta, phi)

    def compute_monostatic_rcs(self, geometry, wave_params, angles,
                               samples_per_lambda=None,
                               parallel=False, n_workers=None,
                               show_progress=True,
                               progress_callback=None,
                               enable_ptd=False, ptd_edge_identifiers=None,
                               cached_mesh_data=None, polarization='VV'):
        if isinstance(geometry, list):
            surfaces = geometry
        else:
            surfaces = [geometry]

        # 准备 PTD 边缘数据
        ptd_edges = []
        if enable_ptd and ptd_edge_identifiers:
            try:
                from solver.manual_edges import extract_manual_edges
                ptd_edges = extract_manual_edges(surfaces, ptd_edge_identifiers)
                if show_progress:
                    print(f"  [PTD] 已启用，提取了 {len(ptd_edges)} 条手动边缘")
            except Exception as e:
                print(f"  [PTD] 边缘提取失败: {e}")

        k_mag = 2 * np.pi * wave_params['frequency'] / C0
        n_angles = len(angles)

        info_msg = (f"计算参数: {len(surfaces)} 个曲面, {n_angles} 个角度, "
                    f"f={wave_params['frequency']/1e9:.2f}GHz")
        if enable_ptd:
            info_msg += f" (PTD: {len(ptd_edges)} edges, {polarization})"

        if show_progress:
            print(info_msg)
        if progress_callback:
            progress_callback(0, n_angles, info_msg)

        # ---------------------------------------------------------------------
        # 优化: 几何预计算 (如果求解器支持)
        # ---------------------------------------------------------------------
        can_cache = hasattr(self.solver, 'precompute_mesh') and hasattr(self.solver, 'integrate_cached')
        geometry_data = surfaces
        is_cached = False

        try:
            if cached_mesh_data:
                # 外部已提供缓存
                if show_progress: print("  使用外部缓存的网格数据...")
                geometry_data = cached_mesh_data
                is_cached = True
            elif can_cache:
                # 尝试预计算
                wavelength = C0 / wave_params['frequency']
                if show_progress: print("  正在预计算几何网格 (加速模式)...")
                geometry_data = [
                    self.solver.precompute_mesh(s, wavelength, samples_per_lambda) 
                    for s in surfaces
                ]
                is_cached = True
        except Exception as e:
            print(f"  预计算失败: {e}, 回退到实时计算模式")
            can_cache = False
            geometry_data = surfaces
            is_cached = False

        if parallel:
            if not is_cached:
                if show_progress: print("  并行模式仅支持可缓存的求解器 (DiscretePO)，回退到串行模式。")
                # Fall through to serial logic
            else:
                # 只有 is_cached=True 才进入并行
                return self._compute_parallel(
                    geometry_data, wave_params, angles, k_mag, 
                    n_workers, show_progress, progress_callback, ptd_edges, polarization,
                    is_cached=is_cached
                )

        return self._compute_serial(
            geometry_data, wave_params, angles, samples_per_lambda,
            k_mag, show_progress, progress_callback, ptd_edges, polarization,
            is_cached=is_cached
        )

    def _compute_serial(self, geometry_data, wave_params, angles,
                        samples_per_lambda, k_mag, show_progress, progress_callback=None, ptd_edges=None, polarization='VV', is_cached=False):
        rcs_list = {'total': [], 'po': [], 'ptd': []}
        n_angles = len(angles)
        
        # 导入 PTD 核心函数 (如果需要)
        compute_ptd = None
        if ptd_edges:
            from physics.ptd_core import compute_ptd_contribution
            compute_ptd = compute_ptd_contribution

        for i, theta in enumerate(angles):
            wave = self._make_wave(wave_params['frequency'], theta, wave_params['phi'])
            
            total_I_po = 0j
            for obj in geometry_data:
                # 根据是否缓存选择不同的积分接口
                if is_cached:
                    total_I_po += self.solver.integrate_cached(obj, wave)
                else:
                    total_I_po += self.solver.integrate_surface(obj, wave, samples_per_lambda=samples_per_lambda)

            total_I_ptd = 0j
            # 添加 PTD
            if ptd_edges and compute_ptd:
                for edge in ptd_edges:
                    total_I_ptd += compute_ptd(edge, wave, polarization=polarization)
            
            total_I = total_I_po + total_I_ptd

            def to_rcs(val):
                sigma = (k_mag**2 / np.pi) * np.abs(val)**2
                return 10.0 * np.log10(max(sigma, 1e-20))

            rcs_list['total'].append(to_rcs(total_I))
            rcs_list['po'].append(to_rcs(total_I_po))
            rcs_list['ptd'].append(to_rcs(total_I_ptd))

            if (i + 1) % max(1, n_angles // 20) == 0 or (i + 1) == n_angles:
                progress = (i + 1) / n_angles * 100
                msg = f"进度: {progress:.0f}% ({i+1}/{n_angles})"
                if show_progress: print(f"  {msg}")
                if progress_callback: progress_callback(i + 1, n_angles, msg)

        done_msg = "计算完成!"
        if show_progress: print(f"  {done_msg}")
        if progress_callback: progress_callback(n_angles, n_angles, done_msg)

        # Convert lists to numpy arrays
        return {k: np.array(v) for k, v in rcs_list.items()}

    def _compute_parallel(self, cached_surfaces, wave_params, angles,
                          k_mag, n_workers, show_progress,
                          progress_callback=None, ptd_edges=None, polarization='VV', is_cached=True):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os

        if n_workers is None: n_workers = os.cpu_count() or 4

        parallel_msg = f"启用并行计算: {n_workers} 个进程 (Cached Mode)"
        if show_progress: print(f"  {parallel_msg}")
        if progress_callback: progress_callback(0, len(angles), parallel_msg)

        # 参数不再包含 samples_per_lambda，因为网格已固定
        args_list = [
            (theta, cached_surfaces, wave_params, k_mag, ptd_edges, polarization)
            for theta in angles
        ]

        rcs_dict = {}
        n_angles = len(angles)

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(self._compute_single_angle_cached, args): i
                    for i, args in enumerate(args_list)
                }

                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    rcs_dict[idx] = future.result() # Now returns dict {'total':, 'po':, 'ptd':}
                    completed += 1

                    if completed % max(1, n_angles // 20) == 0 or completed == n_angles:
                        progress = completed / n_angles * 100
                        msg = f"进度: {progress:.0f}% ({completed}/{n_angles})"
                        if show_progress: print(f"  {msg}")
                        if progress_callback: progress_callback(completed, n_angles, msg)

            # Reassemble results
            final_rcs = {'total': [], 'po': [], 'ptd': []}
            for i in range(n_angles):
                res = rcs_dict[i]
                final_rcs['total'].append(res['total'])
                final_rcs['po'].append(res['po'])
                final_rcs['ptd'].append(res['ptd'])

            done_msg = "并行计算完成!"
            if show_progress: print(f"  {done_msg}")
            if progress_callback: progress_callback(n_angles, n_angles, done_msg)
            
            return {k: np.array(v) for k, v in final_rcs.items()}

        except Exception as e:
            raise RuntimeError(f"并行计算致命错误: {e}")

    def compute_monostatic_rcs_2d(self, geometry, frequency, theta_array, phi_array,
                                   samples_per_lambda=None,
                                   parallel=False, n_workers=None,
                                   show_progress=True,
                                   progress_callback=None,
                                   enable_ptd=False, ptd_edge_identifiers=None,
                                   cached_mesh_data=None, polarization='VV'):
        if isinstance(geometry, list):
            surfaces = geometry
        else:
            surfaces = [geometry]

        # 准备 PTD 边缘数据
        ptd_edges = []
        if enable_ptd and ptd_edge_identifiers:
            try:
                from solver.manual_edges import extract_manual_edges
                ptd_edges = extract_manual_edges(surfaces, ptd_edge_identifiers)
                if show_progress:
                    print(f"  [PTD] 2D扫描启用，提取了 {len(ptd_edges)} 条手动边缘")
            except Exception as e:
                print(f"  [PTD] 边缘提取失败: {e}")

        k_mag = 2 * np.pi * frequency / C0
        n_theta = len(theta_array)
        n_phi = len(phi_array)
        total_points = n_theta * n_phi

        info_msg = (f"2D扫描: {len(surfaces)} 个曲面, "
                    f"{n_theta}×{n_phi}={total_points} 个角度点, "
                    f"f={frequency/1e9:.2f}GHz")
        if enable_ptd:
            info_msg += f" (PTD: Enabled)"

        if show_progress: print(info_msg)
        if progress_callback: progress_callback(0, total_points, info_msg)

        # ---------------------------------------------------------------------
        # 优化: 几何预计算
        # ---------------------------------------------------------------------
        can_cache = hasattr(self.solver, 'precompute_mesh') and hasattr(self.solver, 'integrate_cached')
        geometry_data = surfaces
        is_cached = False

        try:
            if cached_mesh_data:
                if show_progress: print("  使用外部缓存的网格数据...")
                geometry_data = cached_mesh_data
                is_cached = True
            elif can_cache:
                wavelength = C0 / frequency
                if show_progress: print("  正在预计算2D几何网格 (加速模式)...")
                geometry_data = [
                    self.solver.precompute_mesh(s, wavelength, samples_per_lambda) 
                    for s in surfaces
                ]
                is_cached = True
        except Exception as e:
            print(f"  2D预计算失败: {e}, 回退到实时计算")
            can_cache = False
            geometry_data = surfaces
            is_cached = False

        # ---------------------------------------------------------------------
        # 分发计算
        # ---------------------------------------------------------------------
        if parallel:
            if not is_cached:
                if show_progress: print("  并行模式仅支持可缓存的求解器 (DiscretePO)，回退到串行模式。")
                # Fall through to serial logic
            else:
                # 只有 is_cached=True 才进入并行
                return self._compute_parallel_2d(
                    geometry_data, frequency, theta_array, phi_array,
                    k_mag, n_workers, show_progress, progress_callback, ptd_edges, polarization,
                    is_cached=is_cached
                )

        # 串行逻辑
        # 导入 PTD 核心函数
        compute_ptd = None
        if ptd_edges:
            from physics.ptd_core import compute_ptd_contribution
            compute_ptd = compute_ptd_contribution

        rcs_2d = {
            'total': np.zeros((n_theta, n_phi)),
            'po': np.zeros((n_theta, n_phi)),
            'ptd': np.zeros((n_theta, n_phi))
        }
        
        computed = 0
        for i, theta in enumerate(theta_array):
            for j, phi in enumerate(phi_array):
                wave = self._make_wave(frequency, theta, phi)
                
                total_I_po = 0j
                for obj in geometry_data:
                    if is_cached:
                        total_I_po += self.solver.integrate_cached(obj, wave)
                    else:
                        total_I_po += self.solver.integrate_surface(obj, wave, samples_per_lambda=samples_per_lambda)
                
                total_I_ptd = 0j
                # 添加 PTD
                if ptd_edges and compute_ptd:
                    for edge in ptd_edges:
                        total_I_ptd += compute_ptd(edge, wave, polarization=polarization)

                total_I = total_I_po + total_I_ptd

                def to_rcs(val):
                    sigma = (k_mag**2 / np.pi) * np.abs(val)**2
                    return 10.0 * np.log10(max(sigma, 1e-20))

                rcs_2d['total'][i, j] = to_rcs(total_I)
                rcs_2d['po'][i, j] = to_rcs(total_I_po)
                rcs_2d['ptd'][i, j] = to_rcs(total_I_ptd)

                computed += 1
                if computed % max(1, total_points // 20) == 0 or computed == total_points:
                    progress = computed / total_points * 100
                    msg = f"进度: {progress:.0f}% ({computed}/{total_points})"
                    if show_progress: print(f"  {msg}")
                    if progress_callback: progress_callback(computed, total_points, msg)

        done_msg = "2D扫描完成!"
        if show_progress: print(f"  {done_msg}")
        if progress_callback: progress_callback(total_points, total_points, done_msg)
        return rcs_2d

    def _compute_parallel_2d(self, cached_surfaces, frequency, theta_array, phi_array,
                             k_mag, n_workers,
                             show_progress, progress_callback=None, ptd_edges=None, polarization='VV', is_cached=True):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os

        if n_workers is None: n_workers = os.cpu_count() or 4

        n_theta = len(theta_array)
        n_phi = len(phi_array)
        total_points = n_theta * n_phi

        parallel_msg = f"启用2D并行计算: {n_workers} 个进程 (Cached Mode)"
        if show_progress: print(f"  {parallel_msg}")
        if progress_callback: progress_callback(0, total_points, parallel_msg)

        args_list = []
        for i, theta in enumerate(theta_array):
            for j, phi in enumerate(phi_array):
                wave_params = {'frequency': frequency, 'phi': phi}
                # 复用 _compute_single_angle_cached
                args = (theta, cached_surfaces, wave_params, k_mag, ptd_edges, polarization)
                args_list.append(((i, j), args))

        rcs_2d = {
            'total': np.zeros((n_theta, n_phi)),
            'po': np.zeros((n_theta, n_phi)),
            'ptd': np.zeros((n_theta, n_phi))
        }

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(self._compute_single_angle_cached, args): idx_tuple
                    for idx_tuple, args in args_list
                }

                computed = 0
                for future in as_completed(future_to_idx):
                    i, j = future_to_idx[future]
                    res = future.result() # dict
                    rcs_2d['total'][i, j] = res['total']
                    rcs_2d['po'][i, j] = res['po']
                    rcs_2d['ptd'][i, j] = res['ptd']
                    
                    computed += 1

                    if computed % max(1, total_points // 20) == 0 or computed == total_points:
                        progress = computed / total_points * 100
                        msg = f"进度: {progress:.0f}% ({computed}/{total_points})"
                        if show_progress: print(f"  {msg}")
                        if progress_callback: progress_callback(computed, total_points, msg)

            done_msg = "2D并行计算完成!"
            if show_progress: print(f"  {done_msg}")
            if progress_callback: progress_callback(total_points, total_points, done_msg)
            return rcs_2d

        except Exception as e:
            raise RuntimeError(f"2D并行计算致命错误: {e}")


# =============================================================================
# 真正的 Ribbon 积分器
# =============================================================================

class TrueRibbonIntegrator:
    """
    真正的 Ribbon 积分器：v 方向离散，u 方向自适应 Gauss 积分

    与离散 PO 的区别：
    - 离散 PO: 两个方向都离散为网格点，用矩形法则求和
    - True Ribbon: v 方向离散为 nv 条 ribbon，每条 ribbon 沿 u 方向
                   使用自适应分段 Gauss 积分

    关键改进：
    - 自适应分段：将 u 区间分成多个子区间，确保每个子区间内相位变化 < π
    - 这样 Gauss 积分在每个子区间内都能精确工作

    参数:
        nv: v 方向 ribbon 数量（可选，自动估算）
        n_gauss: 每个子区间的 Gauss 积分点数（默认 8）
        samples_per_lambda: 每波长采样数，用于自动估算 nv
        max_phase_per_segment: 每个子区间最大相位变化（默认 π/2）
    """

    # Gauss-Legendre 节点和权重（预计算）
    _gauss_cache = {}

    def __init__(self, nv=None, n_gauss=8, samples_per_lambda=8, max_phase_per_segment=np.pi/2):
        self.nv_manual = nv
        self.n_gauss = n_gauss
        self.samples_per_lambda = samples_per_lambda
        self.max_phase_per_segment = max_phase_per_segment

        # 预计算 Gauss 节点和权重
        if n_gauss not in self._gauss_cache:
            nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
            self._gauss_cache[n_gauss] = (nodes, weights)
        self.gauss_nodes, self.gauss_weights = self._gauss_cache[n_gauss]

    def _estimate_nv(self, surface, wavelength):
        """估算 v 方向需要的 ribbon 数量"""
        if self.nv_manual is not None:
            return self.nv_manual

        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        # 沿 v 方向采样，估算物理长度
        u_mid = (u_min + u_max) / 2
        v_samples = np.linspace(v_min, v_max, 20)
        p_v = surface.evaluate(u_mid, v_samples)
        dist_v = np.sum(np.sqrt(np.sum(np.diff(p_v, axis=0)**2, axis=-1)))

        nv = int(max(10, (dist_v / wavelength) * self.samples_per_lambda))
        return nv

    def _estimate_u_segments(self, surface, wave, v_center):
        """估算 u 方向需要的分段数，基于相位变化范围"""
        u_min, u_max = surface.u_domain
        k_vec = wave.k_vector

        # 采样 u 方向的多个点，估算相位变化范围
        n_sample = 50
        u_samples = np.linspace(u_min, u_max, n_sample)
        v_arr = np.full_like(u_samples, v_center)
        points = surface.evaluate(u_samples, v_arr)

        # 计算相位
        phases = 2.0 * np.sum(points * k_vec, axis=-1)

        # 使用相位的最大变化范围（不是端点差）
        # 对于周期性曲面（如圆柱），端点差可能很小但实际变化很大
        phase_max = np.max(phases)
        phase_min = np.min(phases)
        total_phase_range = phase_max - phase_min

        # 需要多少段才能使每段相位变化 < max_phase
        n_segments = int(np.ceil(total_phase_range / self.max_phase_per_segment))
        n_segments = max(1, n_segments)

        return n_segments

    def _gauss_integrate_segment(self, surface, wave, v_center, u_start, u_end, ref_point):
        """在 [u_start, u_end] 区间上做 Gauss 积分"""
        k_vec = wave.k_vector
        k_dir = wave.k_dir

        # 将 [-1, 1] 映射到 [u_start, u_end]
        u_scale = (u_end - u_start) / 2
        u_shift = (u_end + u_start) / 2
        u_arr = self.gauss_nodes * u_scale + u_shift
        v_arr = np.full_like(u_arr, v_center)

        # 获取几何数据
        points, normals, jacobians = surface.get_data(u_arr, v_arr)

        # 照射检测
        n_dot_k = np.sum(normals * k_dir, axis=-1)
        illumination = np.where(n_dot_k < 0, -n_dot_k, 0.0)

        # 相位（相对于参考点）
        phase_local = 2.0 * np.sum((points - ref_point) * k_vec, axis=-1)

        # Gauss 积分
        integrand = illumination * jacobians * np.exp(1j * phase_local)
        return np.sum(self.gauss_weights * integrand) * u_scale

    def integrate_surface(self, surface, wave, samples_per_lambda=None):
        """
        对曲面进行 Ribbon 积分

        对于每条 ribbon (固定 v)，沿 u 方向做自适应分段 Gauss 积分。
        """
        spl = samples_per_lambda if samples_per_lambda is not None else self.samples_per_lambda
        nv = self._estimate_nv(surface, wave.wavelength)

        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        dv = (v_max - v_min) / nv
        v_centers = np.linspace(v_min + dv/2, v_max - dv/2, nv)

        k_vec = wave.k_vector

        # 相位稳定化：使用曲面中心作为参考点
        u_mid = (u_min + u_max) / 2
        v_mid = (v_min + v_max) / 2
        ref_point = surface.evaluate(u_mid, v_mid)
        phase_ref = 2.0 * np.dot(ref_point.flatten(), k_vec)

        total_integral = 0j
        self._total_segments = 0  # 用于统计

        for v_c in v_centers:
            # 估算这条 ribbon 需要多少 u 分段
            n_segments = self._estimate_u_segments(surface, wave, v_c)
            self._total_segments += n_segments

            # 分段积分
            u_edges = np.linspace(u_min, u_max, n_segments + 1)
            ribbon_integral = 0j

            for i in range(n_segments):
                seg_integral = self._gauss_integrate_segment(
                    surface, wave, v_c,
                    u_edges[i], u_edges[i+1],
                    ref_point
                )
                ribbon_integral += seg_integral

            total_integral += ribbon_integral * dv

        return total_integral * np.exp(1j * phase_ref)

    def get_mesh_size(self, surface, wave, samples_per_lambda=None):
        """返回网格尺寸估算 (n_gauss * avg_segments, nv)"""
        spl = samples_per_lambda if samples_per_lambda is not None else self.samples_per_lambda
        nv = self._estimate_nv(surface, wave.wavelength)

        # 估算平均分段数
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain
        v_mid = (v_min + v_max) / 2
        avg_segments = self._estimate_u_segments(surface, wave, v_mid)

        return self.n_gauss * avg_segments, nv


# =============================================================================
# 真正的 Analytic Ribbon 积分器 (严格按照论文实现)
# =============================================================================

class AnalyticRibbonIntegrator:
    """
    严格按照论文实现的Ribbon积分器 (CADDSCAT, 1995)

    与 TrueRibbonIntegrator 的区别：
    - TrueRibbonIntegrator: 使用自适应Gauss数值积分
    - AnalyticRibbonIntegrator: 使用多项式拟合 + 解析积分 (Ludwig/Gauss)

    主要特点：
    - G(u) = -(n·k)·J 用五阶多项式拟合
    - φ(u) = 2k·P 用三阶多项式拟合
    - 阴影边界通过 G(u)=0 求根精确确定 (精度 1e-6)
    - 只在被照亮区间进行积分
    """

    def __init__(self, nv=None, samples_per_lambda=8,
                 n_fit_samples=16, shadow_tol=1e-6):
        """
        参数:
            nv: v方向ribbon数（自动估算若不指定）
            samples_per_lambda: 每波长采样数
            n_fit_samples: 多项式拟合采样点数 (论文建议针对bi-cubic采样)
            shadow_tol: 阴影边界精度 (论文要求 1e-6)
        """
        self.nv_manual = nv
        self.samples_per_lambda = samples_per_lambda
        self.n_fit_samples = max(10, n_fit_samples)
        self.shadow_tol = shadow_tol

    def _estimate_nv(self, surface, wavelength):
        """估算 v 方向需要的 ribbon 数量"""
        if self.nv_manual is not None:
            return self.nv_manual

        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        # 沿 v 方向采样，估算物理长度
        u_mid = (u_min + u_max) / 2
        v_samples = np.linspace(v_min, v_max, 20)
        p_v = surface.evaluate(u_mid, v_samples)
        dist_v = np.sum(np.sqrt(np.sum(np.diff(p_v, axis=0)**2, axis=-1)))

        nv = int(max(10, (dist_v / wavelength) * self.samples_per_lambda))
        return nv

    def integrate_surface(self, surface, wave, samples_per_lambda=None):
        """
        主入口：对曲面进行 Analytic Ribbon 积分
        """
        from .ribbon_polynomials import RibbonPolynomialCalculator
        from .ribbon_analytic import RibbonAnalyticIntegrator

        spl = samples_per_lambda if samples_per_lambda is not None else self.samples_per_lambda
        nv = self._estimate_nv(surface, wave.wavelength)

        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        dv = (v_max - v_min) / nv
        v_centers = np.linspace(v_min + dv/2, v_max - dv/2, nv)

        k_vec = wave.k_vector
        # 相位稳定化参考点
        u_mid = (u_min + u_max) / 2
        v_mid = (v_min + v_max) / 2
        ref_point = surface.evaluate(u_mid, v_mid)

        total_integral = 0j

        for v_c in v_centers:
            # 1. 计算多项式系数 (G(u) 5阶, phi(u) 3阶)
            G_coeffs, phi_coeffs = RibbonPolynomialCalculator.compute_coefficients(
                surface, v_c, wave, n_samples=self.n_fit_samples
            )

            # 2. 找阴影边界 (G(u)=0 的根)
            shadow_bounds = RibbonPolynomialCalculator.find_shadow_boundaries(
                G_coeffs, u_min, u_max, tol=self.shadow_tol
            )

            # 3. 确定被照亮区间
            lit_intervals = RibbonPolynomialCalculator.get_illuminated_intervals(
                G_coeffs, u_min, u_max, shadow_bounds
            )

            # 4. 对每个区间进行积分
            ribbon_integral = 0j
            for u_a, u_b in lit_intervals:
                seg_integral = RibbonAnalyticIntegrator.integrate_segment(
                    G_coeffs, phi_coeffs, u_a, u_b, ref_point, k_vec
                )
                ribbon_integral += seg_integral

            total_integral += ribbon_integral * dv

        return total_integral

    def get_mesh_size(self, surface, wave, samples_per_lambda=None):
        """返回网格尺寸估算 (拟合采样数, nv)"""
        spl = samples_per_lambda if samples_per_lambda is not None else self.samples_per_lambda
        nv = self._estimate_nv(surface, wave.wavelength)
        return self.n_fit_samples, nv


# =============================================================================
# 向后兼容别名 & 算法选择接口
# =============================================================================

# 保持向后兼容：原来的 RibbonIntegrator 现在是 DiscretePOIntegrator
RibbonIntegrator = DiscretePOIntegrator


# 可用算法列表
AVAILABLE_ALGORITHMS = {
    'discrete_po_none': {
        'name': '离散PO (无校正)',
        'class': DiscretePOIntegrator,
        'kwargs': {'sinc_mode': 'none'},
        'description': '纯离散 PO，无 sinc 校正。需要最多网格点，精度依赖网格密度。',
    },
    'discrete_po_sinc_u': {
        'name': '离散PO (单向Sinc)',
        'class': DiscretePOIntegrator,
        'kwargs': {'sinc_mode': 'u_only'},
        'description': '离散 PO + u方向 sinc 校正。原始 Ribbon 近似，适中精度。',
    },
    'discrete_po_sinc_dual': {
        'name': '离散PO (双向Sinc)',
        'class': DiscretePOIntegrator,
        'kwargs': {'sinc_mode': 'dual'},
        'description': '离散 PO + 双向 sinc 校正。最佳精度，适合斜入射场景。',
    },
    'gauss_ribbon': {
        'name': 'Gauss-Ribbon (自适应)',
        'class': TrueRibbonIntegrator,
        'kwargs': {},
        'description': 'v方向离散，u方向自适应Gauss积分。高效且精确，推荐用于非多项式曲面。',
    },
    'analytic_ribbon': {
        'name': '解析Ribbon (论文算法)',
        'class': AnalyticRibbonIntegrator,
        'kwargs': {},
        'description': '严格按照1995论文实现：多项式拟合+精确阴影边界+解析积分。',
    },
}


def get_integrator(algorithm='discrete_po_sinc_dual', **kwargs):
    """
    算法工厂函数：根据名称获取积分器实例

    参数:
        algorithm: 算法名称，可选:
            - 'discrete_po_none': 纯离散 PO
            - 'discrete_po_sinc_u': 单向 sinc 校正
            - 'discrete_po_sinc_dual': 双向 sinc 校正 (默认)
            - 'gauss_ribbon': Gauss Ribbon (v离散 + u方向Gauss积分)
            - 'analytic_ribbon': 解析 Ribbon (多项式拟合)

    返回:
        积分器实例
    """
    if algorithm not in AVAILABLE_ALGORITHMS:
        raise ValueError(f"未知算法: {algorithm}. 可用: {list(AVAILABLE_ALGORITHMS.keys())}")

    algo_info = AVAILABLE_ALGORITHMS[algorithm]
    # 合并算法预设参数和用户参数（用户参数优先）
    merged_kwargs = {**algo_info.get('kwargs', {}), **kwargs}
    return algo_info['class'](**merged_kwargs)


def list_algorithms():
    """列出所有可用算法"""
    result = []
    for key, info in AVAILABLE_ALGORITHMS.items():
        result.append({
            'id': key,
            'name': info['name'],
            'description': info['description']
        })
    return result