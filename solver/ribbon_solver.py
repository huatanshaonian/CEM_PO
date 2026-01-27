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
        """并行计算任务函数 (使用 CachedMeshData)"""
        theta, cached_surfaces, wave_params, k_mag = args
        wave = self._make_wave(wave_params['frequency'], theta, wave_params['phi'])
        total_I = 0j
        for mesh_data in cached_surfaces:
            total_I += self.solver.integrate_cached(mesh_data, wave)
        sigma = (k_mag**2 / np.pi) * np.abs(total_I)**2
        return 10.0 * np.log10(max(sigma, 1e-20))

    def _make_wave(self, freq, theta, phi):
        from physics.wave import IncidentWave
        return IncidentWave(freq, theta, phi)

    def compute_monostatic_rcs(self, geometry, wave_params, angles,
                               samples_per_lambda=None,
                               parallel=False, n_workers=None,
                               show_progress=True,
                               progress_callback=None,
                               cached_mesh_data=None):
        if isinstance(geometry, list):
            surfaces = geometry
        else:
            surfaces = [geometry]

        k_mag = 2 * np.pi * wave_params['frequency'] / C0
        n_angles = len(angles)
        info_msg = (f"计算参数: {len(surfaces)} 个曲面, {n_angles} 个角度, "
                    f"f={wave_params['frequency']/1e9:.2f}GHz")
        if show_progress: print(info_msg)
        if progress_callback: progress_callback(0, n_angles, info_msg)

        can_cache = hasattr(self.solver, 'precompute_mesh') and hasattr(self.solver, 'integrate_cached')
        geometry_data = surfaces
        is_cached = False
        try:
            if cached_mesh_data:
                if show_progress: print("  使用外部缓存的网格数据...")
                geometry_data = cached_mesh_data
                is_cached = True
            elif can_cache:
                wavelength = C0 / wave_params['frequency']
                if show_progress: print("  正在预计算几何网格 (加速模式)...")
                geometry_data = []
                n_surfs = len(surfaces)
                for i, s in enumerate(surfaces):
                    geometry_data.append(self.solver.precompute_mesh(s, wavelength, samples_per_lambda))
                    if progress_callback and (i % 5 == 0 or i == n_surfs - 1):
                         progress_callback(0, n_angles, f"预计算几何: {i+1}/{n_surfs}")
                is_cached = True
        except Exception as e:
            print(f"  预计算失败: {e}, 回退到实时模式")
            is_cached = False

        if parallel:
            if not is_cached:
                if show_progress: print("  并行模式需支持缓存的求解器，回退到串行。")
                return self._compute_serial(surfaces, wave_params, angles, samples_per_lambda, k_mag, show_progress, progress_callback, is_cached=False)
            return self._compute_parallel(geometry_data, wave_params, angles, k_mag, n_workers, show_progress, progress_callback)
        else:
            return self._compute_serial(geometry_data, wave_params, angles, samples_per_lambda, k_mag, show_progress, progress_callback, is_cached=is_cached)

    def _compute_serial(self, geometry_data, wave_params, angles,
                        samples_per_lambda, k_mag, show_progress, progress_callback=None, is_cached=False):
        rcs_list = []
        n_angles = len(angles)
        for i, theta in enumerate(angles):
            wave = self._make_wave(wave_params['frequency'], theta, wave_params['phi'])
            total_I = 0j
            for obj in geometry_data:
                if is_cached:
                    total_I += self.solver.integrate_cached(obj, wave)
                else:
                    total_I += self.solver.integrate_surface(obj, wave, samples_per_lambda=samples_per_lambda)
            sigma = (k_mag**2 / np.pi) * np.abs(total_I)**2
            rcs_list.append(10.0 * np.log10(max(sigma, 1e-20)))
            if (i + 1) % max(1, n_angles // 20) == 0 or (i + 1) == n_angles:
                progress = (i + 1) / n_angles * 100
                msg = f"进度: {progress:.0f}% ({i+1}/{n_angles})"
                if show_progress: print(f"  {msg}")
                if progress_callback: progress_callback(i + 1, n_angles, msg)
        return np.array(rcs_list)

    def _compute_parallel(self, cached_surfaces, wave_params, angles,
                          k_mag, n_workers, show_progress,
                          progress_callback=None):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os
        if n_workers is None: n_workers = os.cpu_count() or 4
        args_list = [(theta, cached_surfaces, wave_params, k_mag) for theta in angles]
        rcs_dict = {}
        n_angles = len(angles)
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {executor.submit(self._compute_single_angle_cached, args): i for i, args in enumerate(args_list)}
                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    rcs_dict[idx] = future.result()
                    completed += 1
                    if completed % max(1, n_angles // 20) == 0 or completed == n_angles:
                        progress = completed / n_angles * 100
                        msg = f"进度: {progress:.0f}% ({completed}/{n_angles})"
                        if show_progress: print(f"  {msg}")
                        if progress_callback: progress_callback(completed, n_angles, msg)
            return np.array([rcs_dict[i] for i in range(n_angles)])
        except Exception as e:
            raise RuntimeError(f"并行计算致命错误: {e}")

    def compute_monostatic_rcs_2d(self, geometry, frequency, theta_array, phi_array,
                                   samples_per_lambda=None,
                                   parallel=False, n_workers=None,
                                   show_progress=True,
                                   progress_callback=None,
                                   cached_mesh_data=None):
        if isinstance(geometry, list):
            surfaces = geometry
        else:
            surfaces = [geometry]

        k_mag = 2 * np.pi * frequency / C0
        n_theta = len(theta_array)
        n_phi = len(phi_array)
        total_points = n_theta * n_phi
        if show_progress: print(f"2D扫描: {len(surfaces)} 个曲面, {total_points} 个角度点")

        can_cache = hasattr(self.solver, 'precompute_mesh') and hasattr(self.solver, 'integrate_cached')
        geometry_data = surfaces
        is_cached = False
        try:
            if cached_mesh_data:
                geometry_data = cached_mesh_data
                is_cached = True
            elif can_cache:
                wavelength = C0 / frequency
                if show_progress: print("  正在预计算2D几何网格 (加速模式)...")
                
                geometry_data = []
                n_surfs = len(surfaces)
                for i, s in enumerate(surfaces):
                    geometry_data.append(self.solver.precompute_mesh(s, wavelength, samples_per_lambda))
                    
                    # 汇报预计算进度
                    if progress_callback and (i % 5 == 0 or i == n_surfs - 1):
                         progress_callback(0, total_points, f"预计算几何: {i+1}/{n_surfs}")

                is_cached = True
        except Exception as e:
            is_cached = False

        if parallel and is_cached:
            return self._compute_parallel_2d(geometry_data, frequency, theta_array, phi_array, k_mag, n_workers, show_progress, progress_callback)

        rcs_2d = np.zeros((n_theta, n_phi))
        computed = 0
        for i, theta in enumerate(theta_array):
            for j, phi in enumerate(phi_array):
                wave = self._make_wave(frequency, theta, phi)
                total_I = 0j
                for obj in geometry_data:
                    if is_cached:
                        total_I += self.solver.integrate_cached(obj, wave)
                    else:
                        total_I += self.solver.integrate_surface(obj, wave, samples_per_lambda=samples_per_lambda)
                sigma = (k_mag**2 / np.pi) * np.abs(total_I)**2
                rcs_2d[i, j] = 10.0 * np.log10(max(sigma, 1e-20))
                computed += 1
                if computed % max(1, total_points // 20) == 0:
                    if progress_callback: progress_callback(computed, total_points, f"进度: {computed/total_points*100:.0f}%")
        return rcs_2d

    def _compute_parallel_2d(self, cached_surfaces, frequency, theta_array, phi_array,
                             k_mag, n_workers,
                             show_progress, progress_callback=None):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os
        if n_workers is None: n_workers = os.cpu_count() or 4
        n_theta = len(theta_array)
        n_phi = len(phi_array)
        total_points = n_theta * n_phi
        args_list = []
        for i, theta in enumerate(theta_array):
            for j, phi in enumerate(phi_array):
                args_list.append(((i, j), (theta, cached_surfaces, {'frequency': frequency, 'phi': phi}, k_mag)))
        rcs_2d = np.zeros((n_theta, n_phi))
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {executor.submit(self._compute_single_angle_cached, args): idx for idx, args in args_list}
                computed = 0
                for future in as_completed(future_to_idx):
                    i, j = future_to_idx[future]
                    rcs_2d[i, j] = future.result()
                    computed += 1
                    if computed % max(1, total_points // 20) == 0:
                        if progress_callback: progress_callback(completed, total_points, f"进度: {computed/total_points*100:.0f}%")
            return rcs_2d
        except Exception as e:
            raise RuntimeError(f"2D并行计算致命错误: {e}")

class TrueRibbonIntegrator:
    # (此处保留之前的 TrueRibbonIntegrator 完整实现，略过以节省篇幅，确保文件完整性)
    _gauss_cache = {}
    def __init__(self, nv=None, n_gauss=8, samples_per_lambda=8, max_phase_per_segment=np.pi/2):
        self.nv_manual = nv
        self.n_gauss = n_gauss
        self.samples_per_lambda = samples_per_lambda
        self.max_phase_per_segment = max_phase_per_segment
        if n_gauss not in self._gauss_cache:
            nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
            self._gauss_cache[n_gauss] = (nodes, weights)
        self.gauss_nodes, self.gauss_weights = self._gauss_cache[n_gauss]

    def _estimate_nv(self, surface, wavelength):
        if self.nv_manual is not None: return self.nv_manual
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain
        u_mid = (u_min + u_max) / 2
        v_samples = np.linspace(v_min, v_max, 20)
        p_v = surface.evaluate(u_mid, v_samples)
        dist_v = np.sum(np.sqrt(np.sum(np.diff(p_v, axis=0)**2, axis=-1)))
        return int(max(10, (dist_v / wavelength) * self.samples_per_lambda))

    def _estimate_u_segments(self, surface, wave, v_center):
        u_min, u_max = surface.u_domain
        k_vec = wave.k_vector
        u_samples = np.linspace(u_min, u_max, 50)
        points = surface.evaluate(u_samples, np.full_like(u_samples, v_center))
        phases = 2.0 * np.sum(points * k_vec, axis=-1)
        return max(1, int(np.ceil((np.max(phases) - np.min(phases)) / self.max_phase_per_segment)))

    def _gauss_integrate_segment(self, surface, wave, v_center, u_start, u_end, ref_point):
        u_scale = (u_end - u_start) / 2
        u_arr = self.gauss_nodes * u_scale + (u_end + u_start) / 2
        points, normals, jacobians = surface.get_data(u_arr, np.full_like(u_arr, v_center))
        n_dot_k = np.sum(normals * wave.k_dir, axis=-1)
        illumination = np.where(n_dot_k < 0, -n_dot_k, 0.0)
        phase_local = 2.0 * np.sum((points - ref_point) * wave.k_vector, axis=-1)
        return np.sum(self.gauss_weights * illumination * jacobians * np.exp(1j * phase_local)) * u_scale

    def integrate_surface(self, surface, wave, samples_per_lambda=None):
        nv = self._estimate_nv(surface, wave.wavelength)
        v_min, v_max = surface.v_domain
        dv = (v_max - v_min) / nv
        v_centers = np.linspace(v_min + dv/2, v_max - dv/2, nv)
        u_min, u_max = surface.u_domain
        ref_point = surface.evaluate((u_min+u_max)/2, (v_min+v_max)/2)
        phase_ref = 2.0 * np.dot(ref_point.flatten(), wave.k_vector)
        total_integral = 0j
        for v_c in v_centers:
            n_segments = self._estimate_u_segments(surface, wave, v_c)
            u_edges = np.linspace(u_min, u_max, n_segments + 1)
            for i in range(n_segments):
                total_integral += self._gauss_integrate_segment(surface, wave, v_c, u_edges[i], u_edges[i+1], ref_point) * dv
        return total_integral * np.exp(1j * phase_ref)

    def get_mesh_size(self, surface, wave, samples_per_lambda=None):
        nv = self._estimate_nv(surface, wave.wavelength)
        avg_segments = self._estimate_u_segments(surface, wave, np.mean(surface.v_domain))
        return self.n_gauss * avg_segments, nv

# Algorithm Factory
RibbonIntegrator = DiscretePOIntegrator
AVAILABLE_ALGORITHMS = {
    'discrete_po_none': {'name': '离散PO (无校正)', 'class': DiscretePOIntegrator, 'kwargs': {'sinc_mode': 'none'}, 'description': '纯离散 PO，无 sinc 校正。'},
    'discrete_po_sinc_u': {'name': '离散PO (单向Sinc)', 'class': DiscretePOIntegrator, 'kwargs': {'sinc_mode': 'u_only'}, 'description': '离散 PO + u方向 sinc 校正。'},
    'discrete_po_sinc_dual': {'name': '离散PO (双向Sinc)', 'class': DiscretePOIntegrator, 'kwargs': {'sinc_mode': 'dual'}, 'description': '离散 PO + 双向 sinc 校正。'},
    'gauss_ribbon': {'name': 'Gauss-Ribbon (自适应)', 'class': TrueRibbonIntegrator, 'kwargs': {}, 'description': 'v方向离散，u方向自适应Gauss积分。'},
}

def get_integrator(algorithm='discrete_po_sinc_dual', **kwargs):
    if algorithm not in AVAILABLE_ALGORITHMS: raise ValueError(f"Unknown algorithm: {algorithm}")
    info = AVAILABLE_ALGORITHMS[algorithm]
    return info['class'](**{**info.get('kwargs', {}), **kwargs})

def list_algorithms():
    return [{'id': k, 'name': v['name'], 'description': v['description']} for k, v in AVAILABLE_ALGORITHMS.items()]
