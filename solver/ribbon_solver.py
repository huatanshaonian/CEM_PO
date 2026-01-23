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
    预计算的网格数据，用于并行计算传输。
    包含积分所需的所有几何信息（纯 NumPy 数组）。
    """
    def __init__(self, points, normals, jacobians, dP_du, du, dv):
        self.points = points
        self.normals = normals
        self.jacobians = jacobians
        self.dP_du = dP_du  # Partial derivative dP/du for alpha calc
        self.du = du
        self.dv = dv

class RibbonIntegrator:
    """
    使用 Ribbon 方法进行物理光学 (PO) 积分
    支持自适应网格划分 (根据频率和几何尺寸)
    """

    def __init__(self, nu=None, nv=None, samples_per_lambda=10):
        self.nu_manual = nu
        self.nv_manual = nv
        self.default_samples_per_lambda = samples_per_lambda

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
        nu = int(max(20, (dist_u / wavelength) * (samples_per_lambda / 2)))
        
        return nu, nv

    def precompute_mesh(self, surface, wavelength, samples_per_lambda=None):
        """
        预计算网格数据，返回 CachedMeshData 对象。
        此方法应在主进程调用。
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
        eps = du * 1e-4
        p_plus = surface.evaluate(uu + eps, vv)
        p_minus = surface.evaluate(uu - eps, vv)
        dP_du = (p_plus - p_minus) / (2 * eps)
        
        return CachedMeshData(points, normals, jacobians, dP_du, du, dv)

    def integrate_cached(self, cached_data, wave):
        """
        使用预计算的 CachedMeshData 进行积分。
        此方法可在子进程调用，无 SWIG 依赖。
        """
        points = cached_data.points
        normals = cached_data.normals
        jacobians = cached_data.jacobians
        dP_du = cached_data.dP_du
        du = cached_data.du
        dv = cached_data.dv
        
        k_vec = wave.k_vector
        k_dir = wave.k_dir
        
        # 相位
        phase = 2.0 * np.sum(points * k_vec, axis=-1)
        
        # Alpha (d_phase/du)
        # alpha = d(2 * P . k) / du = 2 * (dP/du . k)
        alpha = 2.0 * np.sum(dP_du * k_vec, axis=-1)
        
        # 照射检测
        n_dot_k = np.sum(normals * k_dir, axis=-1)
        lit_mask = n_dot_k < 0
        illumination_factor = -n_dot_k

        # Ribbon 积分
        sinc_term = np.sinc(alpha * du / (2.0 * np.pi))

        contributions = (illumination_factor * jacobians *
                        np.exp(1j * phase) *
                        sinc_term *
                        du * dv)

        return np.sum(contributions[lit_mask])

    def integrate_surface(self, surface, wave, samples_per_lambda=None):
        """(旧接口) 直接计算，不缓存"""
        # 为了保持代码复用，我们在内部进行预计算然后积分
        # 虽然有点多余，但能保证逻辑一致性
        mesh_data = self.precompute_mesh(surface, wave.wavelength, samples_per_lambda)
        return self.integrate_cached(mesh_data, wave)

    def get_mesh_data(self, surface, wave, samples_per_lambda=None):
        """获取可视化数据"""
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
        # ... (保持原样，省略以节省空间，实际写入时应包含) ...
        # (由于这个方法很长且未修改，我这里需要小心。为了确保文件完整性，我必须完整写入。)
        # 让我从上面的 read_file 输出中复制 get_triangle_mesh_cells 的实现。
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
        # 注意：这里我们接收的是 caches 而不是 surfaces
        theta, cached_surfaces, wave_params, k_mag = args
        
        # wave_params = {'frequency': f, 'phi': phi}
        wave = self._make_wave(wave_params['frequency'], theta, wave_params['phi'])

        total_I = 0j
        for mesh_data in cached_surfaces:
            total_I += self.solver.integrate_cached(mesh_data, wave)

        sigma = (k_mag**2 / np.pi) * np.abs(total_I)**2
        return 10.0 * np.log10(max(sigma, 1e-20))

    def _make_wave(self, freq, theta, phi):
        from physics.wave import IncidentWave
        return IncidentWave(freq, theta, phi)

    # 兼容旧接口的串行方法
    def _compute_single_angle(self, args):
        theta, surfaces, wave_params, samples_per_lambda, k_mag = args
        wave = self._make_wave(wave_params['frequency'], theta, wave_params['phi'])
        total_I = 0j
        for surf in surfaces:
            total_I += self.solver.integrate_surface(surf, wave, samples_per_lambda=samples_per_lambda)
        sigma = (k_mag**2 / np.pi) * np.abs(total_I)**2
        return 10.0 * np.log10(max(sigma, 1e-20))

    def compute_monostatic_rcs(self, geometry, wave_params, angles,
                               samples_per_lambda=None,
                               parallel=False, n_workers=None,
                               show_progress=True,
                               progress_callback=None):
        from physics.wave import IncidentWave

        if isinstance(geometry, list):
            surfaces = geometry
        else:
            surfaces = [geometry]

        k_mag = 2 * np.pi * wave_params['frequency'] / C0
        n_angles = len(angles)

        info_msg = (f"计算参数: {len(surfaces)} 个曲面, {n_angles} 个角度, "
                    f"f={wave_params['frequency']/1e9:.2f}GHz")

        if show_progress:
            print(info_msg)
        if progress_callback:
            progress_callback(0, n_angles, info_msg)

        if parallel:
            # 预计算所有网格数据
            try:
                wavelength = C0 / wave_params['frequency']
                if show_progress: print("正在预计算并行网格数据...")
                cached_surfaces = [
                    self.solver.precompute_mesh(s, wavelength, samples_per_lambda) 
                    for s in surfaces
                ]
                
                return self._compute_parallel(
                    cached_surfaces, wave_params, angles, k_mag, 
                    n_workers, show_progress, progress_callback
                )
            except Exception as e:
                if show_progress: print(f"预计算失败: {e}, 回退到串行")
                return self._compute_serial(surfaces, wave_params, angles, samples_per_lambda, k_mag, show_progress, progress_callback)
        else:
            return self._compute_serial(
                surfaces, wave_params, angles, samples_per_lambda,
                k_mag, show_progress, progress_callback
            )

    def _compute_serial(self, surfaces, wave_params, angles,
                        samples_per_lambda, k_mag, show_progress, progress_callback=None):
        rcs_list = []
        n_angles = len(angles)

        for i, theta in enumerate(angles):
            wave = self._make_wave(wave_params['frequency'], theta, wave_params['phi'])
            total_I = 0j
            for surf in surfaces:
                total_I += self.solver.integrate_surface(surf, wave, samples_per_lambda=samples_per_lambda)

            sigma = (k_mag**2 / np.pi) * np.abs(total_I)**2
            rcs_list.append(10.0 * np.log10(max(sigma, 1e-20)))

            if (i + 1) % max(1, n_angles // 20) == 0 or (i + 1) == n_angles:
                progress = (i + 1) / n_angles * 100
                msg = f"进度: {progress:.0f}% ({i+1}/{n_angles})"
                if show_progress: print(f"  {msg}")
                if progress_callback: progress_callback(i + 1, n_angles, msg)

        done_msg = "计算完成!"
        if show_progress: print(f"  {done_msg}")
        if progress_callback: progress_callback(n_angles, n_angles, done_msg)

        return np.array(rcs_list)

    def _compute_parallel(self, cached_surfaces, wave_params, angles,
                          k_mag, n_workers, show_progress,
                          progress_callback=None):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os

        if n_workers is None: n_workers = os.cpu_count() or 4

        parallel_msg = f"启用并行计算: {n_workers} 个进程 (Cached Mode)"
        if show_progress: print(f"  {parallel_msg}")
        if progress_callback: progress_callback(0, len(angles), parallel_msg)

        # 参数不再包含 samples_per_lambda，因为网格已固定
        args_list = [
            (theta, cached_surfaces, wave_params, k_mag)
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
                    rcs_dict[idx] = future.result()
                    completed += 1

                    if completed % max(1, n_angles // 20) == 0 or completed == n_angles:
                        progress = completed / n_angles * 100
                        msg = f"进度: {progress:.0f}% ({completed}/{n_angles})"
                        if show_progress: print(f"  {msg}")
                        if progress_callback: progress_callback(completed, n_angles, msg)

            rcs_list = [rcs_dict[i] for i in range(n_angles)]
            done_msg = "并行计算完成!"
            if show_progress: print(f"  {done_msg}")
            if progress_callback: progress_callback(n_angles, n_angles, done_msg)
            return np.array(rcs_list)

        except Exception as e:
            # 这里如果不幸失败，因为已经没有原始 surfaces 对象了，无法简单回退到串行
            # 除非我们在外部做处理。
            # 但既然是 CachedMeshData，只有 pickle 失败才可能出错，而我们确保了全是 numpy
            raise RuntimeError(f"并行计算致命错误: {e}")

    def compute_monostatic_rcs_2d(self, geometry, frequency, theta_array, phi_array,
                                   samples_per_lambda=None,
                                   parallel=False, n_workers=None,
                                   show_progress=True,
                                   progress_callback=None):
        if isinstance(geometry, list):
            surfaces = geometry
        else:
            surfaces = [geometry]

        k_mag = 2 * np.pi * frequency / C0
        n_theta = len(theta_array)
        n_phi = len(phi_array)
        total_points = n_theta * n_phi

        info_msg = (f"2D扫描: {len(surfaces)} 个曲面, "
                    f"{n_theta}×{n_phi}={total_points} 个角度点, "
                    f"f={frequency/1e9:.2f}GHz")

        if show_progress: print(info_msg)
        if progress_callback: progress_callback(0, total_points, info_msg)

        if parallel:
            try:
                wavelength = C0 / frequency
                if show_progress: print("正在预计算2D并行网格数据...")
                cached_surfaces = [
                    self.solver.precompute_mesh(s, wavelength, samples_per_lambda) 
                    for s in surfaces
                ]
                return self._compute_parallel_2d(
                    cached_surfaces, frequency, theta_array, phi_array,
                    k_mag, n_workers, show_progress, progress_callback
                )
            except Exception as e:
                # 同样的，回退机制在这里比较复杂，我们简单打印错误并尝试串行（如果还能拿到 surfaces）
                # 这里的参数 geometry 是原始对象，所以可以递归调用自己并设 parallel=False
                print(f"2D并行预计算失败: {e}, 回退到串行")
                return self.compute_monostatic_rcs_2d(geometry, frequency, theta_array, phi_array, samples_per_lambda, False, n_workers, show_progress, progress_callback)

        # 串行逻辑
        rcs_2d = np.zeros((n_theta, n_phi))
        computed = 0
        for i, theta in enumerate(theta_array):
            for j, phi in enumerate(phi_array):
                wave = self._make_wave(frequency, theta, phi)
                total_I = 0j
                for surf in surfaces:
                    total_I += self.solver.integrate_surface(surf, wave, samples_per_lambda=samples_per_lambda)
                sigma = (k_mag**2 / np.pi) * np.abs(total_I)**2
                rcs_2d[i, j] = 10.0 * np.log10(max(sigma, 1e-20))
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
                             show_progress, progress_callback=None):
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
                args = (theta, cached_surfaces, wave_params, k_mag)
                args_list.append(((i, j), args))

        rcs_2d = np.zeros((n_theta, n_phi))

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(self._compute_single_angle_cached, args): idx_tuple
                    for idx_tuple, args in args_list
                }

                computed = 0
                for future in as_completed(future_to_idx):
                    i, j = future_to_idx[future]
                    rcs_2d[i, j] = future.result()
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
