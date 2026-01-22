import numpy as np
from physics.constants import ETA0, C0


def detect_degenerate_edge(surface, threshold_ratio=0.001):
    """
    检测曲面的退化边（三角形面）

    返回:
        None: 四边形面（无退化）
        'u_min': u=0 边退化
        'u_max': u=1 边退化
        'v_min': v=0 边退化
        'v_max': v=1 边退化
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
    if max_jac < 1e-10:  # 整个面都退化
        return 'degenerate'

    threshold = max_jac * threshold_ratio
    is_degenerate = [j < threshold for j in jac_values]

    # 判断哪条边退化
    # corners: (0,0), (1,0), (0,1), (1,1)
    if is_degenerate[0] and is_degenerate[2]:  # u_min 边 (0,0)-(0,1)
        return 'u_min'
    if is_degenerate[1] and is_degenerate[3]:  # u_max 边 (1,0)-(1,1)
        return 'u_max'
    if is_degenerate[0] and is_degenerate[1]:  # v_min 边 (0,0)-(1,0)
        return 'v_min'
    if is_degenerate[2] and is_degenerate[3]:  # v_max 边 (0,1)-(1,1)
        return 'v_max'

    return None  # 四边形


class RibbonIntegrator:
    """
    使用 Ribbon 方法进行物理光学 (PO) 积分
    支持自适应网格划分 (根据频率和几何尺寸)
    """

    def __init__(self, nu=None, nv=None, samples_per_lambda=10):
        """
        初始化求解器配置。
        nu, nv: 手动指定网格数 (可选)
        samples_per_lambda: 默认的自适应采样密度 (默认 10)
        """
        self.nu_manual = nu
        self.nv_manual = nv
        self.default_samples_per_lambda = samples_per_lambda

    def _estimate_mesh_density(self, surface, wavelength, samples_per_lambda):
        """
        估算曲面的物理尺寸并决定网格数
        """
        if self.nu_manual is not None and self.nv_manual is not None:
            return self.nu_manual, self.nv_manual
            
        # 采样估算尺寸
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain
        
        # 沿中线采样点来估算弧长
        u_samples = np.linspace(u_min, u_max, 10)
        v_samples = np.linspace(v_min, v_max, 10)
        
        # 估算 v 方向长度 (固定 u_mid)
        u_mid = (u_min + u_max) / 2
        p_v = surface.evaluate(u_mid, v_samples)
        dist_v = np.sum(np.sqrt(np.sum(np.diff(p_v, axis=0)**2, axis=-1)))
        
        # 估算 u 方向长度 (固定 v_mid)
        v_mid = (v_min + v_max) / 2
        p_u = surface.evaluate(u_samples, v_mid)
        dist_u = np.sum(np.sqrt(np.sum(np.diff(p_u, axis=0)**2, axis=-1)))

        # 计算网格数
        # v 方向是数值积分，要求较严
        nv = int(max(20, (dist_v / wavelength) * samples_per_lambda))
        
        # u 方向是解析积分，要求较低 (可以降为 3-5 samples/lambda)
        nu = int(max(20, (dist_u / wavelength) * (samples_per_lambda / 2)))
        
        return nu, nv

    def integrate_surface(self, surface, wave, samples_per_lambda=None):
        """
        计算散射积分，自动检测并处理三角形面

        参数:
        surface: 几何表面对象
        wave: 入射波对象
        samples_per_lambda: (可选) 本次计算的采样密度。如果不提供，使用默认值。
        """
        # 检测退化边
        degen_edge = detect_degenerate_edge(surface)

        if degen_edge == 'degenerate':
            # 整个面退化，跳过
            return 0j
        elif degen_edge is not None:
            # 三角形面，使用条带状积分
            return self._integrate_triangle(surface, wave, samples_per_lambda, degen_edge)
        else:
            # 四边形面，使用标准矩形积分
            return self._integrate_quad(surface, wave, samples_per_lambda)

    def _integrate_quad(self, surface, wave, samples_per_lambda=None):
        """四边形面的标准矩形网格积分"""
        spl = samples_per_lambda if samples_per_lambda is not None else self.default_samples_per_lambda

        nu, nv = self._estimate_mesh_density(surface, wave.wavelength, spl)

        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        du = (u_max - u_min) / nu
        dv = (v_max - v_min) / nv

        u_centers = np.linspace(u_min + du/2, u_max - du/2, nu)
        v_centers = np.linspace(v_min + dv/2, v_max - dv/2, nv)

        uu, vv = np.meshgrid(u_centers, v_centers)

        return self._compute_integral(surface, wave, uu, vv, du, dv)

    def _integrate_triangle(self, surface, wave, samples_per_lambda, degen_edge):
        """
        三角形面的条带状积分

        策略：沿非退化方向分条带，每条带的宽度根据位置变化
        """
        spl = samples_per_lambda if samples_per_lambda is not None else self.default_samples_per_lambda

        nu, nv = self._estimate_mesh_density(surface, wave.wavelength, spl)

        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        total_I = 0j

        if degen_edge == 'u_min':
            # u=0 边退化，沿 v 分条带，u 从小到大
            # 三角形：从 (u_min, v_min)-(u_min, v_max) 这个点延伸到 (u_max, v_min)-(u_max, v_max) 边
            dv = (v_max - v_min) / nv
            for j in range(nv):
                v_center = v_min + (j + 0.5) * dv
                # 条带在 u 方向从 u_min 到 u_max，但跳过靠近退化边的区域
                u_start = u_min + (u_max - u_min) * 0.02  # 跳过前 2%
                du = (u_max - u_start) / nu
                u_centers = np.linspace(u_start + du/2, u_max - du/2, nu)
                v_arr = np.full_like(u_centers, v_center)
                total_I += self._compute_integral_1d(surface, wave, u_centers, v_arr, du, dv)

        elif degen_edge == 'u_max':
            # u=1 边退化
            dv = (v_max - v_min) / nv
            for j in range(nv):
                v_center = v_min + (j + 0.5) * dv
                u_end = u_max - (u_max - u_min) * 0.02
                du = (u_end - u_min) / nu
                u_centers = np.linspace(u_min + du/2, u_end - du/2, nu)
                v_arr = np.full_like(u_centers, v_center)
                total_I += self._compute_integral_1d(surface, wave, u_centers, v_arr, du, dv)

        elif degen_edge == 'v_min':
            # v=0 边退化，沿 u 分条带
            du = (u_max - u_min) / nu
            for i in range(nu):
                u_center = u_min + (i + 0.5) * du
                v_start = v_min + (v_max - v_min) * 0.02
                dv = (v_max - v_start) / nv
                v_centers = np.linspace(v_start + dv/2, v_max - dv/2, nv)
                u_arr = np.full_like(v_centers, u_center)
                total_I += self._compute_integral_1d(surface, wave, u_arr, v_centers, du, dv)

        elif degen_edge == 'v_max':
            # v=1 边退化
            du = (u_max - u_min) / nu
            for i in range(nu):
                u_center = u_min + (i + 0.5) * du
                v_end = v_max - (v_max - v_min) * 0.02
                dv = (v_end - v_min) / nv
                v_centers = np.linspace(v_min + dv/2, v_end - dv/2, nv)
                u_arr = np.full_like(v_centers, u_center)
                total_I += self._compute_integral_1d(surface, wave, u_arr, v_centers, du, dv)

        return total_I

    def _compute_integral_1d(self, surface, wave, u_arr, v_arr, du, dv):
        """计算一维条带的积分贡献"""
        points, normals, jacobians = surface.get_data(u_arr, v_arr)

        k_vec = wave.k_vector
        k_dir = wave.k_dir

        phase = 2.0 * np.sum(points * k_vec, axis=-1)

        # 相位梯度
        eps = du * 1e-4
        p_plus = surface.evaluate(u_arr + eps, v_arr)
        p_minus = surface.evaluate(u_arr - eps, v_arr)
        phi_plus = 2.0 * np.sum(p_plus * k_vec, axis=-1)
        phi_minus = 2.0 * np.sum(p_minus * k_vec, axis=-1)
        alpha = (phi_plus - phi_minus) / (2 * eps)

        n_dot_k = np.sum(normals * k_dir, axis=-1)
        lit_mask = n_dot_k < 0
        illumination_factor = -n_dot_k

        sinc_term = np.sinc(alpha * du / (2.0 * np.pi))

        # 过滤 Jacobian 过小的点
        jac_mask = jacobians > jacobians.max() * 0.001 if jacobians.max() > 0 else np.ones_like(jacobians, dtype=bool)
        valid_mask = lit_mask & jac_mask

        contributions = (illumination_factor * jacobians *
                        np.exp(1j * phase) *
                        sinc_term *
                        du * dv)

        return np.sum(contributions[valid_mask])

    def _compute_integral(self, surface, wave, uu, vv, du, dv):
        """计算二维网格的积分"""
        points, normals, jacobians = surface.get_data(uu, vv)

        k_vec = wave.k_vector
        k_dir = wave.k_dir

        phase = 2.0 * np.sum(points * k_vec, axis=-1)

        eps = du * 1e-4
        p_plus = surface.evaluate(uu + eps, vv)
        p_minus = surface.evaluate(uu - eps, vv)
        phi_plus = 2.0 * np.sum(p_plus * k_vec, axis=-1)
        phi_minus = 2.0 * np.sum(p_minus * k_vec, axis=-1)
        alpha = (phi_plus - phi_minus) / (2 * eps)

        n_dot_k = np.sum(normals * k_dir, axis=-1)
        lit_mask = n_dot_k < 0
        illumination_factor = -n_dot_k

        sinc_term = np.sinc(alpha * du / (2.0 * np.pi))

        contributions = (illumination_factor * jacobians *
                        np.exp(1j * phase) *
                        sinc_term *
                        du * dv)

        return np.sum(contributions[lit_mask])

    def get_mesh_data(self, surface, wave, samples_per_lambda=None):
        """
        获取求解器生成的网格数据 (用于可视化)
        返回: (points, normals, (nu, nv))
        """
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

class RCSAnalyzer:
    """
    RCS 分析器，支持串行和并行计算
    """

    def __init__(self, solver):
        self.solver = solver

    def _compute_single_angle(self, args):
        """
        计算单个角度的RCS（用于并行计算）
        """
        from physics.wave import IncidentWave

        theta, surfaces, wave_params, samples_per_lambda, k_mag = args

        wave = IncidentWave(wave_params['frequency'], theta, wave_params['phi'])

        # 相干叠加所有表面的散射贡献
        total_I = 0j
        for surf in surfaces:
            total_I += self.solver.integrate_surface(surf, wave, samples_per_lambda=samples_per_lambda)

        # σ = (k²/π) × |I_total|²
        sigma = (k_mag**2 / np.pi) * np.abs(total_I)**2

        return 10.0 * np.log10(max(sigma, 1e-20))

    def compute_monostatic_rcs(self, geometry, wave_params, angles,
                               samples_per_lambda=None,
                               parallel=False, n_workers=None,
                               show_progress=True,
                               progress_callback=None):
        """
        计算单站 RCS

        参数:
        geometry: 单个 Surface 或 Surface 列表
        wave_params: {'frequency': freq_hz, 'phi': phi_rad}
        angles: theta 角度数组 (弧度)
        samples_per_lambda: 采样密度 (可选)
        parallel: 是否启用并行计算
        n_workers: 并行进程数 (默认为 CPU 核心数)
        show_progress: 是否显示进度 (命令行)
        progress_callback: 进度回调函数 callback(current, total, message)

        返回:
        RCS 数组 (dBsm)
        """
        from physics.wave import IncidentWave

        # 统一处理为列表
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
            # 并行计算
            return self._compute_parallel(
                surfaces, wave_params, angles, samples_per_lambda,
                k_mag, n_workers, show_progress, progress_callback
            )
        else:
            # 串行计算
            return self._compute_serial(
                surfaces, wave_params, angles, samples_per_lambda,
                k_mag, show_progress, progress_callback
            )

    def _compute_serial(self, surfaces, wave_params, angles,
                        samples_per_lambda, k_mag, show_progress, progress_callback=None):
        """串行计算"""
        from physics.wave import IncidentWave

        rcs_list = []
        n_angles = len(angles)

        for i, theta in enumerate(angles):
            wave = IncidentWave(wave_params['frequency'], theta, wave_params['phi'])

            # 相干叠加所有表面的散射贡献
            total_I = 0j
            for surf in surfaces:
                total_I += self.solver.integrate_surface(
                    surf, wave, samples_per_lambda=samples_per_lambda
                )

            # σ = (k²/π) × |I_total|²
            sigma = (k_mag**2 / np.pi) * np.abs(total_I)**2
            rcs_list.append(10.0 * np.log10(max(sigma, 1e-20)))

            # 进度显示
            if (i + 1) % max(1, n_angles // 20) == 0 or (i + 1) == n_angles:
                progress = (i + 1) / n_angles * 100
                msg = f"进度: {progress:.0f}% ({i+1}/{n_angles})"
                if show_progress:
                    print(f"  {msg}")
                if progress_callback:
                    progress_callback(i + 1, n_angles, msg)

        done_msg = "计算完成!"
        if show_progress:
            print(f"  {done_msg}")
        if progress_callback:
            progress_callback(n_angles, n_angles, done_msg)

        return np.array(rcs_list)

    def _compute_parallel(self, surfaces, wave_params, angles,
                          samples_per_lambda, k_mag, n_workers, show_progress,
                          progress_callback=None):
        """并行计算"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os

        if n_workers is None:
            n_workers = os.cpu_count() or 4

        parallel_msg = f"启用并行计算: {n_workers} 个进程"
        if show_progress:
            print(f"  {parallel_msg}")
        if progress_callback:
            progress_callback(0, len(angles), parallel_msg)

        # 准备参数
        args_list = [
            (theta, surfaces, wave_params, samples_per_lambda, k_mag)
            for theta in angles
        ]

        # 并行执行
        rcs_dict = {}
        n_angles = len(angles)

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # 提交所有任务
                future_to_idx = {
                    executor.submit(self._compute_single_angle, args): i
                    for i, args in enumerate(args_list)
                }

                # 收集结果
                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    rcs_dict[idx] = future.result()
                    completed += 1

                    # 进度显示
                    if completed % max(1, n_angles // 20) == 0 or completed == n_angles:
                        progress = completed / n_angles * 100
                        msg = f"进度: {progress:.0f}% ({completed}/{n_angles})"
                        if show_progress:
                            print(f"  {msg}")
                        if progress_callback:
                            progress_callback(completed, n_angles, msg)

            # 按索引顺序排列结果
            rcs_list = [rcs_dict[i] for i in range(n_angles)]

            done_msg = "并行计算完成!"
            if show_progress:
                print(f"  {done_msg}")
            if progress_callback:
                progress_callback(n_angles, n_angles, done_msg)

            return np.array(rcs_list)

        except Exception as e:
            err_msg = f"并行计算失败，回退到串行模式: {e}"
            if show_progress:
                print(f"  {err_msg}")
            if progress_callback:
                progress_callback(0, n_angles, err_msg)
            return self._compute_serial(
                surfaces, wave_params, angles, samples_per_lambda, k_mag,
                show_progress, progress_callback
            )

    def compute_monostatic_rcs_2d(self, geometry, frequency, theta_array, phi_array,
                                   samples_per_lambda=None,
                                   show_progress=True,
                                   progress_callback=None):
        """
        计算 2D 单站 RCS (theta × phi 扫描)

        参数:
        geometry: 单个 Surface 或 Surface 列表
        frequency: 频率 (Hz)
        theta_array: theta 角度数组 (弧度)
        phi_array: phi 角度数组 (弧度)
        samples_per_lambda: 采样密度 (可选)
        show_progress: 是否显示进度
        progress_callback: 进度回调函数 callback(current, total, message)

        返回:
        rcs_2d: 2D RCS 数组 (dBsm)，shape = (n_theta, n_phi)
        """
        from physics.wave import IncidentWave

        # 统一处理为列表
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

        if show_progress:
            print(info_msg)
        if progress_callback:
            progress_callback(0, total_points, info_msg)

        # 初始化结果数组
        rcs_2d = np.zeros((n_theta, n_phi))

        # 计算
        computed = 0
        for i, theta in enumerate(theta_array):
            for j, phi in enumerate(phi_array):
                wave = IncidentWave(frequency, theta, phi)

                # 相干叠加所有表面的散射贡献
                total_I = 0j
                for surf in surfaces:
                    total_I += self.solver.integrate_surface(
                        surf, wave, samples_per_lambda=samples_per_lambda
                    )

                # σ = (k²/π) × |I_total|²
                sigma = (k_mag**2 / np.pi) * np.abs(total_I)**2
                rcs_2d[i, j] = 10.0 * np.log10(max(sigma, 1e-20))

                computed += 1

                # 进度显示
                if computed % max(1, total_points // 20) == 0 or computed == total_points:
                    progress = computed / total_points * 100
                    msg = f"进度: {progress:.0f}% ({computed}/{total_points})"
                    if show_progress:
                        print(f"  {msg}")
                    if progress_callback:
                        progress_callback(computed, total_points, msg)

        done_msg = "2D扫描完成!"
        if show_progress:
            print(f"  {done_msg}")
        if progress_callback:
            progress_callback(total_points, total_points, done_msg)

        return rcs_2d