import numpy as np

# =============================================================================
# 自适应 Gauss Ribbon 积分器
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
    """

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
        data = surface.get_data(u_arr, v_arr)
        if len(data) == 5:
            points, normals, jacobians, _, _ = data
        else:
            points, normals, jacobians = data

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
        v_min, v_max = surface.v_domain
        v_mid = (v_min + v_max) / 2
        avg_segments = self._estimate_u_segments(surface, wave, v_mid)

        return self.n_gauss * avg_segments, nv


# =============================================================================
# Analytic Ribbon 积分器 (严格按照论文实现)
# =============================================================================

class AnalyticRibbonIntegrator:
    """
    严格按照论文实现的Ribbon积分器 (CADDSCAT, 1995)
    """

    def __init__(self, nv=None, samples_per_lambda=8,
                 n_fit_samples=16, shadow_tol=1e-6):
        self.nv_manual = nv
        self.samples_per_lambda = samples_per_lambda
        self.n_fit_samples = max(10, n_fit_samples)
        self.shadow_tol = shadow_tol

    def _estimate_nv(self, surface, wavelength):
        if self.nv_manual is not None:
            return self.nv_manual
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain
        u_mid = (u_min + u_max) / 2
        v_samples = np.linspace(v_min, v_max, 20)
        p_v = surface.evaluate(u_mid, v_samples)
        dist_v = np.sum(np.sqrt(np.sum(np.diff(p_v, axis=0)**2, axis=-1)))
        nv = int(max(10, (dist_v / wavelength) * self.samples_per_lambda))
        return nv

    def integrate_surface(self, surface, wave, samples_per_lambda=None):
        # 延迟导入以避免在文件头部就依赖这些模块，
        # 假设这些文件会被移动到 solvers 目录下
        from .ribbon_polynomials import RibbonPolynomialCalculator
        from .ribbon_analytic import RibbonAnalyticIntegrator as AnalyticMath

        spl = samples_per_lambda if samples_per_lambda is not None else self.samples_per_lambda
        nv = self._estimate_nv(surface, wave.wavelength)

        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        dv = (v_max - v_min) / nv
        v_centers = np.linspace(v_min + dv/2, v_max - dv/2, nv)

        k_vec = wave.k_vector
        u_mid = (u_min + u_max) / 2
        v_mid = (v_min + v_max) / 2
        ref_point = surface.evaluate(u_mid, v_mid)

        total_integral = 0j

        for v_c in v_centers:
            # 1. 计算多项式系数
            G_coeffs, phi_coeffs = RibbonPolynomialCalculator.compute_coefficients(
                surface, v_c, wave, n_samples=self.n_fit_samples
            )

            # 2. 找阴影边界
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
                seg_integral = AnalyticMath.integrate_segment(
                    G_coeffs, phi_coeffs, u_a, u_b, ref_point, k_vec
                )
                ribbon_integral += seg_integral

            total_integral += ribbon_integral * dv

        return total_integral

    def get_mesh_size(self, surface, wave, samples_per_lambda=None):
        spl = samples_per_lambda if samples_per_lambda is not None else self.samples_per_lambda
        nv = self._estimate_nv(surface, wave.wavelength)
        return self.n_fit_samples, nv
