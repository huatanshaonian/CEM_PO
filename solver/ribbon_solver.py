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
        三角形面的条带状积分（改进版）

        以 u_min 退化为例：
        - E0 (u_max边) 和 E3 (v_max边，斜边) 各划分为 a 份，连接形成条带
        - E1 (v_min边) 划分为 b 份，决定细分数
        - 每个条带内细分数递减：b, b-1, b-2, ...
        """
        spl = samples_per_lambda if samples_per_lambda is not None else self.default_samples_per_lambda

        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        # 估算各边的物理长度来确定 a 和 b
        if degen_edge in ['u_min', 'u_max']:
            # E0 (u_max 或 u_min 的非退化边) 和 E1 (v_min) 的长度
            a, b = self._estimate_triangle_divisions(surface, wave.wavelength, spl, 'u', degen_edge)
        else:  # v_min 或 v_max 退化
            # 角色互换
            a, b = self._estimate_triangle_divisions(surface, wave.wavelength, spl, 'v', degen_edge)

        total_I = 0j

        if degen_edge == 'u_min':
            # u_min 退化：E0=u_max, E3=v_max(斜边), E1=v_min
            total_I = self._integrate_triangle_umin(surface, wave, a, b)
        elif degen_edge == 'u_max':
            # u_max 退化：E0=u_min, E3=v_max(斜边), E1=v_min
            total_I = self._integrate_triangle_umax(surface, wave, a, b)
        elif degen_edge == 'v_min':
            # v_min 退化：角色互换
            total_I = self._integrate_triangle_vmin(surface, wave, a, b)
        elif degen_edge == 'v_max':
            # v_max 退化
            total_I = self._integrate_triangle_vmax(surface, wave, a, b)

        return total_I

    def _estimate_triangle_divisions(self, surface, wavelength, spl, primary_dir, degen_edge):
        """
        估算三角形面的条带数 a 和细分数 b
        """
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        # 采样估算边长
        n_sample = 10

        if primary_dir == 'u':
            # E0 (非退化的 u 边) 的长度 → 决定 a
            if degen_edge == 'u_min':
                u_edge = u_max
            else:
                u_edge = u_min
            v_samples = np.linspace(v_min, v_max, n_sample)
            p_e0 = surface.evaluate(np.full(n_sample, u_edge), v_samples)
            len_e0 = np.sum(np.sqrt(np.sum(np.diff(p_e0, axis=0)**2, axis=-1)))

            # E1 (v_min 边) 的长度 → 决定 b
            u_samples = np.linspace(u_min, u_max, n_sample)
            p_e1 = surface.evaluate(u_samples, np.full(n_sample, v_min))
            len_e1 = np.sum(np.sqrt(np.sum(np.diff(p_e1, axis=0)**2, axis=-1)))

            a = max(5, int(len_e0 / wavelength * spl))
            b = max(5, int(len_e1 / wavelength * spl))

        else:  # primary_dir == 'v'
            # 角色互换
            if degen_edge == 'v_min':
                v_edge = v_max
            else:
                v_edge = v_min
            u_samples = np.linspace(u_min, u_max, n_sample)
            p_e0 = surface.evaluate(u_samples, np.full(n_sample, v_edge))
            len_e0 = np.sum(np.sqrt(np.sum(np.diff(p_e0, axis=0)**2, axis=-1)))

            # E1 (u_min 边) 的长度
            v_samples = np.linspace(v_min, v_max, n_sample)
            p_e1 = surface.evaluate(np.full(n_sample, u_min), v_samples)
            len_e1 = np.sum(np.sqrt(np.sum(np.diff(p_e1, axis=0)**2, axis=-1)))

            a = max(5, int(len_e0 / wavelength * spl))
            b = max(5, int(len_e1 / wavelength * spl))

        return a, b

    def _integrate_triangle_umin(self, surface, wave, a, b):
        """
        u_min 退化的三角形积分

        E2 (u_min) = 退化点
        E0 (u_max) = 划分 a 份
        E3 (v_max) = 斜边，划分 a 份
        E1 (v_min) = 划分 b 份
        """
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        total_I = 0j

        # E0 上的点：(u_max, v_i)，i = 0..a
        # E3 上的点：(u_j, v_max)，j = 0..a，其中 u_0 = u_min (退化点)
        # 连接 E0[i] 和 E3[i] 形成条带边界

        for strip_idx in range(a):
            # 条带边界在参数域中的位置
            # 左边界（靠近退化点）：从 E0[strip_idx] 到 E3[strip_idx]
            # 右边界（远离退化点）：从 E0[strip_idx+1] 到 E3[strip_idx+1]

            # E0 上的 v 坐标
            v_e0_left = v_min + strip_idx * (v_max - v_min) / a
            v_e0_right = v_min + (strip_idx + 1) * (v_max - v_min) / a

            # E3 上的 u 坐标
            u_e3_left = u_min + strip_idx * (u_max - u_min) / a
            u_e3_right = u_min + (strip_idx + 1) * (u_max - u_min) / a

            # 条带内的细分数：从 b 递减到 1
            n_subdivs = max(1, b - strip_idx)

            # 在条带内生成网格点
            for sub_idx in range(n_subdivs):
                # 沿条带方向的插值参数 t（从 E1 侧到斜边侧）
                t_left = sub_idx / n_subdivs
                t_right = (sub_idx + 1) / n_subdivs
                t_center = (t_left + t_right) / 2

                # 计算网格单元的四个角（在参数域中）
                # 左边界上的点：线性插值 (u_max, v_e0_left) -> (u_e3_left, v_max)
                # 右边界上的点：线性插值 (u_max, v_e0_right) -> (u_e3_right, v_max)

                def interp_left(t):
                    u = u_max + t * (u_e3_left - u_max)
                    v = v_e0_left + t * (v_max - v_e0_left)
                    return u, v

                def interp_right(t):
                    u = u_max + t * (u_e3_right - u_max)
                    v = v_e0_right + t * (v_max - v_e0_right)
                    return u, v

                # 网格中心
                u_c, v_c = interp_left(t_center)
                u_c2, v_c2 = interp_right(t_center)
                u_center = (u_c + u_c2) / 2
                v_center = (v_c + v_c2) / 2

                # 估算 du, dv（网格单元大小）
                u_l1, v_l1 = interp_left(t_left)
                u_l2, v_l2 = interp_left(t_right)
                u_r1, v_r1 = interp_right(t_left)
                u_r2, v_r2 = interp_right(t_right)

                # 近似的 du, dv
                du = abs(u_r1 - u_l1 + u_r2 - u_l2) / 2
                dv = abs(v_l2 - v_l1 + v_r2 - v_r1) / 2

                if du < 1e-10 or dv < 1e-10:
                    continue

                # 计算该网格单元的积分贡献
                total_I += self._compute_cell_integral(
                    surface, wave, u_center, v_center, du, dv
                )

        return total_I

    def _integrate_triangle_umax(self, surface, wave, a, b):
        """u_max 退化的三角形积分（与 u_min 对称）"""
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        total_I = 0j

        for strip_idx in range(a):
            v_e0_left = v_min + strip_idx * (v_max - v_min) / a
            v_e0_right = v_min + (strip_idx + 1) * (v_max - v_min) / a

            # E3 在 v_max，从 u_max(退化) 到 u_min
            u_e3_left = u_max - strip_idx * (u_max - u_min) / a
            u_e3_right = u_max - (strip_idx + 1) * (u_max - u_min) / a

            n_subdivs = max(1, b - strip_idx)

            for sub_idx in range(n_subdivs):
                t_center = (sub_idx + 0.5) / n_subdivs

                # 从 (u_min, v_e0) 到 (u_e3, v_max)
                u_c1 = u_min + t_center * (u_e3_left - u_min)
                v_c1 = v_e0_left + t_center * (v_max - v_e0_left)
                u_c2 = u_min + t_center * (u_e3_right - u_min)
                v_c2 = v_e0_right + t_center * (v_max - v_e0_right)

                u_center = (u_c1 + u_c2) / 2
                v_center = (v_c1 + v_c2) / 2

                du = abs(u_c2 - u_c1) + (u_max - u_min) / a / n_subdivs
                dv = (v_max - v_min) / a

                if du < 1e-10 or dv < 1e-10:
                    continue

                total_I += self._compute_cell_integral(surface, wave, u_center, v_center, du, dv)

        return total_I

    def _integrate_triangle_vmin(self, surface, wave, a, b):
        """v_min 退化的三角形积分（u/v 角色互换）"""
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        total_I = 0j

        for strip_idx in range(a):
            u_e0_left = u_min + strip_idx * (u_max - u_min) / a
            u_e0_right = u_min + (strip_idx + 1) * (u_max - u_min) / a

            v_e3_left = v_min + strip_idx * (v_max - v_min) / a
            v_e3_right = v_min + (strip_idx + 1) * (v_max - v_min) / a

            n_subdivs = max(1, b - strip_idx)

            for sub_idx in range(n_subdivs):
                t_center = (sub_idx + 0.5) / n_subdivs

                u_c1 = u_e0_left + t_center * (u_max - u_e0_left)
                v_c1 = v_max + t_center * (v_e3_left - v_max)
                u_c2 = u_e0_right + t_center * (u_max - u_e0_right)
                v_c2 = v_max + t_center * (v_e3_right - v_max)

                u_center = (u_c1 + u_c2) / 2
                v_center = (v_c1 + v_c2) / 2

                du = (u_max - u_min) / a
                dv = abs(v_c2 - v_c1) + (v_max - v_min) / a / n_subdivs

                if du < 1e-10 or dv < 1e-10:
                    continue

                total_I += self._compute_cell_integral(surface, wave, u_center, v_center, du, dv)

        return total_I

    def _integrate_triangle_vmax(self, surface, wave, a, b):
        """v_max 退化的三角形积分"""
        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        total_I = 0j

        for strip_idx in range(a):
            u_e0_left = u_min + strip_idx * (u_max - u_min) / a
            u_e0_right = u_min + (strip_idx + 1) * (u_max - u_min) / a

            v_e3_left = v_max - strip_idx * (v_max - v_min) / a
            v_e3_right = v_max - (strip_idx + 1) * (v_max - v_min) / a

            n_subdivs = max(1, b - strip_idx)

            for sub_idx in range(n_subdivs):
                t_center = (sub_idx + 0.5) / n_subdivs

                u_c1 = u_e0_left + t_center * (u_max - u_e0_left)
                v_c1 = v_min + t_center * (v_e3_left - v_min)
                u_c2 = u_e0_right + t_center * (u_max - u_e0_right)
                v_c2 = v_min + t_center * (v_e3_right - v_min)

                u_center = (u_c1 + u_c2) / 2
                v_center = (v_c1 + v_c2) / 2

                du = (u_max - u_min) / a
                dv = abs(v_c2 - v_c1) + (v_max - v_min) / a / n_subdivs

                if du < 1e-10 or dv < 1e-10:
                    continue

                total_I += self._compute_cell_integral(surface, wave, u_center, v_center, du, dv)

        return total_I

    def _compute_cell_integral(self, surface, wave, u_center, v_center, du, dv):
        """计算单个网格单元的积分贡献"""
        points, normals, jacobians = surface.get_data(
            np.array([[u_center]]), np.array([[v_center]])
        )

        point = points[0, 0]
        normal = normals[0, 0]
        jacobian = jacobians[0, 0]

        if jacobian < 1e-10:
            return 0j

        k_vec = wave.k_vector
        k_dir = wave.k_dir

        # 背面剔除
        n_dot_k = np.dot(normal, k_dir)
        if n_dot_k >= 0:
            return 0j

        illumination_factor = -n_dot_k

        # 相位
        phase = 2.0 * np.dot(point, k_vec)

        # 相位梯度（用于 sinc 项）
        eps = du * 1e-4
        p_plus = surface.evaluate(np.array([[u_center + eps]]), np.array([[v_center]]))[0, 0]
        p_minus = surface.evaluate(np.array([[u_center - eps]]), np.array([[v_center]]))[0, 0]
        phi_plus = 2.0 * np.dot(p_plus, k_vec)
        phi_minus = 2.0 * np.dot(p_minus, k_vec)
        alpha = (phi_plus - phi_minus) / (2 * eps)

        sinc_term = np.sinc(alpha * du / (2.0 * np.pi))

        contribution = (illumination_factor * jacobian *
                       np.exp(1j * phase) *
                       sinc_term *
                       du * dv)

        return contribution

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

    def get_triangle_mesh_cells(self, surface, degen_edge=None, preview_a=15, preview_b=15):
        """
        获取三角形面的网格单元（用于可视化）
        实现同心层递减细分逻辑，确保网格对齐：
        1. 按层生成节点，每层节点数随半径减小而递减。
        2. 层与层之间通过矩形和末端三角形连接。
        """
        if degen_edge is None:
            degen_edge = detect_degenerate_edge(surface)

        if degen_edge is None or degen_edge == 'degenerate':
            return [], 0, 0

        u_min, u_max = surface.u_domain
        v_min, v_max = surface.v_domain

        a, b = preview_a, preview_b
        mesh_cells = []

        # 定义获取某一层节点的函数
        def get_layer_nodes(layer_idx, total_layers, n_subdivs_base, type_edge):
            """
            返回第 layer_idx 层的节点列表 [(u,v), ...]
            layer_idx: 0 (起始边) -> total_layers (退化边)
            """
            nodes = []
            
            # 当前层的段数
            n_segs = max(1, n_subdivs_base - layer_idx)
            
            # 计算当前层的径向位置 (ratio 0 -> 1)
            # layer_idx = 0 对应起始边 (如 u_max)
            # layer_idx = a 对应终止边 (如 u_min)
            r_ratio = layer_idx / total_layers
            
            if type_edge == 'u_min': # u: u_max -> u_min
                u_curr = u_max - r_ratio * (u_max - u_min)
                # v 从 v_min 到 v_max 均匀分布
                for k in range(n_segs + 1):
                    v_curr = v_min + (k / n_segs) * (v_max - v_min)
                    nodes.append((u_curr, v_curr))
                    
            elif type_edge == 'u_max': # u: u_min -> u_max
                u_curr = u_min + r_ratio * (u_max - u_min)
                for k in range(n_segs + 1):
                    v_curr = v_min + (k / n_segs) * (v_max - v_min)
                    nodes.append((u_curr, v_curr))
                    
            elif type_edge == 'v_min': # v: v_max -> v_min
                v_curr = v_max - r_ratio * (v_max - v_min)
                for k in range(n_segs + 1):
                    u_curr = u_min + (k / n_segs) * (u_max - u_min)
                    nodes.append((u_curr, v_curr))
                    
            elif type_edge == 'v_max': # v: v_min -> v_max
                v_curr = v_min + r_ratio * (v_max - v_min)
                for k in range(n_segs + 1):
                    u_curr = u_min + (k / n_segs) * (u_max - u_min)
                    nodes.append((u_curr, v_curr))
                    
            return nodes

        # 生成所有层的节点
        layers_nodes = []
        for i in range(a + 1):
            layers_nodes.append(get_layer_nodes(i, a, b, degen_edge))

        # 构建网格单元
        for i in range(a):
            current_nodes = layers_nodes[i]
            next_nodes = layers_nodes[i+1]
            
            n_curr = len(current_nodes) - 1
            n_next = len(next_nodes) - 1
            
            if n_next < n_curr:
                # 过渡层：1 个三角形 + n_curr-1 个矩形
                # 三角形在起始侧 (v_min / u_min 侧)
                tri_corners = [
                    current_nodes[0],
                    current_nodes[1],
                    next_nodes[0]
                ]
                mesh_cells.append(tri_corners)
                
                # 矩形部分，通过偏移索引保持对齐
                for k in range(1, n_curr):
                    corners = [
                        current_nodes[k],
                        current_nodes[k+1],
                        next_nodes[k],     # 对应下一层的第 k 段终点
                        next_nodes[k-1]    # 对应下一层的第 k 段起点
                    ]
                    mesh_cells.append(corners)
                
            else:
                # 稳定层 (n_next == n_curr): 全是矩形
                for k in range(n_curr):
                    corners = [
                        current_nodes[k],
                        current_nodes[k+1],
                        next_nodes[k+1],
                        next_nodes[k]
                    ]
                    mesh_cells.append(corners)

        return mesh_cells, a, b

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