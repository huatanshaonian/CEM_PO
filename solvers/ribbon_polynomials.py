import numpy as np
from typing import List, Tuple, Optional

class RibbonPolynomialCalculator:
    """
    计算单条ribbon的G(u)和φ(u)多项式系数，并寻找阴影边界。
    
    参考: Elking et al., "A Review of High-Frequency RCS Analysis 
          Capabilities at MDA", IEEE AP Magazine, 1995
    """

    @staticmethod
    def compute_coefficients(surface, v_center: float, wave, n_samples: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        对于bi-cubic曲面，G(u)和φ(u)都是u的多项式。
        
        方法：在u方向采样n_samples个点，最小二乘拟合多项式
        
        G(u) = -(n̂·k̂) × J(u)  (五阶实多项式)
        φ(u) = 2k · P(u)      (三阶多项式)
        
        返回:
        - G_coeffs: [g5, g4, g3, g2, g1, g0] (五阶，numpy convention)
        - phi_coeffs: [p3, p2, p1, p0] (三阶)
        """
        u_min, u_max = surface.u_domain
        u_samples = np.linspace(u_min, u_max, n_samples)
        v_arr = np.full_like(u_samples, v_center)

        # 获取几何数据
        data = surface.get_data(u_samples, v_arr)
        if len(data) == 5:
            points, normals, jacobians, _, _ = data
        else:
            points, normals, jacobians = data

        k_vec = wave.k_vector
        k_dir = wave.k_dir

        # G(u) = -(n·k_dir) * J
        # 注意：论文中 G = (n·k) * J，负值表示被照亮。
        # 这里我们统一用正值表示被照亮，即 G = -(n·k_dir) * J
        n_dot_k = np.sum(normals * k_dir, axis=-1)
        G_values = -n_dot_k * jacobians

        # φ(u) = 2.0 * P · k_vec
        phi_values = 2.0 * np.sum(points * k_vec, axis=-1)

        # 多项式拟合
        G_coeffs = np.polyfit(u_samples, G_values, 5)    # 五阶
        phi_coeffs = np.polyfit(u_samples, phi_values, 5)  # 提高到五阶以更好拟合非多项式曲面

        return G_coeffs, phi_coeffs

    @staticmethod
    def find_shadow_boundaries(G_coeffs: np.ndarray, u_min: float, u_max: float, tol: float = 1e-6) -> List[float]:
        """
        找G(u)=0在[u_min, u_max]内的实根
        
        论文要求：精度达到 one part per million (1e-6)
        """
        roots = np.roots(G_coeffs)
        real_roots = []
        
        dG_coeffs = np.polyder(G_coeffs)
        
        for r in roots:
            # 只考虑近似实数的根
            if np.abs(r.imag) < 1e-4:
                r_real = r.real
                # 稍微放宽范围，以便 Newton-Raphson 能将其拉回区间
                if u_min - 0.05 <= r_real <= u_max + 0.05:
                    r_refined = RibbonPolynomialCalculator._newton_raphson(
                        G_coeffs, dG_coeffs, r_real, u_min, u_max, tol
                    )
                    if r_refined is not None:
                        real_roots.append(r_refined)
        
        # 去重并排序
        unique_roots = []
        if real_roots:
            real_roots.sort()
            unique_roots.append(real_roots[0])
            for r in real_roots[1:]:
                if r - unique_roots[-1] > tol * 10:
                    unique_roots.append(r)
                    
        return unique_roots

    @staticmethod
    def _newton_raphson(poly_coeffs: np.ndarray, dpoly_coeffs: np.ndarray, 
                        x0: float, u_min: float, u_max: float, tol: float,
                        max_iter: int = 20) -> Optional[float]:
        """Newton-Raphson迭代精化根"""
        x = x0
        for _ in range(max_iter):
            f = np.polyval(poly_coeffs, x)
            df = np.polyval(dpoly_coeffs, x)
            if abs(df) < 1e-12:
                # 如果导数太小，尝试微调
                df = 1e-12 if df >= 0 else -1e-12
            
            x_new = x - f / df
            if abs(x_new - x) < tol:
                # 检查是否在区间内（允许极小误差）
                if u_min - tol <= x_new <= u_max + tol:
                    return np.clip(x_new, u_min, u_max)
                return None
            x = x_new
        
        # 如果迭代结束仍未达到容差，但结果在区间内，也可以考虑接受
        if u_min <= x <= u_max and abs(np.polyval(poly_coeffs, x)) < tol * 10:
             return x
             
        return None

    @staticmethod
    def get_illuminated_intervals(G_coeffs: np.ndarray, u_min: float, u_max: float, 
                                 shadow_bounds: List[float]) -> List[Tuple[float, float]]:
        """
        根据阴影边界确定被照亮的区间
        
        判断方法：检查区间中点的G值
        G > 0 表示 -(n·k) > 0，即 n·k < 0，被照亮
        """
        all_bounds = [u_min] + shadow_bounds + [u_max]
        lit_intervals = []

        for i in range(len(all_bounds) - 1):
            u_start, u_end = all_bounds[i], all_bounds[i+1]
            if u_end - u_start < 1e-9:
                continue
                
            u_mid = (u_start + u_end) / 2
            G_mid = np.polyval(G_coeffs, u_mid)

            if G_mid > 0:  # 被照亮
                lit_intervals.append((u_start, u_end))

        return lit_intervals
