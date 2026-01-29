import numpy as np
from scipy.special import fresnel

class RibbonAnalyticIntegrator:
    """
    实现 Ribbon 的解析积分和高阶 Gauss 积分。
    """

    @staticmethod
    def integrate_segment(G_coeffs: np.ndarray, phi_coeffs: np.ndarray, 
                          u_a: float, u_b: float, 
                          ref_point: np.ndarray, k_vec: np.ndarray,
                          method: str = 'auto') -> complex:
        """
        在被照亮区间 [u_a, u_b] 上进行积分。
        
        参数:
            method: 'gauss' | 'ludwig' | 'stationary' | 'auto'
        """
        # 计算相位变化
        phi_a = np.polyval(phi_coeffs, u_a)
        phi_b = np.polyval(phi_coeffs, u_b)
        delta_phi = abs(phi_b - phi_a)

        if method == 'auto':
            if delta_phi < 2.0 * np.pi:
                method = 'gauss'
            else:
                # 检查是否有驻点
                dphi_coeffs = np.polyder(phi_coeffs)
                roots = np.roots(dphi_coeffs)
                has_stationary = False
                for r in roots:
                    if np.abs(r.imag) < 1e-4 and u_a < r.real < u_b:
                        has_stationary = True
                        break
                
                if has_stationary:
                    # method = 'stationary' # 暂时还没实现，先用 gauss 或 ludwig
                    method = 'gauss' 
                else:
                    method = 'gauss' # 暂时统一用高阶 gauss

        if method == 'gauss':
            return RibbonAnalyticIntegrator._gauss_integrate(
                G_coeffs, phi_coeffs, u_a, u_b, ref_point, k_vec
            )
        else:
            # 回退到 gauss
            return RibbonAnalyticIntegrator._gauss_integrate(
                G_coeffs, phi_coeffs, u_a, u_b, ref_point, k_vec
            )

    @staticmethod
    def _gauss_integrate(G_coeffs: np.ndarray, phi_coeffs: np.ndarray, 
                         u_a: float, u_b: float, 
                         ref_point: np.ndarray, k_vec: np.ndarray,
                         n_gauss: int = 16) -> complex:
        """
        使用自适应分段 Gauss 积分。
        确保每个子区间内相位变化不超过指定阈值。
        """
        # 计算端点相位
        phi_a = np.polyval(phi_coeffs, u_a)
        phi_b = np.polyval(phi_coeffs, u_b)
        
        # 估算相位变化范围
        # 对于三阶多项式，我们可以采样更多点来获得更精确的范围
        u_test = np.linspace(u_a, u_b, 10)
        phi_test = np.polyval(phi_coeffs, u_test)
        phi_min, phi_max = np.min(phi_test), np.max(phi_test)
        delta_phi_total = phi_max - phi_min
        
        # 每段最大允许相位变化 (pi/2 是比较保守且精确的)
        max_phase_per_seg = np.pi / 2
        n_segments = int(np.ceil(delta_phi_total / max_phase_per_seg))
        n_segments = max(1, n_segments)
        
        # 预计算 Gauss 节点
        nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
        
        phi_ref = 2.0 * np.dot(ref_point.flatten(), k_vec)
        total_integral = 0j
        
        u_edges = np.linspace(u_a, u_b, n_segments + 1)
        
        for i in range(n_segments):
            u_start, u_end = u_edges[i], u_edges[i+1]
            u_scale = (u_end - u_start) / 2
            u_shift = (u_end + u_start) / 2
            u_arr = nodes * u_scale + u_shift
            
            G_values = np.polyval(G_coeffs, u_arr)
            phi_values = np.polyval(phi_coeffs, u_arr)
            phi_local = phi_values - phi_ref
            
            integrand = G_values * np.exp(1j * phi_local)
            total_integral += np.sum(weights * integrand) * u_scale
            
        return total_integral * np.exp(1j * phi_ref)

    @staticmethod
    def ludwig_integral(G_coeffs, phi_coeffs, u_a, u_b):
        """
        Ludwig积分方法（分部积分递归） - 待实现
        """
        raise NotImplementedError("Ludwig integration is not yet implemented.")
