import numpy as np
from .constants import C0

def compute_ptd_contribution(edge, wave, polarization='VV', debug=False, verbose=False):
    """
    计算整条边缘的 PTD 修正量 (对所有分段求和)

    支持任意楔角 n 的通用 PTD 算法。
    采用 "Split-Average" 处理数值奇点，并使用通用公式扣除 PO 奇异项。

    参数:
        edge: PTDEdge对象 (必须包含正确的 n_param)
        wave: IncidentWave对象
        polarization: 'VV' (Hard/TM) 或 'HH' (Soft/TE)
    """
    total_contrib = 0j

    k_vec = wave.k_vector
    k = np.linalg.norm(k_vec)
    k_dir = wave.k_dir
    s_vec = -k_dir

    n = edge.n_param
    # 自动计算两个面的法向/奇点角度 (Ufimtsev定义)
    # Face 1 at 0. Normal 1 at pi/2.
    # Face 2 at n*pi. Normal 2 at n*pi - pi/2.
    phi_s1 = np.pi / 2
    phi_s2 = n * np.pi - np.pi / 2

    # 预计算常数
    sin_pi_n = np.sin(np.pi / n)
    cos_pi_n = np.cos(np.pi / n)

    # f项 (Incidence Term)
    denom_f = cos_pi_n - 1.0
    if abs(denom_f) > 1e-10:
        f_val = (1.0 / n) * sin_pi_n / denom_f
    else:
        f_val = 0.0

    # 内部系数计算函数
    def eval_D_coeff(phi_rad):
        # 1. UTD Reflection Term (g)
        # 奇异性: cos(2phi/n) = cos(pi/n) -> 2phi/n = +/- pi/n -> phi = pi/2 or n*pi - pi/2
        denom_g = np.cos(np.pi/n) - np.cos(2*phi_rad/n)
        if abs(denom_g) < 1e-14: denom_g = 1e-14 * np.sign(denom_g)
        g_val_local = (1.0/n) * sin_pi_n / denom_g
        
        # 2. PO Correction Terms (General Wedge)
        
        # Face 1 Lit? (0 < phi < pi) - 近似，对于锐角楔形可能不同，但对PO减法来说通常足够
        is_lit1 = (phi_rad > 0) and (phi_rad < np.pi)
        
        # Face 2 Lit? (Normal 2 +/- 90 deg)
        # Normal 2 is at phi_s2. Range: [phi_s2 - pi/2, phi_s2 + pi/2]
        # phi_s2 - pi/2 = n*pi - pi.
        # phi_s2 + pi/2 = n*pi.
        # So range is (n*pi - pi, n*pi).
        is_lit2 = (phi_rad > (n * np.pi - np.pi)) and (phi_rad < n * np.pi)
        
        # D_po1 (Face 1): Singular at phi_s1 (pi/2)
        # Formula: -0.5 * tan(phi - phi_s1 + pi/2) ? No.
        # Standard: -0.5 * tan(phi) is singular at pi/2.
        # Let's stick to stable sin/cos form.
        # Singular when cos(phi) = 0.
        cos_p1 = np.cos(phi_rad)
        if abs(cos_p1) < 1e-14: cos_p1 = 1e-14 * np.sign(cos_p1)
        d_po1 = -0.5 * np.sin(phi_rad) / cos_p1 if is_lit1 else 0.0
        
        # D_po2 (Face 2): Singular at phi_s2 (n*pi - pi/2)
        # We construct a local tan function singular at phi_s2.
        # tan(phi - (phi_s2 - pi/2)) = tan(phi - phi_s2 + pi/2) -> singular at phi_s2.
        # Form: tan(phi - n*pi + pi).
        # Let's derive sign. 
        # UTD at phi_s2 (Reflection Boundary).
        # Check singularity sign of g.
        # phi = phi_s2 + eps = n*pi - pi/2 + eps.
        # 2phi/n = 2pi - pi/n + 2eps/n.
        # cos(2phi/n) = cos(2pi - (pi/n - 2eps/n)) = cos(pi/n - 2eps/n).
        # approx cos(pi/n) + sin(pi/n)*2eps/n.
        # Denom = cos(pi/n) - (cos(pi/n) + ...) = -sin(pi/n)*2eps/n.
        # Denom is Negative for eps > 0.
        # g ~ 1 / (-) = -inf.
        # So UTD is NEGATIVE infinite just after boundary.
        # We need PO to be NEGATIVE infinite to cancel (subtracting -inf).
        # Wait, formula is UTD - PO. (-inf) - (-inf) ok.
        # So we need PO -> -inf at phi_s2 + eps.
        # tan(x) -> -inf at pi/2 + eps.
        # We want argument to be pi/2 at phi_s2.
        # arg = phi - phi_s2 + pi/2.
        # at phi_s2+eps: arg = pi/2 + eps. tan -> -inf.
        # So coeff should be +0.5.
        # D_po2 = 0.5 * tan(phi - phi_s2 + pi/2).
        
        # Shift angle for Face 2 calculation
        phi_local_2 = phi_rad - phi_s2 + np.pi/2
        cos_p2 = np.cos(phi_local_2)
        if abs(cos_p2) < 1e-14: cos_p2 = 1e-14 * np.sign(cos_p2)
        d_po2 = 0.5 * np.sin(phi_local_2) / cos_p2 if is_lit2 else 0.0
        
        if polarization == 'VV':
            # Hard: (f + g) - PO1 - PO2
            return (f_val + g_val_local) - d_po1 - d_po2
        else:
            # Soft: (f - g) + PO1 + PO2
            return (f_val - g_val_local) + d_po1 + d_po2

    for seg in edge.segments:
        t = seg.tangent
        # 优先使用段法向
        n_lit = seg.normal if (hasattr(seg, 'normal') and seg.normal is not None) else edge.n_lit
        
        k_dot_t = np.dot(k_dir, t)
        k_dot_t = np.clip(k_dot_t, -1.0, 1.0)
        sin_beta = np.sqrt(1.0 - k_dot_t**2)
        if sin_beta < 1e-3: continue

        s_dot_t = np.dot(s_vec, t)
        s_perp = s_vec - s_dot_t * t
        len_s_perp = np.linalg.norm(s_perp)
        if len_s_perp < 1e-6: continue
        s_perp /= len_s_perp
        
        cos_alpha = np.dot(s_perp, n_lit)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        
        # 稳健的 phi 计算
        # 注意: 对于任意 STEP，n_face2 可能很难定义(如果不是楔形)
        # 这里假设局部几何近似为楔形，利用 t 和 n_lit 构建右手系
        n_face2_dir = np.cross(t, n_lit) 
        side = np.sign(np.dot(s_perp, n_face2_dir))
        phi = np.pi/2 + side * alpha
        if phi < 0: phi += 2*np.pi
        
        # 奇点检测 (通用 n)
        dist_n1 = abs(phi - phi_s1)
        dist_n2 = abs(phi - phi_s2)
        
        # 0.2度阈值 + 0.05度偏移
        singularity_threshold = np.deg2rad(0.2)
        avg_offset = np.deg2rad(0.05) 
        
        if dist_n1 < singularity_threshold or dist_n2 < singularity_threshold:
            val_minus = eval_D_coeff(phi - avg_offset)
            val_plus = eval_D_coeff(phi + avg_offset)
            D_coeff = (val_minus + val_plus) / 2.0
        else:
            D_coeff = eval_D_coeff(phi)
            
        sinc_arg = k * seg.length * k_dot_t / np.pi
        sinc_val = np.sinc(sinc_arg)
        phase_mid = 2.0 * np.dot(seg.midpoint, k_vec)
        
        pre_factor = (2.0 * np.pi) / (1j * k)
        seg_contrib = pre_factor * D_coeff * seg.length * sinc_val * np.exp(1j * phase_mid)
        total_contrib += seg_contrib

    return total_contrib