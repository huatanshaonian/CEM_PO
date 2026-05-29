"""
Johansen 1996 截断 MEC 边段贡献主函数 ('michaeli_mec_truncated' 算法)

与 mec_core.py 接口完全一致 (PTD 算法注册表平替项), 唯一差别:
    截断 EEC = Michaeli 非截断 - Johansen 修正
    If_T = If_UT - If_cor
    Mf_T = Mf_UT - Mf_cor

依赖:
    - mec_coefficients.compute_total_fringe_currents (Michaeli 非截断)
    - mec_truncated_coefficients.compute_correction_currents (Johansen Eq.26/27)
    - 每个段 seg.l_A 属性 (由 ptd.py 在边提取后预计算)

单站约定 (与 mec_core 一致):
    本代码内 beta_prime, phi_prime 对应 Michaeli β', φ' (源方向夹角与方位)
    单站后向散射 (观察=源方向): beta_obs = π - β', phi_obs = φ'
"""
import numpy as np

from .constants import ETA0
from .mec_coefficients import compute_total_fringe_currents
from .mec_truncated_coefficients import compute_correction_currents


# 发射/接收极化基向量名称映射 (与 mec_core 一致)
_POL_BASIS = {
    'VV': ('theta_hat', 'theta_hat'),
    'HH': ('phi_hat',   'phi_hat'),
    'VH': ('theta_hat', 'phi_hat'),
    'HV': ('phi_hat',   'theta_hat'),
}


def compute_mec_truncated_contribution(edge, wave, polarization='VV'):
    """
    单条边的 Johansen 截断 MEC 边段累加贡献 (与 mec_core 同接口).

    参数:
        edge:         PTDEdge 对象, 每个 segment 需有 .l_A 属性
                      (由 PTDProcessor.extract_edges_from_face_pairs 预计算)
        wave:         IncidentWave 对象
        polarization: 'VV' | 'HH' | 'VH' | 'HV'

    返回:
        complex - 该边对单站散射场积分量 I 的贡献
    """
    if polarization not in _POL_BASIS:
        raise ValueError(
            f"未知极化模式: {polarization}. 可用: {list(_POL_BASIS.keys())}")

    et_name, er_name = _POL_BASIS[polarization]
    e_t = getattr(wave, et_name)
    e_r = getattr(wave, er_name)

    k_dir = wave.k_dir
    k_vec = wave.k_vector
    k = wave.k
    Z = ETA0
    s_dir = -k_dir

    H0_vec = np.cross(k_dir, e_t) / Z

    total = 0.0 + 0.0j

    for seg in edge.segments:
        t = seg.tangent
        n_lit = seg.normal if seg.normal is not None else edge.n_lit

        # ---- 1. 局部坐标系 (与 mec_core 一致) ----
        k_dot_t = float(np.clip(np.dot(k_dir, t), -1.0, 1.0))
        sin_gamma0 = np.sqrt(max(0.0, 1.0 - k_dot_t * k_dot_t))
        if sin_gamma0 < 1e-3:
            continue

        n_dot_t = float(np.dot(n_lit, t))
        y1_raw = n_lit - n_dot_t * t
        y1_len = np.linalg.norm(y1_raw)
        if y1_len < 1e-10:
            continue
        y1 = y1_raw / y1_len

        x1 = np.cross(y1, t)
        x1_len = np.linalg.norm(x1)
        if x1_len < 1e-10:
            continue
        x1 = x1 / x1_len

        inward = getattr(seg, 'inward', None)
        if inward is not None:
            if float(np.dot(x1, inward)) < 0:
                x1 = -x1
        else:
            n_b = seg.normal_b if hasattr(seg, 'normal_b') else None
            if n_b is not None and float(np.dot(x1, n_b)) > 0:
                x1 = -x1

        # ---- 2. β', φ' (Michaeli 源方向约定) ----
        cos_bp = -k_dot_t
        beta_prime = np.arccos(np.clip(cos_bp, -1.0, 1.0))

        s_inc = -k_dir
        s_perp = s_inc - np.dot(s_inc, t) * t
        sp_len = np.linalg.norm(s_perp)
        if sp_len < 1e-10:
            continue
        sx = float(np.dot(s_perp, x1))
        sy = float(np.dot(s_perp, y1))
        phi_prime = np.arctan2(sy, sx)
        if phi_prime < 0:
            phi_prime += 2.0 * np.pi

        N = seg.alpha / np.pi
        if phi_prime > N * np.pi:
            continue

        # ---- 3. 入射切向激励 ----
        E0z = complex(np.dot(e_t, t))
        H0z = complex(np.dot(H0_vec, t))

        # ---- 4. Michaeli 非截断 EEC (现有实现, Michaeli 单站 Eq.26/27) ----
        If_UT, Mf_UT = compute_total_fringe_currents(
            beta_prime, phi_prime, N, E0z, H0z, k, Z)

        # ---- 5. Johansen 截断修正 (Eq.26/27, 双站通式) ----
        # 单站映射: beta_obs = π - β', phi_obs = φ' (mec_coefficients docstring 一致)
        l_A = getattr(seg, 'l_A', None)
        if l_A is None or not np.isfinite(l_A) or l_A < 1e-12:
            # 找不到 trailing edge 或零长度 → 退回非截断 (修正项=0)
            M_cor = 0.0 + 0.0j
            I_cor = 0.0 + 0.0j
        else:
            beta_obs = np.pi - beta_prime
            phi_obs = phi_prime
            M_cor, I_cor = compute_correction_currents(
                beta_prime, phi_prime, beta_obs, phi_obs, l_A,
                E0z, H0z, k, Z)

        # ---- 6. 截断 EEC = UT - cor ----
        If_T = If_UT - I_cor
        Mf_T = Mf_UT - M_cor

        # ---- 7. 接收侧投影 (与 mec_core 一致) ----
        s_cross_t = np.cross(s_dir, t)
        proj_If = float(np.dot(t, e_r))
        proj_Mf = float(np.dot(s_cross_t, e_r))

        amp = -Z * If_T * proj_If + Mf_T * proj_Mf

        # ---- 8. 段积分 + 累加 (与 mec_core 一致的相位约定 + 修正后前置因子) ----
        sinc_arg = k * seg.length * k_dot_t / np.pi
        sinc_val = np.sinc(sinc_arg)
        phase = 2.0 * float(np.dot(seg.midpoint, k_vec))

        # 与 mec_core 同步: pre_factor = -0.5 (Michaeli↔Ufimtsev 时谐约定翻译, commit 6565600)
        seg_contrib = (-0.5) * seg.length * sinc_val * np.exp(1j * phase) * amp
        total += seg_contrib

    return total
