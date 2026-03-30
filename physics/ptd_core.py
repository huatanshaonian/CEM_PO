"""
PTD 边缘积分核心（Ufimtsev EEW, Eq. 7.137）

将边缘分段，每段：
  1. 提取楔形截面内的入射角 angle0 和观察角 angle
  2. 调用 FG_monostatic 得到 (F1_Vt, G1_Vt, G1_phi)
  3. 计算 Keller 锥局部基向量 (β̂, α̂)，将全局极化向量投影得到 (cosχ, sinχ)
  4. 按 D = F1_Vt·cos²χ + G1_phi·sin²χ + G1_Vt·sinχ·cosχ 加权组合
  5. 乘以传播因子 (pre_factor = -j·sinγ₀/k) 和相位，累加到总贡献
"""
import numpy as np
from .ptd_coefficients import FG_monostatic

_SING_THRESH = 1e-4   # 奇点邻域阈值（弧度），与 fun_fg 内部 eps=1e-5 量级匹配
_SING_OFFSET = 3e-5   # 奇点两侧偏移量（弧度）


def compute_ptd_contribution(edge, wave, polarization='VV'):
    """
    计算整条边缘的 PTD 修正量（对所有分段求和）。

    参数:
        edge:         PTDEdge 对象
        wave:         IncidentWave 对象（需有 theta_hat, phi_hat 属性）
        polarization: 'VV' 或 'HH'
    返回:
        complex: 该边缘对散射场积分 I 的贡献量
    """
    total_contrib = 0j

    k_dir = wave.k_dir      # 入射方向单位向量（指向目标）
    k_vec = wave.k_vector   # k·k_dir
    k = wave.k
    s_dir = -k_dir          # 单站模式：散射方向 = 入射反向

    # 全局极化向量
    e_pol = wave.theta_hat if polarization == 'VV' else wave.phi_hat

    for seg in edge.segments:
        alfa = seg.alpha        # 楔形外角 α = n·π（每段独立）
        t = seg.tangent
        n_lit = seg.normal if (seg.normal is not None) else edge.n_lit

        # ── 1. 计算斜入射角 γ₀（入射方向与棱边的夹角）──
        k_dot_t = np.dot(k_dir, t)
        k_dot_t = np.clip(k_dot_t, -1.0, 1.0)
        sin_gamma0 = np.sqrt(1.0 - k_dot_t ** 2)
        if sin_gamma0 < 1e-3:
            # 入射方向近乎平行棱边，PTD 无贡献
            continue

        # ── 2. 建立楔形截面局部坐标系（⊥ t 平面内）──
        # 曲面棱边上 n_lit 可能有沿 t 的分量，必须投影到截面平面
        n_dot_t = np.dot(n_lit, t)
        e1_raw = n_lit - n_dot_t * t
        e1_len = np.linalg.norm(e1_raw)
        if e1_len < 1e-10:
            continue
        e1 = e1_raw / e1_len

        e2_raw = np.cross(t, e1)
        e2_len = np.linalg.norm(e2_raw)
        if e2_len < 1e-10:
            continue
        e2 = e2_raw / e2_len

        # 消除 e2 符号歧义（e2 应指向 Face A 表面、远离 Face B）
        n_b = seg.normal_b if hasattr(seg, 'normal_b') else None
        if n_b is not None and np.dot(e2, n_b) > 0:
            e2 = -e2

        # ── 3. 计算入射方向在截面内的投影 → angle0 ──
        k_perp = k_dir - k_dot_t * t
        k_perp_len = np.linalg.norm(k_perp)
        if k_perp_len < 1e-10:
            continue
        k_perp_unit = k_perp / k_perp_len

        inc_unit = -k_perp_unit
        i_e1 = np.dot(inc_unit, e1)
        i_e2 = np.dot(inc_unit, e2)
        angle0_raw = np.arctan2(i_e1, i_e2)
        if angle0_raw < -1e-6:
            angle0_raw += 2.0 * np.pi
        if angle0_raw < -1e-6 or angle0_raw > alfa + 1e-6:
            continue
        angle0 = np.clip(angle0_raw, 0.0, alfa)

        # ── 3b. 散射方向投影 → angle_obs ──
        s_dot_t = np.dot(s_dir, t)
        s_perp = s_dir - s_dot_t * t
        s_perp_len = np.linalg.norm(s_perp)
        if s_perp_len < 1e-10:
            continue
        s_perp_unit = s_perp / s_perp_len

        s_e1 = np.dot(s_perp_unit, e1)
        s_e2 = np.dot(s_perp_unit, e2)
        angle_raw = np.arctan2(s_e1, s_e2)
        if angle_raw < -1e-6:
            angle_raw += 2.0 * np.pi
        angle_obs = np.clip(angle_raw, 0.0, alfa)

        # ── 4. Keller 锥局部基向量 ──
        # β̂ = ϑ̂_local: t̂ 在垂直于观测方向 ŝ 的平面上的投影（归一化）
        beta_raw = t - np.dot(t, s_dir) * s_dir
        beta_len = np.linalg.norm(beta_raw)
        if beta_len < 1e-10:
            continue
        beta_hat = beta_raw / beta_len
        # α̂ = φ̂_local: ŝ × β̂
        alpha_hat = np.cross(s_dir, beta_hat)

        # 极化投影角 χ
        cos_chi = np.dot(e_pol, beta_hat)
        sin_chi = np.dot(e_pol, alpha_hat)

        # ── 5. 计算 FG 衍射系数（含奇点处理）──
        gamma0 = np.arcsin(sin_gamma0)

        psi1 = angle_obs - angle0
        psi2 = angle_obs + angle0
        near_sing = (abs(psi1 % (2 * np.pi) - np.pi) < _SING_THRESH or
                     abs(psi2 % (2 * np.pi) - np.pi) < _SING_THRESH or
                     abs(2 * alfa - psi2 - np.pi) < _SING_THRESH)

        if near_sing:
            F1a, G1Va, G1pa = FG_monostatic(
                angle_obs - _SING_OFFSET, angle0, gamma0, alfa)
            F1b, G1Vb, G1pb = FG_monostatic(
                angle_obs + _SING_OFFSET, angle0, gamma0, alfa)
            F1_Vt  = 0.5 * (F1a + F1b)
            G1_Vt  = 0.5 * (G1Va + G1Vb)
            G1_phi = 0.5 * (G1pa + G1pb)
        else:
            F1_Vt, G1_Vt, G1_phi = FG_monostatic(
                angle_obs, angle0, gamma0, alfa)

        # ── 6. 加权组合衍射系数 D ──
        # Eq. 7.137: D = F1_Vt·cos²χ + G1_phi·sin²χ + G1_Vt·sinχ·cosχ
        D = (F1_Vt * cos_chi**2
             + G1_phi * sin_chi**2
             + G1_Vt * sin_chi * cos_chi)

        # ── 7. 边缘段积分 ──
        sinc_arg = k * seg.length * k_dot_t / np.pi
        sinc_val = np.sinc(sinc_arg)

        # 单站相位
        phase_mid = 2.0 * np.dot(seg.midpoint, k_vec)

        # 传播因子：Ufimtsev EEW (Eq. 7.137)
        # σ = (k²/π)|I_PO + I_edge|²
        # J_PO=2n̂×H 的因子2 已吸收进 k²/π 系数，
        # EEW 边缘积分与 PO 共享相同归一化，无需额外 /2
        pre_factor = -1j * sin_gamma0 / k

        seg_contrib = pre_factor * D * seg.length * sinc_val * np.exp(1j * phase_mid)
        total_contrib += seg_contrib

    return total_contrib
