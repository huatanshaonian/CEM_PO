"""
PTD 边缘积分核心

将边缘分段，每段：
  1. 提取楔形截面内的入射角 angle0 和观察角 angle
  2. 调用 fun_fg 得到 Ufimtsev 修正系数 (f1, g1)
  3. 按极化选取系数，乘以传播因子，累加到总贡献
"""
import numpy as np
from .ptd_coefficients import fun_fg, FG_monostatic

_SING_THRESH = 1e-3   # 奇点邻域阈值（弧度，约 0.057°）
_SING_OFFSET = 3e-4   # 奇点两侧偏移量（弧度）


def compute_ptd_contribution(edge, wave, polarization='VV'):
    """
    计算整条边缘的 PTD 修正量（对所有分段求和）。

    参数:
        edge:         PTDEdge 对象
        wave:         IncidentWave 对象
        polarization: 'VV'（Hard/TM，使用 g1）或 'HH'（Soft/TE，使用 f1）
    返回:
        complex: 该边缘对散射场积分 I 的贡献量
    """
    total_contrib = 0j

    k_dir = wave.k_dir      # 入射方向单位向量（指向目标）
    k_vec = wave.k_vector   # k·k_dir
    k = wave.k
    s_dir = -k_dir          # 单站模式：散射方向 = 入射反向

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

        # ── 2. 建立楔形截面局部坐标系 ──
        # 基向量 e1：Face A 的外法向（n_lit），对应楔形坐标 φ = π/2
        # 基向量 e2：沿 Face A 表面、从棱边向外（φ = 0 方向）
        e1 = n_lit
        e2_raw = np.cross(t, n_lit)
        e2_len = np.linalg.norm(e2_raw)
        if e2_len < 1e-10:
            continue
        e2 = e2_raw / e2_len

        # 消除 e2 符号歧义：e2 应沿 Face A 表面远离 Face B
        # 正确的 e2 满足 dot(e2, n_b) ≤ 0（对 α ∈ (π, 2π) 恒成立）
        n_b = seg.normal_b if hasattr(seg, 'normal_b') else None
        if n_b is not None and np.dot(e2, n_b) > 0:
            e2 = -e2

        # ── 3. 计算入射方向和散射方向在截面内的投影 ──
        # 入射方向在截面平面（垂直于 t）的分量
        k_perp = k_dir - k_dot_t * t
        k_perp_len = np.linalg.norm(k_perp)
        if k_perp_len < 1e-10:
            continue
        k_perp_unit = k_perp / k_perp_len

        # angle0: 入射来源方向在截面内相对于 Face A 表面（e2 方向）的角度
        # 使用来源方向 -k_dir（从目标指向源），使 φ=0 对应沿 Face A 掠射
        inc_unit = -k_perp_unit            # 入射来源方向（从目标指向源）
        i_e1 = np.dot(inc_unit, e1)        # n_lit 分量
        i_e2 = np.dot(inc_unit, e2)        # Face A 切向分量
        angle0_raw = np.arctan2(i_e1, i_e2)  # atan2 → [−π, π]
        # atan2 对 α > π 的楔形，合法角 (π, α] 映射为负值，需展开到 [0, 2π)
        if angle0_raw < -1e-6:
            angle0_raw += 2.0 * np.pi
        if angle0_raw < -1e-6 or angle0_raw > alfa + 1e-6:
            continue   # 入射方向来自楔形内部（材料侧），跳过该段
        angle0 = np.clip(angle0_raw, 0.0, alfa)

        # ── 3b. 散射方向投影到截面局部坐标，求观察角 angle_obs ──
        # 散射方向 s_dir 先去除沿棱边的分量，得到截面内分量，
        # 再投影到 (e2, e1) 坐标系，方式与 angle0 完全对称。
        # 单站：s_dir = -k_dir，截面投影等同 inc_unit，angle_obs = angle0。
        # 双站：s_dir 为独立的散射观察方向，需由调用方传入（当前未实现）。
        s_dot_t = np.dot(s_dir, t)
        s_perp = s_dir - s_dot_t * t
        s_perp_len = np.linalg.norm(s_perp)
        if s_perp_len < 1e-10:
            continue
        s_perp_unit = s_perp / s_perp_len

        s_e1 = np.dot(s_perp_unit, e1)
        s_e2 = np.dot(s_perp_unit, e2)
        angle_raw = np.arctan2(s_e1, s_e2)

        # 双站观察角阴影处理（暂未使用，开发双站时参考）：
        #
        # angle_raw ∈ [0, α]：观察点在楔形亮区，当前 PTD fringe 公式适用。
        #
        # angle_raw 在 [0, α] 之外：观察点在某个面的几何阴影区（仍是自由空间）。
        # 此时衍射场是"阴影形成波"，与入射场等幅反相，使总场连续过渡到零。
        # PTD fringe 在阴影边界（angle_raw ≈ 0 或 ≈ α）处发散，需要 UTD Fresnel
        # 过渡函数正则化；深阴影区（远离边界）fringe 贡献趋于零，主要由 PO=0
        # 和完整衍射系数 D_total 提供场值。
        # 因此双站实现必须升级为 UTD（含 Fresnel 函数），不能直接 skip。
        #
        # if angle_raw < -1e-6 or angle_raw > alfa + 1e-6:
        #     # TODO: 双站阴影区处理（UTD Fresnel 过渡）
        #     pass
        # angle_obs = np.clip(angle_raw, 0.0, alfa)

        # 与 angle0 同样的展开处理
        if angle_raw < -1e-6:
            angle_raw += 2.0 * np.pi
        angle_obs = np.clip(angle_raw, 0.0, alfa)

        # ── 4. 计算衍射系数 D ──
        gamma0 = np.arcsin(sin_gamma0)
        D = _compute_D(angle_obs, angle0, gamma0, alfa, polarization)

        # ── 5. 边缘段积分 ──
        # sinc 项：考虑棱边延伸方向造成的干涉
        sinc_arg = k * seg.length * k_dot_t / np.pi
        sinc_val = np.sinc(sinc_arg)   # numpy.sinc 已含 π 归一化

        # 单站相位：exp(i·2k·r_mid)
        phase_mid = 2.0 * np.dot(seg.midpoint, k_vec)

        # 传播因子：含 1/sin(γ₀) 斜入射修正
        # 2D→3D SPA 反演：A = D/(2π)
        # 3D 微元场: dE_s/E_i = [D/(2π sinγ₀)] · (e^{-jkR}/R) · e^{-j·phase} · dζ
        # 匹配代码约定 E_s/E_i = (jk/2πR)e^{-jkR}·I → dI = (-j/k sinγ₀)·D·e^{-j·phase}·dζ
        # 代码 PO 用 e^{+j·phase} (共轭), PTD 取共轭对齐 → pre_factor = j/(k sinγ₀)
        # 验证: σ = (k²/π)|pre·D·L|² = |D|²L²/(π sin²γ₀) (Keller GTD) ✓
        pre_factor = 1j / (k * sin_gamma0)

        seg_contrib = pre_factor * D * seg.length * sinc_val * np.exp(1j * phase_mid)
        total_contrib += seg_contrib

    return total_contrib


def _compute_D(angle, angle0, gamma0, alfa, polarization):
    """
    用 FG_monostatic 计算衍射系数 D，含斜入射耦合项。

    奇点处（ψ₁≈π 或 ψ₂≈π）用两侧偏移平均处理数值稳定性。
    """
    psi1 = angle - angle0
    psi2 = angle + angle0
    near_sing = (abs(psi1 % (2 * np.pi) - np.pi) < _SING_THRESH or
                 abs(psi2 % (2 * np.pi) - np.pi) < _SING_THRESH or
                 abs(2 * alfa - psi2 - np.pi) < _SING_THRESH)

    if near_sing:
        F1a, G1Va, G1pa = FG_monostatic(angle - _SING_OFFSET, angle0, gamma0, alfa)
        F1b, G1Vb, G1pb = FG_monostatic(angle + _SING_OFFSET, angle0, gamma0, alfa)
        F1_Vt  = 0.5 * (F1a + F1b)
        G1_Vt  = 0.5 * (G1Va + G1Vb)
        G1_phi = 0.5 * (G1pa + G1pb)
    else:
        F1_Vt, G1_Vt, G1_phi = FG_monostatic(angle, angle0, gamma0, alfa)

    if polarization == 'HH':
        return -F1_Vt           # = f1 (soft / TE)
    else:
        return -G1_phi + G1_Vt  # = g1 + 斜入射耦合项 (hard / TM, default 'VV')
