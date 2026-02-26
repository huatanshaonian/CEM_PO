"""
PTD 边缘积分核心

将边缘分段，每段：
  1. 提取楔形截面内的入射角 angle0 和观察角 angle
  2. 调用 fun_fg 得到 Ufimtsev 修正系数 (f1, g1)
  3. 按极化选取系数，乘以传播因子，累加到总贡献
"""
import numpy as np
from .ptd_coefficients import fun_fg

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

    alfa = edge.alpha       # 楔形外角 α = n·π

    for seg in edge.segments:
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
        # 基向量 e1：Face 1 的外法向（n_lit），对应楔形坐标 φ = π/2
        # 基向量 e2：t × n_lit，沿 Face 1 表面（φ = 0 方向）
        e1 = n_lit
        e2_raw = np.cross(t, n_lit)
        e2_len = np.linalg.norm(e2_raw)
        if e2_len < 1e-10:
            continue
        e2 = e2_raw / e2_len

        # ── 3. 计算入射方向和散射方向在截面内的投影 ──
        # 入射方向在截面平面（垂直于 t）的分量
        k_perp = k_dir - k_dot_t * t
        k_perp_len = np.linalg.norm(k_perp)
        if k_perp_len < 1e-10:
            continue
        k_perp_unit = k_perp / k_perp_len

        # angle0: 入射来源方向在截面内相对于 Face 1（e2 方向）的角度
        # 注意：使用来源方向 -k_dir（而非传播方向 k_dir），使 φ=0 对应沿 Face 1 入射
        inc_unit = -k_perp_unit            # 入射来源方向（从目标指向源）
        i_e1 = np.dot(inc_unit, e1)        # n_lit 分量
        i_e2 = np.dot(inc_unit, e2)        # Face 1 切向分量
        angle0_raw = np.arctan2(i_e1, i_e2)  # atan2(e1分量, e2分量) → [−π, π]
        angle0 = angle0_raw % alfa            # 映射到 [0, α)

        # 单站模式：散射方向 = 入射来源方向（波原路返回）
        # s_dir = -k_dir，截面投影同 inc_unit
        s_dot_t = np.dot(s_dir, t)
        s_perp = s_dir - s_dot_t * t
        s_perp_len = np.linalg.norm(s_perp)
        if s_perp_len < 1e-10:
            continue
        s_perp_unit = s_perp / s_perp_len

        s_e1 = np.dot(s_perp_unit, e1)
        s_e2 = np.dot(s_perp_unit, e2)
        angle_raw = np.arctan2(s_e1, s_e2)
        angle_obs = angle_raw % alfa           # 映射到 [0, α)

        # ── 4. 计算衍射系数 D ──
        D = _compute_D(angle_obs, angle0, alfa, polarization)

        # ── 5. 边缘段积分 ──
        # sinc 项：考虑棱边延伸方向造成的干涉
        sinc_arg = k * seg.length * k_dot_t / np.pi
        sinc_val = np.sinc(sinc_arg)   # numpy.sinc 已含 π 归一化

        # 单站相位：exp(i·2k·r_mid)
        phase_mid = 2.0 * np.dot(seg.midpoint, k_vec)

        # 传播因子（与 PO 积分保持相同量纲约定）
        pre_factor = (2.0 * np.pi) / (1j * k)

        seg_contrib = pre_factor * D * seg.length * sinc_val * np.exp(1j * phase_mid)
        total_contrib += seg_contrib

    return total_contrib


def _compute_D(angle, angle0, alfa, polarization):
    """
    从 fun_fg 获取 f1/g1，按极化选取衍射系数 D。

    奇点处（angle 或 angle0 接近 0/α，或 angle ≈ angle0 → ψ₁≈0）
    用两侧偏移平均处理数值稳定性。
    """
    # 检查是否需要奇点平滑
    psi1 = angle - angle0
    psi2 = angle + angle0
    near_sing = (abs(psi1 % (2 * np.pi) - np.pi) < _SING_THRESH or
                 abs(psi2 % (2 * np.pi) - np.pi) < _SING_THRESH or
                 abs(2 * alfa - psi2 - np.pi) < _SING_THRESH)

    if near_sing:
        # 用两侧偏移的平均值平滑奇点
        f1a, g1a = fun_fg(angle - _SING_OFFSET, angle0, alfa)
        f1b, g1b = fun_fg(angle + _SING_OFFSET, angle0, alfa)
        f1 = 0.5 * (f1a + f1b)
        g1 = 0.5 * (g1a + g1b)
    else:
        f1, g1 = fun_fg(angle, angle0, alfa)

    if polarization == 'HH':
        return f1   # Soft boundary / TE
    else:
        return g1   # Hard boundary / TM  (default 'VV')
