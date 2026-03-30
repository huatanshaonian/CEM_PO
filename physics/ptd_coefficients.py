"""
PTD 衍射系数核心函数

直接对应乌菲姆采夫教材 MATLAB 代码：
  eps_x          ← eps_x.m   (Eq. 7.48)
  sigma12        ← sigma12.m (Eq. 7.76-7.77)
  fun_fg         ← fun_fg.m  (Eq. 4.20-4.21)
  FG_monostatic  ← FG.m      (Eq. 7.137, 单站路径)
"""
import numpy as np


_EPS_ANGLE = 1e-5  # 角度判断容差（弧度），与 MATLAB fun_fg.m 保持一致


def eps_x(x):
    """阶跃函数 Eq.(7.48)：0 < x ≤ π 时返回 1，否则返回 0"""
    return 1.0 if (x > 0) and (x <= np.pi) else 0.0


def sigma12(beta, gamma):
    """
    计算 σ₁₂(β, γ)，对应 Eq.(7.76) 和 (7.77)。

    参数:
        beta:  β ∈ [0, π]
        gamma: γ ∈ [0, π]
    返回:
        实数（β ≤ β_k）或复数（β > β_k）
    """
    if 0 <= gamma <= np.pi / 2:
        betak = 2.0 * gamma
    else:
        betak = 2.0 * (np.pi - gamma)

    if 0 <= beta <= betak:
        # Eq.(7.76): 实数分支
        sin2g = np.sin(gamma) ** 2
        arg = np.clip((np.cos(beta) - np.cos(gamma) ** 2) / sin2g, -1.0, 1.0)
        return np.pi - np.arccos(arg)
    else:
        # Eq.(7.77): 复数分支
        # MATLAB: 1i*log(cos(gamma)^2 - cos(beta) + sqrt(...)) - 1i*2*log(sin(gamma))
        cos2g = np.cos(gamma) ** 2
        sin_g = np.sin(gamma)
        delta = np.cos(beta) - cos2g
        under_sqrt = delta ** 2 - sin_g ** 4
        # 数值保护：under_sqrt 可能因浮点误差略负
        under_sqrt = max(under_sqrt, 0.0)
        numerator = cos2g - np.cos(beta) + np.sqrt(under_sqrt)
        return 1j * np.log(numerator) - 1j * 2.0 * np.log(sin_g)


def fun_fg(angle, angle0, alfa):
    """
    计算 Ufimtsev PTD 修正衍射系数 f1, g1，对应 Eq.(4.20) 和 (4.21)。

    等同于 MATLAB fun_fg.m。

    参数:
        angle:  观察角 φ（弧度），相对于楔形 Face 1，∈ (0, α)
        angle0: 入射角 φ₀（弧度），相对于楔形 Face 1，∈ (0, α-π) 或 (α-π, π)
        alfa:   楔形外角 α = n·π（弧度），对应 MATLAB 中的 alfa
    返回:
        (f1, g1): 软边界(TE)和硬边界(TM)修正系数（实数或浮点数）
    """
    eps = _EPS_ANGLE
    n = alfa / np.pi
    psi1 = angle - angle0
    psi2 = angle + angle0

    # 四个中间角度（对应 MATLAB angle1..angle4）
    a1 = (np.pi - psi2) / (2.0 * n)
    a2 = (np.pi - psi1) / (2.0 * n)
    a3 = (np.pi + psi2) / (2.0 * n)
    a4 = (np.pi + psi1) / (2.0 * n)

    if (angle0 >= eps) and (angle0 <= alfa - np.pi - eps):
        # ──── SSI: 单侧照射 ────
        if (psi1 <= np.pi + eps) and (psi1 >= np.pi - eps):
            # psi1 ≈ π：第 1、2 项奇异，用解析极限替代
            f1 = (1.0 / (2.0 * n)) * (_cot(a1) + _cot(a3) - _cot(a4)) \
                 - 0.5 * _cot((np.pi - psi2) / 2.0)
            g1 = -(1.0 / (2.0 * n)) * (_cot(a1) + _cot(a3) + _cot(a4)) \
                 + 0.5 * _cot((np.pi - psi2) / 2.0)
        elif (psi2 <= np.pi + eps) and (psi2 >= np.pi - eps):
            # psi2 ≈ π：第 3、4 项奇异
            f1 = (1.0 / (2.0 * n)) * (-_cot(a2) + _cot(a3) - _cot(a4)) \
                 + 0.5 * _cot((np.pi - psi1) / 2.0)
            g1 = -(1.0 / (2.0 * n)) * (_cot(a2) + _cot(a3) + _cot(a4)) \
                 + 0.5 * _cot((np.pi - psi1) / 2.0)
        else:
            # 一般情况：GTD 项减去 PO 项
            f = (1.0 / (2.0 * n)) * (_cot(a1) - _cot(a2) + _cot(a3) - _cot(a4))
            g = -(1.0 / (2.0 * n)) * (_cot(a1) + _cot(a2) + _cot(a3) + _cot(a4))
            f0 = 0.5 * (_cot((np.pi - psi2) / 2.0) - _cot((np.pi - psi1) / 2.0))
            g0 = -0.5 * (_cot((np.pi - psi2) / 2.0) + _cot((np.pi - psi1) / 2.0))
            f1 = f - f0
            g1 = g - g0

    elif (angle0 >= alfa - np.pi + eps) and (angle0 <= np.pi - eps):
        # ──── DSI: 双侧照射 ────
        if (psi2 <= np.pi + eps) and (psi2 >= np.pi - eps):
            # ψ₂≈π 奇点：cot(a1) 与 cot((π-ψ₂)/2) 均→∞，组合极限=0，移除两者
            f1 = (1.0 / (2.0 * n)) * (-_cot(a2) + _cot(a3) - _cot(a4)) \
                 - 0.5 * _cot((np.pi - 2.0 * alfa + psi2) / 2.0)
            g1 = -(1.0 / (2.0 * n)) * (_cot(a2) + _cot(a3) + _cot(a4)) \
                 + 0.5 * _cot((np.pi - 2.0 * alfa + psi2) / 2.0)
        elif (2.0 * alfa - psi2 <= np.pi + eps) and (2.0 * alfa - psi2 >= np.pi - eps):
            # 2α-ψ₂≈π 奇点：cot(a3) 与 cot((π-2α+ψ₂)/2) 均→∞，组合极限=0，移除两者
            f1 = (1.0 / (2.0 * n)) * (_cot(a1) - _cot(a2) - _cot(a4)) \
                 - 0.5 * _cot((np.pi - psi2) / 2.0)
            g1 = -(1.0 / (2.0 * n)) * (_cot(a1) + _cot(a2) + _cot(a4)) \
                 + 0.5 * _cot((np.pi - psi2) / 2.0)
        else:
            # 一般 DSI：GTD 项减去两个面的 PO 项
            # cot((π-ψ₁)/2) + cot((π+ψ₁)/2) ≡ 0，故 f0/g0 仅含 ψ₂ 相关的两项
            f = (1.0 / (2.0 * n)) * (_cot(a1) - _cot(a2) + _cot(a3) - _cot(a4))
            g = -(1.0 / (2.0 * n)) * (_cot(a1) + _cot(a2) + _cot(a3) + _cot(a4))
            f0 = 0.5 * (_cot((np.pi - psi2) / 2.0)
                        + _cot((np.pi - 2.0 * alfa + psi2) / 2.0))
            g0 = -0.5 * (_cot((np.pi - psi2) / 2.0)
                         + _cot((np.pi - 2.0 * alfa + psi2) / 2.0))
            f1 = f - f0
            g1 = g - g0
    elif (abs(angle0 - np.pi / 2.0) < eps) and (abs(angle - np.pi / 2.0) < eps):
        # ──── 特殊情况：angle0 = angle = π/2（MATLAB fun_fg.m 对应分支）────
        f1 = (1.0 / (2.0 * n)) * (-_cot(a2) + _cot(a3) - _cot(a4))
        g1 = -(1.0 / (2.0 * n)) * (_cot(a2) + _cot(a3) + _cot(a4))
    else:
        # angle0 在范围之外（掠入射或不合法）
        f1 = 0.0
        g1 = 0.0

    return float(np.real(f1)), float(np.real(g1))


def FG_monostatic(angle, angle0, gamma0, alfa):
    """
    Ufimtsev 完整 3D 衍射系数，单站情况 (Vtheta = pi - gamma0)。
    对应 FG.m Eq.(7.137) 的 Vtheta == pi-gamma0 分支。

    参数:
        angle:  观察角 φ（弧度），∈ (0, α)
        angle0: 入射角 φ₀（弧度），∈ (0, α)
        gamma0: 斜入射角（弧度），入射方向与棱边的夹角，∈ (0, π)
        alfa:   楔形外角 α（弧度）

    返回:
        F1_Vt:  soft (f1) 的 θ 分量 = -f1
        G1_Vt:  hard (g1) 的 θ 分量（极化耦合项，正入射时为零）
        G1_phi: hard (g1) 的 φ 分量 = -g1
    """
    f1, g1 = fun_fg(angle, angle0, alfa)
    sin_g = np.sin(gamma0)
    # 单站 Keller 锥：Vtheta = pi - gamma0
    F1_Vt  = -f1 / sin_g                   # Eq.(7.148)
    G1_Vt  = (eps_x(angle0) - eps_x(alfa - angle0)) * (np.cos(gamma0) / sin_g)  # Eq.(7.153)
    G1_phi = g1 / sin_g                    # Eq.(7.151)，注意正号
    return F1_Vt, G1_Vt, G1_phi


# ──────────────── 辅助函数 ────────────────

def _cot(x):
    """余切函数，含数值保护"""
    sx = np.sin(x)
    if abs(sx) < 1e-14:
        sx = 1e-14 * np.sign(sx) if sx != 0 else 1e-14
    return np.cos(x) / sx
