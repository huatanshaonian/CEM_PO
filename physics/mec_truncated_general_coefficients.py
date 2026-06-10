"""
Johansen 1996 通用 N 截断 MEC 修正等效边电流 M_cor^A, I_cor^A.

参考: Peter M. Johansen, "Uniform Physical Theory of Diffraction Equivalent
Edge Currents for Truncated Wedge Strips," IEEE TAP, vol.44, no.7, pp.989-995,
July 1996. 直接 PDF 对照 (Eq.21/22 from page 992 / 993).

实现范围:
    - 任意外角参数 N (论文假设 1 < N <= 2)
    - 任意双站方位 (β_0, φ_0; β, φ 独立)
    - **Face A only** —— Face B 由调用方通过坐标替换得到:
        {β_0 → π-β_0, β → π-β, φ_0 → Nπ-φ_0, φ → Nπ-φ, l^A → l^B}
    - 时谐约定: Johansen 原文 e^{+jωt} (虚单位 j).
      项目其他模块用 e^{-iωt}; 调用方负责对返回值整支 conj 完成约定翻译.

核心公式 (Johansen Eq.6, 17, 21, 22; L 在 Eq.19 上下文定义):
    L     = k · l^A · sin²β_0
    μ     = (sin β sin β_0 cos φ + cos β_0 (cos β - cos β_0)) / sin²β_0       (Eq.6)
    F(z)  = sqrt(j/π) · exp(j·z²) · ∫_z^∞ exp(-j·t²) dt   (modified Fresnel)  (Eq.17)

    M_cor^A:  (Eq.21)
        前因子 = (2 Z H_z0 sin φ exp(jL(μ-1))) / (jk sin β sin β_0)
        F1 项 = (-sign(cos(φ_0/2)) / (μ + cos φ_0)) · F(sqrt(2L)|cos(φ_0/2)|)
        F2 项 = (sqrt(1-μ) / (sqrt(2) (μ + cos φ_0) cos(φ_0/2))
                - sqrt(2) sin(π/N) / (N sqrt(1-μ) (cos(π/N) - cos(φ_0/N)))) · F(sqrt(L(1-μ)))
        M_cor^A = 前因子 · (F1 项 + F2 项)

    I_cor^A:  (Eq.22)
        前因子 = (2 exp(jL(μ-1))) / (jk sin β_0 (μ + cos φ_0))
        F1 项 = sign(cos(φ_0/2)) · ((E_z0 sin φ_0)/(Z sin β_0)
                                     - H_z0 (cot β_0 cos φ_0 + cot β cos φ))
                · F(sqrt(2L)|cos(φ_0/2)|)
        F2 项 = sqrt(2(1-μ)) ·
                 (-(E_z0 sin(φ_0/2))/(Z sin β_0)
                  + H_z0 / (2 cos(φ_0/2)) · (cot β_0 cos φ_0 + cot β cos φ)
                  + (H_z0 sin(π/N) (μ + cos φ_0) (cot β_0 - cot β cos φ))
                    / (N (cos(π/N) - cos(φ_0/N)) (1-μ)))
                · F(sqrt(L(1-μ)))
        I_cor^A = 前因子 · (F1 项 + F2 项)

N=2 化简 (Eq.26, 27): 把 sin(π/N)=1, cos(π/N)=0 代入 + Face A+B 求和后即得.
本文件实现 Eq.21/22 (general N, Face A only); N=2 自动正确, 不需要特殊分支.

奇点处理:
    - μ + cos φ_0 → 0  (Ufimtsev 奇点)   : 返回 (0, 0), 让 M_T ← M_UT 退化
    - 1 - μ → 0       (Keller 锥反射边界): 返回 (0, 0), 同上
    - cos(π/N) - cos(φ_0/N) → 0 (Keller related): 返回 (0, 0)
    论文 Section III-B 证明 M_UT 在这些点有同型奇点, 减法后有限. 数值上
    let M_cor=0 让 M_T 退回 M_UT, 在远离奇点的角度恢复 (近奇点 ~1° 内有偏差).
"""
import numpy as np

from .uniform_transition import modified_fresnel

_DENOM_EPS = 1e-9     # 奇点判定阈值
_MU1_EPS = 1e-9
_DENOM_E_EPS = 1e-9
_SING_OFFSET = 3e-5   # 奇点 ±偏移量, 与 mec_coefficients._SING_OFFSET 一致以保证
                      # M_UT 与 M_cor 在奇点处的 1/(μ+cosφ_0) 等分量"相同方式抵消"


def _mu_from_angles(beta_0, phi_0, beta, phi):
    sb0 = np.sin(beta_0); sb = np.sin(beta)
    if abs(sb0) < 1e-12:
        return 0.0
    return (sb * sb0 * np.cos(phi) + np.cos(beta_0) * (np.cos(beta) - np.cos(beta_0))) / (sb0 * sb0)


def _is_singular(beta_0, phi_0, beta, phi, N):
    """与 mec_coefficients._is_singular_bistatic 同口径: 检测 3 类奇点
    (Ufimtsev μ+cosφ_0=0; Keller 锥反射 μ=1; sub-Keller cos(π/N)=cos(φ_0/N))."""
    mu = _mu_from_angles(beta_0, phi_0, beta, phi)
    if abs(np.cos(phi_0) + mu) < _DENOM_EPS:
        return True
    if abs(1.0 - mu) < _MU1_EPS:
        return True
    cos_piN = np.cos(np.pi / N)
    cos_p0N = np.cos(phi_0 / N)
    if abs(cos_piN - cos_p0N) < _DENOM_E_EPS:
        return True
    return False


def compute_correction_face_A_general(beta_0, phi_0, beta, phi, N, l_A,
                                       E0z, H0z, k, Z):
    """
    Johansen Eq.21/22: Face A 的截断修正 (M_cor^A, I_cor^A), 任意 N.

    参数:
        beta_0, phi_0: 入射方位 (Michaeli 局部坐标, rad)
        beta, phi    : 观察方位 (同坐标系, rad)
        N            : 外角参数 (1 < N <= 2)
        l_A          : Face A 上从棱边到 trailing edge 的距离 (m)
        E0z, H0z     : 入射场切向投影 (复数, ê_t·t̂ 与 (k̂×ê_t)·t̂/Z)
        k, Z         : 波数 (rad/m), 介质波阻抗 (Ω)

    返回:
        (M_cor_A, I_cor_A): 复数标量, Johansen 原文 e^{+jωt} 数值,
                            调用方需 conj 翻到 e^{-iωt}.

    奇点处理 (与 mec_coefficients 同口径):
        在 3 类奇点附近 (Ufimtsev / Keller 锥 / sub-Keller), 沿 φ 做 ±_SING_OFFSET
        偏移并取均值. 与 M_UT 共用同一 _SING_OFFSET 是关键: 这样 1/(μ+cosφ_0) 等
        发散项在两侧反号 → 平均后 → 0, M_UT 和 M_cor 各自消掉相同的发散贡献,
        二者的有限残量在 M_T = M_UT - M_cor 中正确相减.
    """
    if _is_singular(beta_0, phi_0, beta, phi, N):
        # 平均 ±_SING_OFFSET 处的值, 与 M_UT 的奇点保护同口径
        Mc_m, Ic_m = _compute_correction_face_A_general_raw(
            beta_0, phi_0, beta, phi - _SING_OFFSET, N, l_A, E0z, H0z, k, Z)
        Mc_p, Ic_p = _compute_correction_face_A_general_raw(
            beta_0, phi_0, beta, phi + _SING_OFFSET, N, l_A, E0z, H0z, k, Z)
        return 0.5 * (Mc_m + Mc_p), 0.5 * (Ic_m + Ic_p)
    return _compute_correction_face_A_general_raw(
        beta_0, phi_0, beta, phi, N, l_A, E0z, H0z, k, Z)


def _compute_correction_face_A_general_raw(beta_0, phi_0, beta, phi, N, l_A,
                                            E0z, H0z, k, Z):
    """无奇点保护的内核, 直接按 Eq.21/22 算. 由 wrapper 处理奇点偏移."""
    sb0 = np.sin(beta_0)
    sb = np.sin(beta)
    if abs(sb0) < 1e-12 or abs(sb) < 1e-12:
        return 0.0 + 0.0j, 0.0 + 0.0j

    cb0 = np.cos(beta_0)
    cb = np.cos(beta)
    cot_b0 = cb0 / sb0
    cot_b = cb / sb

    cos_phi0 = np.cos(phi_0)
    sin_phi0 = np.sin(phi_0)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # μ (Eq.6)
    mu = (sb * sb0 * cos_phi + cb0 * (cb - cb0)) / (sb0 * sb0)

    # L = k l^A sin²β_0  (Eq.19/20 上下文)
    L = k * l_A * sb0 * sb0
    if L < 1e-12:
        return 0.0 + 0.0j, 0.0 + 0.0j   # 零截断, M_cor = M_UT, M_T = 0

    # cos(φ_0/2) 及其 sign / abs
    cos_half = np.cos(phi_0 / 2.0)
    abs_cos_half = abs(cos_half)
    sin_phi0_half = np.sin(phi_0 / 2.0)
    if cos_half > 0:
        sign_cos_half = 1.0
    elif cos_half < 0:
        sign_cos_half = -1.0
    else:
        sign_cos_half = 0.0  # φ_0 = π

    # ── 几何量 (奇点保护由 wrapper 经 ±_SING_OFFSET 平均处理) ──
    denom_a = mu + cos_phi0
    one_minus_mu = 1.0 - mu
    cos_piN = np.cos(np.pi / N)
    cos_p0N = np.cos(phi_0 / N)
    denom_e = cos_piN - cos_p0N
    sin_piN = np.sin(np.pi / N)

    # 硬保护: 极端 (one_minus_mu ≈ 0, denom_a ≈ 0, denom_e ≈ 0, cos_half ≈ 0) 退回 0
    # wrapper 已经做 ±SING_OFFSET 偏移, 但偏移后仍可能踩 0 (例如 μ = cos(φ_o)
    # 在 φ_o = 0 处 dμ/dφ = 0, 偏移不能避开). 此处给安全分母, 避免 NaN.
    _SAFE = 1e-15
    if abs(one_minus_mu) < _SAFE:
        one_minus_mu = _SAFE if one_minus_mu >= 0 else -_SAFE
    if abs(denom_a) < _SAFE:
        denom_a = _SAFE if denom_a >= 0 else -_SAFE
    if abs(denom_e) < _SAFE:
        denom_e = _SAFE if denom_e >= 0 else -_SAFE
    if abs(cos_half) < _SAFE:
        # φ_0 = π 是真实物理奇点, 不能 escape; 返 0 让 M_T 退化到 M_UT
        return 0.0 + 0.0j, 0.0 + 0.0j

    # sqrt(1-μ) 可能复 (μ > 1)
    sqrt_1mmu = np.lib.scimath.sqrt(one_minus_mu)

    # F 参量
    F1_arg = np.sqrt(2.0 * L) * abs_cos_half        # 实数 ≥ 0
    F2_arg = np.lib.scimath.sqrt(L * one_minus_mu)  # μ > 1 时纯虚

    F1 = modified_fresnel(F1_arg)
    F2 = modified_fresnel(F2_arg)

    # 公共指数因子 exp(jL(μ-1))
    exp_factor = np.exp(1j * L * (mu - 1.0))

    # ──────────────── Eq.21: M_cor^A ────────────────
    M_pre = (2.0 * Z * H0z * sin_phi * exp_factor) / (1j * k * sb * sb0)

    # F1 项: (-sign(cos(φ_0/2)) / (μ + cos φ_0)) · F1
    M_F1_coef = -sign_cos_half / denom_a

    # F2 项: (sqrt(1-μ) / (sqrt(2)(μ + cos φ_0) cos(φ_0/2))
    #         - sqrt(2) sin(π/N) / (N sqrt(1-μ) (cos(π/N) - cos(φ_0/N))))
    if abs(cos_half) < 1e-12:
        # φ_0 = π → cos(φ_0/2)=0, term1 → ∞. 退回 0 (M_UT 也有同型奇点).
        return 0.0 + 0.0j, 0.0 + 0.0j

    M_F2_term1 = sqrt_1mmu / (np.sqrt(2.0) * denom_a * cos_half)
    M_F2_term2 = np.sqrt(2.0) * sin_piN / (N * sqrt_1mmu * denom_e)
    M_F2_coef = M_F2_term1 - M_F2_term2

    M_cor_A = M_pre * (M_F1_coef * F1 + M_F2_coef * F2)

    # ──────────────── Eq.22: I_cor^A ────────────────
    I_pre = (2.0 * exp_factor) / (1j * k * sb0 * denom_a)

    # F1 项: sign(cos(φ_0/2)) · ((E_z0 sin φ_0)/(Z sin β_0)
    #                            - H_z0 (cot β_0 cos φ_0 + cot β cos φ)) · F1
    I_F1_bracket = (E0z * sin_phi0 / (Z * sb0)
                    - H0z * (cot_b0 * cos_phi0 + cot_b * cos_phi))
    I_F1_coef = sign_cos_half * I_F1_bracket

    # F2 项: sqrt(2(1-μ)) · (...) · F2
    sqrt_2_1mmu = np.lib.scimath.sqrt(2.0 * one_minus_mu)

    I_F2_term1 = -E0z * sin_phi0_half / (Z * sb0)
    I_F2_term2 = (H0z / (2.0 * cos_half)
                  * (cot_b0 * cos_phi0 + cot_b * cos_phi))
    I_F2_term3 = (H0z * sin_piN * denom_a * (cot_b0 - cot_b * cos_phi)
                  / (N * denom_e * one_minus_mu))
    I_F2_coef = sqrt_2_1mmu * (I_F2_term1 + I_F2_term2 + I_F2_term3)

    I_cor_A = I_pre * (I_F1_coef * F1 + I_F2_coef * F2)

    return complex(M_cor_A), complex(I_cor_A)


def compute_total_correction_general(beta_0, phi_0, beta, phi, N, l_A, l_B,
                                      E0z, H0z, k, Z):
    """
    Johansen Eq.2: 总截断修正 = Face A + Face B.
    Face B 由 Johansen 替换得到 {β_0→π-β_0, β→π-β, φ_0→Nπ-φ_0, φ→Nπ-φ, l^A→l^B}.

    返回 (M_cor, I_cor) = (M_cor^A + M_cor^B, I_cor^A + I_cor^B).
    奇点处自动 ±_SING_OFFSET 平均 (与 mec_coefficients 平滑配对).
    """
    Mc_A, Ic_A = compute_correction_face_A_general(
        beta_0, phi_0, beta, phi, N, l_A, E0z, H0z, k, Z)
    Mc_B, Ic_B = compute_correction_face_A_general(
        np.pi - beta_0, N * np.pi - phi_0,
        np.pi - beta,   N * np.pi - phi,
        N, l_B, E0z, H0z, k, Z)
    return Mc_A + Mc_B, Ic_A + Ic_B


def compute_total_correction_general_raw(beta_0, phi_0, beta, phi, N, l_A, l_B,
                                          E0z, H0z, k, Z):
    """
    "裸"版本: 直接调 _compute_correction_face_A_general_raw, 跳过 ±_SING_OFFSET 平均.

    与 compute_total_fringe_currents_bistatic_raw 配对使用:
        M_UT_raw 与 M_cor_raw 在奇点处发散项口径一致 (都是论文公式直接代入,
        无独立平滑), 差分 M_UT_raw - M_cor_raw 自动消去发散, 得到 Johansen
        论文 III-B 节证明的有限极限.
    """
    Mc_A, Ic_A = _compute_correction_face_A_general_raw(
        beta_0, phi_0, beta, phi, N, l_A, E0z, H0z, k, Z)
    Mc_B, Ic_B = _compute_correction_face_A_general_raw(
        np.pi - beta_0, N * np.pi - phi_0,
        np.pi - beta,   N * np.pi - phi,
        N, l_B, E0z, H0z, k, Z)
    return Mc_A + Mc_B, Ic_A + Ic_B
