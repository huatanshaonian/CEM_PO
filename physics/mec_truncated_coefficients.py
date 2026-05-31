"""
Johansen 1996 截断 MEC 修正等效边电流 M_cor, I_cor (半平面 N=2 闭式)

参考:
    P. M. Johansen, "Uniform Physical Theory of Diffraction Equivalent Edge
    Currents for Truncated Wedge Strips", IEEE TAP, vol.44, no.7, July 1996.

实现范围:
    - 半平面 (N = 2, 外角 α = 2π) — 即薄板刀刃边 / 各种 strip
    - 双站通式 (β_0, φ_0 入射;β, φ 观察) — 不预先做单站化简, 保留拓展性
    - 时谐约定: 本函数返回 **Johansen 原文 e^{+jωt}** 原生值 (含 exp(+jL(μ-1))、
      原版 modified_fresnel)。调用方 (mec_truncated_core) 对返回值整支 np.conj
      即得 Ufimtsev e^{-iωt} 值, 与 conj 后的 Michaeli 非截断 EEC 同约定相减。
      (因 μ/L/β/φ 等参数全为实数, "求值后 conj" 与 "公式内每个 j→-i 全替换"
       逐项恒等; 见 mec_core 同款处理与 docs。)

核心思想 (Eq.3):
    M_T = M_UT - M_cor,  I_T = I_UT - I_cor
    M_UT, I_UT 是 Michaeli 1986 非截断 EEC (本仓库 mec_coefficients.py)
    M_cor, I_cor 是 Johansen 推的"虚源尾部"修正

Ufimtsev 奇点 (φ_0=π, μ=1) 在 M_UT 和 M_cor 各自发散但**精确抵消**, 由减法处理.
Johansen 1996 Section III.B/IV-C 证明.

公式 (Johansen 1996 Eq.26, 27, 半平面 N=2, 面 A+B 已合并):
    L  = k · l^A · sin²β_0                                    (截断参量)
    μ  = (sinβ·sinβ_0·cosφ + cosβ_0·(cosβ-cosβ_0)) / sin²β_0   (Michaeli, Eq.6)
    F  = modified_fresnel  (physics.uniform_transition)

    M_cor = [4·Z·H_z0·sinφ·exp(jL(μ-1))] / [jk·sinβ·sinβ_0·(μ+cosφ_0)]
            × { -sign(cos(φ_0/2))·F(√(2L)|cos(φ_0/2)|)
                + √2·cos(φ_0/2)/√(1-μ)·F(√(L(1-μ))) }

    I_cor = E_z0 项 + H_z0 项
        E_z0 项 = [4·E_z0·sin(φ_0/2)·exp(jL(μ-1))] / [jkZ·sin²β_0·(μ+cosφ_0)]
                × { 2·cos(φ_0/2)·F(√(2L)|cos(φ_0/2)|)
                    - √(2(1-μ))·F(√(L(1-μ))) }
        H_z0 项 = [4·H_z0·exp(jL(μ-1))] / [jk·sinβ_0·(μ+cosφ_0)]
                × { -sign(cos(φ_0/2))·(cotβ_0·cosφ_0+cotβ·cosφ)·F(√(2L)|cos(φ_0/2)|)
                    + √2·cos(φ_0/2)/√(1-μ)·(cotβ·cosφ-μ·cotβ_0)·F(√(L(1-μ))) }
"""
import numpy as np
from .uniform_transition import modified_fresnel


_DENOM_EPS = 1e-9    # 防 (μ+cosφ_0) → 0 数值除零 (Ufimtsev 奇点附近)
_MU1_EPS = 1e-9      # 防 sqrt(1-μ) → 0 数值除零 (μ→1, 即 Keller 锥反射边界)


def compute_correction_currents(beta_0, phi_0, beta, phi, l_A,
                                 E0z, H0z, k, Z):
    """
    Johansen 1996 Eq.26/27 半平面 (N=2) 修正等效边电流 (面 A+B 合计).

    参数:
        beta_0:  入射方向与边切线夹角 (弧度), 0 < β_0 < π
        phi_0:   入射方位角 (Michaeli Fig.1 局部坐标系内, 弧度)
        beta:    观察方向与边切线夹角 (弧度), 0 < β < π
        phi:     观察方位角 (弧度)
        l_A:     截断长度 (米) — 沿 û^A (Keller 锥与面 A 的交) 从当前边到 trailing edge
        E0z:     入射电场沿边切线分量 ê_t·t̂ (复数, 单位幅值的入射波约定)
        H0z:     入射磁场沿边切线分量 (k̂×ê_t)·t̂ / Z (复数)
        k:       波数 (rad/m)
        Z:       介质波阻抗 (Ω)

    返回:
        (M_cor, I_cor): 复数标量, 修正等效边电流的"待减去"分量
                        最终截断 EEC = Michaeli 非截断 - 本函数返回值

    数值边界:
        l_A → 0:         L → 0, F(0)=0.5 → M_cor → M_UT, I_cor → I_UT (Johansen p.993)
                          故 M_T = I_T = 0 (零长度边无电流, 物理正确)
        μ > 1 (锥外):     √(1-μ) 复数, modified_fresnel 自动延拓
        μ + cosφ_0 → 0:   Ufimtsev 奇点, M_cor 与 M_UT 各发散; 由减法层 (调用者) 抵消
                          本函数仅在 |分母| < _DENOM_EPS 时把 M_cor/I_cor 置零,
                          让上层退化到非截断 (Johansen 在 §III.B 用 Taylor 展开严格处理,
                          首版用简化截断, 后续如需可在此处加 ε-展开)
    """
    # ---- 1. 基础几何量 ----
    sb0 = np.sin(beta_0)
    sb = np.sin(beta)
    if abs(sb0) < 1e-12:
        return 0.0 + 0.0j, 0.0 + 0.0j

    cos_phi0 = np.cos(phi_0)
    cos_phi = np.cos(phi)

    # Michaeli 参数 μ (Eq.6) — 双站完整形式
    mu = (sb * sb0 * cos_phi + np.cos(beta_0) * (np.cos(beta) - np.cos(beta_0))) / (sb0 * sb0)

    # ---- 2. Ufimtsev 奇点检测: μ + cosφ_0 ≈ 0 ----
    denom = mu + cos_phi0
    if abs(denom) < _DENOM_EPS:
        # 奇点: M_cor/I_cor 与 M_UT/I_UT 各发散; 让上层用 M_UT/I_UT (即返 0)
        return 0.0 + 0.0j, 0.0 + 0.0j

    # ---- 3. 截断参量 L = k·l^A·sin²β_0 ----
    L = k * l_A * sb0 * sb0

    # ---- 4. Fresnel 参量 ----
    cos_half = np.cos(phi_0 / 2.0)
    abs_cos_half = abs(cos_half)
    if cos_half > 0:
        sign_cos_half = 1.0
    elif cos_half < 0:
        sign_cos_half = -1.0
    else:
        sign_cos_half = 0.0  # φ_0 = π 时 cos(π/2)=0

    # F 的两个参量
    F1_arg = np.sqrt(2.0 * L) * abs_cos_half          # 实数 ≥ 0
    # √(1-μ): μ ≤ 1 时为实数, μ > 1 时纯虚 (Johansen 接 modified_fresnel 自动延拓)
    # μ → 1 时分母 → 0; 此时 cos(φ_0/2) 项的系数 √2·cos(φ_0/2)/√(1-μ) 极限存在
    # (Johansen §III.B), 但直接 floating-point 求值给 NaN. 用 ε-floor 防数值崩.
    one_minus_mu = 1.0 - mu
    if abs(one_minus_mu) < _MU1_EPS:
        # μ ≈ 1 (Keller 锥反射边界): 系数有限极限, 但数值上分母太小. 退回非截断.
        return 0.0 + 0.0j, 0.0 + 0.0j
    sqrt_1mmu = np.lib.scimath.sqrt(one_minus_mu)    # 处理负数 → 纯虚
    F2_arg = np.lib.scimath.sqrt(L * one_minus_mu)

    F1 = modified_fresnel(F1_arg)
    F2 = modified_fresnel(F2_arg)

    # ---- 5. 公共指数因子 exp(jL(μ-1)) ----
    # μ > 1 时 L(μ-1) > 0, 是真实相位; μ < 1 时 L(μ-1) < 0
    exp_factor = np.exp(1j * L * (mu - 1.0))

    # ---- 6. M_cor (Eq.26) ----
    M_pre = (4.0 * Z * H0z * np.sin(phi) * exp_factor) / (1j * k * sb * sb0 * denom)
    M_bracket = (-sign_cos_half * F1
                 + (np.sqrt(2.0) * cos_half / sqrt_1mmu) * F2)
    M_cor = M_pre * M_bracket

    # ---- 7. I_cor (Eq.27) E_z0 项 ----
    I_E_pre = (4.0 * E0z * np.sin(phi_0 / 2.0) * exp_factor) / (1j * k * Z * sb0 * sb0 * denom)
    I_E_bracket = (2.0 * cos_half * F1
                   - np.lib.scimath.sqrt(2.0 * one_minus_mu) * F2)
    I_E = I_E_pre * I_E_bracket

    # ---- 8. I_cor (Eq.27) H_z0 项 ----
    # cot β = cosβ/sinβ; 这里 sin 已 >0 (前面检 sb0, sb), cosβ 可正可负
    cot_b0 = np.cos(beta_0) / sb0
    cot_b = np.cos(beta) / sb

    coef1_H = -sign_cos_half * (cot_b0 * cos_phi0 + cot_b * cos_phi)
    coef2_H = (np.sqrt(2.0) * cos_half / sqrt_1mmu) * (cot_b * cos_phi - mu * cot_b0)

    I_H_pre = (4.0 * H0z * exp_factor) / (1j * k * sb0 * denom)
    I_H_bracket = coef1_H * F1 + coef2_H * F2
    I_H = I_H_pre * I_H_bracket

    I_cor = I_E + I_H

    return complex(M_cor), complex(I_cor)
