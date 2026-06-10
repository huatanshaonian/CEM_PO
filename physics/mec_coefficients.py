"""
Michaeli MEC (1986 Part I) 等效边电流闭式系数

参考文献:
  A. Michaeli, "Elimination of Infinities in Equivalent Edge Currents,
  Part I: Fringe Current Components", IEEE TAP, 1986.

实现范围:
  ── 单站特化路径 (compute_face_currents / compute_total_fringe_currents) ──
    远场单站后向散射 (β = π-β', φ = φ', 论文 Eq. 26 化简后的 μ)
    保留作为快速分支, 与生产 mec_core / mec_truncated_core 链路无缝衔接。

  ── 双站通用路径 (compute_face_currents_bistatic / compute_total_fringe_currents_bistatic) ──
    任意 (β_0, φ_0; β, φ) 4 角输入, 走完整 Eq. 4-7 (I_1^PO, M_1^PO, I_1 total, M_1 total),
    fringe = I_1 - I_1^PO  和  M_1^f = M_1 - M_1^PO (Eq. 3).
    单站等价时 (β=π-β_0, φ=φ_0) 数值上等于单站特化路径 (已自检 < 1e-9, 见
    physics/bistatic_assembly.run_self_test)。

  共同约束:
    PEC; 仅 fringe 分量 (PO 表面积分由其他模块覆盖)。

时谐约定:
  Michaeli 1986 原文 e^{+jωt}; 本库统一 Ufimtsev e^{-iωt}, 由调用方
  (mec_core / mec_truncated_core / bistatic_assembly) 对返回值整支取 conj 完成
  约定翻译。本模块返回 Michaeli 原相量, 不做 conj。

单站特化核心 (Eq. 27, 28 + Eq. 26 化简后的 μ):
  μ_mono = cos φ' - 2 cot²β'
  α_c = arccos(μ)  (复数, μ 可能 |μ|>1)

  I_1^f = pre_E·[sin φ'·U/(cos φ'+μ) + (1/N)sin(φ'/N)/D_e] · E_0z
        + pre_H·[cot β'(μ+cos φ')/(-D_e)] · H_0z
  M_1^f = pre_M·[U/(cos φ'+μ) + sin((π-α_c)/N)/(sin α_c · N · D_e)] · H_0z

  其中 D_e = cos((π-α_c)/N) - cos(φ'/N)

双站通用核心 (Eq. 4-7 直接相减, 无算式合并以保正确性):
  cos γ = sin β_0 sin β cos φ + cos β_0 cos β             (Eq. 18)
  μ     = (cos γ - cos²β_0) / sin²β_0                      (Eq. 22)
  α     = arccos μ  (复数)                                  (Eq. 8)

  I_1^PO = (2j U(π-φ_0))/(k sinβ_0 (cosφ_0+μ)) ·
           [sinφ_0/(Z sinβ_0) · E_0z - (cotβ_0 cosφ_0 + cotβ cosφ) · H_0z]    (Eq. 4)
  M_1^PO = (-2j Z sinφ U(π-φ_0))/(k sinβ sinβ_0 (cosφ_0+μ)) · H_0z              (Eq. 5)

  I_1    = (2j/(k sinβ_0)) · (1/N)/(cos(φ_0/N) - cos((π-α)/N)) ·
           {sin(φ_0/N)/(Z sinβ_0) E_0z + sin((π-α)/N)/sinα · (μ cotβ_0 - cotβ cosφ) H_0z}
           - (2j cotβ_0)/(kN sinβ_0) · H_0z                                     (Eq. 6)
  M_1    = (2j Z sinφ)/(k sinβ_0 sinβ) · (1/N) sin((π-α)/N) cscα /
           (cos((π-α)/N) - cos(φ_0/N)) · H_0z                                    (Eq. 7)

  I_1^f = I_1 - I_1^PO,  M_1^f = M_1 - M_1^PO                                    (Eq. 3)

总电流 (Eq. 23 + Eq. 1 面 2 几何替换):
  单站特化: I_2^raw = compute_face_currents(π-β_0, Nπ-φ_0, ...)
  双站通用: I_2^raw = compute_face_currents_bistatic(π-β_0, Nπ-φ_0, π-β, Nπ-φ, ...)
  I^f = I_1^f + I_2^raw,  M^f = M_1^f + M_2^raw
"""
import numpy as np

_SB_MIN = 1e-3        # 边端入射阈值: sin β' 过小直接返回 0
_DENOM_EPS = 1e-9     # 分母奇点阈值
_SING_OFFSET = 3e-5   # 奇点邻域偏移量 (与 ptd_core.py 风格一致)


def _arccos_complex(mu):
    """复数 arccos: mu 可能 |mu|>1, 结果含虚部。"""
    mu_c = complex(mu)
    return -1j * np.log(mu_c + 1j * np.lib.scimath.sqrt(1.0 - mu_c * mu_c))


def _is_singular(phi_prime, mu, N):
    """检测三个潜在奇点: cos φ'+μ ≈ 0, D_e ≈ 0, sin α_c ≈ 0。"""
    cos_pp = np.cos(phi_prime)
    if abs(cos_pp + mu) < _DENOM_EPS:
        return True

    alpha_c = _arccos_complex(mu)
    if abs(np.sin(alpha_c)) < _DENOM_EPS:
        return True

    pa_N = (np.pi - alpha_c) / N
    pp_N = phi_prime / N
    if abs(np.cos(pa_N) - np.cos(pp_N)) < _DENOM_EPS:
        return True

    return False


def _compute_face_currents_raw(beta_prime, phi_prime, N, E0z, H0z, k, Z):
    """
    面 1 的 (I_1^f, M_1^f) 闭式计算 - 不含奇点保护。

    参数:
        beta_prime: 入射角 (与边切线夹角, 弧度), 0 < β' < π
        phi_prime:  入射方位角 (在面 1 局部坐标系内, 弧度), 0 ≤ φ' ≤ Nπ
        N:          外角参数, 外部楔角 = N·π (N > 1)
        E0z, H0z:   入射场切向投影 (复数), E·t̂ 与 H·t̂
        k:          波数 (rad/m)
        Z:          介质波阻抗 (Ω)

    返回:
        (If, Mf) - 复数标量
    """
    sb = np.sin(beta_prime)
    if abs(sb) < _SB_MIN:
        return 0.0 + 0.0j, 0.0 + 0.0j

    cot_b = np.cos(beta_prime) / sb
    cos_pp = np.cos(phi_prime)
    sin_pp = np.sin(phi_prime)

    # Eq. 26: 单站化简后的 μ
    mu = cos_pp - 2.0 * cot_b * cot_b

    # 复数 α_c = arccos(μ)
    alpha_c = _arccos_complex(mu)

    pa_N = (np.pi - alpha_c) / N
    pp_N = phi_prime / N
    cos_pa_N = np.cos(pa_N)
    sin_pa_N = np.sin(pa_N)
    cos_pp_N = np.cos(pp_N)
    sin_pp_N = np.sin(pp_N)
    sin_alpha_c = np.sin(alpha_c)

    # 单位阶跃: φ' < π 时面 1 被照亮
    U = 1.0 if phi_prime < np.pi else 0.0

    denom_a = cos_pp + mu                  # 第一类分母
    denom_e = cos_pa_N - cos_pp_N          # 第二类分母 (D_e)

    Y = 1.0 / Z

    # ---- I_1^f (Eq. 27) ----
    pre_E = (-2j * Y) / (k * sb * sb)
    bracket_E = sin_pp * U / denom_a + sin_pp_N / (N * denom_e)

    pre_H = (2j * sin_pa_N) / (N * k * sb * sin_alpha_c)
    # 原公式分母 cos(φ'/N) - cos((π-α_c)/N) = -denom_e
    coef_H = cot_b * (mu + cos_pp) / (-denom_e)

    If = pre_E * bracket_E * E0z + pre_H * coef_H * H0z

    # ---- M_1^f (Eq. 28) ----
    pre_M = (2j * Z * sin_pp) / (k * sb * sb)
    # 第二项原分母 cos(φ'/N) - cos((π-α_c)/N) = -denom_e
    # -(1/N) sin_pa_N csc(α_c) / (-denom_e) = sin_pa_N / (sin α_c · N · denom_e)
    bracket_M = U / denom_a + sin_pa_N / (sin_alpha_c * N * denom_e)

    Mf = pre_M * bracket_M * H0z

    return If, Mf


def compute_face_currents(beta_prime, phi_prime, N, E0z, H0z, k, Z):
    """
    面 1 的 (I_1^f, M_1^f), 含奇点邻域偏移取均值保护。

    奇点检测覆盖:
      - cos φ' + μ ≈ 0    (Eq. 27/28 第一项分母)
      - sin α_c ≈ 0        (μ → ±1, Ufimtsev 奇点延伸)
      - D_e ≈ 0            (Keller 锥相关)
    """
    sb = np.sin(beta_prime)
    if abs(sb) < _SB_MIN:
        return 0.0 + 0.0j, 0.0 + 0.0j

    cb = np.cos(beta_prime)
    cot2 = (cb * cb) / (sb * sb)
    mu = np.cos(phi_prime) - 2.0 * cot2

    if not _is_singular(phi_prime, mu, N):
        return _compute_face_currents_raw(beta_prime, phi_prime, N, E0z, H0z, k, Z)

    # 奇点附近: 对 φ' 做 ±_SING_OFFSET 偏移取均值
    if1, mf1 = _compute_face_currents_raw(
        beta_prime, phi_prime - _SING_OFFSET, N, E0z, H0z, k, Z)
    if2, mf2 = _compute_face_currents_raw(
        beta_prime, phi_prime + _SING_OFFSET, N, E0z, H0z, k, Z)
    return 0.5 * (if1 + if2), 0.5 * (mf1 + mf2)


def compute_total_fringe_currents(beta_prime, phi_prime, N, E0z, H0z, k, Z):
    """
    完整 Fringe 等效边电流: I^f = I_1·ẑ + I_2·(-ẑ) 沿全局 ẑ 的标量。

    论文 Eq. 1 对称替换 {1→2}: ẑ→-ẑ, β'→π-β', φ'→Nπ-φ'。
    面 2 自身坐标下的输入应为 (-E0z, -H0z)。但 If/Mf 关于 (E0z, H0z) 线性,
    故面 2 标量电流 I_2 = -I_1(π-β', Nπ-φ', E0z, H0z)。

    总电流沿全局 ẑ:
        I^f = I_1·1 + I_2·(-1) = I_1 - I_2 = I_1 - (-I_2_raw) = I_1 + I_2_raw
    其中 I_2_raw = compute_face_currents(π-β', Nπ-φ', E0z, H0z)。
    """
    If1, Mf1 = compute_face_currents(beta_prime, phi_prime, N, E0z, H0z, k, Z)
    If2_raw, Mf2_raw = compute_face_currents(
        np.pi - beta_prime, N * np.pi - phi_prime, N, E0z, H0z, k, Z)
    return If1 + If2_raw, Mf1 + Mf2_raw


# ════════════════════════════════════════════════════════════════════════════
# 双站通用路径 (Michaeli 1986 Eq. 4-7 完整形式)
# ════════════════════════════════════════════════════════════════════════════

def _mu_bistatic(beta_0, phi_0, beta, phi):
    """Eq. 18 + Eq. 22: 任意 (β_0,φ_0; β,φ) 的 μ。
    单站 β=π-β_0, φ=φ_0 时退化为 μ = cosφ_0 - 2cot²β_0 (Eq. 26)。"""
    sb0 = np.sin(beta_0)
    sb = np.sin(beta)
    cb0 = np.cos(beta_0)
    cb = np.cos(beta)
    cos_gamma = sb0 * sb * np.cos(phi) + cb0 * cb       # Eq. 18
    return (cos_gamma - cb0 * cb0) / (sb0 * sb0)        # Eq. 22


def _is_singular_bistatic(beta_0, phi_0, beta, phi, mu, N):
    """检测双站下的三个潜在奇点:
       (1) cosφ_0 + μ ≈ 0       — I_1^PO, M_1^PO 第一类分母 (Ufimtsev 奇点)
       (2) sin α ≈ 0            — μ → ±1, Keller 锥反射边界
       (3) cos((π-α)/N) - cos(φ_0/N) ≈ 0 — I_1, M_1 第二类分母 (Keller 关联)
       注意: 全部判定都用 φ_0, 不用 φ — 与单站特化保持口径一致。
    """
    if abs(np.cos(phi_0) + mu) < _DENOM_EPS:
        return True
    alpha_c = _arccos_complex(mu)
    if abs(np.sin(alpha_c)) < _DENOM_EPS:
        return True
    pa_N = (np.pi - alpha_c) / N
    pp_N = phi_0 / N
    if abs(np.cos(pa_N) - np.cos(pp_N)) < _DENOM_EPS:
        return True
    return False


def _compute_face_currents_bistatic_raw(beta_0, phi_0, beta, phi, N, E0z, H0z, k, Z):
    """
    面 1 的双站 (I_1^f, M_1^f) = (I_1, M_1) - (I_1^PO, M_1^PO).
    严格按 Michaeli 1986 Eq. 4-7 直接相减, 不做算式合并。

    返回 (If, Mf), 不含奇点保护 (由调用方 compute_face_currents_bistatic 处理)。
    """
    sb0 = np.sin(beta_0)
    sb  = np.sin(beta)
    if abs(sb0) < _SB_MIN or abs(sb) < _SB_MIN:
        return 0.0 + 0.0j, 0.0 + 0.0j

    cb0 = np.cos(beta_0)
    cb  = np.cos(beta)
    cot_b0 = cb0 / sb0
    cot_b  = cb  / sb
    cos_phi0 = np.cos(phi_0)
    sin_phi0 = np.sin(phi_0)
    cos_phi  = np.cos(phi)
    sin_phi  = np.sin(phi)

    # μ, α (Eq. 18 + 22 + 8)
    mu = _mu_bistatic(beta_0, phi_0, beta, phi)
    alpha_c = _arccos_complex(mu)

    pa_N = (np.pi - alpha_c) / N
    pp_N = phi_0 / N
    sin_pa_N = np.sin(pa_N)
    cos_pa_N = np.cos(pa_N)
    sin_pp_N = np.sin(pp_N)
    cos_pp_N = np.cos(pp_N)
    sin_alpha_c = np.sin(alpha_c)

    Y = 1.0 / Z
    U = 1.0 if phi_0 < np.pi else 0.0
    denom_a    = cos_phi0 + mu              # Eq. 4/5 第一类分母 (cos φ_0 + μ)
    denom_I_eq6 = cos_pp_N - cos_pa_N       # Eq. 6  分母: cos(φ_0/N) - cos((π-α)/N)
    denom_M_eq7 = cos_pa_N - cos_pp_N       # Eq. 7  分母: cos((π-α)/N) - cos(φ_0/N)  (= -denom_I_eq6)

    # 极硬保护: 仅防 exact-zero 引发的 ZeroDivisionError; 保留大值的奇点行为.
    _HARD_ZERO = 1e-300
    if abs(denom_a) < _HARD_ZERO:
        denom_a = _HARD_ZERO if denom_a.real >= 0 else -_HARD_ZERO if hasattr(denom_a, 'real') else _HARD_ZERO
    if abs(sin_alpha_c) < _HARD_ZERO:
        sin_alpha_c = _HARD_ZERO + 0j
    if abs(denom_I_eq6) < _HARD_ZERO:
        denom_I_eq6 = _HARD_ZERO + 0j
    if abs(denom_M_eq7) < _HARD_ZERO:
        denom_M_eq7 = _HARD_ZERO + 0j

    # ── Eq. 4: I_1^PO ──
    pre_PO_I = (2j * U) / (k * sb0 * denom_a)
    I1_PO = pre_PO_I * (sin_phi0 / (Z * sb0) * E0z
                        - (cot_b0 * cos_phi0 + cot_b * cos_phi) * H0z)

    # ── Eq. 5: M_1^PO ──
    M1_PO = (-2j * Z * sin_phi * U) / (k * sb * sb0 * denom_a) * H0z

    # ── Eq. 6: I_1 total ──
    # 关键: 分别处理 E_0z 和 H_0z 项, 避免 IEEE 754 中 0·inf = NaN.
    # 当 H_0z = 0 (例如 TE 极化 E沿z 时) 但 sin α = 0 (μ=1), 数学上 H_0z 项 = 0,
    # 但浮点 (sin_pa_N/sin_alpha_c) * 0 * 0 = inf · 0 · 0 = NaN.
    pre_I = (2j / (k * sb0)) * (1.0 / N) / denom_I_eq6
    bracket_I_E = sin_pp_N / (Z * sb0) * E0z
    if abs(H0z) > 1e-30:
        H_factor = mu * cot_b0 - cot_b * cos_phi
        bracket_I_H = sin_pa_N / sin_alpha_c * H_factor * H0z
    else:
        bracket_I_H = 0.0 + 0.0j
    bracket_I = bracket_I_E + bracket_I_H
    tail_I = -(2j * cot_b0) / (k * N * sb0) * H0z
    I1 = pre_I * bracket_I + tail_I

    # ── Eq. 7: M_1 total ──
    if abs(H0z) > 1e-30:
        M1 = (2j * Z * sin_phi) / (k * sb0 * sb) \
             * (sin_pa_N / sin_alpha_c) / (N * denom_M_eq7) * H0z
    else:
        M1 = 0.0 + 0.0j

    # ── Eq. 3: fringe = total - PO ──
    return complex(I1 - I1_PO), complex(M1 - M1_PO)


def compute_face_currents_bistatic(beta_0, phi_0, beta, phi, N, E0z, H0z, k, Z):
    """
    面 1 的双站 (I_1^f, M_1^f), 含奇点邻域偏移取均值保护。

    奇点判定基于 (β_0, φ_0, β, φ) 算出的 μ; 触发时对 φ 做 ±_SING_OFFSET
    偏移取均值 (与 φ_0 偏移相比, 改 φ 不破坏 Face 1 照亮判据 U(π-φ_0))。
    """
    sb0 = np.sin(beta_0)
    if abs(sb0) < _SB_MIN:
        return 0.0 + 0.0j, 0.0 + 0.0j

    mu = _mu_bistatic(beta_0, phi_0, beta, phi)
    if not _is_singular_bistatic(beta_0, phi_0, beta, phi, mu, N):
        return _compute_face_currents_bistatic_raw(
            beta_0, phi_0, beta, phi, N, E0z, H0z, k, Z)

    # 奇点附近: 沿 φ 方向 ±_SING_OFFSET 取均值
    if1, mf1 = _compute_face_currents_bistatic_raw(
        beta_0, phi_0, beta, phi - _SING_OFFSET, N, E0z, H0z, k, Z)
    if2, mf2 = _compute_face_currents_bistatic_raw(
        beta_0, phi_0, beta, phi + _SING_OFFSET, N, E0z, H0z, k, Z)
    return 0.5 * (if1 + if2), 0.5 * (mf1 + mf2)


def compute_total_fringe_currents_bistatic(beta_0, phi_0, beta, phi, N, E0z, H0z, k, Z):
    """
    双站完整 Fringe 等效边电流: 面 1 + 面 2_raw 累加沿全局 ẑ 标量。

    面 2 几何替换 (Eq. 1, 含 ẑ→-ẑ 引起的电流再翻号, 净效果 = 加而非减):
        β_0 → π - β_0,  φ_0 → Nπ - φ_0
        β   → π - β,    φ   → Nπ - φ
    单站特化时退化等于 compute_total_fringe_currents (β=π-β_0, φ=φ_0 → 自动满足
    "面 2 角恰为面 1 角的 π-/Nπ- 镜像"), 数值上保持一致。
    """
    If1, Mf1 = compute_face_currents_bistatic(
        beta_0, phi_0, beta, phi, N, E0z, H0z, k, Z)
    If2_raw, Mf2_raw = compute_face_currents_bistatic(
        np.pi - beta_0, N * np.pi - phi_0,
        np.pi - beta,   N * np.pi - phi,
        N, E0z, H0z, k, Z)
    return If1 + If2_raw, Mf1 + Mf2_raw


def compute_total_fringe_currents_bistatic_raw(beta_0, phi_0, beta, phi, N, E0z, H0z, k, Z):
    """
    双站完整 Fringe 等效边电流 —— "裸"版本 (无奇点平滑保护).

    与 compute_total_fringe_currents_bistatic 公式一致, 但直接调
    _compute_face_currents_bistatic_raw 跳过 ±_SING_OFFSET 平均.

    用途:
        - Johansen 1996 Fig.4 复现 (非截断 EEC, 必须保留 Ufimtsev 奇点)
        - 与 compute_total_correction_general 的"裸" M_cor 配套, 保证 M_UT 与 M_cor
          在奇点处的发散项口径一致, 让 M_T = M_UT - M_cor 数值消除奇点

    缺点:
        奇点处 (μ+cosφ_0=0 或 μ=1) 返回非常大的复数 (~1e16), 单独使用会爆掉
        RCS 数值. 必须与配套的 M_cor 相减才能用.
    """
    If1, Mf1 = _compute_face_currents_bistatic_raw(
        beta_0, phi_0, beta, phi, N, E0z, H0z, k, Z)
    If2_raw, Mf2_raw = _compute_face_currents_bistatic_raw(
        np.pi - beta_0, N * np.pi - phi_0,
        np.pi - beta,   N * np.pi - phi,
        N, E0z, H0z, k, Z)
    return If1 + If2_raw, Mf1 + Mf2_raw
