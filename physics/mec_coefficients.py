"""
Michaeli MEC (1986 Part I) 等效边电流闭式系数

参考文献:
  A. Michaeli, "Elimination of Infinities in Equivalent Edge Currents,
  Part I: Fringe Current Components", IEEE TAP, 1986.

实现范围:
  - 远场单站后向散射 (β = π-β', φ = φ', 论文 Eq. 26)
  - PEC 目标
  - 仅 Fringe 分量 (PO 分量由表面 PO 积分覆盖, 不在此实现)

核心公式 (Eq. 27, 28 + Eq. 26 化简后的 μ):
  μ = cos φ' - 2 cot²β'
  α_c = arccos(μ)  (复数, μ 可能 |μ|>1)

  I_1^f = pre_E·[sin φ'·U/(cos φ'+μ) + (1/N)sin(φ'/N)/D_e] · E_0z
        + pre_H·[cot β'(μ+cos φ')/(-D_e)] · H_0z
  M_1^f = pre_M·[U/(cos φ'+μ) + sin((π-α_c)/N)/(sin α_c · N · D_e)] · H_0z

  其中 D_e = cos((π-α_c)/N) - cos(φ'/N)

总电流 (Eq. 23 + Eq. 1 替换):
  I^f = I_1^f - I_2^f, M^f = M_1^f - M_2^f
  面 2 通过 {β'→π-β', φ'→Nπ-φ'} 替换得到。
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
