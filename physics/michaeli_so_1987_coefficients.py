"""
Michaeli 1987 二阶 EEC 端点等效电流闭式系数 (薄板 N=2 特化).

参考: A. Michaeli, "Equivalent Currents for Second-Order Diffraction by the Edges
of Perfectly Conducting Polygonal Surfaces", IEEE TAP 35(2), 1987.

实现范围:
    - 薄平板 (外角 alpha = 2 pi, N = 2)
    - Eq.21 给出 grazing 终点 O_2 处中间矢量 K_2^f 的 N=2 闭式
    - Eq.12 把 K_2^f 投影到端点二阶等效电/磁电流标量 (I_2^f, M_2^f)

K_2^f 物理对象 (NotebookLM 解析确认):
    - 是单个 3D 矢量, 不是张量
    - 在 O_1 处局部 (x_1, y_1, z_1) 系内表达, 即代码里 x1_hat, z1_hat
      作为全局笛卡尔下的 3D 单位矢量传入
    - 矢量结构: K_2^f = scalar_pre * bracket_vec
      其中 |sigma_hat x z2_hat| 是 *模长* (标量), 不是矢量叉积

时谐约定:
    本函数内部按 Michaeli 论文 e^{+jomega t} 公式实现 (j 用 numpy 的 1j).
    返回值是 e^{+jomega t} 相量. 调用方对最终标量 (I_2^f, M_2^f) 整支 np.conj
    翻到本仓库 e^{-iomega t} 约定, 与 mec_core / mec_truncated_core 同路径.

实现细节 (符号与 Michaeli 1986 一阶 MEC 一致):
    beta'   = 入射方向与边 A 切线夹角
    phi'    = 入射波在 O_1 局部 (x_1, y_1) 平面方位角
    mu      = Michaeli 单站参量 = cos phi' - 2 cot^2(beta')   (Michaeli 1986 Eq.26)
    L       = k * l * sin^2(beta')                            (无量纲大参数)
    F(x)    = modified Fresnel, physics/uniform_transition

Eq.21 矢量结构 (本文件内 docstring 标注以避免重复论文公式):
    K_2^f = pre_scalar * { (F1 / sqrt(1-mu)) * bracket_F1
                          - (F2 / sqrt(2))   * bracket_F2 }
    pre_scalar  = 4 sqrt(2) * |sigma_hat x z2_hat| * exp(-j k l)
                  / [ j k sin^2(beta') (mu + cos phi') ]
    F1          = F( sqrt( L (1 - mu) ) )
    F2          = F( sqrt( L (1 + cos phi') ) )
    bracket_F1, bracket_F2: x1_hat 与 z1_hat 加权的 3D 矢量 (各分量系数见下方代码)

奇点保护 (与 mec_truncated_coefficients 同口径):
    sin(beta') < _SB_MIN              -> 返 0 矢量
    |mu + cos phi'| < _DENOM_EPS      -> 返 0 (Ufimtsev 奇点; M_UT 已发散)
    |1 - mu| < _MU1_EPS               -> 返 0 (Keller 锥反射边界)
    L < _L_MIN                        -> 返 0 (l -> 0 退化)
"""
import numpy as np

from .uniform_transition import modified_fresnel


_SB_MIN = 1e-3
_DENOM_EPS = 1e-9
_MU1_EPS = 1e-9
_L_MIN = 1e-15


def compute_K2_plate_N2(beta_p, phi_p, l, sigma_hat, x1_hat, z1_hat, z2_hat,
                         E0_vec, H0_vec, k, Z):
    """
    Eq.21 (N=2 薄板) 计算 grazing 终点 O_2 处的中间 3D 矢量 K_2^f.

    参数:
        beta_p:    入射方向与边 A 切线夹角 (rad)
        phi_p:     入射方位角 phi' in (x_1, y_1) 平面 (rad)
        l:         O_1 -> O_2 沿共享面 grazing 射线的距离 (m)
        sigma_hat: grazing 射线方向单位矢量 (3,) 全局笛卡尔
        x1_hat:    O_1 处 x_1 轴 = 在面内 ⊥ 边 A, 指向面内 (3,)
        z1_hat:    O_1 处 z_1 轴 = 边 A 切线 t_A (3,)
        z2_hat:    O_2 处 z_2 轴 = 边 B 切线 t_B (3,)
        E0_vec:    入射波电场矢量在 O_1 处复值 (3, complex)
        H0_vec:    入射波磁场矢量在 O_1 处复值 (3, complex)
        k:         波数 (rad/m)
        Z:         介质波阻抗 (Ohm)

    返回:
        K2: (3,) complex, e^{+jomega t} 约定原生数值; 调用方 conj 翻到 e^{-iomega t}.
    """
    sb = np.sin(beta_p)
    if abs(sb) < _SB_MIN:
        return np.zeros(3, dtype=complex)

    cb = np.cos(beta_p)
    cot_b = cb / sb
    csc_b = 1.0 / sb
    cos_pp = np.cos(phi_p)
    sin_pp = np.sin(phi_p)
    cos_half = np.cos(phi_p / 2.0)
    sin_half = np.sin(phi_p / 2.0)

    # Michaeli 单站 N=2 mu (1986 Eq.26): mu = cos phi' - 2 cot^2 beta'
    mu = cos_pp - 2.0 * cot_b * cot_b

    denom_a = mu + cos_pp
    if abs(denom_a) < _DENOM_EPS:
        return np.zeros(3, dtype=complex)

    one_minus_mu = 1.0 - mu
    if abs(one_minus_mu) < _MU1_EPS:
        return np.zeros(3, dtype=complex)

    L = k * l * sb * sb
    if L < _L_MIN:
        return np.zeros(3, dtype=complex)

    # Fresnel 参量 (mu>1 时 sqrt 给纯虚, F 自动延拓)
    sqrt_1mmu = np.lib.scimath.sqrt(one_minus_mu)
    F1_arg = np.lib.scimath.sqrt(L * one_minus_mu)
    F2_arg = np.lib.scimath.sqrt(L * (1.0 + cos_pp))

    F1 = modified_fresnel(F1_arg)
    F2 = modified_fresnel(F2_arg)

    Y = 1.0 / Z

    # 入射场切向投影到 z_1 (= 边 A 切线): 复标量
    Ez1 = complex(np.dot(z1_hat, E0_vec))
    Hz1 = complex(np.dot(z1_hat, H0_vec))

    # |sigma_hat x z2_hat| 是标量模长, 表示 grazing 射线与边 B 夹角的 sin
    sin_sigma_z2 = float(np.linalg.norm(np.cross(sigma_hat, z2_hat)))

    pre_scalar = (4.0 * np.sqrt(2.0) * sin_sigma_z2
                  * np.exp(-1j * k * l)
                  / (1j * k * sb * sb * denom_a))

    # bracket_F1 (x_1 + z_1 加权 3D 矢量)
    bracket_F1 = (
        x1_hat * (Hz1 * cos_half)
        + z1_hat * (Ez1 * Y * one_minus_mu * csc_b * sin_half)
        + z1_hat * (Hz1 * mu * cot_b * cos_half)
    )

    # bracket_F2 (x_1 + z_1 加权 3D 矢量)
    bracket_F2 = (
        x1_hat * Hz1
        + z1_hat * (Ez1 * Y * csc_b * sin_pp)
        - z1_hat * (Hz1 * cot_b * cos_pp)
    )

    K2 = pre_scalar * (
        (F1 / sqrt_1mmu) * bracket_F1
        - (F2 / np.sqrt(2.0)) * bracket_F2
    )

    return K2


def project_K2_to_EEC(K2, beta_2, phi_2, x2_hat, z2_hat, Z):
    """
    Eq.12: 把 O_2 处 K_2^f 投影到端点等效电流标量.

        M_2^f = -Z * (x_2 . K_2^f) * sin(phi_2) / sin(beta_2)
        I_2^f =  z_2 . K_2^f - (x_2 . K_2^f) * cot(beta_2) * cos(phi_2)

    K_2^f 在 O_1 系算出但用全局笛卡尔表示, 故 x_2 . K_2^f 用 np.dot 直接做,
    坐标变换隐含在点积里 (因 x_2 也是全局笛卡尔的).

    参数:
        K2:     (3,) complex, compute_K2_plate_N2 返回
        beta_2: 观察方向 s_hat 与边 B 切线夹角 (rad)
        phi_2:  s_hat 在 O_2 处 (x_2, y_2) 平面方位角 (rad)
        x2_hat: O_2 处 x_2 轴 = 在面内 ⊥ 边 B 指向面内 (3,)
        z2_hat: O_2 处 z_2 轴 = 边 B 切线 (3,)
        Z:      介质波阻抗

    返回:
        (If2, Mf2): 复标量, 仍为 e^{+jomega t} 约定. 调用方 conj.
    """
    sb2 = np.sin(beta_2)
    if abs(sb2) < _SB_MIN:
        return 0.0 + 0.0j, 0.0 + 0.0j

    cot_b2 = np.cos(beta_2) / sb2
    sin_p2 = np.sin(phi_2)
    cos_p2 = np.cos(phi_2)

    K2_x2 = complex(np.dot(x2_hat, K2))
    K2_z2 = complex(np.dot(z2_hat, K2))

    Mf2 = -Z * K2_x2 * sin_p2 / sb2
    If2 = K2_z2 - K2_x2 * cot_b2 * cos_p2

    return If2, Mf2
