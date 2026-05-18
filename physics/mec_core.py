"""
Michaeli MEC (1986 Part I) 边段贡献主函数

接口与 physics/ptd_core.py::compute_ptd_contribution 对齐 - 可作为 PTD 算法注册表的平替项。

支持极化模式:
    'VV' / 'HH' / 'VH' / 'HV'  (Michaeli 公式天然支持完整极化矩阵)

每条边的处理流程:
    1. 建立段局部坐标系 (z=t̂, y_1=面 1 法向 ⊥t, x_1=y_1 × z)
    2. 计算入射方向参数: β' = arccos(-k_dir · t̂), φ' = atan2(k·y_1, k·x_1)
    3. 由发射极化 ê_t 算激励切向分量: E_0z = ê_t · t̂, H_0z = (k_dir × ê_t)·t̂ / Z
    4. 调用 compute_total_fringe_currents 得 (I^f, M^f)
    5. 由接收极化 ê_r 投影 + sinc 校正 + 单站相位累加 ΔI

相位约定: 与 physics/ptd_core.py 一致, exp(+j 2 k_vec · r_c)。
"""
import numpy as np

from .constants import ETA0
from .mec_coefficients import compute_total_fringe_currents

# 发射/接收极化基向量名称映射 (按 IncidentWave 属性查表)
_POL_BASIS = {
    'VV': ('theta_hat', 'theta_hat'),
    'HH': ('phi_hat',   'phi_hat'),
    'VH': ('theta_hat', 'phi_hat'),
    'HV': ('phi_hat',   'theta_hat'),
}


def compute_mec_contribution(edge, wave, polarization='VV'):
    """
    单条边的 MEC 边段累加贡献。

    参数:
        edge:         PTDEdge 对象 (含 .segments, 每段有 alpha/normal/normal_b/...)
        wave:         IncidentWave 对象 (k_dir, k, k_vector, theta_hat, phi_hat)
        polarization: 'VV' | 'HH' | 'VH' | 'HV'

    返回:
        complex - 该边对单站散射场积分量 I 的贡献
    """
    if polarization not in _POL_BASIS:
        raise ValueError(
            f"未知极化模式: {polarization}. 可用: {list(_POL_BASIS.keys())}")

    et_name, er_name = _POL_BASIS[polarization]
    e_t = getattr(wave, et_name)   # 发射极化向量
    e_r = getattr(wave, er_name)   # 接收极化向量

    k_dir = wave.k_dir
    k_vec = wave.k_vector
    k = wave.k
    Z = ETA0
    s_dir = -k_dir   # 单站观察方向

    # 入射场切向投影 (与边段切线 t 无关, 但 t 在每段计算时不同)
    # E^i_0 = e_t (单位幅值), H^i_0 = (k_dir × e_t) / Z
    H0_vec = np.cross(k_dir, e_t) / Z

    total = 0.0 + 0.0j

    for seg in edge.segments:
        t = seg.tangent
        n_lit = seg.normal if seg.normal is not None else edge.n_lit

        # ---- 1. 入射方向 ⊥ t 检查 (边端入射 => 无衍射) ----
        k_dot_t = float(np.clip(np.dot(k_dir, t), -1.0, 1.0))
        sin_gamma0 = np.sqrt(max(0.0, 1.0 - k_dot_t * k_dot_t))
        if sin_gamma0 < 1e-3:
            continue

        # ---- 2. 建立局部坐标系 (与 ptd_core.py 完全一致) ----
        # y_1 = n_lit 投影到 ⊥t 平面后归一化
        n_dot_t = float(np.dot(n_lit, t))
        y1_raw = n_lit - n_dot_t * t
        y1_len = np.linalg.norm(y1_raw)
        if y1_len < 1e-10:
            continue
        y1 = y1_raw / y1_len

        # x_1 = y_1 × t (在面 1 内, ⊥ t, 指向外部)
        x1 = np.cross(y1, t)
        x1_len = np.linalg.norm(x1)
        if x1_len < 1e-10:
            continue
        x1 = x1 / x1_len

        # 消除 x_1 符号歧义: x_1 应远离面 2 (即 x_1·n_b 应 ≤ 0)
        n_b = seg.normal_b if hasattr(seg, 'normal_b') else None
        if n_b is not None and float(np.dot(x1, n_b)) > 0:
            x1 = -x1

        # ---- 3. β' 与 φ' (Michaeli Fig.1 坐标) ----
        # β' = 入射方向 ŝ' 与边切线 t 的空间夹角
        # ŝ' = k_dir, 故 cos β' = k_dir · t
        # 但 Michaeli Fig.1 中 ŝ' 指向源 (= -k_dir), 有 cos β' = -k_dir · t
        # 现有 ptd_core.py 中 angle0 用的是 inc_unit = -k_perp_unit, 相当于源方向
        # 故采用源方向约定: cos β' = -k_dir · t
        cos_bp = -k_dot_t
        # 限定 β' ∈ (0, π)
        beta_prime = np.arccos(np.clip(cos_bp, -1.0, 1.0))

        # φ' = ŝ' 在 (x_1, y_1) 平面内方位角 (从 x_1 轴起)
        s_inc = -k_dir   # 源方向 (Fig.1 约定)
        s_perp = s_inc - np.dot(s_inc, t) * t
        sp_len = np.linalg.norm(s_perp)
        if sp_len < 1e-10:
            continue
        sx = float(np.dot(s_perp, x1))
        sy = float(np.dot(s_perp, y1))
        phi_prime = np.arctan2(sy, sx)
        if phi_prime < 0:
            phi_prime += 2.0 * np.pi

        # 跳过照射方向落在阴影区 (φ' > Nπ) 的段
        N = seg.alpha / np.pi
        if phi_prime > N * np.pi:
            continue

        # ---- 4. 入射场切向分量 ----
        E0z = complex(np.dot(e_t, t))
        H0z = complex(np.dot(H0_vec, t))

        # ---- 5. 闭式 Fringe 等效电流 ----
        If, Mf = compute_total_fringe_currents(
            beta_prime, phi_prime, N, E0z, H0z, k, Z)

        # ---- 6. 接收侧投影 ----
        # ΔE^d ∝ -Z·I^f·(ŝ × (ŝ × t̂)) + M^f·(ŝ × t̂)
        # 标量投影: -Z·I^f·(t̂·ê_r) + M^f·((ŝ × t̂)·ê_r)
        # (利用 ê_r ⊥ ŝ, 故 ŝ × (ŝ × t̂) · ê_r = -t̂·ê_r)
        s_cross_t = np.cross(s_dir, t)
        proj_If = float(np.dot(t, e_r))
        proj_Mf = float(np.dot(s_cross_t, e_r))

        amp = -Z * If * proj_If + Mf * proj_Mf

        # ---- 7. 段长 sinc 校正 + 单站相位 + 累加 ----
        sinc_arg = k * seg.length * k_dot_t / np.pi
        sinc_val = np.sinc(sinc_arg)
        phase = 2.0 * float(np.dot(seg.midpoint, k_vec))

        # MEC 远场 E 场 → Ufimtsev I 量换算:
        #   σ = (k²/π)|I|², |E^s| = k|I|/(2πr) ⇒ I = 2πr·E^s/k
        # 代入 Knott 标准 MEC 远场表达式 E^s = -jk/(4πr) ∫ [...] dl, 化简得
        # 单段贡献前置因子 = -j/2 (替代原先错误的 +jk, 差 K = -1/(2k))
        seg_contrib = (-0.5j) * seg.length * sinc_val * np.exp(1j * phase) * amp
        total += seg_contrib

    return total
