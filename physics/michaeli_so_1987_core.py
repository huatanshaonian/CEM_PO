"""
Michaeli 1987 二阶 EEC 主入口 (薄板 N=2 单基地后向 beta'=pi/2 简化).

接口: 与 mec_core 同款 (edge, wave, polarization) -> complex.
内部通过 edge._all_edges 拿全部边做边对枚举.
(这是为了不破坏现有 rcs_analyzer 的 "逐边循环累加" 调度. 第一次调用
[edge._so_index == 0] 时算所有边对的二阶贡献和返回; 后续调用直接返 0.)

几何前置 (单基地 beta'=pi/2 简化):
    对每对 (A, B), A != B:
      O_2 = B 段 midpoint
      sigma_hat = -inward_B   (从 B 向面内反方向 = 沿面指向 A 一侧)
      ray-cast: 从 O_2 沿 sigma_hat 走 -> 碰到 A 上一点 O_1, 距离 l
      (x_1, z_1) 系: z_1 = t_A, x_1 = inward_A
      (x_2, z_2) 系: z_2 = t_B, x_2 = inward_B
      (beta', phi') 在 O_1 系算 (与 mec_core 同款)
      (beta_2, phi_2) 在 O_2 系算 (观察方向 s_hat = -k_dir)

时谐: 内部 e^{+jomega t} (论文约定), 最终 conj 翻到 e^{-iomega t}.

远场算子: 复用 mec_core 同款标量积分量 I 装配
    I += seg_B.length * sinc_corr * exp(2j k . O_2) * amp
    amp = -Z * If2_conj * (t_B . e_r) + Mf2_conj * ((s_hat x t_B) . e_r)
    前因子 +0.5 (Ufimtsev e^{-iomega t} 约定下与 EEW 自洽).
"""
import numpy as np

from .constants import ETA0
from .michaeli_so_1987_coefficients import (
    compute_K2_plate_N2, project_K2_to_EEC
)


_POL_BASIS = {
    'VV': ('theta_hat', 'theta_hat'),
    'HH': ('phi_hat',   'phi_hat'),
}

_INTERSECTION_EPS = 1e-9
_RAY_FORWARD_EPS = 1e-9
_BETA_MIN = 1e-3


def _ray_segment_dist(origin, direction, target_start, target_end):
    """3D 射线 (origin + s dir, s>=0) 与线段相交距离 s. 无效返 -1."""
    target_dir = target_end - target_start
    target_len = np.linalg.norm(target_dir)
    if target_len < 1e-12:
        return -1.0, None
    t_unit = target_dir / target_len

    n = np.cross(direction, t_unit)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return -1.0, None
    n_unit = n / n_norm

    delta = target_start - origin
    if abs(np.dot(delta, n_unit)) > _INTERSECTION_EPS:
        return -1.0, None

    dd = np.dot(direction, t_unit)
    rhs1 = np.dot(delta, direction)
    rhs2 = np.dot(delta, t_unit)
    det = 1.0 - dd * dd
    if abs(det) < 1e-12:
        return -1.0, None

    s = (rhs1 - dd * rhs2) / det
    t = (dd * rhs1 - rhs2) / det

    if s > _RAY_FORWARD_EPS and -_INTERSECTION_EPS <= t <= target_len + _INTERSECTION_EPS:
        # 交点在 A 段内
        hit = origin + s * direction
        return s, hit
    return -1.0, None


def _find_O1_on_edge(O2, sigma_hat, edge_A):
    """
    从 O_2 沿 sigma_hat 走, 找在 edge_A 上的最近交点 O_1.

    返回 (l, O_1, hit_seg) 或 (None, None, None) 不相交.
    """
    best_l = np.inf
    best_O1 = None
    best_seg = None
    for seg in edge_A.segments:
        l, hit = _ray_segment_dist(O2, sigma_hat, seg.start, seg.end)
        if l > 0 and l < best_l:
            best_l = l
            best_O1 = hit
            best_seg = seg
    if not np.isfinite(best_l):
        return None, None, None
    return best_l, best_O1, best_seg


def _local_frame_A(seg_A, k_dir):
    """
    边 A 端点 O_1 处局部坐标系 + (beta', phi'), 与 mec_core 同款约定.

    返回 (ok, beta_p, phi_p, x1_hat, z1_hat) 或 (False, ...).
    """
    z1 = seg_A.tangent
    n_A = seg_A.normal
    if n_A is None:
        return False, 0, 0, None, None

    k_dot_t = float(np.clip(np.dot(k_dir, z1), -1.0, 1.0))
    sin_g = np.sqrt(max(0.0, 1.0 - k_dot_t * k_dot_t))
    if sin_g < _BETA_MIN:
        return False, 0, 0, None, None

    # x_1: 在面 A 内 ⊥ z_1, 由 inward 决定朝向
    inward = seg_A.inward
    if inward is None:
        # fallback: 用 n_A 投影
        x1_raw = n_A - np.dot(n_A, z1) * z1
        x1_len = np.linalg.norm(x1_raw)
        if x1_len < 1e-10:
            return False, 0, 0, None, None
        x1 = x1_raw / x1_len
    else:
        x1 = inward - np.dot(inward, z1) * z1
        x1_len = np.linalg.norm(x1)
        if x1_len < 1e-10:
            return False, 0, 0, None, None
        x1 = x1 / x1_len

    # y_1 = z_1 x x_1  (face normal direction)
    y1 = np.cross(z1, x1)
    y1_len = np.linalg.norm(y1)
    if y1_len < 1e-10:
        return False, 0, 0, None, None
    y1 = y1 / y1_len

    # beta' = arccos(-k_dir . z_1)  (Michaeli 源方向约定)
    cos_bp = -k_dot_t
    beta_p = float(np.arccos(np.clip(cos_bp, -1.0, 1.0)))

    # phi' = 入射在 (x_1, y_1) 平面方位角
    s_inc = -k_dir
    s_perp = s_inc - np.dot(s_inc, z1) * z1
    sp_len = np.linalg.norm(s_perp)
    if sp_len < 1e-10:
        return False, 0, 0, None, None
    sx = float(np.dot(s_perp, x1))
    sy = float(np.dot(s_perp, y1))
    phi_p = float(np.arctan2(sy, sx))
    if phi_p < 0:
        phi_p += 2.0 * np.pi

    return True, beta_p, phi_p, x1, z1


def _local_frame_B(seg_B, s_dir):
    """
    边 B 端点 O_2 处局部坐标系 + (beta_2, phi_2) 对观察方向 s_dir.

    返回 (ok, beta_2, phi_2, x2_hat, z2_hat).
    """
    z2 = seg_B.tangent
    inward = seg_B.inward
    if inward is None:
        return False, 0, 0, None, None

    x2 = inward - np.dot(inward, z2) * z2
    x2_len = np.linalg.norm(x2)
    if x2_len < 1e-10:
        return False, 0, 0, None, None
    x2 = x2 / x2_len

    y2 = np.cross(z2, x2)
    y2_len = np.linalg.norm(y2)
    if y2_len < 1e-10:
        return False, 0, 0, None, None
    y2 = y2 / y2_len

    # beta_2: 观察方向 与 z_2 夹角
    s_dot_t = float(np.clip(np.dot(s_dir, z2), -1.0, 1.0))
    sin_b2 = np.sqrt(max(0.0, 1.0 - s_dot_t * s_dot_t))
    if sin_b2 < _BETA_MIN:
        return False, 0, 0, None, None
    beta_2 = float(np.arccos(s_dot_t))

    s_perp = s_dir - s_dot_t * z2
    sp_len = np.linalg.norm(s_perp)
    if sp_len < 1e-10:
        return False, 0, 0, None, None
    sx = float(np.dot(s_perp, x2))
    sy = float(np.dot(s_perp, y2))
    phi_2 = float(np.arctan2(sy, sx))
    if phi_2 < 0:
        phi_2 += 2.0 * np.pi

    return True, beta_2, phi_2, x2, z2


def _compute_pair_contribution(edge_A, edge_B, wave, polarization):
    """
    单个边对 (A, B) 的二阶贡献累加 (在 B 上 segment-by-segment).
    """
    if polarization not in _POL_BASIS:
        return 0j

    et_name, er_name = _POL_BASIS[polarization]
    e_t = getattr(wave, et_name)
    e_r = getattr(wave, er_name)

    k = wave.k
    k_dir = wave.k_dir
    k_vec = wave.k_vector
    s_dir = -k_dir
    Z = ETA0

    # 入射平面波幅值 (单位幅) 矢量化: H_inc = (k_dir x e_t) / Z (Z*H = s x E)
    # 但平面波: H = (k_dir x E)/Z 在 e^{-iomega t} 约定下, 与 mec_core 同款
    H_pol = np.cross(k_dir, e_t) / Z

    total = 0j
    for seg_B in edge_B.segments:
        O2 = seg_B.midpoint
        if seg_B.inward is None:
            continue
        # ray-cast 方向: 从 O_2 沿 +inward_B (朝三角形内) 反向追到 A 上 O_1.
        # 然后 sigma_hat = (O_2 - O_1)/l 是公式里 grazing 射线方向 (A -> B).
        # 几何上 sigma_hat = -ray_dir = -inward_B.
        ray_dir = seg_B.inward
        rd_norm = np.linalg.norm(ray_dir)
        if rd_norm < 1e-10:
            continue
        ray_dir = ray_dir / rd_norm

        # 找 A 上的 O_1 (从 O_2 沿 ray_dir 走)
        l, O1, seg_A = _find_O1_on_edge(O2, ray_dir, edge_A)
        if l is None or seg_A is None:
            continue
        sigma_hat = -ray_dir  # 公式约定: 从 O_1 到 O_2

        # O_1 处局部坐标系 + (beta', phi')
        okA, beta_p, phi_p, x1, z1 = _local_frame_A(seg_A, k_dir)
        if not okA:
            continue

        # 阴影判定 (face A 是否被照亮): phi' < pi 才被亮面照
        N = seg_A.alpha / np.pi
        if phi_p > N * np.pi:
            continue

        # O_2 处局部坐标系 + (beta_2, phi_2)
        okB, beta_2, phi_2, x2, z2 = _local_frame_B(seg_B, s_dir)
        if not okB:
            continue

        # 入射波在 O_1 处的复值
        phase_O1 = float(np.dot(O1, k_vec))
        E0_at_O1 = e_t * np.exp(1j * phase_O1)
        H0_at_O1 = H_pol * np.exp(1j * phase_O1)

        # K_2^f (e^{+jomega t} 约定)
        K2 = compute_K2_plate_N2(
            beta_p, phi_p, l, sigma_hat, x1, z1, z2,
            E0_at_O1, H0_at_O1, k, Z
        )
        if not np.any(K2):
            continue

        # Eq.12 投影
        If2_m, Mf2_m = project_K2_to_EEC(K2, beta_2, phi_2, x2, z2, Z)

        # 翻到 e^{-iomega t}: 整支 conj (与 mec_core 同款)
        If2 = np.conj(If2_m)
        Mf2 = np.conj(Mf2_m)

        # 接收侧投影 (与 mec_core 同款标量 amp)
        s_cross_t = np.cross(s_dir, z2)
        proj_If = float(np.dot(z2, e_r))
        proj_Mf = float(np.dot(s_cross_t, e_r))
        amp = -Z * If2 * proj_If + Mf2 * proj_Mf

        # 段长 sinc + 单站相位 (与 mec_core 同款)
        sinc_arg = k * seg_B.length * float(np.dot(k_dir, z2)) / np.pi
        sinc_val = np.sinc(sinc_arg)
        phase_O2 = 2.0 * float(np.dot(O2, k_vec))

        seg_contrib = 0.5 * seg_B.length * sinc_val * np.exp(1j * phase_O2) * amp
        total += seg_contrib

    return total


def compute_michaeli_so_1987_contribution(edge, wave, polarization='VV'):
    """
    Michaeli 1987 二阶 EEC 贡献 (单边接口, 通过 edge._all_edges 拿全部边).

    第一次调用 (edge._so_index == 0) 时算所有边对总贡献; 其它索引返 0,
    避免被 rcs_analyzer 的 "逐边循环累加" 重复加.

    支持极化: VV, HH (论文也未给 cross-pol).
    """
    if polarization not in _POL_BASIS:
        return 0j

    # 必须有 edge._all_edges 和 edge._so_index (由 PTDProcessor 注入)
    all_edges = getattr(edge, '_all_edges', None)
    so_index = getattr(edge, '_so_index', None)
    if all_edges is None or so_index is None:
        # fallback: 当作只有这一条边, 没有边对, 二阶为 0
        return 0j

    if so_index != 0:
        return 0j

    total = 0j
    for edge_A in all_edges:
        for edge_B in all_edges:
            if edge_A is edge_B:
                continue
            total += _compute_pair_contribution(edge_A, edge_B, wave, polarization)
    return total
