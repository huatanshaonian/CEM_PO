"""
Johansen 截断 MEC 的几何辅助: 计算每个边段的截断长度 l^A.

Johansen 1996 定义:
    l^A = 沿 û^A 方向 (Keller 锥与面 A 的交线) 从当前边到 trailing edge 的距离

简化:
    本仓库当前实现假定 β_0 = π/2 (入射方向 ⊥ 棱边), 此时 Keller 锥退化为
    ⊥棱边平面, û^A = seg.inward (沿亮面、由边指向面内). 这对**主平面单站后向
    散射**(如平板 φ=0 任意 θ)是精确的, 因为 k_dir 无棱边切向分量, β_0=π/2.

    对斜入射 / 双站, 严格 û^A 应在 face A 内、与 t̂ 成角 β_0 旋转向 -k_dir 投影.
    后续如需要支持非主平面, 可在此处替换 û^A 计算.

后续扩展接口 (Task #13 远期):
    method='ray_cast' (本版本, 适用平面/折线面)
    method='geodesic' (NURBS / 样条曲面, 用 potpourri3d 库的离散测地线)

参考: Johansen 1996 IEEE TAP, p.990 (定义), p.991 (Eq.11 积分变量)
"""
import numpy as np


_INTERSECTION_EPS = 1e-9     # 共面性容差
_RAY_FORWARD_EPS = 1e-9      # 视为"前向"的最小距离 (避开自交)


def compute_truncation_length(seg, k_dir, all_edges,
                              surfaces=None, method='ray_cast'):
    """
    计算 PTD 边段的 Johansen 截断长度 l^A.

    参数:
        seg:        PTDSegment 当前边段 (需 .midpoint, .inward)
        k_dir:      入射方向 (np.ndarray shape=(3,), 指向目标), 当前实现未用
                    (β_0=π/2 假定下 û^A 与 k_dir 无关). 留作未来斜入射扩展接口.
        all_edges:  list of PTDEdge — 几何中所有 PTD 边
        surfaces:   list of Surface, 当前未用; 未来 method='geodesic' 时用于
                    在曲面上做测地线追踪.
        method:     'ray_cast' (默认, 平面折线面) | 'geodesic' (NURBS, Task #13)

    返回:
        l_A: 浮点, 沿 û^A 到首条相交边的距离 (米). 找不到 trailing edge 返 np.inf.

    实现细节 (method='ray_cast'):
        1. û^A = seg.inward (β_0=π/2 简化)
        2. 从 seg.midpoint 沿 û^A 射线追踪
        3. 对所有 *其他* 边的每个段, 计算射线-段最近距离 + 段内参数
        4. 过滤: 共面 (法距 ≤ ε) + 前向 (s > 0) + 段内 (0 ≤ t ≤ seg_len)
        5. 返回最小前向距离
    """
    if method == 'ray_cast':
        return _ray_cast_l_A(seg, all_edges)
    elif method == 'geodesic':
        raise NotImplementedError(
            "geodesic 方法待 Task #13 实现 (potpourri3d for NURBS surfaces)")
    else:
        raise ValueError(f"未知 method: {method}")


def _ray_cast_l_A(seg, all_edges):
    """β_0=π/2 简化下的射线-段几何 l^A 计算."""
    if seg.inward is None:
        return np.inf

    u_A_norm = np.linalg.norm(seg.inward)
    if u_A_norm < 1e-12:
        return np.inf
    u_A = seg.inward / u_A_norm
    origin = np.asarray(seg.midpoint, dtype=float)

    min_dist = np.inf
    for other_edge in all_edges:
        for other_seg in other_edge.segments:
            if other_seg is seg:
                continue
            dist = _ray_segment_intersection(origin, u_A, other_seg)
            if 0 < dist < min_dist:
                min_dist = dist
    return min_dist


def _ray_segment_intersection(origin, direction, target_seg):
    """
    3D 射线 (origin + s·direction, s≥0) 与线段 target_seg (start→end) 的相交距离.

    要求射线和线段近似共面 (法距 ≤ _INTERSECTION_EPS); 否则视为不相交.

    返回:
        s ≥ 0 (距离), 或 -1 (不相交 / 出平面 / 平行 / 背向)
    """
    start = np.asarray(target_seg.start, dtype=float)
    end = np.asarray(target_seg.end, dtype=float)
    target_dir = end - start
    target_len = np.linalg.norm(target_dir)
    if target_len < 1e-12:
        return -1.0
    target_dir_unit = target_dir / target_len

    # 共面性检查: direction × target_dir 应与 (start - origin) 正交
    n = np.cross(direction, target_dir_unit)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        # 平行 (或反向平行) — 不构成"穿过"相交
        return -1.0
    n_unit = n / n_norm

    delta = start - origin
    out_of_plane = abs(np.dot(delta, n_unit))
    if out_of_plane > _INTERSECTION_EPS:
        # 不共面 — 在折线/平面几何里视为不相交
        return -1.0

    # 共面 — 解 2D 参数: origin + s·direction = start + t·target_dir_unit
    # s·direction - t·target_dir_unit = start - origin = delta
    dd = np.dot(direction, target_dir_unit)
    rhs1 = np.dot(delta, direction)
    rhs2 = np.dot(delta, target_dir_unit)

    det = 1.0 - dd * dd
    if abs(det) < 1e-12:
        return -1.0

    s = (rhs1 - dd * rhs2) / det
    t = (dd * rhs1 - rhs2) / det

    # 验收: 前向射线 (s > eps) 且交点在段内 (0 ≤ t ≤ target_len)
    if s > _RAY_FORWARD_EPS and -_INTERSECTION_EPS <= t <= target_len + _INTERSECTION_EPS:
        return s
    return -1.0
