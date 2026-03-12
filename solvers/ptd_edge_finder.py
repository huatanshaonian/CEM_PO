"""
PTD 自动边提取：给定两个相邻面，自动找共享边并计算外部二面角。
"""
import numpy as np

_NORMAL_OFFSET = 1e-5  # 法向量采样的参数域内偏量（固定绝对值，约 0.01mm for meter-scale surfaces）


def find_shared_edge(face_a, face_b, n_samples=60, n_grid=40):
    """
    自动寻找 face_a 与 face_b 的共享边，计算外部二面角。

    算法：
    1. 采样 face_b 密集网格作为近邻参考，遍历 face_a 的 4 条边界找最近的
    2. 若 face_a 的边均不满足容差，再试 face_b 的 4 条边
    3. 对 other_face 同样扫 4 条边界找距离 edge_pts 最近的，用 _get_boundary_normals 取法向
    4. 两侧法向均略微向面内偏移 _NORMAL_OFFSET，避免退化边界

    参数:
        face_a:    第一个 Surface 对象
        face_b:    第二个 Surface 对象
        n_samples: 边缘离散采样点数
        n_grid:    寻边时参考网格密度（每方向）

    返回:
        edge_points (N,3):     共享边采样点
        normals_a (N,3):       face_a 在该边略内侧的法向量
        normals_b (N,3):       face_b 在该边略内侧的法向量
        exterior_angle_rad:    外部二面角（弧度）

    异常:
        ValueError: 两面没有共享边
    """
    scale = _estimate_face_scale(face_a)
    tol = 0.05 * scale

    # ── Step 1: 找 face_a 哪条边界最近 ──
    ref_b = _sample_surface_grid(face_b, n_grid)
    best_dist = np.inf
    best_edge_idx = None
    best_from_a = True

    for idx in range(4):
        pts = _get_boundary_edge(face_a, idx, n_samples)
        d = _avg_min_dist(pts, ref_b)
        if d < best_dist:
            best_dist = d
            best_edge_idx = idx

    if best_dist > tol:
        # face_a 的边都不够近，改试 face_b 的 4 条边
        ref_a = _sample_surface_grid(face_a, n_grid)
        for idx in range(4):
            pts = _get_boundary_edge(face_b, idx, n_samples)
            d = _avg_min_dist(pts, ref_a)
            if d < best_dist:
                best_dist = d
                best_edge_idx = idx
                best_from_a = False

    if best_dist > tol:
        raise ValueError(
            f"找不到共享边（最小平均距离 {best_dist:.4f} > 容差 {tol:.4f}）"
        )

    # ── Step 2: 取边缘点和 face_a 侧法向量 ──
    if best_from_a:
        edge_pts  = _get_boundary_edge(face_a, best_edge_idx, n_samples)
        normals_a = _get_boundary_normals(face_a, best_edge_idx, n_samples)
        other_face = face_b
    else:
        edge_pts  = _get_boundary_edge(face_b, best_edge_idx, n_samples)
        normals_a = _get_boundary_normals(face_b, best_edge_idx, n_samples)
        other_face = face_a

    # ── Step 3: 对 other_face 同样扫 4 条边界，找距离 edge_pts 最近的，取法向 ──
    other_best_dist = np.inf
    other_best_idx = 0
    for idx in range(4):
        bnd = _get_boundary_edge(other_face, idx, n_samples)
        d = _avg_min_dist(edge_pts, bnd)
        if d < other_best_dist:
            other_best_dist = d
            other_best_idx = idx
    normals_b = _get_boundary_normals(other_face, other_best_idx, n_samples)

    # ── Step 4: 外部二面角 α = π + arccos(mean(n_A · n_B)) ──
    cos_vals = np.einsum('ij,ij->i', normals_a, normals_b)
    cos_mean = float(np.clip(np.mean(cos_vals), -1.0, 1.0))
    exterior_angle = np.pi + np.arccos(cos_mean)

    return edge_pts, normals_a, normals_b, exterior_angle


# ──────────────────────── 内部辅助函数 ────────────────────────

def _get_boundary_edge(face, idx, n_samples):
    """获取 face 的第 idx 条边界边（0=u_min,1=u_max,2=v_min,3=v_max）。"""
    if hasattr(face, 'get_edge_by_index'):
        return face.get_edge_by_index(idx, n_samples=n_samples)

    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    t = np.linspace(0.0, 1.0, n_samples)

    if idx == 0:   # u = u_min
        u = np.full(n_samples, u0)
        v = v0 + t * (v1 - v0)
    elif idx == 1:  # u = u_max
        u = np.full(n_samples, u1)
        v = v0 + t * (v1 - v0)
    elif idx == 2:  # v = v_min
        u = u0 + t * (u1 - u0)
        v = np.full(n_samples, v0)
    else:           # v = v_max
        u = u0 + t * (u1 - u0)
        v = np.full(n_samples, v1)

    pts = face.evaluate(u, v)
    return pts.reshape(n_samples, 3)


def _get_boundary_normals(face, idx, n_samples):
    """获取 face 第 idx 条边界略内侧的单位法向量数组 (N, 3)。

    固定向内偏移 _NORMAL_OFFSET（绝对参数值），避免在退化边界处取法向。
    """
    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    t = np.linspace(0.0, 1.0, n_samples)

    if idx == 0:   # u = u_min，向内偏
        u = np.full(n_samples, u0 + _NORMAL_OFFSET)
        v = v0 + t * (v1 - v0)
    elif idx == 1:  # u = u_max，向内偏
        u = np.full(n_samples, u1 - _NORMAL_OFFSET)
        v = v0 + t * (v1 - v0)
    elif idx == 2:  # v = v_min，向内偏
        u = u0 + t * (u1 - u0)
        v = np.full(n_samples, v0 + _NORMAL_OFFSET)
    else:           # v = v_max，向内偏
        u = u0 + t * (u1 - u0)
        v = np.full(n_samples, v1 - _NORMAL_OFFSET)

    normals = face.get_normal(u, v).reshape(n_samples, 3)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return normals / norms


def _sample_surface_grid(face, n_grid):
    """在面上均匀采样 n_grid×n_grid 点，返回 (n_grid²,3)。仅用于寻边距离比较。"""
    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    u = np.linspace(u0, u1, n_grid)
    v = np.linspace(v0, v1, n_grid)
    ug, vg = np.meshgrid(u, v)
    pts = face.evaluate(ug, vg)
    return pts.reshape(-1, 3)


def _avg_min_dist(query, ref):
    """计算 query (N,3) 中每个点到 ref (M,3) 最近邻距离的均值。"""
    diff = query[:, None, :] - ref[None, :, :]   # (N, M, 3)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))   # (N, M)
    return float(np.mean(np.min(dists, axis=1)))


def _estimate_face_scale(face):
    """估算面的特征尺寸（四角连线对角线长度的均值）。"""
    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    corners = [
        face.evaluate(u0, v0),
        face.evaluate(u0, v1),
        face.evaluate(u1, v0),
        face.evaluate(u1, v1),
    ]
    corners = [c.flatten() for c in corners]
    d1 = np.linalg.norm(corners[3] - corners[0])
    d2 = np.linalg.norm(corners[2] - corners[1])
    return max((d1 + d2) / 2.0, 1e-6)
