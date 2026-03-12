"""
PTD 自动边提取：给定两个相邻面，自动找共享边并计算外部二面角。
"""
import numpy as np


def find_shared_edge(face_a, face_b, n_samples=60, n_grid=40):
    """
    自动寻找 face_a 与 face_b 的共享边，计算外部二面角。

    算法：
    1. 采样 face_b 密集网格作为近邻参考
    2. 遍历 face_a 的 4 条参数域边界，找平均最近邻距离最小的边
    3. 若 face_a 的边均不满足容差，再试 face_b 的 4 条边
    4. 从两面获取该边上的法向量，计算外部二面角

    参数:
        face_a:    第一个 Surface 对象
        face_b:    第二个 Surface 对象
        n_samples: 边缘离散采样点数
        n_grid:    近邻参考网格密度（每方向）

    返回:
        edge_points (N,3):     共享边采样点（来自 face_a 侧）
        normals_a (N,3):       face_a 在该边上的法向量
        exterior_angle_rad:    外部二面角（弧度）

    异常:
        ValueError: 两面没有共享边
    """
    scale = _estimate_face_scale(face_a)
    tol = 0.05 * scale

    # 尝试 face_a 的 4 条边与 face_b 的网格比较
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
        # 尝试 face_b 的 4 条边与 face_a 的网格比较
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

    # 取出边缘点和 face_a 侧法向量
    if best_from_a:
        edge_pts = _get_boundary_edge(face_a, best_edge_idx, n_samples)
        normals_a = _get_boundary_normals(face_a, best_edge_idx, n_samples)
    else:
        edge_pts = _get_boundary_edge(face_b, best_edge_idx, n_samples)
        normals_a = _get_boundary_normals(face_b, best_edge_idx, n_samples)

    # 在 face_b（或 face_a）上查询对应点的法向量
    other_face = face_b if best_from_a else face_a
    normals_b = _get_normals_at_points(other_face, edge_pts, n_grid)

    # 外部二面角：α = π + arccos(mean(n_A · n_B))
    # 推导：内角 = π - arccos(n_A·n_B)，外角 = 2π - 内角 = π + arccos(n_A·n_B)
    cos_vals = np.einsum('ij,ij->i', normals_a, normals_b)
    cos_mean = float(np.clip(np.mean(cos_vals), -1.0, 1.0))
    exterior_angle = np.pi + np.arccos(cos_mean)

    return edge_pts, normals_a, exterior_angle


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
    """获取 face 第 idx 条边界上的单位法向量数组 (N, 3)。"""
    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    t = np.linspace(0.0, 1.0, n_samples)

    if idx == 0:
        u = np.full(n_samples, u0)
        v = v0 + t * (v1 - v0)
    elif idx == 1:
        u = np.full(n_samples, u1)
        v = v0 + t * (v1 - v0)
    elif idx == 2:
        u = u0 + t * (u1 - u0)
        v = np.full(n_samples, v0)
    else:
        u = u0 + t * (u1 - u0)
        v = np.full(n_samples, v1)

    normals = face.get_normal(u, v).reshape(n_samples, 3)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return normals / norms


def _sample_surface_grid(face, n_grid):
    """在面上均匀采样 n_grid×n_grid 点，返回 (n_grid²,3)。"""
    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    u = np.linspace(u0, u1, n_grid)
    v = np.linspace(v0, v1, n_grid)
    ug, vg = np.meshgrid(u, v)
    pts = face.evaluate(ug, vg)
    return pts.reshape(-1, 3)


def _get_normals_at_points(face, query_pts, n_grid):
    """
    在 face 上采样密集网格，对每个 query_pt 取最近邻的法向量。
    返回 (N, 3)。
    """
    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    u = np.linspace(u0, u1, n_grid)
    v = np.linspace(v0, v1, n_grid)
    ug, vg = np.meshgrid(u, v)
    ref_pts = face.evaluate(ug, vg).reshape(-1, 3)
    ref_normals = face.get_normal(ug, vg).reshape(-1, 3)

    # 归一化
    norms = np.linalg.norm(ref_normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    ref_normals = ref_normals / norms

    # 最近邻查询（向量化）
    diff = query_pts[:, None, :] - ref_pts[None, :, :]  # (N, M, 3)
    dists_sq = np.sum(diff ** 2, axis=2)                # (N, M)
    nn_idx = np.argmin(dists_sq, axis=1)                # (N,)
    return ref_normals[nn_idx]


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
    d1 = np.linalg.norm(corners[3] - corners[0])  # 主对角线
    d2 = np.linalg.norm(corners[2] - corners[1])  # 副对角线
    return max((d1 + d2) / 2.0, 1e-6)
