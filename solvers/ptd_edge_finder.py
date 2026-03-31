"""
PTD 自动边提取：给定两个相邻面，自动找共享边并计算外部二面角。

采样策略：先均匀采 n_initial 点，用 _adaptive_breakpoints 合并（删点），
若合并后仍有段超标则插入中点（最多 3 轮），返回最终自适应点集。
"""
import numpy as np

_NORMAL_OFFSET = 1e-5  # 法向量采样的参数域内偏量


def _num_edges(face):
    """获取面的边数。OCC 面用 n_edges 属性，其余默认 4（参数域边界）。"""
    return getattr(face, 'n_edges', 4)


def _arc_length(pts):
    """采样点折线的总弧长。退化边弧长趋近 0。"""
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def _find_shared_edge_pair(face_a, face_b, n_samples=20, tol=1e-3,
                           min_edge_len=1e-6):
    """
    在 face_a 和 face_b 的边界边中找到共享边对。

    判定条件（全部满足才算共享边）：
    - 边弧长 >= min_edge_len（排除退化边）
    - ea 的两端点和中点到 eb 的最近距离都 <= tol（排除仅共顶点）

    返回:
        (idx_a, idx_b):  共享边在各自面上的索引
        None:            找不到共享边
    """
    edges_a = [_get_boundary_edge(face_a, idx, n_samples)
               for idx in range(_num_edges(face_a))]
    edges_b = [_get_boundary_edge(face_b, idx, n_samples)
               for idx in range(_num_edges(face_b))]

    best_dist = np.inf
    best_pair = None
    for ia, ea in enumerate(edges_a):
        if _arc_length(ea) < min_edge_len:
            continue
        for ib, eb in enumerate(edges_b):
            if _arc_length(eb) < min_edge_len:
                continue
            # 两端点 + 中点都要匹配
            d0 = np.min(np.linalg.norm(eb - ea[0], axis=1))
            d1 = np.min(np.linalg.norm(eb - ea[-1], axis=1))
            dm = np.min(np.linalg.norm(eb - ea[len(ea) // 2], axis=1))
            if d0 > tol or d1 > tol or dm > tol:
                continue
            d = _avg_min_dist(ea, eb)
            if d < best_dist:
                best_dist = d
                best_pair = (ia, ib)

    return best_pair


def find_shared_edge(face_a, face_b, max_angle_deg=2.0, n_initial=100,
                     max_pts=500):
    """
    自动寻找 face_a 与 face_b 的共享边，返回自适应采样的边缘点及法向量。

    返回:
        edge_points (N,3):     共享边采样点
        normals_a (N,3):       face_a 在该边略内侧的法向量
        normals_b (N,3):       face_b 在该边略内侧的法向量
        exterior_angle_rad:    外部二面角（弧度）
        warning_msg:           str 或 None，超标警告信息

    异常:
        ValueError: 两面没有共享边
    """
    pair = _find_shared_edge_pair(face_a, face_b)
    if pair is None:
        raise ValueError("找不到共享边")
    best_idx_a, best_idx_b = pair

    max_angle_rad = np.deg2rad(max_angle_deg)
    n_pts = n_initial
    warning_msg = None

    for round_idx in range(4):
        edge_pts, normals_a = _get_edge_points_and_normals(
            face_a, best_idx_a, n_pts)
        _, normals_b = _get_edge_points_and_normals(
            face_b, best_idx_b, n_pts)

        bp = _adaptive_breakpoints(edge_pts, max_angle_rad)
        bp_arr = np.array(bp)
        spans = bp_arr[1:] - bp_arr[:-1]
        n_single = int(np.sum(spans == 1))

        if n_single == 0:
            break

        if round_idx == 3:
            warning_msg = (
                f"3 轮细化后仍有 {n_single} 段仅跨单区间（初始采样不足），"
                f"请增大 PTD Seg. Angle 或检查边缘几何"
            )
            break

        n_pts = min(n_pts * 2, max_pts)
        if n_pts >= max_pts:
            warning_msg = (
                f"达到最大采样点数 {max_pts}，仍有 {n_single} 段初始采样不足"
            )
            break

    bp = _adaptive_breakpoints(edge_pts, max_angle_rad)
    pts_sel = edge_pts[bp]
    na_sel  = normals_a[bp]
    nb_sel  = normals_b[bp]

    cos_vals = np.einsum('ij,ij->i', na_sel, nb_sel)
    cos_mean = float(np.clip(np.mean(cos_vals), -1.0, 1.0))
    exterior_angle = np.pi + np.arccos(cos_mean)

    return pts_sel, na_sel, nb_sel, exterior_angle, warning_msg



# ──────────────────────── 自适应合并 ────────────────────────

def _adaptive_breakpoints(pts, max_angle_rad):
    """
    根据切线方向累积变化量，在折线 pts 上选取断点索引（始终包含 0 和 N-1）。
    用于从密集点集中选出满足每段累积转角 ≤ max_angle_rad 的最少点子集。
    """
    N = len(pts)
    if N < 3:
        return list(range(N))

    breakpoints = [0]
    accum_angle = 0.0

    tangents = pts[1:] - pts[:-1]
    lengths  = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths  = np.where(lengths < 1e-12, 1.0, lengths)
    tangents = tangents / lengths

    for i in range(1, N - 1):
        cos_a = np.clip(np.dot(tangents[i - 1], tangents[i]), -1.0, 1.0)
        delta = np.arccos(cos_a)
        accum_angle += delta
        if accum_angle >= max_angle_rad:
            breakpoints.append(i)
            accum_angle = 0.0

    if breakpoints[-1] != N - 1:
        breakpoints.append(N - 1)

    return breakpoints


# ──────────────────────── 参数域评估辅助函数 ────────────────────────

def _eval_boundary_edge_at_t(face, idx, t_vals):
    """在给定 t 值（0→1）处评估面的第 idx 条边界点，返回 (N, 3)。"""
    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    t = np.asarray(t_vals)

    if idx == 0:
        u = np.full(len(t), u0);  v = v0 + t * (v1 - v0)
    elif idx == 1:
        u = np.full(len(t), u1);  v = v0 + t * (v1 - v0)
    elif idx == 2:
        u = u0 + t * (u1 - u0);   v = np.full(len(t), v0)
    else:
        u = u0 + t * (u1 - u0);   v = np.full(len(t), v1)

    return face.evaluate(u, v).reshape(len(t), 3)


def _eval_boundary_normals_at_t(face, idx, t_vals):
    """在给定 t 值处评估面的第 idx 条边界略内侧的单位法向量数组 (N, 3)。"""
    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    t = np.asarray(t_vals)

    if idx == 0:
        u = np.full(len(t), u0 + _NORMAL_OFFSET);  v = v0 + t * (v1 - v0)
    elif idx == 1:
        u = np.full(len(t), u1 - _NORMAL_OFFSET);  v = v0 + t * (v1 - v0)
    elif idx == 2:
        u = u0 + t * (u1 - u0);   v = np.full(len(t), v0 + _NORMAL_OFFSET)
    else:
        u = u0 + t * (u1 - u0);   v = np.full(len(t), v1 - _NORMAL_OFFSET)

    normals = face.get_normal(u, v).reshape(len(t), 3)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    norms   = np.where(norms < 1e-12, 1.0, norms)
    return normals / norms


# ──────────────────────── 统一采样函数 ────────────────────────

def _get_boundary_edge(face, idx, n_samples):
    """获取 face 的第 idx 条边界边均匀采样点。"""
    if hasattr(face, 'get_edge_by_index'):
        return face.get_edge_by_index(idx, n_samples=n_samples)
    t = np.linspace(0.0, 1.0, n_samples)
    return _eval_boundary_edge_at_t(face, idx, t)


def _get_edge_points_and_normals(face, idx, n_samples):
    """
    获取 face 第 idx 条边的采样点和法向量，统一走 get_edge_by_index 路径。
    返回 (points (N,3), normals (N,3))。
    """
    if hasattr(face, 'get_edge_by_index_with_normals'):
        try:
            return face.get_edge_by_index_with_normals(idx, n_samples=n_samples)
        except NotImplementedError:
            pass

    if hasattr(face, 'get_edge_by_index'):
        pts = face.get_edge_by_index(idx, n_samples=n_samples)
        normals = _eval_boundary_normals_at_t(face, idx,
                                              np.linspace(0.0, 1.0, n_samples))
        return pts, normals

    t = np.linspace(0.0, 1.0, n_samples)
    pts = _eval_boundary_edge_at_t(face, idx, t)
    normals = _eval_boundary_normals_at_t(face, idx, t)
    return pts, normals



def _avg_min_dist(query, ref):
    """计算 query (N,3) 中每个点到 ref (M,3) 最近邻距离的均值。"""
    diff  = query[:, None, :] - ref[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    return float(np.mean(np.min(dists, axis=1)))


