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


def _find_shared_edge_pairs(face_a, face_b, n_samples=20, tol=1e-3,
                            min_edge_len=1e-6):
    """
    在 face_a 和 face_b 的边界边中找到所有共享边对。

    判定条件（全部满足才算共享边）：
    - 边弧长 >= min_edge_len（排除退化边）
    - ea 的两端点和中点到 eb 的最近距离都 <= tol（排除仅共顶点）

    返回:
        List[(idx_a, idx_b)]:  通过共享判定的所有边对（顺序与 face_a 边序对齐）
                               空列表表示找不到共享边
    """
    edges_a = [_get_boundary_edge(face_a, idx, n_samples)
               for idx in range(_num_edges(face_a))]
    edges_b = [_get_boundary_edge(face_b, idx, n_samples)
               for idx in range(_num_edges(face_b))]

    pairs = []
    used_b = set()  # 避免 face_b 的同一条边被多次匹配（共面镜像板 4 边一一对应）
    for ia, ea in enumerate(edges_a):
        if _arc_length(ea) < min_edge_len:
            continue
        best_d = np.inf
        best_ib = None
        for ib, eb in enumerate(edges_b):
            if ib in used_b:
                continue
            if _arc_length(eb) < min_edge_len:
                continue
            # 两端点 + 中点都要匹配
            d0 = np.min(np.linalg.norm(eb - ea[0], axis=1))
            d1 = np.min(np.linalg.norm(eb - ea[-1], axis=1))
            dm = np.min(np.linalg.norm(eb - ea[len(ea) // 2], axis=1))
            if d0 > tol or d1 > tol or dm > tol:
                continue
            d = _avg_min_dist(ea, eb)
            if d < best_d:
                best_d = d
                best_ib = ib
        if best_ib is not None:
            pairs.append((ia, best_ib))
            used_b.add(best_ib)

    return pairs


def find_shared_edges(face_a, face_b, max_angle_deg=2.0, n_initial=100,
                      max_pts=500):
    """
    自动寻找 face_a 与 face_b 的所有共享边，返回每条边的自适应采样数据。

    用于覆盖：
    - 一般情况下两面只共享 1 条边（Brick / Wedge 相邻面）
    - 共面镜像情况下两面共享多条边（双面薄板的 4 条边界）

    返回:
        List[Tuple]:  每条共享边一个元组
            (edge_points (N,3), normals_a (N,3), normals_b (N,3),
             inwards_a (N,3), exterior_angle_rad, warning_msg)

        inwards_a: face_a 上 "由边指向面内" 的单位向量；ptd_core 用来
                   定楔形截面 e2 朝向（对 α=2π 刀刃边尤其关键）。

    异常:
        ValueError: 两面没有共享边
    """
    pairs = _find_shared_edge_pairs(face_a, face_b)
    if not pairs:
        raise ValueError("找不到共享边")

    max_angle_rad = np.deg2rad(max_angle_deg)
    results = []

    for idx_a, idx_b in pairs:
        n_pts = n_initial
        warning_msg = None

        for round_idx in range(4):
            edge_pts, normals_a, inwards_a = _get_edge_points_and_normals(
                face_a, idx_a, n_pts)
            _, normals_b, _ = _get_edge_points_and_normals(
                face_b, idx_b, n_pts)

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
        in_sel  = inwards_a[bp]

        cos_vals = np.einsum('ij,ij->i', na_sel, nb_sel)
        cos_mean = float(np.clip(np.mean(cos_vals), -1.0, 1.0))
        exterior_angle = np.pi + np.arccos(cos_mean)

        results.append((pts_sel, na_sel, nb_sel, in_sel, exterior_angle, warning_msg))

    return results


def find_shared_edge(face_a, face_b, max_angle_deg=2.0, n_initial=100,
                     max_pts=500):
    """
    向后兼容包装：返回 face_a / face_b 的第一条共享边。

    新代码请使用 find_shared_edges（复数）以获取全部共享边。
    """
    edges = find_shared_edges(face_a, face_b, max_angle_deg=max_angle_deg,
                              n_initial=n_initial, max_pts=max_pts)
    return edges[0]



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


def _eval_boundary_inwards_at_t(face, idx, t_vals):
    """
    给定 t 值，返回第 idx 条边界点的 "向面内" 单位向量 (N, 3)。

    用有限差分 (略向参数域内偏移点 - 边界点) 得到，等价于 ±∂P/∂u 或 ±∂P/∂v。
    这是 ptd_core 用来定 e2 朝向的关键量：必须从边指向 Face A 表面内部。
    """
    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    t = np.asarray(t_vals)

    if idx == 0:
        u_edge = np.full(len(t), u0)
        v_edge = v0 + t * (v1 - v0)
        u_in   = np.full(len(t), u0 + _NORMAL_OFFSET)
        v_in   = v_edge
    elif idx == 1:
        u_edge = np.full(len(t), u1)
        v_edge = v0 + t * (v1 - v0)
        u_in   = np.full(len(t), u1 - _NORMAL_OFFSET)
        v_in   = v_edge
    elif idx == 2:
        u_edge = u0 + t * (u1 - u0)
        v_edge = np.full(len(t), v0)
        u_in   = u_edge
        v_in   = np.full(len(t), v0 + _NORMAL_OFFSET)
    else:
        u_edge = u0 + t * (u1 - u0)
        v_edge = np.full(len(t), v1)
        u_in   = u_edge
        v_in   = np.full(len(t), v1 - _NORMAL_OFFSET)

    p_edge = face.evaluate(u_edge, v_edge).reshape(len(t), 3)
    p_in   = face.evaluate(u_in,   v_in  ).reshape(len(t), 3)
    delta  = p_in - p_edge
    dnorm  = np.linalg.norm(delta, axis=1, keepdims=True)
    dnorm  = np.where(dnorm < 1e-12, 1.0, dnorm)
    return delta / dnorm


# ──────────────────────── 统一采样函数 ────────────────────────

def _get_boundary_edge(face, idx, n_samples):
    """获取 face 的第 idx 条边界边均匀采样点。"""
    if hasattr(face, 'get_edge_by_index'):
        return face.get_edge_by_index(idx, n_samples=n_samples)
    t = np.linspace(0.0, 1.0, n_samples)
    return _eval_boundary_edge_at_t(face, idx, t)


def _get_edge_points_and_normals(face, idx, n_samples):
    """
    获取 face 第 idx 条边的采样点、法向量、面内方向。
    返回 (points (N,3), normals (N,3), inwards (N,3))。

    inwards: 从边界点指向面内的单位向量；ptd_core 用来定 e2 朝向。
    若 face 提供 get_edge_by_index_with_normals 但无 inward，则 inward 用
    参数化有限差分回退求出。
    """
    t = np.linspace(0.0, 1.0, n_samples)

    if hasattr(face, 'get_edge_by_index_with_normals'):
        try:
            res = face.get_edge_by_index_with_normals(idx, n_samples=n_samples)
            if len(res) >= 3:
                return res[0], res[1], res[2]
            pts, normals = res
            inwards = _eval_boundary_inwards_at_t(face, idx, t)
            return pts, normals, inwards
        except NotImplementedError:
            pass

    if hasattr(face, 'get_edge_by_index'):
        pts = face.get_edge_by_index(idx, n_samples=n_samples)
        normals = _eval_boundary_normals_at_t(face, idx, t)
        inwards = _eval_boundary_inwards_at_t(face, idx, t)
        return pts, normals, inwards

    pts = _eval_boundary_edge_at_t(face, idx, t)
    normals = _eval_boundary_normals_at_t(face, idx, t)
    inwards = _eval_boundary_inwards_at_t(face, idx, t)
    return pts, normals, inwards



def _avg_min_dist(query, ref):
    """计算 query (N,3) 中每个点到 ref (M,3) 最近邻距离的均值。"""
    diff  = query[:, None, :] - ref[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    return float(np.mean(np.min(dists, axis=1)))


