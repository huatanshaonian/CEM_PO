"""
PTD 自动边提取：给定两个相邻面，自动找共享边并计算外部二面角。

采样策略：先均匀采 n_initial 点，用 _adaptive_breakpoints 合并（删点），
若合并后仍有段超标则插入中点（最多 3 轮），返回最终自适应点集。
"""
import numpy as np

_NORMAL_OFFSET = 1e-5  # 法向量采样的参数域内偏量


def find_shared_edge(face_a, face_b, max_angle_deg=2.0, n_initial=100,
                     n_search=15, max_pts=500):
    """
    自动寻找 face_a 与 face_b 的共享边，返回自适应采样的边缘点及法向量。

    采样策略：
    1. 粗搜索（n_search 点）找哪条边界最近
    2. 均匀采 n_initial 点
    3. _adaptive_breakpoints 合并：删去冗余点，使每段累积切线转角 ≤ max_angle_deg
    4. 检查：若仍有段超标（n_initial 不够密时），插入中点并重新合并，最多 3 轮
    5. 3 轮后仍超标则返回警告字符串（由调用方打印到 log）

    返回:
        edge_points (N,3):     共享边采样点
        normals_a (N,3):       face_a 在该边略内侧的法向量
        normals_b (N,3):       face_b 在该边略内侧的法向量
        exterior_angle_rad:    外部二面角（弧度）
        warning_msg:           str 或 None，超标警告信息

    异常:
        ValueError: 两面没有共享边
    """
    max_angle_rad = np.deg2rad(max_angle_deg)
    scale = _estimate_face_scale(face_a)
    tol = 0.05 * scale

    # ── Step 1: 粗搜索找最近边 ──
    ref_b = _sample_surface_grid(face_b, n_search)
    best_dist = np.inf
    best_edge_idx = None
    best_from_a = True

    for idx in range(4):
        pts = _get_boundary_edge(face_a, idx, n_search)
        d = _avg_min_dist(pts, ref_b)
        if d < best_dist:
            best_dist = d
            best_edge_idx = idx

    if best_dist > tol:
        ref_a = _sample_surface_grid(face_a, n_search)
        for idx in range(4):
            pts = _get_boundary_edge(face_b, idx, n_search)
            d = _avg_min_dist(pts, ref_a)
            if d < best_dist:
                best_dist = d
                best_edge_idx = idx
                best_from_a = False

    if best_dist > tol:
        raise ValueError(
            f"找不到共享边（最小平均距离 {best_dist:.4f} > 容差 {tol:.4f}）"
        )

    # ── Step 2: 确定所在面和边界索引 ──
    if best_from_a:
        edge_face   = face_a
        edge_idx    = best_edge_idx
        other_face  = face_b
    else:
        edge_face   = face_b
        edge_idx    = best_edge_idx
        other_face  = face_a

    # 找 other_face 最近边界（粗搜索阶段，t 均匀）
    coarse_pts = _get_boundary_edge(edge_face, edge_idx, n_search)
    other_best_dist = np.inf
    other_best_idx  = 0
    for idx in range(4):
        bnd = _get_boundary_edge(other_face, idx, n_search)
        d = _avg_min_dist(coarse_pts, bnd)
        if d < other_best_dist:
            other_best_dist = d
            other_best_idx  = idx

    # ── Step 3: 均匀采 n_initial 点 ──
    t_vals = np.linspace(0.0, 1.0, n_initial)
    edge_pts  = _eval_boundary_edge_at_t(edge_face,  edge_idx,      t_vals)
    normals_a = _eval_boundary_normals_at_t(edge_face,  edge_idx,      t_vals)
    normals_b = _eval_boundary_normals_at_t(other_face, other_best_idx, t_vals)

    # ── Step 4: 合并 + 最多 3 轮插入 ──
    warning_msg = None
    for round_idx in range(4):   # round 0 = 纯合并；round 1-3 = 插入后再合并
        # 用 _adaptive_breakpoints 合并
        bp = _adaptive_breakpoints(edge_pts, max_angle_rad)
        t_sel   = t_vals[bp]
        pts_sel = edge_pts[bp]
        na_sel  = normals_a[bp]
        nb_sel  = normals_b[bp]

        # 检查合并后各段是否仍超标（仅当 bp 相邻点间距 > 1 个初始段时才可能）
        bad_segs = _find_bad_segments(pts_sel, max_angle_rad)

        if len(bad_segs) == 0:
            break  # 全部合格

        if round_idx == 3:
            # 3 轮插入后仍超标
            warning_msg = (
                f"3 轮细化后仍有 {len(bad_segs)} 段切线转角超过 {max_angle_deg:.1f}°，"
                f"请增大 PTD Seg. Angle 或检查边缘几何"
            )
            break

        # 在超标段中点处插入新 t 值（在原始密集 t_vals 中插值）
        new_t_set = set(t_vals.tolist())
        for seg_i in bad_segs:
            t_mid = (t_sel[seg_i] + t_sel[seg_i + 1]) / 2.0
            new_t_set.add(float(t_mid))

        new_t = np.array(sorted(new_t_set))
        if len(new_t) >= max_pts:
            warning_msg = (
                f"达到最大采样点数 {max_pts}，仍有 {len(bad_segs)} 段切线转角超过 "
                f"{max_angle_deg:.1f}°"
            )
            break

        t_vals    = new_t
        edge_pts  = _eval_boundary_edge_at_t(edge_face,  edge_idx,      t_vals)
        normals_a = _eval_boundary_normals_at_t(edge_face,  edge_idx,      t_vals)
        normals_b = _eval_boundary_normals_at_t(other_face, other_best_idx, t_vals)
    else:
        # for 循环正常结束（不应发生，因为 round 0 就会 break 或在内部 break）
        bp = _adaptive_breakpoints(edge_pts, max_angle_rad)
        t_sel   = t_vals[bp]
        pts_sel = edge_pts[bp]
        na_sel  = normals_a[bp]
        nb_sel  = normals_b[bp]

    # ── Step 5: 外部二面角 ──
    cos_vals = np.einsum('ij,ij->i', na_sel, nb_sel)
    cos_mean = float(np.clip(np.mean(cos_vals), -1.0, 1.0))
    exterior_angle = np.pi + np.arccos(cos_mean)

    return pts_sel, na_sel, nb_sel, exterior_angle, warning_msg


def faces_share_edge(face_a, face_b, n_samples=15, n_grid=15):
    """
    快速判断两个面是否共享边（不计算法向量，仅做距离检查）。
    用于 GUI 添加 PTD 面对前的预过滤。
    """
    scale = _estimate_face_scale(face_a)
    tol = 0.05 * scale

    ref_b = _sample_surface_grid(face_b, n_grid)
    best_dist = min(
        _avg_min_dist(_get_boundary_edge(face_a, idx, n_samples), ref_b)
        for idx in range(4)
    )
    if best_dist <= tol:
        return True

    ref_a = _sample_surface_grid(face_a, n_grid)
    best_dist = min(
        _avg_min_dist(_get_boundary_edge(face_b, idx, n_samples), ref_a)
        for idx in range(4)
    )
    return best_dist <= tol


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


def _find_bad_segments(pts, max_angle_rad):
    """
    检查点集 pts 中哪些段（相邻两节点间）的切线转角（在中间节点处）超过阈值。
    返回超标的段索引列表（段 i = pts[i]→pts[i+1]，在节点 i+1 处检查）。
    """
    if len(pts) < 3:
        return []

    tangents = pts[1:] - pts[:-1]
    lengths  = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths  = np.where(lengths < 1e-12, 1.0, lengths)
    tangents = tangents / lengths

    cos_a  = np.clip(np.einsum('ij,ij->i', tangents[:-1], tangents[1:]), -1.0, 1.0)
    angles = np.arccos(cos_a)   # (N-2,)：节点 1..N-2 处的转角

    # 节点 k+1 超标 → 段 k 和段 k+1 需要细分（返回较小的段索引）
    bad = set()
    for k in np.where(angles > max_angle_rad)[0]:
        bad.add(int(k))
        bad.add(int(k + 1))
    return sorted(bad)


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


# ──────────────────────── 粗搜索辅助函数 ────────────────────────

def _get_boundary_edge(face, idx, n_samples):
    """获取 face 的第 idx 条边界边均匀采样点（用于粗搜索）。"""
    if hasattr(face, 'get_edge_by_index'):
        return face.get_edge_by_index(idx, n_samples=n_samples)
    t = np.linspace(0.0, 1.0, n_samples)
    return _eval_boundary_edge_at_t(face, idx, t)


def _sample_surface_grid(face, n_grid):
    """在面上均匀采样 n_grid×n_grid 点，返回 (n_grid²,3)。仅用于寻边距离比较。"""
    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    u = np.linspace(u0, u1, n_grid)
    v = np.linspace(v0, v1, n_grid)
    ug, vg = np.meshgrid(u, v)
    return face.evaluate(ug, vg).reshape(-1, 3)


def _avg_min_dist(query, ref):
    """计算 query (N,3) 中每个点到 ref (M,3) 最近邻距离的均值。"""
    diff  = query[:, None, :] - ref[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    return float(np.mean(np.min(dists, axis=1)))


def _estimate_face_scale(face):
    """估算面的特征尺寸（四角连线对角线长度的均值）。"""
    u0, u1 = face.u_domain
    v0, v1 = face.v_domain
    corners = [face.evaluate(u0, v0), face.evaluate(u0, v1),
               face.evaluate(u1, v0), face.evaluate(u1, v1)]
    corners = [c.flatten() for c in corners]
    d1 = np.linalg.norm(corners[3] - corners[0])
    d2 = np.linalg.norm(corners[2] - corners[1])
    return max((d1 + d2) / 2.0, 1e-6)
