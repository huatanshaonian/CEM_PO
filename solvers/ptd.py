import re
import numpy as np
from .ptd_structures import PTDEdge
from .ptd_edge_finder import find_shared_edges
from physics.ptd_algorithms import get_ptd_function, DEFAULT_PTD_ALGORITHM
from physics.mec_truncated_geometry import compute_truncation_length


def _refine_polyline(pts, normals_a, normals_b, inwards_a, max_step):
    """
    把折线 (N, 3) 重采样到段长 <= max_step. 法向/inward 线性插值后归一化.

    用途: Johansen 截断 MEC 在直边长度上 l_A 沿边变化时, 把"整条边一个 segment"
    细化成多个 λ/L 级别的子段, 让 seg.l_A 在段内"常数假设"更准确.
    """
    if max_step is None or max_step <= 0 or len(pts) < 2:
        return pts, normals_a, normals_b, inwards_a

    new_pts = [pts[0]]
    new_na = [normals_a[0]]
    new_nb = [normals_b[0]] if normals_b is not None else None
    new_iw = [inwards_a[0]] if inwards_a is not None else None

    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        na0, na1 = normals_a[i], normals_a[i + 1]
        nb0, nb1 = (normals_b[i], normals_b[i + 1]) if normals_b is not None else (None, None)
        iw0, iw1 = (inwards_a[i], inwards_a[i + 1]) if inwards_a is not None else (None, None)

        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len <= max_step:
            new_pts.append(p1)
            new_na.append(na1)
            if new_nb is not None: new_nb.append(nb1)
            if new_iw is not None: new_iw.append(iw1)
            continue

        n_sub = int(np.ceil(seg_len / max_step))
        for j in range(1, n_sub + 1):
            t = j / n_sub
            new_pts.append(p0 + t * (p1 - p0))

            na_t = (1 - t) * na0 + t * na1
            na_t = na_t / max(np.linalg.norm(na_t), 1e-12)
            new_na.append(na_t)
            if new_nb is not None:
                nb_t = (1 - t) * nb0 + t * nb1
                nb_t = nb_t / max(np.linalg.norm(nb_t), 1e-12)
                new_nb.append(nb_t)
            if new_iw is not None:
                iw_t = (1 - t) * iw0 + t * iw1
                iw_t = iw_t / max(np.linalg.norm(iw_t), 1e-12)
                new_iw.append(iw_t)

    return (
        np.array(new_pts),
        np.array(new_na),
        np.array(new_nb) if new_nb is not None else None,
        np.array(new_iw) if new_iw is not None else None,
    )


class PTDProcessor:
    """
    PTD (物理绕射理论) 处理器
    负责边缘提取和贡献计算。
    """

    @staticmethod
    def extract_edges_from_face_pairs(surfaces, pairs_text, max_angle_deg=2.0,
                                      verbose=True, max_seg_length=None):
        """
        解析 "(0,1);(1,2)" 格式的面对字符串，自动找共享边并计算外部二面角。

        参数:
            surfaces:      List[Surface]
            pairs_text:    str，格式 "(a,b);(c,d)"，括号可选
            max_angle_deg: 每段最大切线转角（度），超过则细化
            verbose:       是否打印每条边的提取日志。求解器默认 True；
                           GUI 可视化高频调用可传 False 避免刷屏。
            max_seg_length: 物理段长上限（米）。非 None 时, 在自适应切向角细分
                           之后再按弧长强制重采样: 任何段长 > 该值的会插入子节点。
                           典型设 λ/8 让 Johansen 截断 MEC 在 l_A 沿边变化的
                           非对称几何上更稳。None = 不再细分。

        返回:
            List[PTDEdge]
        """
        ptd_edges = []
        pairs = re.findall(r'(\d+)\s*,\s*(\d+)', str(pairs_text))
        if not pairs:
            return ptd_edges

        for a_str, b_str in pairs:
            a, b = int(a_str), int(b_str)
            if a >= len(surfaces) or b >= len(surfaces):
                if verbose:
                    print(f"Warning: Face index out of range ({a},{b}), max={len(surfaces)-1}")
                continue

            try:
                edges_data = find_shared_edges(
                    surfaces[a], surfaces[b], max_angle_deg=max_angle_deg
                )
                multi = len(edges_data) > 1
                for k, (edge_pts, normals_a, normals_b, inwards_a, ext_angle, warn) in enumerate(edges_data):
                    tag = f"({a},{b})#{k}" if multi else f"({a},{b})"
                    if warn and verbose:
                        print(f"  [PTD] 警告 面对 {tag}: {warn}")
                    # 按弧长强制细分 (λ/N 级别). 让 Johansen 截断 MEC l_A 段内常数
                    # 假设在非对称几何上更准确.
                    if max_seg_length is not None and max_seg_length > 0:
                        edge_pts, normals_a, normals_b, inwards_a = _refine_polyline(
                            edge_pts, normals_a, normals_b, inwards_a, max_seg_length)
                    lit_normal = np.mean(normals_a, axis=0)
                    edge = PTDEdge(
                        name=tag,
                        points=edge_pts,
                        lit_face_normal=lit_normal,
                        exterior_angle_rad=ext_angle,
                        point_normals=normals_a,
                        point_normals_b=normals_b,
                        point_inwards=inwards_a,
                    )
                    ptd_edges.append(edge)
                    if verbose:
                        print(f"  [PTD] 面对 {tag}: 外角 = {np.degrees(ext_angle):.1f}°，"
                              f"段数 = {len(edge.segments)}")
            except Exception as e:
                if verbose:
                    print(f"  [PTD] 面对 ({a},{b}) 边提取失败: {e}")

        # 预计算 Johansen 截断 MEC 所需的 l_A (沿 inward 方向到 trailing edge 距离)
        # β_0=π/2 简化下 l_A 与 k_dir 无关, 仅几何; 用 'michaeli_mec_truncated' 算法时必需,
        # 其他算法无害 (只是多算一遍几何). 不存为属性会被截断 MEC 检测并退化到非截断.
        for edge in ptd_edges:
            for seg in edge.segments:
                seg.l_A = compute_truncation_length(seg, None, ptd_edges)

        # 注入 Michaeli 1987 二阶 EEC 所需的 (_all_edges, _so_index):
        # 二阶是边对算法, 但 PTD 调度按"逐边累加"循环. 让第 0 条边算所有边对总和,
        # 其它边返 0, 避免在 rcs_analyzer 的 for-loop 里被重复加.
        for i, edge in enumerate(ptd_edges):
            edge._all_edges = ptd_edges
            edge._so_index = i

        return ptd_edges

    @staticmethod
    def compute_contribution(edge, wave, polarization='VV',
                             algorithm=DEFAULT_PTD_ALGORITHM):
        """
        计算单条边缘的 PTD 贡献。

        参数:
            edge:         PTDEdge 对象
            wave:         IncidentWave 对象
            polarization: 'VV' | 'HH' | 'VH' | 'HV'
                          ('VH'/'HV' 仅 michaeli_mec 支持)
            algorithm:    PTD 算法 ID, 见 physics/ptd_algorithms.py
        """
        if algorithm == 'ufimtsev_eew' and polarization not in ('VV', 'HH'):
            raise NotImplementedError(
                f"Ufimtsev EEW 暂不支持极化模式 '{polarization}'。"
                f"请改用 algorithm='michaeli_mec', 或将 polarization 设为 'VV'/'HH'。"
            )
        func = get_ptd_function(algorithm)
        return func(edge, wave, polarization)
