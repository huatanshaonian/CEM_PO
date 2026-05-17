import re
import numpy as np
from .ptd_structures import PTDEdge
from .ptd_edge_finder import find_shared_edge
from physics.ptd_algorithms import get_ptd_function, DEFAULT_PTD_ALGORITHM


class PTDProcessor:
    """
    PTD (物理绕射理论) 处理器
    负责边缘提取和贡献计算。
    """

    @staticmethod
    def extract_edges_from_face_pairs(surfaces, pairs_text, max_angle_deg=2.0):
        """
        解析 "(0,1);(1,2)" 格式的面对字符串，自动找共享边并计算外部二面角。

        参数:
            surfaces:      List[Surface]
            pairs_text:    str，格式 "(a,b);(c,d)"，括号可选
            max_angle_deg: 每段最大切线转角（度），超过则细化

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
                print(f"Warning: Face index out of range ({a},{b}), max={len(surfaces)-1}")
                continue

            try:
                edge_pts, normals_a, normals_b, ext_angle, warn = find_shared_edge(
                    surfaces[a], surfaces[b], max_angle_deg=max_angle_deg
                )
                if warn:
                    print(f"  [PTD] 警告 面对 ({a},{b}): {warn}")
                lit_normal = np.mean(normals_a, axis=0)
                edge = PTDEdge(
                    name=f"({a},{b})",
                    points=edge_pts,
                    lit_face_normal=lit_normal,
                    exterior_angle_rad=ext_angle,
                    point_normals=normals_a,
                    point_normals_b=normals_b,
                )
                ptd_edges.append(edge)
                print(f"  [PTD] 面对 ({a},{b}): 外角 = {np.degrees(ext_angle):.1f}°，"
                      f"段数 = {len(edge.segments)}")
            except Exception as e:
                print(f"  [PTD] 面对 ({a},{b}) 边提取失败: {e}")

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
