import re
import numpy as np
from .ptd_structures import PTDEdge
from .ptd_edge_finder import find_shared_edge
from physics.ptd_core import compute_ptd_contribution


class PTDProcessor:
    """
    PTD (物理绕射理论) 处理器
    负责边缘提取和贡献计算。
    """

    @staticmethod
    def extract_edges_from_face_pairs(surfaces, pairs_text, n_samples=120):
        """
        解析 "(0,1);(1,2)" 格式的面对字符串，自动找共享边并计算外部二面角。

        参数:
            surfaces:   List[Surface]
            pairs_text: str，格式 "(a,b);(c,d)"，括号可选
            n_samples:  边缘离散采样点数

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
                edge_pts, normals_a, normals_b, ext_angle = find_shared_edge(
                    surfaces[a], surfaces[b], n_samples=n_samples
                )
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
                print(f"  [PTD] 面对 ({a},{b}): 外角 = {np.degrees(ext_angle):.1f}°")
            except Exception as e:
                print(f"  [PTD] 面对 ({a},{b}) 边提取失败: {e}")

        return ptd_edges

    @staticmethod
    def compute_contribution(edge, wave, polarization='VV'):
        """
        计算单条边缘的 PTD 贡献
        """
        return compute_ptd_contribution(edge, wave, polarization)
