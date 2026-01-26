import re
import numpy as np
from .ptd_structures import PTDEdge

def extract_manual_edges(surfaces, edge_identifiers):
    """
    根据标识符列表提取 PTD 边缘对象

    参数:
        surfaces: List[OCCFaceSurface] 几何面列表
        edge_identifiers: List[str] 标识符列表 (如 ["F0E1", "F1E2"])

    返回:
        List[PTDEdge]
    """
    ptd_edges = []

    for ident in edge_identifiers:
        # 解析 FxEy 格式 (Face x, Edge y)
        match = re.match(r'F(\d+)E(\d+)', ident)
        if not match:
            print(f"Warning: Invalid edge identifier format: {ident}")
            continue

        f_idx = int(match.group(1))
        e_idx = int(match.group(2))

        if f_idx < 0 or f_idx >= len(surfaces):
            print(f"Warning: Face index {f_idx} out of range (max {len(surfaces)-1})")
            continue

        face = surfaces[f_idx]

        try:
            # 1. 获取边缘离散点
            if hasattr(face, 'get_edge_by_index'):
                # 获取点序列 (N, 3)
                points = face.get_edge_by_index(e_idx, n_samples=60)
            else:
                print(f"Warning: Surface object does not support edge extraction: {type(face)}")
                continue

            # 2. 获取亮面法向 (Lit Face Normal)
            # 使用面中心的法向作为近似
            u_min, u_max = face.u_domain
            v_min, v_max = face.v_domain
            mid_u = (u_min + u_max) / 2
            mid_v = (v_min + v_max) / 2
            _, normal, _ = face.get_data(mid_u, mid_v)
            normal = normal.flatten()
            
            # 3. 创建 PTD Edge 对象
            # 注意: PTDEdge 构造函数签名是 (name, points, lit_face_normal, wedge_angle_deg)
            edge = PTDEdge(ident, points, normal, wedge_angle_deg=90.0)
            ptd_edges.append(edge)

        except Exception as e:
            print(f"Error extracting edge {ident}: {e}")
            import traceback
            traceback.print_exc()

    return ptd_edges