import numpy as np
from .plate import AnalyticPlate
from .surface import TransformedSurface

def create_analytic_brick(length=10.0, width=5.0, height=3.0):
    """
    创建解析长方体 (Brick / Cuboid)
    中心在原点。
    
    参数:
        length: X 轴方向长度
        width:  Y 轴方向长度
        height: Z 轴方向长度
        
    返回:
        surfaces: list of Surface objects (6 faces)
        ptd_ids: str, PTD 边缘定义字符串 (12 edges)
    """
    surfaces = []
    
    # 定义 6 个面
    # Face 0: Top (+Z)
    s0 = TransformedSurface(AnalyticPlate(length, width), translation=[0, 0, height/2])
    # Face 1: Bottom (-Z) - Normal needs to be -Z. Plate normal is +Z. Rotate 180 around Y.
    # Rx(180): y->-y, z->-z.
    R_x180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    s1 = TransformedSurface(AnalyticPlate(length, width), rotation_matrix=R_x180, translation=[0, 0, -height/2])
    
    # Face 2: Front (+X) - Plate in YZ plane. Normal +X.
    # Rotate +90 around Y? No, Plate +Z -> +X. Ry(90): z->x, x->-z.
    R_y90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    s2 = TransformedSurface(AnalyticPlate(height, width), rotation_matrix=R_y90, translation=[length/2, 0, 0])
    # Note: AnalyticPlate dims are (width, height) -> (u, v).
    # Here mapped to (z, y) or similar.
    
    # Face 3: Back (-X) - Normal -X.
    # Ry(-90): z->-x.
    R_yn90 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    s3 = TransformedSurface(AnalyticPlate(height, width), rotation_matrix=R_yn90, translation=[-length/2, 0, 0])
    
    # Face 4: Right (+Y) - Plate in XZ plane. Normal +Y.
    # Rx(-90): z->y.
    R_xn90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    s4 = TransformedSurface(AnalyticPlate(length, height), rotation_matrix=R_xn90, translation=[0, width/2, 0])
    
    # Face 5: Left (-Y) - Normal -Y.
    # Rx(90): z->-y.
    R_x90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    s5 = TransformedSurface(AnalyticPlate(length, height), rotation_matrix=R_x90, translation=[0, -width/2, 0])
    
    surfaces = [s0, s1, s2, s3, s4, s5]
    
    # PTD Edges: 12 edges
    # Face 0 (Top) edges: 
    #   E0: u=-0.5 (Left, x=-L/2) -> Shared with Face 3 (Back)
    #   E1: u=+0.5 (Right, x=+L/2) -> Shared with Face 2 (Front)
    #   E2: v=-0.5 (Bottom, y=-W/2) -> Shared with Face 5 (Left)
    #   E3: v=+0.5 (Top, y=+W/2) -> Shared with Face 4 (Right)
    
    # Face 1 (Bottom) edges:
    #   Similar...
    
    # Vertical edges: Shared by Side faces.
    
    # Construct identifier string
    # We select edges from specific faces to avoid duplicates.
    # Top Face (F0) all 4 edges: F0E0, F0E1, F0E2, F0E3
    # Bottom Face (F1) all 4 edges: F1E0, F1E1, F1E2, F1E3
    # Vertical edges:
    #   Front-Right: F2E3 (Face 2 Top, mapped to +y?)
    #   Let's trace orientation.
    #   Instead of guessing, we can just list ALL edges of ALL faces?
    #   No, that would double count (24 edges).
    #   We need 12 unique edges.
    
    # Simple set:
    # Top Loop: F0E0, F0E1, F0E2, F0E3
    # Bottom Loop: F1E0, F1E1, F1E2, F1E3
    # Vertical Posts:
    #   Front-Right (+X, +Y): Intersection of F2 and F4.
    #   Front-Left (+X, -Y): Intersection of F2 and F5.
    #   Back-Right (-X, +Y): Intersection of F3 and F4.
    #   Back-Left (-X, -Y): Intersection of F3 and F5.
    
    # Let's verify Face 2 (Front +X) edges.
    # Plate(height, width).
    # Ry(90) mapping: Local X(u) -> Global Z (or -Z?). Local Y(v) -> Global Y.
    # So u is Z, v is Y.
    # Edges u=+-0.5 are Top/Bottom (Z).
    # Edges v=+-0.5 are Left/Right (Y).
    # We want Vertical edges (along Z). These are u=+-0.5 edges?
    # No, vertical edges run along Z. So u varies. v is constant.
    # So edges are v=+-0.5 (indices 2 and 3).
    
    edges = [
        "F0E0", "F0E1", "F0E2", "F0E3", # Top 4
        "F1E0", "F1E1", "F1E2", "F1E3", # Bottom 4
        "F2E2", "F2E3",                 # Front verticals
        "F3E2", "F3E3"                  # Back verticals
    ]
    
    ptd_id = ",".join(edges)
    
    return surfaces, ptd_id
