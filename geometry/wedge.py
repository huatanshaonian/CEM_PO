import numpy as np
from .plate import AnalyticPlate
from .surface import TransformedSurface

def create_analytic_wedge(length=10.0, width=5.0, height=5.0):
    """
    创建解析直角楔形 (Right Angle Wedge)
    
    几何定义:
    - 棱边 (Edge) 位于 X 轴，从 x=-length/2 到 x=+length/2
    - Face 0 (Horizontal): XY 平面, y > 0. (0 < y < width)
      法向: +Z (0, 0, 1)
    - Face 1 (Vertical): XZ 平面, z < 0. (-height < z < 0)
      法向: -Y (0, -1, 0)
      
    返回:
        surfaces: list of Surface objects
        ptd_edge_ids: str, PTD 边缘定义字符串 (e.g. "F0E2")
    """
    
    # Face 0: Horizontal Plate
    # Local: x in [-L/2, L/2], y in [-W/2, W/2]
    # Shifted: y += W/2 -> y in [0, W]
    # Edge at y=0 is index 2 (v=-0.5)
    p1 = AnalyticPlate(length, width)
    s1 = TransformedSurface(p1, translation=[0, width/2, 0])
    
    # Face 1: Vertical Plate
    # Local: x in [-L/2, L/2], y in [-H/2, H/2]
    # Target: y=0, z in [-H, 0]. Normal -Y.
    # Rotation: +90 deg around X axis.
    #   y_local -> z_global
    #   z_local -> -y_global
    #   Normal +z_local -> -y_global (Correct)
    # Translation:
    #   We want z_global range [-H, 0].
    #   y_local range [-H/2, H/2] maps to z_global [-H/2, H/2].
    #   We need to shift z_global by -H/2.
    #   So translation = [0, 0, -H/2].
    # Edge at z=0 (global) corresponds to y_local = +H/2.
    #   This is index 3 (v=+0.5).
    
    R = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]]) # Rx(90) logic: y->z, z->-y ?
                  # Col 1: X axis -> (1,0,0)
                  # Col 2: Y axis -> (0,0,1) (+Z)
                  # Col 3: Z axis -> (0,-1,0) (-Y)
                  
    p2 = AnalyticPlate(length, height)
    s2 = TransformedSurface(p2, rotation_matrix=R, translation=[0, 0, -height/2])
    
    surfaces = [s1, s2]
    
    # PTD Edge: Common edge is at the corner.
    # For s1 (Face 0), edge at y=0 is Edge 2.
    # For s2 (Face 1), edge at z=0 is Edge 3.
    # We can pick either. Let's pick F0E2.
    ptd_id = "F0E2" 
    
    return surfaces, ptd_id
