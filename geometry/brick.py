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
    
    # PTD 面对：12 条棱，每条棱由两个相邻面自动提取
    # 面索引：0=Top, 1=Bottom, 2=Front(+X), 3=Back(-X), 4=Right(+Y), 5=Left(-Y)
    ptd_id = "(0,2);(0,3);(0,4);(0,5);(1,2);(1,3);(1,4);(1,5);(2,4);(2,5);(3,4);(3,5)"
    
    return surfaces, ptd_id
