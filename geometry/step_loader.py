import os
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool

from .occ_surface import OCCSurface

def load_step_file(filename):
    """
    读取 STEP 文件并返回 OCCSurface 对象列表。
    
    参数:
        filename: STEP 文件路径 (.stp / .step)
        
    返回:
        List[OCCSurface]: 包含文件中所有面的列表
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"STEP file not found: {filename}")
        
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    
    if status != IFSelect_RetDone:
        raise ValueError(f"Error reading STEP file: {filename}")
        
    # 转换所有根形状
    # TransferRoots() 将 STEP 实体转换为 OpenCascade 的 TopoDS_Shape
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    surfaces = []
    
    # 遍历形状中的所有面 (Faces)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        
        # 将 TopoDS_Face 转换为 Geom_Surface
        # BRep_Tool.Surface(face) 返回 handle<Geom_Surface>
        occ_geom = BRep_Tool.Surface(face)
        
        # 注意：这里我们忽略了面的修剪 (P-Curves/Wires)
        # BRep_Tool.Surface 返回的是底层的几何曲面（可能是无限的）。
        # 如果 STEP 中的面是修剪过的（例如带孔的板），仅仅使用 underlying surface 是不准确的。
        # 完整的实现需要处理参数域裁剪。
        # 但对于简单的几何体（如完整的圆柱面、立方体的面），这通常是可用的，
        # 只要 OCCSurface 能够读取到正确的 Bounds。
        # 
        # 实际上，TopoDS_Face 包含了裁剪信息。
        # 我们的 OCCSurface 目前是基于 Geom_Surface 的。
        # 若要支持裁剪，需要更复杂的逻辑 (检查 IsUPeriodic 等，或使用 BRepAdaptor_Surface)。
        
        # 暂时方案：直接使用底层曲面。这对于验证标准体（Box, Cylinder）通常是 OK 的，
        # 只要它们是 Parametric Rectangular 的。
        
        surfaces.append(OCCSurface(occ_geom))
        
        exp.Next()
        
    print(f"Loaded {len(surfaces)} surfaces from {filename}")
    return surfaces
