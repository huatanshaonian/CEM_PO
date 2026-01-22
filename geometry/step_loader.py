import os
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods

from .occ_surface import OCCFaceSurface

def load_step_file(filename, max_param_range=100):
    """
    读取 STEP 文件并返回 OCCFaceSurface 对象列表。
    使用 BRepAdaptor_Surface 正确处理 trimming 边界。
    自动过滤参数域过大的面（如无限平面）。

    参数:
        filename: STEP 文件路径 (.stp / .step)
        max_param_range: 参数域范围阈值，超过此值的面会被跳过

    返回:
        List[OCCFaceSurface]: 包含文件中有效面的列表
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"STEP file not found: {filename}")

    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)

    if status != IFSelect_RetDone:
        raise ValueError(f"Error reading STEP file: {filename}")

    # 转换所有根形状
    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    surfaces = []
    skipped = 0

    # 遍历形状中的所有面 (Faces)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0
    while exp.More():
        # 转换为 TopoDS_Face
        face = topods.Face(exp.Current())
        surf = OCCFaceSurface(face)

        # 检查参数域范围
        u_range = surf.u_max - surf.u_min
        v_range = surf.v_max - surf.v_min

        if u_range > max_param_range or v_range > max_param_range:
            print(f"  Skipping face {face_idx}: param range too large (u={u_range:.1f}, v={v_range:.1f})")
            skipped += 1
        else:
            surfaces.append(surf)

        face_idx += 1
        exp.Next()

    print(f"Loaded {len(surfaces)} surfaces from {filename} (skipped {skipped} invalid faces)")
    return surfaces
