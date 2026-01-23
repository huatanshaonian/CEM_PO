import os
import re
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods

from .occ_surface import OCCFaceSurface


def _parse_step_entity_ids(filename):
    """
    直接解析 STEP 文件，提取 ADVANCED_FACE 和 EDGE_CURVE 的实体 ID。

    返回:
        face_ids: list of int, ADVANCED_FACE 实体的 #ID（按文件顺序）
        edge_ids: list of int, EDGE_CURVE 实体的 #ID（按文件顺序）
    """
    face_ids = []
    edge_ids = []

    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                # 匹配 #123=ADVANCED_FACE(...) 格式
                match = re.match(r'#(\d+)\s*=\s*ADVANCED_FACE\s*\(', line)
                if match:
                    face_ids.append(int(match.group(1)))
                    continue
                # 匹配 #123=EDGE_CURVE(...) 格式
                match = re.match(r'#(\d+)\s*=\s*EDGE_CURVE\s*\(', line)
                if match:
                    edge_ids.append(int(match.group(1)))
    except Exception as e:
        print(f"Warning: Could not parse STEP file: {e}")

    return face_ids, edge_ids


def load_step_file(filename, max_param_range=100, scale=1.0):
    """
    读取 STEP 文件并返回 OCCFaceSurface 对象列表。
    使用 BRepAdaptor_Surface 正确处理 trimming 边界。
    自动过滤参数域过大的面（如无限平面）。

    参数:
        filename: STEP 文件路径 (.stp / .step)
        max_param_range: 参数域范围阈值，超过此值的面会被跳过
        scale: 坐标缩放系数（例如 0.001 将 mm 转换为 m）

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

    # 从 STEP 文件解析实体 ID
    step_face_ids, step_edge_ids = _parse_step_entity_ids(filename)
    print(f"  Parsed {len(step_face_ids)} ADVANCED_FACE, {len(step_edge_ids)} EDGE_CURVE from STEP file")

    surfaces = []
    skipped = 0

    # 遍历形状中的所有面 (Faces)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0
    while exp.More():
        # 转换为 TopoDS_Face
        face = topods.Face(exp.Current())
        surf = OCCFaceSurface(face, scale=scale)

        # 获取 STEP 实体 ID（假设遍历顺序与文件顺序一致）
        step_id = step_face_ids[face_idx] if face_idx < len(step_face_ids) else -1
        surf.step_id = step_id

        # 获取该面所有边的局部索引（边的对应关系更复杂，暂用局部索引）
        edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
        edge_local_ids = []
        edge_idx = 0
        while edge_exp.More():
            edge_local_ids.append(edge_idx)
            edge_idx += 1
            edge_exp.Next()
        surf.edge_step_ids = edge_local_ids
        surf.n_edges = edge_idx

        # 检查参数域范围
        u_range = surf.u_max - surf.u_min
        v_range = surf.v_max - surf.v_min

        if u_range > max_param_range or v_range > max_param_range:
            print(f"  Skipping face {face_idx} (#{step_id}): param range too large (u={u_range:.1f}, v={v_range:.1f})")
            skipped += 1
        else:
            surfaces.append(surf)

        face_idx += 1
        exp.Next()

    print(f"Loaded {len(surfaces)} surfaces from {filename} (skipped {skipped} invalid faces)")
    return surfaces
