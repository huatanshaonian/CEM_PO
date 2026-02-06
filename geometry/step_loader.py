import os
import re
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopoDS import topods

from .occ_surface import OCCFaceSurface
from .iges_fix import fix_iges, needs_fix


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


def load_step_file(filename, max_param_range=100, scale=1.0, invert_indices=None):
    """
    读取 STEP 文件并返回 OCCFaceSurface 对象列表。
    使用 BRepAdaptor_Surface 正确处理 trimming 边界。
    自动过滤参数域过大的面（如无限平面）。

    参数:
        filename: STEP 文件路径 (.stp / .step)
        max_param_range: 参数域范围阈值，超过此值的面会被跳过
        scale: 坐标缩放系数（例如 0.001 将 mm 转换为 m）
        invert_indices: 需要翻转法向量的面索引列表 (0-based sequence index)

    返回:
        List[OCCFaceSurface]: 包含文件中有效面的列表
    """
    if invert_indices is None:
        invert_indices = []

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
    valid_idx = 0
    while exp.More():
        # 转换为 TopoDS_Face
        face = topods.Face(exp.Current())
        
        # 默认不翻转，待确认有效后根据 valid_idx 设置
        surf = OCCFaceSurface(face, scale=scale, invert_normal=False)

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
            # 使用 valid_idx (有效面索引) 来判断是否翻转，与 GUI 显示的序号保持一致
            should_invert = valid_idx in invert_indices
            if should_invert:
                print(f"  Inverting normal for valid face index {valid_idx} (#{step_id})")
                surf.invert_normal = True # 直接设置属性
            
            surfaces.append(surf)
            valid_idx += 1

        face_idx += 1
        exp.Next()

    print(f"Loaded {len(surfaces)} surfaces from {filename} (skipped {skipped} invalid faces)")
    return surfaces


def _iges_read_shapes(filepath):
    """用 OCC 读取 IGES 文件，返回 (faces 列表, 成功标志)。"""
    reader = IGESControl_Reader()
    status = reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        return [], False

    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        return [], True

    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()
    return faces, True


def _mirror_face(face, mirror_plane, scale):
    """对 TopoDS_Face 做镜像，返回镜像后的 OCCFaceSurface。"""
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Trsf
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    normals = {'X=0': (1, 0, 0), 'Y=0': (0, 1, 0), 'Z=0': (0, 0, 1)}
    if mirror_plane not in normals:
        raise ValueError(f"Invalid mirror plane: {mirror_plane}, use X=0/Y=0/Z=0")
    nx, ny, nz = normals[mirror_plane]

    trsf = gp_Trsf()
    trsf.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(nx, ny, nz)))

    builder = BRepBuilderAPI_Transform(face, trsf, True)
    mirrored_face = topods.Face(builder.Shape())
    return OCCFaceSurface(mirrored_face, scale=scale, invert_normal=False)


def _rotate_faces(faces, rotation_deg):
    """
    对 TopoDS_Face 列表绕原点旋转（依次绕 X → Y → Z 轴）。
    rotation_deg: (rx, ry, rz) 旋转角度，单位为度。
    返回旋转后的 face 列表。
    """
    import math
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Trsf
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    rx, ry, rz = rotation_deg
    if abs(rx) < 1e-10 and abs(ry) < 1e-10 and abs(rz) < 1e-10:
        return faces

    origin = gp_Pnt(0, 0, 0)
    trsf = gp_Trsf()

    # 依次绕 X → Y → Z
    for angle_deg, direction in [(rx, (1, 0, 0)), (ry, (0, 1, 0)), (rz, (0, 0, 1))]:
        if abs(angle_deg) < 1e-10:
            continue
        t = gp_Trsf()
        t.SetRotation(gp_Ax1(origin, gp_Dir(*direction)), math.radians(angle_deg))
        trsf = t.Multiplied(trsf)

    rotated = []
    for face in faces:
        builder = BRepBuilderAPI_Transform(face, trsf, True)
        rotated.append(topods.Face(builder.Shape()))
    return rotated


def load_iges_file(filename, max_param_range=100, scale=1.0,
                   invert_indices=None, delete_indices=None,
                   mirror_plane=None, rotation=None):
    """
    读取 IGES 文件并返回 OCCFaceSurface 对象列表。
    对非标准 IGES（如 Tecplot 导出）自动进行格式预处理后再交给 OCC 读取。

    处理顺序: 读取 → 删面 → 翻转法向 → 镜像 → 旋转

    参数:
        filename: IGES 文件路径 (.igs / .iges)
        max_param_range: 参数域范围阈值，超过此值的面会被跳过
        scale: 坐标缩放系数（例如 0.001 将 mm 转换为 m）
        invert_indices: 需要翻转法向量的面索引列表 (0-based valid index)
        delete_indices: 需要删除的面索引列表 (0-based valid index)
        mirror_plane: 对称面，'X=0'/'Y=0'/'Z=0' 或 None；设置后对所有剩余面生成镜像副本
        rotation: 绕原点旋转角度 (rx, ry, rz)，单位为度，None 表示不旋转

    返回:
        List[OCCFaceSurface]: 包含文件中有效面的列表
    """
    if invert_indices is None:
        invert_indices = []
    if delete_indices is None:
        delete_indices = []

    if not os.path.exists(filename):
        raise FileNotFoundError(f"IGES file not found: {filename}")

    # 检测并修复非标准 IGES 格式（如 Tecplot 导出的缺逗号/分号文件）
    if needs_fix(filename):
        backup = filename + '.bak'
        if not os.path.exists(backup):
            import shutil
            shutil.copy2(filename, backup)
            print(f"  IGES: backed up original to {os.path.basename(backup)}")
        fix_iges(filename, filename)
        print(f"  IGES: fixed non-standard P-section format in-place")

    faces, ok = _iges_read_shapes(filename)
    if not ok:
        raise ValueError(f"Error reading IGES file: {filename}")

    if len(faces) == 0:
        raise ValueError(f"Could not extract any faces from IGES file: {filename}")

    # 从 TopoDS_Face 构建 OCCFaceSurface
    surfaces = []
    skipped = 0
    deleted = 0
    valid_idx = 0

    for face_idx, face in enumerate(faces):
        surf = OCCFaceSurface(face, scale=scale, invert_normal=False)
        surf.step_id = face_idx

        edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
        edge_count = 0
        edge_local_ids = []
        while edge_exp.More():
            edge_local_ids.append(edge_count)
            edge_count += 1
            edge_exp.Next()
        surf.edge_step_ids = edge_local_ids
        surf.n_edges = edge_count

        u_range = surf.u_max - surf.u_min
        v_range = surf.v_max - surf.v_min

        if u_range > max_param_range or v_range > max_param_range:
            print(f"  Skipping face {face_idx}: param range too large (u={u_range:.1f}, v={v_range:.1f})")
            skipped += 1
        else:
            if valid_idx in delete_indices:
                print(f"  Deleted face index {valid_idx}")
                deleted += 1
                valid_idx += 1
                continue

            if valid_idx in invert_indices:
                print(f"  Inverting normal for face index {valid_idx}")
                surf.invert_normal = True

            surfaces.append(surf)
            valid_idx += 1

    # 镜像：对所有剩余面生成镜像副本
    if mirror_plane:
        n_orig = len(surfaces)
        for surf in surfaces[:n_orig]:
            try:
                mirrored = _mirror_face(surf.face, mirror_plane, scale)
                surfaces.append(mirrored)
            except Exception as e:
                print(f"  Warning: mirror failed for face: {e}")
        print(f"  Mirrored {len(surfaces) - n_orig} faces across {mirror_plane}")

    # 旋转：对所有面（含镜像）统一旋转，保证镜像在原始坐标系完成
    if rotation:
        all_faces = [s.face for s in surfaces]
        rotated_faces = _rotate_faces(all_faces, rotation)
        for i, rf in enumerate(rotated_faces):
            inv = surfaces[i].invert_normal
            surfaces[i] = OCCFaceSurface(rf, scale=scale, invert_normal=inv)
        print(f"  Rotated all {len(surfaces)} faces by ({rotation[0]}, {rotation[1]}, {rotation[2]}) deg")

    msg = f"Loaded {len(surfaces)} surfaces from {os.path.basename(filename)}"
    if deleted:
        msg += f" (deleted {deleted})"
    if skipped:
        msg += f" (skipped {skipped})"
    print(msg)
    return surfaces


def load_cad_file(filename, max_param_range=100, scale=1.0, invert_indices=None):
    """
    统一的 CAD 文件加载接口，根据文件扩展名自动选择 STEP 或 IGES 读取器。

    参数:
        filename: CAD 文件路径 (.step/.stp 或 .iges/.igs)
        max_param_range: 参数域范围阈值
        scale: 坐标缩放系数
        invert_indices: 需要翻转法向量的面索引列表

    返回:
        List[OCCFaceSurface]: 包含文件中有效面的列表
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in ('.step', '.stp'):
        return load_step_file(filename, max_param_range, scale, invert_indices)
    elif ext in ('.iges', '.igs'):
        return load_iges_file(filename, max_param_range, scale, invert_indices)
    else:
        raise ValueError(f"Unsupported CAD file format: {ext}. Supported: .step, .stp, .iges, .igs")
