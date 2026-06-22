import os
import numpy as np
from geometry.plate import AnalyticPlate, create_double_sided_plate
from geometry.triangle import create_double_sided_triangle
from geometry.sphere import AnalyticSphere
from geometry.cylinder import AnalyticCylinder
from geometry.occ_surface import OCCSurface
from geometry.step_loader import load_step_file, load_iges_file
from geometry.wedge import create_analytic_wedge
from geometry.brick import create_analytic_brick
from geometry.infinite_wedge import create_infinite_wedge
from geometry.occ_primitives import create_occ_cylinder

class GeometryFactory:
    """
    几何构建工厂，将参数字典转换为实体的 Surface 对象列表。
    可供不同的 GUI 框架(Tkinter/Qt)或 CLI 脚本共享。
    """
    @staticmethod
    def create_geometry(geo_type, params):
        """
        Args:
            geo_type (str): 'Cylinder', 'Plate', 'Triangle', 'Sphere', 'STEP File', 'Wedge', 'Brick', 'Infinite Wedge'
            params (dict): 包含 radius, height, width 等参数
        Returns:
            list: Surface 对象列表
        """
        if geo_type == "Cylinder":
            r = float(params.get('radius', 1.0))
            h = float(params.get('height', 2.0))
            return [AnalyticCylinder(r, h)]
        
        elif geo_type == "Plate":
            w = float(params.get('width', 5.0))
            l = float(params.get('length', 10.0))
            surfaces, ptd_id = create_double_sided_plate(w, l)
            return surfaces, ptd_id

        elif geo_type == "Triangle":
            p1 = params.get('p1', [0.0, 0.0, 0.0])
            p2 = params.get('p2', [1.0, 0.0, 0.0])
            p3 = params.get('p3', [0.0, 1.0, 0.0])
            surfaces, ptd_id = create_double_sided_triangle(p1, p2, p3)
            return surfaces, ptd_id

        elif geo_type == "Sphere":
            r = float(params.get('radius', 1.0))
            return [AnalyticSphere(r)]
        
        elif geo_type == "Wedge":
            w = float(params.get('width', 2.0))
            l = float(params.get('length', 5.0))
            h = float(params.get('height', 3.0))
            surfaces, ptd_id = create_analytic_wedge(l, w, h)
            return surfaces, ptd_id  # 返回元组以便 GUI 获取 PTD 边信息

        elif geo_type == "Brick":
            w = float(params.get('width', 2.0))
            l = float(params.get('length', 5.0))
            h = float(params.get('height', 3.0))
            surfaces, ptd_id = create_analytic_brick(l, w, h)
            return surfaces, ptd_id  # 返回元组以便 GUI 获取 PTD 边信息

        elif geo_type == "Infinite Wedge":
            edge_length = float(params.get('edge_length', 5.0))
            exterior_angle_deg = float(params.get('exterior_angle', 270.0))
            surfaces, ptd_id = create_infinite_wedge(edge_length, exterior_angle_deg)
            return surfaces, ptd_id  # 返回元组以便 GUI 获取 PTD 边信息

        elif geo_type == "OCC Cylinder (NURBS)":
            r = float(params.get('radius', 1.0))
            h = float(params.get('height', 2.0))
            occ_geom = create_occ_cylinder(r, h)
            return [OCCSurface(occ_geom)]

        elif geo_type == "STEP File":
            file_path = params.get('file_path')
            if not file_path or not os.path.exists(file_path):
                raise ValueError(f"STEP file not found: {file_path}")

            unit = params.get('unit', 'mm')
            scale = 1.0
            if unit == 'mm':
                scale = 0.001
            elif unit == 'cm':
                scale = 0.01

            invert_indices = params.get('invert_indices', [])
            max_param_range = params.get('max_param_range', 1e9)

            surfaces = load_step_file(file_path, max_param_range=max_param_range,
                                      scale=scale, invert_indices=invert_indices)

            return surfaces

        elif geo_type == "IGES File":
            unit_to_scale = {'mm': 0.001, 'cm': 0.01, 'm': 1.0}

            files = params.get('files')
            if not files:
                file_path = params.get('file_path')
                if file_path:
                    files = [{
                        'path': file_path,
                        'unit': params.get('unit', 'mm'),
                        'invert_indices': params.get('invert_indices', []),
                        'delete_indices': params.get('delete_indices', []),
                        'mirror_plane': params.get('mirror_plane'),
                        'rotation': params.get('rotation'),
                    }]

            if not files:
                raise ValueError("IGES File: no file specified (need 'files' list or 'file_path')")

            surfaces = []
            for spec in files:
                fpath = spec.get('path')
                if not fpath or not os.path.exists(fpath):
                    raise ValueError(f"IGES file not found: {fpath}")

                scale = unit_to_scale.get(spec.get('unit', 'mm'), 1.0)

                print(f"\n=== Loading {os.path.basename(fpath)} ===")
                file_surfaces = load_iges_file(
                    fpath,
                    scale=scale,
                    invert_indices=spec.get('invert_indices', []),
                    delete_indices=spec.get('delete_indices', []),
                    mirror_plane=spec.get('mirror_plane'),
                    rotation=spec.get('rotation'),
                )
                surfaces.extend(file_surfaces)

            return surfaces

        else:
            raise ValueError(f"Unknown geometry type: {geo_type}")
