import os
import numpy as np
from geometry.plate import AnalyticPlate
from geometry.sphere import AnalyticSphere
from geometry.cylinder import AnalyticCylinder
from geometry.occ_surface import OCCSurface
from geometry.step_loader import load_step_file
from geometry.wedge import create_analytic_wedge
from geometry.brick import create_analytic_brick
from tools.visualize_mesh import create_occ_cylinder

class GeometryFactory:
    """
    几何构建工厂，将参数字典转换为实体的 Surface 对象列表。
    可供不同的 GUI 框架(Tkinter/Qt)或 CLI 脚本共享。
    """
    @staticmethod
    def create_geometry(geo_type, params):
        """
        Args:
            geo_type (str): 'Cylinder', 'Plate', 'Sphere', 'STEP File', 'Wedge', 'Brick'
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
            return [AnalyticPlate(w, l)]
        
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

            # 调用原有的 STEP 加载逻辑，直接传递 invert_indices
            surfaces = load_step_file(file_path, scale=scale, invert_indices=invert_indices)

            return surfaces
        
        else:
            raise ValueError(f"Unknown geometry type: {geo_type}")
