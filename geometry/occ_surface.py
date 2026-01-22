import numpy as np
from OCC.Core.Geom import Geom_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from .surface import Surface

class OCCSurface(Surface):
    """
    基于 PythonOCC (OpenCascade) 实现的通用曲面支持。
    支持所有来自 OpenCascade 的几何曲面（NURBS, Cylinder, Plane 等）。
    """

    def __init__(self, occ_surface: Geom_Surface):
        """
        occ_surface: 一个 OCC 的 Geom_Surface 或其子类实例。
        """
        self.surf = occ_surface
        # 获取参数范围
        self.u_min, self.u_max, self.v_min, self.v_max = self.surf.Bounds()

    @property
    def u_domain(self):
        return (self.u_min, self.u_max)

    @property
    def v_domain(self):
        return (self.v_min, self.v_max)

    def evaluate(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        shape = u.shape
        u_flat = u.ravel()
        v_flat = v.ravel()
        
        points = []
        from OCC.Core.gp import gp_Pnt
        pnt = gp_Pnt()
        
        for ui, vi in zip(u_flat, v_flat):
            self.surf.D0(ui, vi, pnt)
            points.append([pnt.X(), pnt.Y(), pnt.Z()])
            
        return np.array(points).reshape(shape + (3,))

    def get_normal(self, u, v):
        return self.get_data(u, v)[1]

    def get_jacobian(self, u, v):
        return self.get_data(u, v)[2]

    def get_data(self, u_grid, v_grid):
        """
        利用 OCC 的属性计算类 (SLProps) 一次性获取点、法线和一阶偏导。
        """
        u, v = np.broadcast_arrays(u_grid, v_grid)
        shape = u.shape
        u_flat = u.ravel()
        v_flat = v.ravel()
        
        points = []
        normals = []
        jacobians = []
        
        # SLProps (Surface Local Properties)
        # 参数 2 表示我们要计算到二阶导数（虽然这里只需一阶）
        # 1e-7 是解析精度
        props = GeomLProp_SLProps(self.surf, 2, 1e-7)
        
        for ui, vi in zip(u_flat, v_flat):
            props.SetParameters(ui, vi)
            
            # 1. 获取坐标点
            pnt = props.Value()
            points.append([pnt.X(), pnt.Y(), pnt.Z()])
            
            # 2. 获取法线 (OCC 会处理法线朝向问题)
            if props.IsNormalDefined():
                n_dir = props.Normal()
                normals.append([n_dir.X(), n_dir.Y(), n_dir.Z()])
            else:
                normals.append([0.0, 0.0, 0.0])
            
            # 3. 计算 Jacobian |Du x Dv|
            du = props.D1U()
            dv = props.D1V()
            # 叉乘模长
            # OCC 的向量没有内置 cross 后的模长？我们转为 numpy 计算
            du_vec = np.array([du.X(), du.Y(), du.Z()])
            dv_vec = np.array([dv.X(), dv.Y(), dv.Z()])
            cross_prod = np.cross(du_vec, dv_vec)
            jacobians.append(np.linalg.norm(cross_prod))
            
        points = np.array(points).reshape(shape + (3,))
        normals = np.array(normals).reshape(shape + (3,))
        jacobians = np.array(jacobians).reshape(shape)
        
        return points, normals, jacobians
