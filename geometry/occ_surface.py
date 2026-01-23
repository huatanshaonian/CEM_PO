import numpy as np
from OCC.Core.Geom import Geom_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.TopoDS import TopoDS_Face, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
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


class OCCFaceSurface(Surface):
    """
    基于 TopoDS_Face 的曲面，正确处理 trimming 边界。
    使用 BRepAdaptor_Surface 获取实际的参数域。
    """

    def __init__(self, face: TopoDS_Face, scale: float = 1.0):
        """
        face: OCC 的 TopoDS_Face 对象
        scale: 坐标缩放系数（例如 0.001 将 mm 转换为 m）
        """
        self.face = face
        self.adaptor = BRepAdaptor_Surface(face)
        self.scale = scale

        # 获取实际的参数边界（考虑 trimming）
        self.u_min = self.adaptor.FirstUParameter()
        self.u_max = self.adaptor.LastUParameter()
        self.v_min = self.adaptor.FirstVParameter()
        self.v_max = self.adaptor.LastVParameter()

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

        for ui, vi in zip(u_flat, v_flat):
            pnt = self.adaptor.Value(ui, vi)
            points.append([pnt.X() * self.scale, pnt.Y() * self.scale, pnt.Z() * self.scale])

        return np.array(points).reshape(shape + (3,))

    def get_normal(self, u, v):
        return self.get_data(u, v)[1]

    def get_jacobian(self, u, v):
        return self.get_data(u, v)[2]

    def get_data(self, u_grid, v_grid):
        """
        使用 BRepLProp_SLProps 获取点、法线和 Jacobian。
        坐标和 Jacobian 会根据 scale 参数进行缩放。
        """
        u, v = np.broadcast_arrays(u_grid, v_grid)
        shape = u.shape
        u_flat = u.ravel()
        v_flat = v.ravel()

        points = []
        normals = []
        jacobians = []

        # BRepLProp_SLProps 用于 BRepAdaptor_Surface
        props = BRepLProp_SLProps(self.adaptor, 2, 1e-7)

        for ui, vi in zip(u_flat, v_flat):
            props.SetParameters(ui, vi)

            # 1. 获取坐标点（应用缩放）
            pnt = props.Value()
            points.append([pnt.X() * self.scale, pnt.Y() * self.scale, pnt.Z() * self.scale])

            # 2. 获取法线（单位向量，不需要缩放）
            if props.IsNormalDefined():
                n_dir = props.Normal()
                normals.append([n_dir.X(), n_dir.Y(), n_dir.Z()])
            else:
                normals.append([0.0, 0.0, 0.0])

            # 3. 计算 Jacobian |Du x Dv|（应用 scale^2 因为是面积元素）
            du = props.D1U()
            dv = props.D1V()
            du_vec = np.array([du.X(), du.Y(), du.Z()])
            dv_vec = np.array([dv.X(), dv.Y(), dv.Z()])
            cross_prod = np.cross(du_vec, dv_vec)
            jacobians.append(np.linalg.norm(cross_prod) * self.scale * self.scale)

        points = np.array(points).reshape(shape + (3,))
        normals = np.array(normals).reshape(shape + (3,))
        jacobians = np.array(jacobians).reshape(shape)

        return points, normals, jacobians

    def get_edges(self, n_samples=20):
        """
        获取面的所有边界边，返回每条边的采样点和中点。

        返回:
            list of dict: 每条边的信息，包含:
                - 'points': (n_samples, 3) 边上的采样点
                - 'midpoint': (3,) 边的中点坐标
        """
        edges_data = []

        exp = TopExp_Explorer(self.face, TopAbs_EDGE)
        while exp.More():
            edge = topods.Edge(exp.Current())
            curve = BRepAdaptor_Curve(edge)

            t_min = curve.FirstParameter()
            t_max = curve.LastParameter()

            # 采样边上的点（应用缩放）
            t_vals = np.linspace(t_min, t_max, n_samples)
            points = []
            for t in t_vals:
                pnt = curve.Value(t)
                points.append([pnt.X() * self.scale, pnt.Y() * self.scale, pnt.Z() * self.scale])

            points = np.array(points)
            midpoint = points[n_samples // 2]

            edges_data.append({
                'points': points,
                'midpoint': midpoint
            })

            exp.Next()

        return edges_data
