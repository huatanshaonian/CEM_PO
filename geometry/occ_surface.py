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

    def __init__(self, occ_surface: Geom_Surface, invert_normal: bool = False):
        """
        occ_surface: 一个 OCC 的 Geom_Surface 或其子类实例。
        invert_normal: 是否翻转法向量
        """
        self.surf = occ_surface
        self.invert_normal = invert_normal
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
                n_vec = [n_dir.X(), n_dir.Y(), n_dir.Z()]
                if self.invert_normal:
                    n_vec = [-x for x in n_vec]
                normals.append(n_vec)
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

    def __init__(self, face: TopoDS_Face, scale: float = 1.0, invert_normal: bool = False):
        """
        face: OCC 的 TopoDS_Face 对象
        scale: 坐标缩放系数（例如 0.001 将 mm 转换为 m）
        invert_normal: 是否翻转法向量
        """
        self.face = face
        self.adaptor = BRepAdaptor_Surface(face)
        self.scale = scale
        self.invert_normal = invert_normal

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
                n_vec = [n_dir.X(), n_dir.Y(), n_dir.Z()]
                if self.invert_normal:
                    n_vec = [-x for x in n_vec]
                normals.append(n_vec)
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

    def get_edge_by_index(self, index, n_samples=40):
        """
        根据索引获取特定的边，并进行离散化采样
        
        参数:
            index: 边的局部索引 (0-based)
            n_samples: 采样点数
            
        返回:
            points: numpy array (n_samples, 3) 包含边上的有序点序列
        """
        exp = TopExp_Explorer(self.face, TopAbs_EDGE)
        current_idx = 0
        while exp.More():
            if current_idx == index:
                edge = topods.Edge(exp.Current())
                curve = BRepAdaptor_Curve(edge)
                
                t_min = curve.FirstParameter()
                t_max = curve.LastParameter()
                
                # 均匀参数采样
                t_vals = np.linspace(t_min, t_max, n_samples)
                points = []
                for t in t_vals:
                    p = curve.Value(t)
                    points.append([p.X() * self.scale, p.Y() * self.scale, p.Z() * self.scale])
                
                return np.array(points)
                
            current_idx += 1
            exp.Next()
            
        raise IndexError(f"Edge index {index} out of range (max {current_idx-1})")

    def get_edge_by_index_with_normals(self, index, n_samples=40):
        """
        获取边缘的采样点及其对应的面法向

        参数:
            index: 边的局部索引 (0-based)
            n_samples: 采样点数

        返回:
            points: numpy array (n_samples, 3) 边上的有序点序列
            normals: numpy array (n_samples, 3) 每个点对应的面法向
        """
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.Geom2dAdaptor import Geom2dAdaptor_Curve

        exp = TopExp_Explorer(self.face, TopAbs_EDGE)
        current_idx = 0
        while exp.More():
            if current_idx == index:
                edge = topods.Edge(exp.Current())

                # 获取边缘的3D曲线
                curve = BRepAdaptor_Curve(edge)
                t_min = curve.FirstParameter()
                t_max = curve.LastParameter()

                # 尝试获取PCurve（边缘在面参数空间中的曲线）
                pcurve, first, last = BRep_Tool.CurveOnSurface(edge, self.face)
                has_pcurve = pcurve is not None

                # 均匀参数采样
                t_vals = np.linspace(t_min, t_max, n_samples)
                points = []
                normals = []

                # 创建SLProps对象用于计算法向
                props = BRepLProp_SLProps(self.adaptor, 1, 1e-6)

                for i, t in enumerate(t_vals):
                    # 获取3D点
                    p = curve.Value(t)
                    points.append([p.X() * self.scale, p.Y() * self.scale, p.Z() * self.scale])

                    # 获取法向
                    if has_pcurve:
                        # 通过PCurve获取参数坐标
                        # 需要将t映射到PCurve的参数范围
                        t_pcurve = first + (t - t_min) / (t_max - t_min) * (last - first)
                        p2d = pcurve.Value(t_pcurve)
                        u, v = p2d.X(), p2d.Y()
                    else:
                        # 如果没有PCurve，使用面中心的法向
                        u = (self.u_min + self.u_max) / 2
                        v = (self.v_min + self.v_max) / 2

                    props.SetParameters(u, v)
                    if props.IsNormalDefined():
                        n = props.Normal()
                        normal = np.array([n.X(), n.Y(), n.Z()])
                        if self.invert_normal:
                            normal = -normal
                    else:
                        normal = np.array([0.0, 0.0, 1.0])

                    normals.append(normal)

                return np.array(points), np.array(normals)

            current_idx += 1
            exp.Next()

        raise IndexError(f"Edge index {index} out of range (max {current_idx-1})")

