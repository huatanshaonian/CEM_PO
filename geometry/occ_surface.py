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
        利用 OCC 的属性计算类 (SLProps) 一次性获取点、法线、Jacobian 以及导数。
        """
        u, v = np.broadcast_arrays(u_grid, v_grid)
        shape = u.shape
        u_flat = u.ravel()
        v_flat = v.ravel()
        
        points = []
        normals = []
        jacobians = []
        dP_du_list = []
        dP_dv_list = []
        
        # SLProps (Surface Local Properties)
        props = GeomLProp_SLProps(self.surf, 2, 1e-7)
        
        for ui, vi in zip(u_flat, v_flat):
            props.SetParameters(ui, vi)
            
            # 1. 获取坐标点
            pnt = props.Value()
            points.append([pnt.X(), pnt.Y(), pnt.Z()])
            
            # 2. 获取切向量
            du = props.D1U()
            dv = props.D1V()
            du_vec = np.array([du.X(), du.Y(), du.Z()])
            dv_vec = np.array([dv.X(), dv.Y(), dv.Z()])
            dP_du_list.append(du_vec)
            dP_dv_list.append(dv_vec)
            
            # 3. 获取法线
            if props.IsNormalDefined():
                n_dir = props.Normal()
                n_vec = [n_dir.X(), n_dir.Y(), n_dir.Z()]
                if self.invert_normal:
                    n_vec = [-x for x in n_vec]
                normals.append(n_vec)
            else:
                normals.append([0.0, 0.0, 0.0])
            
            # 4. 计算 Jacobian |Du x Dv|
            cross_prod = np.cross(du_vec, dv_vec)
            jacobians.append(np.linalg.norm(cross_prod))
            
        points = np.array(points).reshape(shape + (3,))
        normals = np.array(normals).reshape(shape + (3,))
        jacobians = np.array(jacobians).reshape(shape)
        dP_du = np.array(dP_du_list).reshape(shape + (3,))
        dP_dv = np.array(dP_dv_list).reshape(shape + (3,))

        return points, normals, jacobians, dP_du, dP_dv

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
        使用 BRepLProp_SLProps 获取点、法线、Jacobian 以及导数。
        优化：预分配 numpy 数组以减少循环内开销。
        """
        u, v = np.broadcast_arrays(u_grid, v_grid)
        num_pts = u.size
        shape = u.shape
        u_flat = u.ravel()
        v_flat = v.ravel()

        points = np.empty((num_pts, 3))
        normals = np.empty((num_pts, 3))
        jacobians = np.empty(num_pts)
        dP_du = np.empty((num_pts, 3))
        dP_dv = np.empty((num_pts, 3))

        # BRepLProp_SLProps 用于 BRepAdaptor_Surface
        props = BRepLProp_SLProps(self.adaptor, 2, 1e-7)

        for i in range(num_pts):
            props.SetParameters(u_flat[i], v_flat[i])

            # 1. 获取坐标点
            pnt = props.Value()
            points[i, 0] = pnt.X() * self.scale
            points[i, 1] = pnt.Y() * self.scale
            points[i, 2] = pnt.Z() * self.scale

            # 2. 获取切向量
            du = props.D1U()
            dv = props.D1V()
            du_x, du_y, du_z = du.X() * self.scale, du.Y() * self.scale, du.Z() * self.scale
            dv_x, dv_y, dv_z = dv.X() * self.scale, dv.Y() * self.scale, dv.Z() * self.scale
            
            dP_du[i, 0] = du_x
            dP_du[i, 1] = du_y
            dP_du[i, 2] = du_z
            dP_dv[i, 0] = dv_x
            dP_dv[i, 1] = dv_y
            dP_dv[i, 2] = dv_z

            # 3. 获取法线
            if props.IsNormalDefined():
                n_dir = props.Normal()
                normals[i, 0] = n_dir.X()
                normals[i, 1] = n_dir.Y()
                normals[i, 2] = n_dir.Z()
            else:
                normals[i, :] = 0.0

            # 4. 计算 Jacobian (叉乘模长)
            # 直接展开计算以减少 np.cross 的调用开销
            cx = du_y * dv_z - du_z * dv_y
            cy = du_z * dv_x - du_x * dv_z
            cz = du_x * dv_y - du_y * dv_x
            jacobians[i] = (cx*cx + cy*cy + cz*cz)**0.5

        if self.invert_normal:
            normals *= -1.0

        return (points.reshape(shape + (3,)), 
                normals.reshape(shape + (3,)), 
                jacobians.reshape(shape), 
                dP_du.reshape(shape + (3,)), 
                dP_dv.reshape(shape + (3,)))

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

