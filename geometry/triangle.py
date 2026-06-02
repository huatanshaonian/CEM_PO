import numpy as np
from .surface import Surface, TransformedSurface


class AnalyticTriangle(Surface):
    """
    解析三角形面片，由三个顶点 p1, p2, p3 定义（笛卡尔坐标）。

    参数化采用 Duffy 变换：将单位正方形 (u,v) ∈ [0,1]² 映射到三角形内部，
        s = u·(1 - v),  t = u·v,
        P(u,v) = p1 + s·e1 + t·e2,     e1 = p2 - p1,  e2 = p3 - p1.
    Duffy 雅可比 = u；面元雅可比 = |e1 × e2| = 2·Area。

    退化结构（PO 求解器/预览自动识别）：
        u = 0 整条边塌缩到 p1（u_min 退化边）。
        Jacobian(u, *) = u·|e1×e2| → u=0 处为 0，detect_degenerate_edge
        会判定为 'u_min' 并自动切到条带状网格。

    三条物理边索引（n_edges = 3）：
        0: p1 → p2   (v = 0)
        1: p2 → p3   (u = 1)
        2: p3 → p1   (v = 1, 反向)
    """

    n_edges = 3

    def __init__(self, p1, p2, p3):
        self.p1 = np.asarray(p1, dtype=float).reshape(3)
        self.p2 = np.asarray(p2, dtype=float).reshape(3)
        self.p3 = np.asarray(p3, dtype=float).reshape(3)
        self.e1 = self.p2 - self.p1
        self.e2 = self.p3 - self.p1
        cross = np.cross(self.e1, self.e2)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm < 1e-14:
            raise ValueError("AnalyticTriangle: 三个顶点共线或重合，无法构成三角形")
        self._n_const = cross / cross_norm
        self._jac_const = cross_norm  # = 2·Area

    @property
    def u_domain(self):
        return (0.0, 1.0)

    @property
    def v_domain(self):
        return (0.0, 1.0)

    def evaluate(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float),
                                   np.asarray(v, dtype=float))
        s = u * (1.0 - v)
        t = u * v
        x = self.p1[0] + s * self.e1[0] + t * self.e2[0]
        y = self.p1[1] + s * self.e1[1] + t * self.e2[1]
        z = self.p1[2] + s * self.e1[2] + t * self.e2[2]
        return np.stack([x, y, z], axis=-1)

    def get_normal(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float),
                                   np.asarray(v, dtype=float))
        nx = np.full_like(u, self._n_const[0])
        ny = np.full_like(u, self._n_const[1])
        nz = np.full_like(u, self._n_const[2])
        return np.stack([nx, ny, nz], axis=-1)

    def get_jacobian(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float),
                                   np.asarray(v, dtype=float))
        return u * self._jac_const

    def get_edge_by_index(self, index, n_samples=2):
        """
        三角形物理边采样点。返回 (n_samples, 3)。

        0: p1 → p2   (v=0)
        1: p2 → p3   (u=1)
        2: p3 → p1   (v=1, 参数化 u: 1→0 即可得到反向)
        """
        if index == 0:
            u_vals = np.linspace(0.0, 1.0, n_samples)
            v_vals = np.zeros(n_samples)
        elif index == 1:
            u_vals = np.ones(n_samples)
            v_vals = np.linspace(0.0, 1.0, n_samples)
        elif index == 2:
            u_vals = np.linspace(1.0, 0.0, n_samples)
            v_vals = np.ones(n_samples)
        else:
            raise IndexError(f"Triangle edge index {index} out of range (0-2)")
        return self.evaluate(u_vals, v_vals)


def create_double_sided_triangle(p1, p2, p3):
    """
    构建三角形薄板的双面模型：顶面 (p1,p2,p3) + 底面 (p1,p3,p2) 反序顶点。

    两面共占同一空间但法向相反，PO 端逐单元 n·k<0 判定保证任一入射角下
    只有一面被照亮，不会重复积分。

    PTD 端：顶/底两面共享全部 3 条边界（共面镜像），find_shared_edges
    会自动识别 3 条 knife-edge（外角 α = 2π，n = 2），即 Ufimtsev 半平面情形。

    返回:
        surfaces: [top, bottom]   两个 TransformedSurface
        ptd_id:   "(0,1)"          顶/底面对
    """
    top = TransformedSurface(AnalyticTriangle(p1, p2, p3))
    bottom = TransformedSurface(AnalyticTriangle(p1, p3, p2))
    return [top, bottom], "(0,1)"
