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

    def tessellate(self, resolution=30):
        """
        生成预览用的 barycentric 三角网格 (points, faces)，PyVista 格式。

        关键约定：把 (p1, p2, p3) 三个物理顶点按字典序排序后再做细分，
        这样 create_double_sided_triangle 里顶面 (p1,p2,p3) 和底面
        (p1,p3,p2) 输出的 (points, faces) 完全相同 → GUI 预览叠加后不会
        看到两套方向不一致的对角线（用户先前观察到的"全三角形"现象）。

        细分: 沿 a→b 和 a→c 各 resolution 段，每行 N-i 个梯形再切成
        2 个三角形（最末层 1 个三角形），共 resolution^2 个 face。
        """
        verts = sorted([tuple(self.p1), tuple(self.p2), tuple(self.p3)])
        a = np.asarray(verts[0])
        b = np.asarray(verts[1])
        c = np.asarray(verts[2])
        N = max(1, int(resolution))

        # barycentric 节点 P(i,j) = a + (i/N)·(b-a) + (j/N)·(c-a), i+j <= N
        coords = []
        index = {}
        for i in range(N + 1):
            for j in range(N + 1 - i):
                index[(i, j)] = len(coords)
                coords.append(a + (i / N) * (b - a) + (j / N) * (c - a))
        points = np.array(coords)

        faces = []
        for i in range(N):
            for j in range(N - i):
                v00 = index[(i, j)]
                v10 = index[(i + 1, j)]
                v01 = index[(i, j + 1)]
                faces.extend([3, v00, v10, v01])
                if (i + 1, j + 1) in index:
                    v11 = index[(i + 1, j + 1)]
                    faces.extend([3, v10, v11, v01])
        return points, np.array(faces)

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

    def get_edge_by_index_with_normals(self, index, n_samples=40):
        """
        返回 (pts, normals, inwards) 三元组, 让 ptd_edge_finder 拿到正确的
        inward (在面内 ⊥ 边切线, 指向三角形重心), 绕开默认 _eval_boundary_inwards_at_t
        基于矩形参数空间的兜底 (那条路径对 Duffy 三角形的边索引不对应, 会给出
        奇异 / 错误方向的 inward, 是 l_A ray-cast 失败 / ±θ 不对称的根因).

        三条物理边索引与 get_edge_by_index 完全一致 (0/1/2 = p1->p2, p2->p3, p3->p1).
        """
        if index == 0:
            P_start, P_end = self.p1, self.p2
        elif index == 1:
            P_start, P_end = self.p2, self.p3
        elif index == 2:
            P_start, P_end = self.p3, self.p1
        else:
            raise IndexError(f"Triangle edge index {index} out of range (0-2)")

        pts = np.linspace(P_start, P_end, n_samples)
        normals = np.tile(self._n_const, (n_samples, 1))

        # inward: 面内 ⊥ 边切线, 指向重心
        t_vec = P_end - P_start
        t_len = np.linalg.norm(t_vec)
        if t_len < 1e-12:
            inward_unit = np.zeros(3)
        else:
            t_hat = t_vec / t_len
            inward_raw = np.cross(self._n_const, t_hat)
            inward_len = np.linalg.norm(inward_raw)
            if inward_len < 1e-12:
                inward_unit = np.zeros(3)
            else:
                inward_unit = inward_raw / inward_len
                centroid = (self.p1 + self.p2 + self.p3) / 3.0
                midpoint = 0.5 * (P_start + P_end)
                if np.dot(inward_unit, centroid - midpoint) < 0:
                    inward_unit = -inward_unit
        inwards = np.tile(inward_unit, (n_samples, 1))

        return pts, normals, inwards


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
