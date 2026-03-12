import numpy as np
from .surface import Surface


class InfiniteWedgeSurface(Surface):
    """
    无限楔角 Face 1（水平面）。
    位于 z=0，从 x=0 延伸到 x=plate_width，法向 = +z。
    PTD棱边（Edge 0）沿 y 轴，位于 x=0 处。

    坐标系（与 ptd_core.py 一致）：
        棱边切向 t = ŷ，亮面法向 n_lit = ẑ
        e1 = ẑ, e2 = x̂，angle0 = π/2 - theta_GUI
    """

    def __init__(self, edge_length, plate_width):
        self.edge_length = edge_length
        self.plate_width = plate_width

    @property
    def u_domain(self):
        return (0.0, 1.0)   # u=0 → PTD棱边(x=0), u=1 → 远端

    @property
    def v_domain(self):
        return (-0.5, 0.5)  # v → y ∈ [-L/2, +L/2]

    def evaluate(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        x = self.plate_width * u
        y = self.edge_length * v
        z = np.zeros_like(u)
        return np.stack([x, y, z], axis=-1)

    def get_normal(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        return np.stack([np.zeros_like(u), np.zeros_like(u), np.ones_like(u)], axis=-1)

    def get_jacobian(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        return self.plate_width * self.edge_length * np.ones_like(u)

    def get_edge_by_index(self, index, n_samples=60):
        """
        0: u=0 (x=0) — PTD棱边，沿y轴
        1: u=1 (x=plate_width) — 远端边
        2: v=-0.5 — 下边
        3: v=+0.5 — 上边
        """
        if index == 0:
            return self.evaluate(np.zeros(n_samples), np.linspace(-0.5, 0.5, n_samples))
        elif index == 1:
            return self.evaluate(np.ones(n_samples), np.linspace(-0.5, 0.5, n_samples))
        elif index == 2:
            return self.evaluate(np.linspace(0.0, 1.0, n_samples), np.full(n_samples, -0.5))
        elif index == 3:
            return self.evaluate(np.linspace(0.0, 1.0, n_samples), np.full(n_samples, 0.5))
        else:
            raise IndexError(f"InfiniteWedgeSurface edge index {index} out of range (0-3)")


class WedgeFace2Surface(Surface):
    """
    无限楔角 Face 2（斜面/阴影面），方向由外楔角 alfa 决定。

    Face 2 从 PTD 棱边（x=0，y轴）出发，沿方向
        d₂ = cos(alfa)*x̂ + sin(alfa)*ẑ
    延伸 plate_width 距离。法向（指向外部/外楔区域）：
        n₂ = -sin(alfa)*x̂ + cos(alfa)*ẑ

    典型情况（外楔角 alfa = 270°，内角 90°）：
        d₂ = -ẑ → Face 2 竖直向下（x=0，z<0）
        n₂ = +x̂ → 法向朝 +x（指向外部区域）
    """

    def __init__(self, edge_length, plate_width, exterior_angle_rad):
        self.edge_length = edge_length
        self.plate_width = plate_width
        self.alfa = exterior_angle_rad
        # 面延伸方向（在 xz 平面内）
        self._dx = np.cos(exterior_angle_rad)
        self._dz = np.sin(exterior_angle_rad)
        # 外法向 = d₂ 顺时针旋转 90°，指向外部区域（波传播侧）
        # n = (sin α, 0, -cos α)
        self._nx =  np.sin(exterior_angle_rad)
        self._nz = -np.cos(exterior_angle_rad)

    @property
    def u_domain(self):
        return (0.0, 1.0)   # u=0 → PTD棱边, u=1 → 远端

    @property
    def v_domain(self):
        return (-0.5, 0.5)

    def evaluate(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        x = self.plate_width * u * self._dx
        y = self.edge_length * v
        z = self.plate_width * u * self._dz
        return np.stack([x, y, z], axis=-1)

    def get_normal(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        nx = self._nx * np.ones_like(u)
        ny = np.zeros_like(u)
        nz = self._nz * np.ones_like(u)
        return np.stack([nx, ny, nz], axis=-1)

    def get_jacobian(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        return self.plate_width * self.edge_length * np.ones_like(u)

    def get_edge_by_index(self, index, n_samples=60):
        """
        0: u=0 — PTD棱边（与 Face 1 共享）
        1: u=1 — 远端边
        2: v=-0.5 — 下边
        3: v=+0.5 — 上边
        """
        if index == 0:
            return self.evaluate(np.zeros(n_samples), np.linspace(-0.5, 0.5, n_samples))
        elif index == 1:
            return self.evaluate(np.ones(n_samples), np.linspace(-0.5, 0.5, n_samples))
        elif index == 2:
            return self.evaluate(np.linspace(0.0, 1.0, n_samples), np.full(n_samples, -0.5))
        elif index == 3:
            return self.evaluate(np.linspace(0.0, 1.0, n_samples), np.full(n_samples, 0.5))
        else:
            raise IndexError(f"WedgeFace2Surface edge index {index} out of range (0-3)")


def create_infinite_wedge(edge_length, exterior_angle_deg=270.0):
    """
    创建无限楔角几何（两面，PTD棱边在两面的交线处）。

    Face 0（水平面）：z=0，x∈[0, plate_width]，法向 = +z
    Face 1（斜面）  ：从棱边出发，沿外楔角方向延伸，法向指向外部

    plate_width 自动设为 edge_length / 2，不作为用户参数。

    参数:
        edge_length:        棱边长度（y方向，m）
        exterior_angle_deg: 外楔角 α（度），例：90°内角 → alfa=270°

    返回:
        surfaces: [InfiniteWedgeSurface, WedgeFace2Surface]
        ptd_id:   '(0,1)'（Face 0 和 Face 1 的共享边）
    """
    plate_width = edge_length / 2.0
    alfa = np.radians(exterior_angle_deg)

    face1 = InfiniteWedgeSurface(edge_length, plate_width)
    face2 = WedgeFace2Surface(edge_length, plate_width, alfa)

    return [face1, face2], "(0,1)"
