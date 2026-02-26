import numpy as np
from .surface import Surface


class InfiniteWedgeSurface(Surface):
    """
    无限楔角单面几何（矩形平板近似），用于PTD算法验证。

    几何：矩形平板，位于z=0平面，从x=0延伸到x=plate_width。
    PTD棱边（Edge 0）沿y轴，位于x=0处，法向 = +z。

    坐标系（与ptd_core.py一致）：
        棱边切向 t = ŷ
        亮面法向 n_lit = ẑ
        e1 = ẑ, e2 = t × n_lit = ŷ × ẑ = x̂
        观测角 angle0 = π/2 - theta_GUI（theta为GUI扫描角，弧度）

    参数：
        edge_length: 棱边长度（y方向，m）
        plate_width: 平板延伸宽度（x方向，m）
    """

    def __init__(self, edge_length, plate_width):
        self.edge_length = edge_length
        self.plate_width = plate_width

    @property
    def u_domain(self):
        # u=0 → PTD棱边(x=0), u=1 → 远端(x=plate_width)
        return (0.0, 1.0)

    @property
    def v_domain(self):
        # v → y从 -edge_length/2 到 +edge_length/2
        return (-0.5, 0.5)

    def evaluate(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        x = self.plate_width * u
        y = self.edge_length * v
        z = np.zeros_like(u)
        return np.stack([x, y, z], axis=-1)

    def get_normal(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        nx = np.zeros_like(u)
        ny = np.zeros_like(u)
        nz = np.ones_like(u)
        return np.stack([nx, ny, nz], axis=-1)

    def get_jacobian(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        return self.plate_width * self.edge_length * np.ones_like(u)

    def get_edge_by_index(self, index, n_samples=60):
        """
        获取边缘点序列。
        0: u=0 (x=0) — PTD棱边，沿y轴 ← 用于PTD计算
        1: u=1 (x=plate_width) — 远端边
        2: v=-0.5 (y=-edge_length/2) — 下边
        3: v= 0.5 (y=+edge_length/2) — 上边
        """
        if index == 0:
            u_vals = np.zeros(n_samples)
            v_vals = np.linspace(-0.5, 0.5, n_samples)
        elif index == 1:
            u_vals = np.ones(n_samples)
            v_vals = np.linspace(-0.5, 0.5, n_samples)
        elif index == 2:
            u_vals = np.linspace(0.0, 1.0, n_samples)
            v_vals = np.full(n_samples, -0.5)
        elif index == 3:
            u_vals = np.linspace(0.0, 1.0, n_samples)
            v_vals = np.full(n_samples, 0.5)
        else:
            raise IndexError(f"InfiniteWedgeSurface edge index {index} out of range (0-3)")
        return self.evaluate(u_vals, v_vals)


def create_infinite_wedge(edge_length, plate_width):
    """
    创建无限楔角几何（单面平板近似）。

    返回:
        surfaces: [InfiniteWedgeSurface]
        ptd_id:   'F0E0'（PTD棱边位于Face 0的Edge 0处，即x=0的棱边）
    """
    surface = InfiniteWedgeSurface(edge_length, plate_width)
    return [surface], "F0E0"
