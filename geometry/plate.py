import numpy as np
from .surface import Surface

class AnalyticPlate(Surface):
    """
    解析矩形平板
    位于 z=0 平面，中心在原点。
    参数 u, v: [-0.5, 0.5]
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def u_domain(self):
        return (-0.5, 0.5)

    @property
    def v_domain(self):
        return (-0.5, 0.5)

    def evaluate(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        x = self.width * u
        y = self.height * v
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
        return self.width * self.height * np.ones_like(u)
