import numpy as np
from .surface import Surface

class AnalyticCylinder(Surface):
    """
    解析圆柱体表面
    参数 u: [0, 1] 对应角度 [0, 2*pi]
    参数 v: [-0.5, 0.5] 对应高度 [-H/2, H/2]
    """

    def __init__(self, radius, height):
        self.radius = radius
        self.height = height

    @property
    def u_domain(self):
        return (0.0, 1.0)

    @property
    def v_domain(self):
        return (-0.5, 0.5)

    def evaluate(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        x = self.radius * np.cos(2 * np.pi * u)
        y = self.radius * np.sin(2 * np.pi * u)
        z = self.height * v
        return np.stack([x, y, z], axis=-1)

    def get_normal(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        nx = np.cos(2 * np.pi * u)
        ny = np.sin(2 * np.pi * u)
        nz = np.zeros_like(u)
        return np.stack([nx, ny, nz], axis=-1)

    def get_jacobian(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        return 2 * np.pi * self.radius * self.height * np.ones_like(u)
