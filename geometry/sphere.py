import numpy as np
from .surface import Surface

class AnalyticSphere(Surface):
    """
    解析球体
    参数 u: [0, 1] 对应方位角 phi [0, 2pi]
    参数 v: [0, 1] 对应极角 theta [0, pi]
    """

    def __init__(self, radius):
        self.radius = radius

    @property
    def u_domain(self):
        return (0.0, 1.0)

    @property
    def v_domain(self):
        return (0.0, 1.0)

    def evaluate(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        phi = 2 * np.pi * u
        theta = np.pi * v
        
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)
        
        return np.stack([x, y, z], axis=-1)

    def get_normal(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        phi = 2 * np.pi * u
        theta = np.pi * v
        
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)
        
        return np.stack([nx, ny, nz], axis=-1)

    def get_jacobian(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        theta = np.pi * v
        return 2 * (np.pi**2) * (self.radius**2) * np.sin(theta)
