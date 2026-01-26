import numpy as np
from abc import ABC, abstractmethod

class Surface(ABC):
    """
    表面抽象基类
    """
    
    @property
    @abstractmethod
    def u_domain(self):
        """
        返回 u 的参数范围 (u_min, u_max)
        """
        pass

    @property
    @abstractmethod
    def v_domain(self):
        """
        返回 v 的参数范围 (v_min, v_max)
        """
        pass

    @abstractmethod
    def evaluate(self, u, v):
        """
        返回坐标点 (x, y, z)
        """
        pass

    @abstractmethod
    def get_normal(self, u, v):
        """
        返回法向量 (nx, ny, nz)
        """
        pass

    @abstractmethod
    def get_jacobian(self, u, v):
        """
        返回雅可比行列式 |dS/dudv|
        """
        pass

    def get_data(self, u_grid, v_grid):
        """
        为给定的网格返回点、法线和雅可比。
        要求向量化实现以提高性能。
        """
        points = self.evaluate(u_grid, v_grid)
        normals = self.get_normal(u_grid, v_grid)
        jacobians = self.get_jacobian(u_grid, v_grid)
        return points, normals, jacobians


class TransformedSurface(Surface):
    """
    对基础曲面进行旋转和平移
    """
    def __init__(self, base_surface, rotation_matrix=None, translation=None):
        self.base = base_surface
        self.R = rotation_matrix if rotation_matrix is not None else np.eye(3)
        self.T = np.array(translation) if translation is not None else np.zeros(3)

    @property
    def u_domain(self):
        return self.base.u_domain

    @property
    def v_domain(self):
        return self.base.v_domain

    def evaluate(self, u, v):
        pts_local = self.base.evaluate(u, v)
        # Apply rotation: (N, 3) dot (3, 3) -> (N, 3)
        # Assuming R transforms local to global
        return np.dot(pts_local, self.R.T) + self.T

    def get_normal(self, u, v):
        n_local = self.base.get_normal(u, v)
        return np.dot(n_local, self.R.T)

    def get_jacobian(self, u, v):
        return self.base.get_jacobian(u, v) # Rotation/Translation doesn't change area element magnitude

    def get_edge_by_index(self, index, n_samples=40):
        if hasattr(self.base, 'get_edge_by_index'):
            pts_local = self.base.get_edge_by_index(index, n_samples)
            return np.dot(pts_local, self.R.T) + self.T
        raise NotImplementedError("Base surface does not support edge extraction")