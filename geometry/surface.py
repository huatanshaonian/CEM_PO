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