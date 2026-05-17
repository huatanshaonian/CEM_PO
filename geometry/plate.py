import numpy as np
from .surface import Surface, TransformedSurface

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

    def get_edge_by_index(self, index, n_samples=2):
        """
        获取矩形板的边缘
        0: u=-0.5 (x = -width/2)
        1: u= 0.5 (x = +width/2)
        2: v=-0.5 (y = -height/2)
        3: v= 0.5 (y = +height/2)
        """
        u_vals = None
        v_vals = None
        
        if index == 0: # u = -0.5
            u_vals = np.full(n_samples, -0.5)
            v_vals = np.linspace(-0.5, 0.5, n_samples)
        elif index == 1: # u = 0.5
            u_vals = np.full(n_samples, 0.5)
            v_vals = np.linspace(-0.5, 0.5, n_samples)
        elif index == 2: # v = -0.5
            u_vals = np.linspace(-0.5, 0.5, n_samples)
            v_vals = np.full(n_samples, -0.5)
        elif index == 3: # v = 0.5
            u_vals = np.linspace(-0.5, 0.5, n_samples)
            v_vals = np.full(n_samples, 0.5)
        else:
            raise IndexError(f"Plate edge index {index} out of range (0-3)")

        return self.evaluate(u_vals, v_vals)


def create_double_sided_plate(width, length):
    """
    构建薄板的双面模型：两片共面、法向相反的矩形板。

    顶面法向 +ẑ，底面法向 -ẑ（绕 X 轴旋 180°），两面都位于 z=0。
    PO 端逐单元做 n·k<0 照射判定，任一入射角下只有一面被照亮，
    天然不会重复积分。

    PTD 端通过共享 4 条边界（顶/底两面共面镜像）自动产出 4 条
    knife-edge：外角 α = 2π，n = 2，正是 Ufimtsev 半平面情形。

    返回:
        surfaces: [top, bottom]  两个 TransformedSurface
        ptd_id:   "(0,1)"  顶底面对，find_shared_edges 自动返回 4 条边
    """
    top = TransformedSurface(AnalyticPlate(width, length))
    R_x180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    bottom = TransformedSurface(AnalyticPlate(width, length),
                                rotation_matrix=R_x180)
    return [top, bottom], "(0,1)"
