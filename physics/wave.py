import numpy as np
import warnings
from .constants import C0

class IncidentWave:
    """
    平面入射波类

    坐标系定义：
    - theta: 极角，与 +z 轴的夹角，范围 [0, π]
    - phi: 方位角，与 +x 轴的夹角，范围 [0, 2π]
    - 波从 (theta, phi) 方向传播到原点
    """

    def __init__(self, frequency, theta, phi=0.0):
        """
        参数:
        frequency: 频率 (Hz)
        theta: 极角 (弧度)，相对于 +z 轴，范围 [0, π]
        phi: 方位角 (弧度)，相对于 +x 轴，范围 [0, 2π]
        """
        # 频率验证
        if frequency <= 0:
            raise ValueError(f"频率必须为正数，当前值: {frequency}")

        self.frequency = frequency
        self.wavelength = C0 / frequency
        self.k = 2 * np.pi / self.wavelength

        # 角度规范化处理
        # theta 应在 [0, π]，phi 应在 [0, 2π]
        theta_normalized = float(theta)
        phi_normalized = float(phi)

        # 处理负角度和超范围角度
        if theta_normalized < 0 or theta_normalized > np.pi:
            # 将 theta 映射到 [0, π]
            # 负角度等效于 phi 偏移 180°
            if theta_normalized < 0:
                theta_normalized = -theta_normalized
                phi_normalized = phi_normalized + np.pi
            # theta > π 等效于 2π - theta，phi 偏移 180°
            if theta_normalized > np.pi:
                theta_normalized = 2 * np.pi - theta_normalized
                phi_normalized = phi_normalized + np.pi

        # phi 规范化到 [0, 2π]
        phi_normalized = phi_normalized % (2 * np.pi)

        self.theta = theta_normalized
        self.phi = phi_normalized

        # k 向量的方向 (传播方向，从源指向原点)
        # 源位于 (theta, phi) 方向的远场，波传播方向为负径向
        self.k_dir = -np.array([
            np.sin(self.theta) * np.cos(self.phi),
            np.sin(self.theta) * np.sin(self.phi),
            np.cos(self.theta)
        ])

        # 传播向量 k = k * k_dir
        self.k_vector = self.k * self.k_dir

    def __repr__(self):
        return (f"IncidentWave(f={self.frequency/1e9:.3f}GHz, "
                f"θ={np.degrees(self.theta):.1f}°, "
                f"φ={np.degrees(self.phi):.1f}°)")
