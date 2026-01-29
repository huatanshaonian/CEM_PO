import numpy as np

class PTDSegment:
    """
    描述 PTD 边缘的一个微小线段
    """
    def __init__(self, start, end, normal=None):
        self.start = start
        self.end = end
        self.midpoint = (start + end) / 2.0

        vec = end - start
        self.length = np.linalg.norm(vec)
        if self.length > 1e-12:
            self.tangent = vec / self.length
        else:
            self.tangent = np.array([0, 0, 0])

        # 该segment对应的面法向（用于遮挡检测）
        if normal is not None:
            n = np.array(normal)
            norm = np.linalg.norm(n)
            self.normal = n / norm if norm > 1e-12 else np.array([0, 0, 1.0])
        else:
            self.normal = None

class PTDEdge:
    """
    描述用于 PTD 计算的物理边缘 (支持曲线)
    """
    def __init__(self, name, points, lit_face_normal, wedge_angle_deg=90.0, point_normals=None):
        """
        参数:
            name: 标识符 (如 "F0E1")
            points: numpy array (N, 3) 边上的有序采样点
            lit_face_normal: 照亮面(Face A)的法向量（用作默认值）
            wedge_angle_deg: 楔角角度 (默认 90度)
            point_normals: numpy array (N, 3) 每个点对应的面法向（可选）
        """
        self.name = name
        self.points = points

        # 默认法向量（面中心的法向）
        self.n_lit = np.array(lit_face_normal)
        norm = np.linalg.norm(self.n_lit)
        if norm > 1e-12:
            self.n_lit = self.n_lit / norm
        else:
            self.n_lit = np.array([0, 0, 1.0])

        # 构建分段，每个分段有自己的法向
        self.segments = []
        for i in range(len(points) - 1):
            # 计算该segment的法向（取两端点法向的平均）
            if point_normals is not None:
                seg_normal = (point_normals[i] + point_normals[i+1]) / 2.0
            else:
                seg_normal = self.n_lit

            seg = PTDSegment(points[i], points[i+1], normal=seg_normal)
            if seg.length > 1e-12:  # 忽略极短的段
                self.segments.append(seg)

        # 楔形参数
        self.wedge_angle = np.deg2rad(wedge_angle_deg)
        self.alpha = 2 * np.pi - self.wedge_angle  # 外部角
        self.n_param = self.alpha / np.pi
