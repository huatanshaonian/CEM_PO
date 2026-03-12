import numpy as np


def _adaptive_breakpoints(pts, max_angle_rad):
    """
    根据切线方向变化量，自适应地在折线 pts 上选取分段断点索引。

    参数:
        pts:           numpy array (N, 3)，边上的有序采样点
        max_angle_rad: 累积切线转角超过此阈值时插入断点

    返回:
        list[int]，断点索引列表（始终包含 0 和 N-1）
    """
    N = len(pts)
    if N < 3:
        return [0, N - 1]

    breakpoints = [0]
    accum_angle = 0.0

    # 计算各段切线方向
    tangents = pts[1:] - pts[:-1]  # (N-1, 3)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    tangents = tangents / lengths  # 归一化

    for i in range(1, N - 1):
        # 相邻切线夹角
        cos_a = np.clip(np.dot(tangents[i - 1], tangents[i]), -1.0, 1.0)
        delta = np.arccos(cos_a)
        accum_angle += delta
        if accum_angle >= max_angle_rad:
            breakpoints.append(i)
            accum_angle = 0.0

    if breakpoints[-1] != N - 1:
        breakpoints.append(N - 1)

    return breakpoints


class PTDSegment:
    """
    描述 PTD 边缘的一个微小线段
    """
    def __init__(self, start, end, normal=None, alpha=None):
        self.start = start
        self.end = end
        self.midpoint = (start + end) / 2.0
        self.alpha = alpha

        vec = end - start
        self.length = np.linalg.norm(vec)
        if self.length > 1e-12:
            self.tangent = vec / self.length
        else:
            self.tangent = np.array([0, 0, 0])

        # 该 segment 对应的面法向（用于遮挡检测）
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
    def __init__(self, name, points, lit_face_normal,
                 exterior_angle_rad=None, wedge_angle_deg=90.0,
                 point_normals=None, point_normals_b=None,
                 max_segment_angle_deg=2.0):
        """
        参数:
            name:                 标识符（如 "(0,1)"）
            points:               numpy array (N, 3) 边上的有序采样点
            lit_face_normal:      照亮面法向量（用作默认值）
            exterior_angle_rad:   外部二面角（弧度），优先于 wedge_angle_deg
            wedge_angle_deg:      楔角内角（度），仅在 exterior_angle_rad=None 时使用
            point_normals:        numpy array (N, 3) 每个点对应的 face_a 法向（可选）
            point_normals_b:      numpy array (N, 3) 每个点对应的 face_b 法向（可选）
            max_segment_angle_deg: 自适应分段的最大累积切线转角（度）
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

        # 楔形参数（全局回退值）
        if exterior_angle_rad is not None:
            self.alpha = float(exterior_angle_rad)
        else:
            self.alpha = 2 * np.pi - np.deg2rad(wedge_angle_deg)
        self.n_param = self.alpha / np.pi

        pts = np.asarray(points)
        self.segments = []

        if point_normals is not None:
            # 自适应分段
            bp = _adaptive_breakpoints(pts, np.deg2rad(max_segment_angle_deg))
            for k in range(len(bp) - 1):
                i0, i1 = bp[k], bp[k + 1]
                # face_a 法向：取区间均值并归一化
                na_slice = point_normals[i0:i1 + 1]
                na_mean = na_slice.mean(axis=0)
                na_len = np.linalg.norm(na_mean)
                na_seg = na_mean / na_len if na_len > 1e-12 else self.n_lit

                # 每段的 alpha
                if point_normals_b is not None:
                    nb_slice = point_normals_b[i0:i1 + 1]
                    nb_mean = nb_slice.mean(axis=0)
                    nb_len = np.linalg.norm(nb_mean)
                    nb_seg = nb_mean / nb_len if nb_len > 1e-12 else self.n_lit
                    seg_alpha = np.pi + np.arccos(
                        np.clip(np.dot(na_seg, nb_seg), -1.0, 1.0))
                else:
                    seg_alpha = self.alpha  # 全局回退

                seg = PTDSegment(pts[i0], pts[i1], normal=na_seg, alpha=seg_alpha)
                if seg.length > 1e-12:
                    self.segments.append(seg)
        else:
            # 无逐点法向：逐点建段，使用全局 alpha
            for i in range(len(pts) - 1):
                seg = PTDSegment(pts[i], pts[i + 1],
                                 normal=self.n_lit, alpha=self.alpha)
                if seg.length > 1e-12:
                    self.segments.append(seg)
