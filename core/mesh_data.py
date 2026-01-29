import numpy as np
import sys
import os

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

def detect_degenerate_edge(surface, threshold_ratio=0.01):
    """
    检测曲面的退化边（三角形面或纺锤形面）
    """
    u_min, u_max = surface.u_domain
    v_min, v_max = surface.v_domain

    # 检查四个角的 Jacobian
    corners = [
        (u_min, v_min),  # (0,0)
        (u_max, v_min),  # (1,0)
        (u_min, v_max),  # (0,1)
        (u_max, v_max),  # (1,1)
    ]

    jac_values = []
    for u, v in corners:
        data_res = surface.get_data(np.array([[u]]), np.array([[v]]))
        if len(data_res) == 5:
            jac = data_res[2]
        else:
            jac = data_res[2]
        jac_values.append(jac[0, 0])

    max_jac = max(jac_values)
    if max_jac < 1e-10:
        return 'degenerate'

    threshold = max_jac * threshold_ratio
    is_degenerate = [j < threshold for j in jac_values]

    # 判断哪条边退化
    u_min_deg = is_degenerate[0] and is_degenerate[2]  # (0,0)-(0,1)
    u_max_deg = is_degenerate[1] and is_degenerate[3]  # (1,0)-(1,1)
    v_min_deg = is_degenerate[0] and is_degenerate[1]  # (0,0)-(1,0)
    v_max_deg = is_degenerate[2] and is_degenerate[3]  # (0,1)-(1,1)

    if u_min_deg and u_max_deg: return 'u_both'
    if v_min_deg and v_max_deg: return 'v_both'
    if u_min_deg: return 'u_min'
    if u_max_deg: return 'u_max'
    if v_min_deg: return 'v_min'
    if v_max_deg: return 'v_max'

    return None

class CachedMeshData:
    """
    预计算的网格数据，用于加速计算和并行传输。
    包含积分所需的所有几何信息（纯 NumPy 数组）。
    """
    def __init__(self, points, normals, jacobians, dP_du, dP_dv, du, dv):
        self.points = points
        self.normals = normals
        self.jacobians = jacobians
        self.dP_du = dP_du  # Partial derivative dP/du for alpha calc
        self.dP_dv = dP_dv  # Partial derivative dP/dv for beta calc
        self.du = du
        self.dv = dv
        self.direct_derivatives = False # 标记是否直接从几何体获取了导数

    def to_gpu(self):
        """将数据迁移到 GPU"""
        if not HAS_GPU:
            return self
        self.points = cp.asarray(self.points)
        self.normals = cp.asarray(self.normals)
        self.jacobians = cp.asarray(self.jacobians)
        self.dP_du = cp.asarray(self.dP_du)
        self.dP_dv = cp.asarray(self.dP_dv)
        # du, dv 是标量，通常不需要 cp.asarray，除非需要参与 GPU 运算
        return self

    def to_cpu(self):
        """将数据迁移回 CPU"""
        if hasattr(self.points, 'get'):
            self.points = self.points.get()
            self.normals = self.normals.get()
            self.jacobians = self.jacobians.get()
            self.dP_du = self.dP_du.get()
            self.dP_dv = self.dP_dv.get()
        return self

class MergedMeshData:
    """
    [GPU 优化核心] 合并后的网格数据
    利用大显存优势，将多个小曲面的网格合并为一个巨大的数组。
    这样 GPU 只需要执行一次 Kernel Launch 即可计算所有曲面的贡献，
    极大地减少了 Python 循环开销和 CPU-GPU 通信延迟。
    """
    def __init__(self, mesh_list, to_gpu=True):
        self.num_surfaces = len(mesh_list)
        
        # 1. 收集所有数据并扁平化
        # 注意：这里我们只保留积分所需的必要数据，减少显存占用
        
        # Flatten all arrays to (N, 3) or (N,)
        all_points = [m.points.reshape(-1, 3) for m in mesh_list]
        all_normals = [m.normals.reshape(-1, 3) for m in mesh_list]
        
        # 处理标量权重：jacobians * du * dv
        all_weights = []
        all_dP_du = []
        all_dP_dv = []
        all_du = []
        all_dv = []
        
        has_derivs = hasattr(mesh_list[0], 'dP_du')
        
        for m in mesh_list:
            # 预乘面积元 du * dv
            w = m.jacobians * m.du * m.dv
            all_weights.append(w.reshape(-1))
            
            if has_derivs:
                all_dP_du.append(m.dP_du.reshape(-1, 3))
                all_dP_dv.append(m.dP_dv.reshape(-1, 3))
                # 为了向量化，我们需要把 du, dv 扩展到每个点上
                all_du.append(np.full(m.points.shape[:2], m.du).reshape(-1))
                all_dv.append(np.full(m.points.shape[:2], m.dv).reshape(-1))

        # 2. 合并大数组 (CPU 端操作)
        self.points = np.vstack(all_points)
        self.normals = np.vstack(all_normals)
        self.weights = np.hstack(all_weights) # 1D array
        
        self.has_derivs = has_derivs
        if has_derivs:
            self.dP_du = np.vstack(all_dP_du)
            self.dP_dv = np.vstack(all_dP_dv)
            self.du = np.hstack(all_du)
            self.dv = np.hstack(all_dv)

        # 3. 立即传输到 GPU (如果请求)
        if to_gpu and HAS_GPU:
            self.points = cp.asarray(self.points)
            self.normals = cp.asarray(self.normals)
            self.weights = cp.asarray(self.weights)
            if has_derivs:
                self.dP_du = cp.asarray(self.dP_du)
                self.dP_dv = cp.asarray(self.dP_dv)
                self.du = cp.asarray(self.du)
                self.dv = cp.asarray(self.dv)

def merge_meshes(mesh_list, to_gpu=True):
    """工具函数：将网格列表合并为 MergedMeshData"""
    if not mesh_list:
        return None
    return MergedMeshData(mesh_list, to_gpu)
