"""
统一 PO 积分核 —— 单频扫角、多曲面合并、批量频扫的唯一数学实现.

设计:
- 输入永远是 list[CachedMeshData] (CPU FP64 或 GPU FP64), 自动检测设备
- k_mags 维度统一: 单频时 (1,), 频扫时 (Nf,), 输出 (Nf,) complex128
- 阴影面元用 max(-n·k, 0) 自然清零, 无 mask 无阈值
- 每曲面用自己的照射加权重心作参考点 (FP32 长基线下精度安全)
- 精度模式: double=FP64 全程, mixed=FP32 算子 + FP64 reduce, single=FP32 全程

设备决策由调用方负责: 想在 GPU 上算, 调用前 mesh.to_gpu(); 想在 CPU 上算, 保持 CPU.
kernel 内部 cp.get_array_module(mesh.points) 自动选 xp.
"""

import numpy as np
from core.env import HAS_GPU, cp


_PRECISION_DTYPES = {
    'double': (np.float64, np.complex128, np.complex128),
    'mixed':  (np.float32, np.complex64,  np.complex128),
    'single': (np.float32, np.complex64,  np.complex64),
}


def _xp_of(arr):
    """检测 array 所属模块 (numpy / cupy)."""
    if HAS_GPU and cp.get_array_module(arr) is cp:
        return cp
    return np


def _flatten_mesh(cached, xp, real_dtype):
    """从 CachedMeshData 抽出扁平的 (N,)/(N,3) 数组, 已 cast 到目标 dtype.

    带缓存: 同一 (设备, 精度) 组合下网格只 flatten + cast 一次, 后续 1801 次扫角
    全部命中. mesh.to_gpu/to_cpu 时缓存自动失效 (见 CachedMeshData).
    """
    # 缓存命中检测: (device_id, real_dtype name)
    cache = getattr(cached, '_precision_views', None)
    if cache is not None:
        key = (id(xp), np.dtype(real_dtype).name)
        if key in cache:
            return cache[key]

    pts  = xp.asarray(cached.points)
    nrm  = xp.asarray(cached.normals)
    jac  = xp.asarray(cached.jacobians)
    dpdu = xp.asarray(cached.dP_du)
    dpdv = xp.asarray(cached.dP_dv)

    is_degen = (pts.ndim == 2)

    if not is_degen:
        pts  = pts.reshape(-1, 3)
        nrm  = nrm.reshape(-1, 3)
        jac  = jac.reshape(-1)
        dpdu = dpdu.reshape(-1, 3)
        dpdv = dpdv.reshape(-1, 3)
        du_scalar = float(cached.du)
        dv_scalar = float(cached.dv)
        weights_base = jac * du_scalar * dv_scalar
        N = pts.shape[0]
        du_sinc = xp.full(N, du_scalar)
        dv_sinc = xp.full(N, dv_scalar)
    else:
        du_arr = xp.asarray(cached.du)
        dv_arr = xp.asarray(cached.dv)
        weights_base = jac * du_arr * dv_arr
        N = pts.shape[0]
        if hasattr(cached, 'sinc_du'):
            du_sinc = xp.asarray(cached.sinc_du)
            dv_sinc = xp.asarray(cached.sinc_dv)
        elif hasattr(cached, 'avg_du'):
            du_sinc = xp.full(N, float(cached.avg_du))
            dv_sinc = xp.full(N, float(cached.avg_dv))
        else:
            du_sinc = xp.full(N, float(xp.mean(du_arr).item()
                                       if hasattr(xp.mean(du_arr), 'item')
                                       else xp.mean(du_arr)))
            dv_sinc = xp.full(N, 1.0)

    pts_x  = pts.astype(real_dtype,  copy=False)
    nrm_x  = nrm.astype(real_dtype,  copy=False)
    w_x    = weights_base.astype(real_dtype, copy=False)
    dpdu_x = dpdu.astype(real_dtype, copy=False)
    dpdv_x = dpdv.astype(real_dtype, copy=False)
    dus_x  = du_sinc.astype(real_dtype, copy=False)
    dvs_x  = dv_sinc.astype(real_dtype, copy=False)
    result = (pts_x, nrm_x, w_x, dpdu_x, dpdv_x, dus_x, dvs_x)

    if cache is not None:
        cache[key] = result
    return result


def _to_cpu_complex128(arr, xp):
    if xp is np:
        return np.asarray(arr, dtype=np.complex128)
    return cp.asnumpy(arr).astype(np.complex128, copy=False)


def po_integrate(mesh_list, k_dir, k_mags, sinc_mode='dual',
                 precision='double'):
    """统一 PO 积分核.

    参数:
        mesh_list:  list[CachedMeshData], 可以在 CPU 也可以在 GPU.
                    设备由 mesh.points 自动检测.
        k_dir:      (3,) 入射方向单位向量 (numpy 即可, kernel 内会迁移).
        k_mags:     (Nf,) 波数数组. 单频时传 array([k]).
        sinc_mode:  'none' | 'u_only' | 'dual'
        precision:  'double' | 'mixed' | 'single'

    返回:
        I_total: (Nf,) complex128, 各频率的 PO 散射积分 (始终 CPU).
    """
    if sinc_mode not in ('none', 'u_only', 'dual'):
        raise ValueError(f"sinc_mode 非法: {sinc_mode!r}")
    if precision not in _PRECISION_DTYPES:
        raise ValueError(f"precision 非法: {precision!r}")
    if not mesh_list:
        Nf = max(1, np.atleast_1d(np.asarray(k_mags)).size)
        return np.zeros(Nf, dtype=np.complex128)

    real_dtype, op_cdtype, accum_cdtype = _PRECISION_DTYPES[precision]
    k_mags_cpu = np.atleast_1d(np.asarray(k_mags, dtype=np.float64))
    Nf = k_mags_cpu.size
    I_total = np.zeros(Nf, dtype=np.complex128)

    # 用目标 dtype 的虚数单位, 避免 `1j * float32` 强制 promote 到 complex128
    op_j     = op_cdtype(1j)
    accum_j  = accum_cdtype(1j)

    for cached in mesh_list:
        xp = _xp_of(cached.points)
        (pts, nrm, weights_base, dpdu, dpdv,
         du_sinc, dv_sinc) = _flatten_mesh(cached, xp, real_dtype)

        k_dir_x  = xp.asarray(k_dir,      dtype=real_dtype)        # (3,)
        k_mags_x = xp.asarray(k_mags_cpu, dtype=real_dtype)        # (Nf,)
        pi_x = real_dtype(np.pi)

        n_dot_k = nrm @ k_dir_x                                    # (N,)
        illum   = xp.maximum(-n_dot_k, real_dtype(0.0))            # (N,)

        if not bool(xp.any(illum > 0)):
            continue

        w = weights_base * illum                                   # (N,)
        w_sum = xp.sum(w)
        ref_point = (pts * w[:, None]).sum(axis=0) / w_sum         # (3,)

        d_local  = (pts - ref_point) @ k_dir_x                     # (N,)
        ref_proj_x = ref_point @ k_dir_x                           # scalar xp

        # 把 2k * 1j_op 预乘成 op_cdtype, 避免中间 phase_mat 实数 → complex 转换的
        # 临时分配 (旧版 `1j * phase_mat` 产生 complex128 中间结果, FP32 模式下浪费)
        two_jk = (2.0 * k_mags_x).astype(op_cdtype) * op_j         # (Nf,) op_cdtype

        # (Nf, N) op_cdtype: 直接构造 complex 相位
        phase_arg_c = two_jk[:, None] * d_local.astype(op_cdtype)[None, :]
        phase_exp = xp.exp(phase_arg_c)                            # (Nf, N) op_cdtype

        if sinc_mode == 'none':
            contrib = phase_exp
        else:
            alpha = (dpdu @ k_dir_x) * du_sinc                     # (N,) real
            sinc_u = xp.sinc(k_mags_x[:, None] / pi_x * alpha[None, :])  # (Nf,N) real
            if sinc_mode == 'dual':
                beta = (dpdv @ k_dir_x) * dv_sinc
                sinc_v = xp.sinc(k_mags_x[:, None] / pi_x * beta[None, :])
                sinc_mat = sinc_u * sinc_v
            else:
                sinc_mat = sinc_u
            contrib = sinc_mat.astype(op_cdtype) * phase_exp       # (Nf, N) op_cdtype

        w_op = w.astype(op_cdtype)
        I_surf = contrib @ w_op                                    # (Nf,) op_cdtype

        # 升精度做 ref-phase 旋转, 累加成 complex128 CPU
        ref_phase_c = (2.0 * k_mags_x * ref_proj_x).astype(accum_cdtype) * accum_j
        I_surf_ac = I_surf.astype(accum_cdtype) * xp.exp(ref_phase_c)
        I_total += _to_cpu_complex128(I_surf_ac, xp)

    return I_total
