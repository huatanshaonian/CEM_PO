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


def po_integrate_batch(mesh_list, k_dirs, k_mags, sinc_mode='dual',
                       precision='double'):
    """批量 PO 积分核 —— 一次 GPU 调用处理 Nbatch 个入射方向.

    数学上等价于对每个 k_dir 独立调用 po_integrate, 但通过把
    (M, 3) @ (3, Nbatch) 这种 GEMV 升级成 GEMM, 减少 launch overhead 并
    提升 cuBLAS 算力利用率. 在大量角度的扫描场景下 (例如 91×91 外形迭代)
    比单角度循环快 ~2-3x.

    参数:
        mesh_list:  list[CachedMeshData]
        k_dirs:     (Nbatch, 3) 入射方向数组. 单角度可传 (3,) 自动展开
        k_mags:     (Nf,) 波数数组. 单频时传 array([k])
        sinc_mode:  'none' | 'u_only' | 'dual'
        precision:  'double' | 'mixed' | 'single'

    返回:
        I_total: (Nbatch, Nf) complex128, 始终 CPU 数组.
    """
    if sinc_mode not in ('none', 'u_only', 'dual'):
        raise ValueError(f"sinc_mode 非法: {sinc_mode!r}")
    if precision not in _PRECISION_DTYPES:
        raise ValueError(f"precision 非法: {precision!r}")

    k_dirs_cpu = np.asarray(k_dirs, dtype=np.float64)
    if k_dirs_cpu.ndim == 1:
        k_dirs_cpu = k_dirs_cpu[None, :]
    Nbatch = k_dirs_cpu.shape[0]

    k_mags_cpu = np.atleast_1d(np.asarray(k_mags, dtype=np.float64))
    Nf = k_mags_cpu.size

    I_total = np.zeros((Nbatch, Nf), dtype=np.complex128)
    if not mesh_list:
        return I_total

    real_dtype, op_cdtype, accum_cdtype = _PRECISION_DTYPES[precision]
    op_j    = op_cdtype(1j)
    accum_j = accum_cdtype(1j)

    for cached in mesh_list:
        xp = _xp_of(cached.points)
        (pts, nrm, weights_base, dpdu, dpdv,
         du_sinc, dv_sinc) = _flatten_mesh(cached, xp, real_dtype)
        M = pts.shape[0]

        k_dirs_x = xp.asarray(k_dirs_cpu, dtype=real_dtype)         # (Nbatch, 3)
        k_mags_x = xp.asarray(k_mags_cpu, dtype=real_dtype)         # (Nf,)
        pi_x = real_dtype(np.pi)

        # 4 个核心 GEMM: (M, 3) @ (3, Nbatch) → (M, Nbatch)
        # cuBLAS 走 SGEMM/DGEMM 高吞吐, 取代每角度独立 GEMV
        kT = k_dirs_x.T                                              # (3, Nbatch)
        n_dot_k    = nrm  @ kT                                       # (M, Nbatch)
        pts_dot_k  = pts  @ kT                                       # (M, Nbatch)

        illum = xp.maximum(-n_dot_k, real_dtype(0.0))                # (M, Nbatch)

        # 按 batch 截掉全阴影方向, 避免后续除以 0
        # (现实场景几乎不会出现, 但 sphere 之类对称几何可能)
        w_full = weights_base[:, None] * illum                       # (M, Nbatch)
        w_sum  = xp.sum(w_full, axis=0)                              # (Nbatch,)
        active = w_sum > real_dtype(0.0)                             # (Nbatch,) bool
        # 防 0 除 (非 active 方向稍后乘 0 也不影响结果, 但仍要避免 NaN)
        w_sum_safe = xp.where(active, w_sum, real_dtype(1.0))

        # per-batch 加权重心 ref[b,d] = Σ_m pts[m,d] * w[m,b] / w_sum[b]
        # = (pts.T @ w_full) / w_sum.   shape (3, Nbatch) → 转置成 (Nbatch, 3)
        ref_points = (pts.T @ w_full) / w_sum_safe[None, :]          # (3, Nbatch)
        ref_proj   = xp.sum(ref_points * kT, axis=0)                 # (Nbatch,)

        # d_local[m, b] = (pts[m,:] - ref[b,:]) · k_dirs[b,:]
        #              = pts_dot_k[m,b] - ref_proj[b]
        d_local = pts_dot_k - ref_proj[None, :]                      # (M, Nbatch)

        # (Nbatch, Nf, M) 大数组 — Nbatch=64, Nf=1, M=240k, complex64 → 122 MB
        # two_jk: (Nf, Nbatch) op_cdtype
        two_jk = (2.0 * xp.outer(k_mags_x, xp.ones(
            Nbatch, dtype=real_dtype))).astype(op_cdtype) * op_j     # (Nf, Nbatch)
        # phase_arg[f, m, b] = two_jk[f, b] * d_local[m, b]
        d_local_c = d_local.astype(op_cdtype)                        # (M, Nbatch)
        # einsum 高效: 'fb,mb->fmb' 不可, 直接广播 (Nf,1,Nbatch)*(M,Nbatch)
        phase_arg = two_jk[:, None, :] * d_local_c[None, :, :]       # (Nf, M, Nbatch)
        phase_exp = xp.exp(phase_arg)                                # (Nf, M, Nbatch)

        if sinc_mode == 'none':
            contrib = phase_exp
        else:
            # alpha[m, b] = (dpdu[m,:] · k_dirs[b,:]) * du_sinc[m]
            dpdu_dot_k = dpdu @ kT                                   # (M, Nbatch) GEMM
            alpha = dpdu_dot_k * du_sinc[:, None]                    # (M, Nbatch)
            sinc_u = xp.sinc(k_mags_x[:, None, None] / pi_x *
                             alpha[None, :, :])                      # (Nf, M, Nbatch)
            if sinc_mode == 'dual':
                dpdv_dot_k = dpdv @ kT                               # (M, Nbatch)
                beta = dpdv_dot_k * dv_sinc[:, None]
                sinc_v = xp.sinc(k_mags_x[:, None, None] / pi_x *
                                 beta[None, :, :])
                sinc_mat = sinc_u * sinc_v
            else:
                sinc_mat = sinc_u
            contrib = sinc_mat.astype(op_cdtype) * phase_exp         # (Nf, M, Nbatch)

        # I_surf[f, b] = Σ_m contrib[f, m, b] * w_full[m, b]
        # einsum 路径走 cuBLAS 批量 GEMV / GEMM
        w_full_c = w_full.astype(op_cdtype)                          # (M, Nbatch)
        I_surf = xp.einsum('fmb,mb->fb', contrib, w_full_c)          # (Nf, Nbatch)

        # ref-phase 旋转 + 累加为 complex128 CPU
        # ref_phase[f, b] = 2 k_mags[f] * ref_proj[b]
        ref_phase_c = (2.0 * xp.outer(k_mags_x, ref_proj)).astype(
            accum_cdtype) * accum_j                                  # (Nf, Nbatch)
        I_surf_ac = I_surf.astype(accum_cdtype) * xp.exp(ref_phase_c)
        # 全阴影方向乘 0 mask
        I_surf_ac = I_surf_ac * active.astype(accum_cdtype)[None, :]

        I_cpu = _to_cpu_complex128(I_surf_ac, xp)                    # (Nf, Nbatch)
        I_total += I_cpu.T                                           # (Nbatch, Nf)

    return I_total


def po_integrate(mesh_list, k_dir, k_mags, sinc_mode='dual',
                 precision='double'):
    """单角度 PO 积分入口 (po_integrate_batch 的 batch=1 包装).

    保留独立函数是为了向后兼容旧调用方; 数学等价于
    po_integrate_batch(mesh, k_dir[None, :], k_mags, ...)[0].

    返回 (Nf,) complex128.
    """
    I_batch = po_integrate_batch(mesh_list, np.asarray(k_dir)[None, :],
                                 k_mags, sinc_mode=sinc_mode,
                                 precision=precision)
    return I_batch[0]
