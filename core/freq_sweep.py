"""
频率扫描（相位旋转法）核心计算模块。

PO 频扫：预算几何量一次，批量向量化处理所有频率。
PTD 频扫：预算衍射系数和几何量，向量化处理。
距离像：IFFT + 窗函数 + 零填充。
"""

import numpy as np
from scipy.signal.windows import chebwin, taylor
from physics.constants import C0
from core.env import HAS_GPU, cp


def compute_po_freq_sweep(mesh_list, k_dir, frequencies, sinc_mode='dual', use_gpu=False):
    """
    PO 相位旋转法频率扫描（固定入射方向，扫频率）。

    参数:
        mesh_list:   list[CachedMeshData]，每个面的预计算网格
        k_dir:       ndarray (3,)，入射方向单位向量（频率无关）
        frequencies: ndarray (Nf,)，频率数组 (Hz)
        sinc_mode:   'none' | 'u_only' | 'dual'
        use_gpu:     是否使用 GPU (CuPy)

    返回:
        I_po: ndarray complex128 (Nf,)，各频率的 PO 散射积分（始终 CPU 数组）
    """
    xp = cp if (use_gpu and HAS_GPU) else np
    k_values_cpu = 2.0 * np.pi * np.asarray(frequencies) / C0  # (Nf,) CPU
    Nf = len(k_values_cpu)
    I_total = np.zeros(Nf, dtype=np.complex128)

    k_values_xp = xp.asarray(k_values_cpu)  # (Nf,) on device

    for cached in mesh_list:
        # --- 确保所有网格数据在 CPU numpy ---
        pts  = np.asarray(cached.points)
        nrm  = np.asarray(cached.normals)
        jac  = np.asarray(cached.jacobians)
        dpdu = np.asarray(cached.dP_du)
        dpdv = np.asarray(cached.dP_dv)

        is_degen = (pts.ndim == 2)

        if not is_degen:
            # 规则网格：(nv, nu, 3) → (N, 3)
            pts  = pts.reshape(-1, 3)
            nrm  = nrm.reshape(-1, 3)
            jac  = jac.reshape(-1)
            dpdu = dpdu.reshape(-1, 3)
            dpdv = dpdv.reshape(-1, 3)
            du_scalar = float(cached.du)
            dv_scalar = float(cached.dv)
            weights_base = jac * du_scalar * dv_scalar  # (N,)
            # sinc 校正步长
            N = len(pts)
            du_sinc = np.full(N, du_scalar)
            dv_sinc = np.full(N, dv_scalar)
        else:
            # 退化网格：(N, 3)
            du_arr = np.asarray(cached.du)   # cell_areas
            dv_arr = np.asarray(cached.dv)   # ones
            weights_base = jac * du_arr * dv_arr  # (N,)
            N = len(pts)
            if hasattr(cached, 'sinc_du'):
                du_sinc = np.asarray(cached.sinc_du)
                dv_sinc = np.asarray(cached.sinc_dv)
            elif hasattr(cached, 'avg_du'):
                du_sinc = np.full(N, cached.avg_du)
                dv_sinc = np.full(N, cached.avg_dv)
            else:
                du_sinc = np.full(N, float(np.mean(du_arr)))
                dv_sinc = np.full(N, 1.0)

        # --- 照射检测（与频率无关）---
        n_dot_k = np.einsum('ij,j->i', nrm, k_dir)  # (N,)
        lit_mask = n_dot_k < -1e-6
        if not np.any(lit_mask):
            continue

        pts_lit   = pts[lit_mask]
        dpdu_lit  = dpdu[lit_mask]
        dpdv_lit  = dpdv[lit_mask]
        illum_lit = -n_dot_k[lit_mask]                # (N_lit,) > 0
        w_lit     = weights_base[lit_mask] * illum_lit  # (N_lit,)
        du_sinc_lit = du_sinc[lit_mask]
        dv_sinc_lit = dv_sinc[lit_mask]

        # --- 参考点相位稳定化 ---
        mid_idx = len(pts_lit) // 2
        ref_point = pts_lit[mid_idx]
        d_lit = np.einsum('ij,j->i', pts_lit - ref_point, k_dir)  # (N_lit,)
        r_ref_proj = float(np.dot(ref_point, k_dir))

        # --- 迁移到 xp ---
        d_xp = xp.asarray(d_lit)       # (N_lit,)
        w_xp = xp.asarray(w_lit)       # (N_lit,)

        # phase_mat[i, n] = 2 * k[i] * d[n]  →  shape (Nf, N_lit)
        phase_mat = 2.0 * xp.outer(k_values_xp, d_xp)
        phase_ref = 2.0 * k_values_xp * r_ref_proj  # (Nf,)

        if sinc_mode == 'none':
            # 无 sinc 校正
            contrib_mat = xp.exp(1j * phase_mat)          # (Nf, N_lit)
        else:
            # alpha_base[n] = dP_du[n] · k_dir  （频率无关）
            alpha_base = np.einsum('ij,j->i', dpdu_lit, k_dir)  # (N_lit,)
            alpha_xp   = xp.asarray(alpha_base * du_sinc_lit)    # (N_lit,)
            # sinc_u_arg[i,n] = k[i] * alpha[n] / π
            sinc_u_arg = xp.outer(k_values_xp / xp.pi, alpha_xp)  # (Nf, N_lit)
            sinc_mat   = xp.sinc(sinc_u_arg)

            if sinc_mode != 'u_only':  # dual
                beta_base  = np.einsum('ij,j->i', dpdv_lit, k_dir)  # (N_lit,)
                beta_xp    = xp.asarray(beta_base * dv_sinc_lit)
                sinc_v_arg = xp.outer(k_values_xp / xp.pi, beta_xp)
                sinc_mat   = sinc_mat * xp.sinc(sinc_v_arg)

            contrib_mat = sinc_mat * xp.exp(1j * phase_mat)    # (Nf, N_lit)

        I_surf = contrib_mat @ w_xp                              # (Nf,)
        I_surf = I_surf * xp.exp(1j * phase_ref)

        # 拉回 CPU 累加
        if HAS_GPU and use_gpu and hasattr(I_surf, 'get'):
            I_total += I_surf.get()
        else:
            I_total += np.asarray(I_surf)

    return I_total


def compute_ptd_freq_sweep(ptd_edges, k_dir, frequencies, polarization='VV', use_gpu=False):
    """
    PTD 相位旋转法频率扫描（固定入射方向，扫频率）。

    衍射系数 D 完全频率无关，向量化处理所有频率。

    参数:
        ptd_edges:   list[PTDEdge]，已提取的边缘列表
        k_dir:       ndarray (3,)，入射方向单位向量
        frequencies: ndarray (Nf,)，频率数组 (Hz)
        polarization: 'VV' 或 'HH'
        use_gpu:     是否使用 GPU

    返回:
        I_ptd: ndarray complex128 (Nf,)（始终 CPU 数组）
    """
    from physics.ptd_core import _compute_D

    xp = cp if (use_gpu and HAS_GPU) else np
    k_values_cpu = 2.0 * np.pi * np.asarray(frequencies) / C0  # (Nf,)
    Nf = len(k_values_cpu)
    I_ptd = np.zeros(Nf, dtype=np.complex128)

    k_arr_xp = xp.asarray(k_values_cpu)  # (Nf,)

    for edge in ptd_edges:
        for seg in edge.segments:
            alfa  = seg.alpha
            t     = seg.tangent
            n_lit = seg.normal if (seg.normal is not None) else edge.n_lit

            # ── 1. 斜入射角 γ₀ ──
            k_dot_t  = float(np.clip(np.dot(k_dir, t), -1.0, 1.0))
            sin_gamma0 = float(np.sqrt(max(0.0, 1.0 - k_dot_t ** 2)))
            if sin_gamma0 < 1e-3:
                continue

            # ── 2. 局部坐标系 ──
            e1 = n_lit
            e2_raw = np.cross(t, n_lit)
            e2_len = float(np.linalg.norm(e2_raw))
            if e2_len < 1e-10:
                continue
            e2 = e2_raw / e2_len

            # ── 3. 入射角 angle0 ──
            k_perp = k_dir - k_dot_t * t
            k_perp_len = float(np.linalg.norm(k_perp))
            if k_perp_len < 1e-10:
                continue
            k_perp_unit = k_perp / k_perp_len
            inc_unit = -k_perp_unit
            i_e1 = float(np.dot(inc_unit, e1))
            i_e2 = float(np.dot(inc_unit, e2))
            angle0_raw = float(np.arctan2(i_e1, i_e2))
            if angle0_raw < -1e-6 or angle0_raw > alfa + 1e-6:
                continue
            angle0 = float(np.clip(angle0_raw, 0.0, alfa))

            # ── 3b. 散射角 angle_obs（单站：= angle0）──
            s_dir = -k_dir
            s_dot_t = float(np.dot(s_dir, t))
            s_perp = s_dir - s_dot_t * t
            s_perp_len = float(np.linalg.norm(s_perp))
            if s_perp_len < 1e-10:
                continue
            s_perp_unit = s_perp / s_perp_len
            s_e1 = float(np.dot(s_perp_unit, e1))
            s_e2 = float(np.dot(s_perp_unit, e2))
            angle_obs = float(np.arctan2(s_e1, s_e2)) % alfa

            # ── 4. 衍射系数 D（频率无关）──
            gamma0 = float(np.arcsin(sin_gamma0))
            D = _compute_D(angle_obs, angle0, gamma0, alfa, polarization)

            # ── 5. 频率无关几何量 ──
            L      = float(seg.length)
            r_proj = float(np.dot(seg.midpoint, k_dir))  # r_mid · k_dir

            # ── 6. 向量化频率 ──
            # pre_arr[i]   = 2π / (1j * k[i] * sin_gamma0)
            # sinc_arr[i]  = sinc(k[i] * L * k_dot_t / π)
            # phase_arr[i] = exp(2j * k[i] * r_proj)
            pre_arr   = (2.0 * np.pi) / (1j * k_arr_xp * sin_gamma0)   # (Nf,)
            sinc_arg  = k_arr_xp * L * k_dot_t / float(np.pi)          # (Nf,)
            sinc_arr  = xp.sinc(sinc_arg)                                # (Nf,)
            phase_arr = xp.exp(2j * k_arr_xp * r_proj)                  # (Nf,)

            seg_contrib = pre_arr * complex(D) * L * sinc_arr * phase_arr   # (Nf,)

            if HAS_GPU and use_gpu and hasattr(seg_contrib, 'get'):
                I_ptd += seg_contrib.get()
            else:
                I_ptd += np.asarray(seg_contrib)

    return I_ptd


def compute_range_profile(I_freq, frequencies, window='hamming', zero_pad=4, cheby_at=40.0,
                          taylor_nbar=4, taylor_sll=30.0):
    """
    从频域散射数据计算距离像（IFFT 方法）。

    参数:
        I_freq:       ndarray complex (Nf,)，频域散射积分
        frequencies:  ndarray (Nf,)，频率数组 (Hz)
        window:       窗函数类型 'hamming' | 'hanning' | 'blackman' | 'chebyshev' |
                      'taylor' | 'rectangular'
        zero_pad:     零填充倍数（整数）
        cheby_at:     Chebyshev 窗副瓣衰减 (dB)
        taylor_nbar:  Taylor 窗等副瓣段数（通常 4~8）
        taylor_sll:   Taylor 窗副瓣电平 (dB, 正值)

    返回:
        (profile_db, range_axis, profile_complex, stats_dict)
        - profile_db:      ndarray (N_pad,)，归一化 dB 距离像
        - range_axis:      ndarray (N_pad,)，对应距离轴 (m)
        - profile_complex: ndarray complex (N_pad,)，复数距离像
        - stats_dict:      dict，统计量
    """
    I_freq = np.asarray(I_freq)
    frequencies = np.asarray(frequencies)
    Nf = len(frequencies)

    if Nf < 2:
        dummy = np.zeros(max(1, zero_pad))
        stats = {
            'range_resolution_m': 0.0,
            'max_range_m': 0.0,
            'bandwidth_mhz': 0.0,
            'n_freqs': Nf,
            'delta_f_mhz': 0.0,
        }
        return dummy, np.zeros(len(dummy)), dummy.astype(complex), stats

    delta_f = (frequencies[-1] - frequencies[0]) / (Nf - 1)
    bandwidth = frequencies[-1] - frequencies[0]

    # 窗函数
    if window == 'hamming':
        win = np.hamming(Nf)
    elif window == 'hanning':
        win = np.hanning(Nf)
    elif window == 'blackman':
        win = np.blackman(Nf)
    elif window == 'chebyshev':
        win = chebwin(Nf, at=float(cheby_at))
    elif window == 'taylor':
        win = taylor(Nf, nbar=int(taylor_nbar), sll=float(taylor_sll), norm=True)
    else:  # rectangular
        win = np.ones(Nf)

    S_windowed = I_freq * win

    # 零填充
    N_pad = Nf * int(zero_pad)
    S_padded = np.zeros(N_pad, dtype=np.complex128)
    S_padded[:Nf] = S_windowed

    # IFFT → 距离像
    profile_complex = np.fft.ifft(S_padded)

    # 距离轴：Δr = c/(2·N_pad·Δf)，第 n 个 bin 对应 r = n * c/(2·N_pad·Δf)
    range_axis = np.arange(N_pad) * C0 / (2.0 * N_pad * delta_f)

    # dB 归一化（相对峰值）
    profile_mag = np.abs(profile_complex)
    peak = np.max(profile_mag)
    if peak < 1e-30:
        peak = 1e-30
    profile_db = 20.0 * np.log10(profile_mag / peak + 1e-30)

    range_resolution = C0 / (2.0 * bandwidth)
    max_range = C0 / (2.0 * delta_f)

    stats = {
        'range_resolution_m': range_resolution,
        'max_range_m': max_range,
        'bandwidth_mhz': bandwidth / 1e6,
        'n_freqs': Nf,
        'delta_f_mhz': delta_f / 1e6,
    }

    return profile_db, range_axis, profile_complex, stats


def _is_new_format_csv(path):
    """检测 CSV 是否为新长格式（含 Frequency (MHz) 列头）。"""
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            return 'Frequency (MHz)' in line
    return False


def _load_new_format_freq_sweep_csv(path):
    """
    读取新长格式频扫 RCS CSV（由 export_freq_sweep_rcs_csv 生成）。

    格式：
      - 以 '# Key, Value' 表示元数据
      - 列头：Theta (deg), Phi (deg), RCS Total (dBsm), [...,] I Total (Re), I Total (Im), Frequency (MHz)
      - 每行对应一个 (角度, 频率) 组合

    返回与 run_freq_sweep() 兼容的 result dict，带 'source': 'csv'。
    range_axis / profile_matrix 为正距离半段（由 I Total 重算距离像）。
    """
    import pandas as pd

    meta = {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\n').strip()
            if line.startswith('#'):
                content = line[1:].strip()
                if ',' in content:
                    key, _, val = content.partition(',')
                    meta[key.strip()] = val.strip()

    df = pd.read_csv(path, comment='#')

    # 提取唯一角度和频率
    freq_mhz_arr = np.sort(df['Frequency (MHz)'].unique())
    freq_arr     = freq_mhz_arr * 1e6   # Hz
    Nf           = len(freq_arr)

    theta_arr = np.sort(df['Theta (deg)'].unique())
    phi_arr   = np.sort(df['Phi (deg)'].unique())
    angle_list = [(th, ph) for th in theta_arr for ph in phi_arr]
    N_angles   = len(angle_list)

    rcs_matrix     = np.full((N_angles, Nf), -999.0)
    I_total_matrix = np.zeros((N_angles, Nf), dtype=np.complex128)
    has_i_total    = ('I Total (Re)' in df.columns and 'I Total (Im)' in df.columns)

    # 用 pivot_table 向量化填充（快速）
    rcs_pivot = df.pivot_table(
        index=['Theta (deg)', 'Phi (deg)'],
        columns='Frequency (MHz)',
        values='RCS Total (dBsm)',
        aggfunc='mean',
    )
    for i, (th, ph) in enumerate(angle_list):
        key = (th, ph)
        if key in rcs_pivot.index:
            for jj, fmhz in enumerate(freq_mhz_arr):
                if fmhz in rcs_pivot.columns:
                    rcs_matrix[i, jj] = rcs_pivot.loc[key, fmhz]

    if has_i_total:
        re_pivot = df.pivot_table(
            index=['Theta (deg)', 'Phi (deg)'],
            columns='Frequency (MHz)',
            values='I Total (Re)',
            aggfunc='mean',
        )
        im_pivot = df.pivot_table(
            index=['Theta (deg)', 'Phi (deg)'],
            columns='Frequency (MHz)',
            values='I Total (Im)',
            aggfunc='mean',
        )
        for i, (th, ph) in enumerate(angle_list):
            key = (th, ph)
            if key in re_pivot.index:
                for jj, fmhz in enumerate(freq_mhz_arr):
                    if fmhz in re_pivot.columns:
                        I_total_matrix[i, jj] = complex(re_pivot.loc[key, fmhz],
                                                         im_pivot.loc[key, fmhz])

    # 重算距离像（正距离半段）
    window      = meta.get('Window', 'hamming')
    zero_pad    = int(float(meta.get('Zero Pad', 4)))
    cheby_at    = float(meta.get('Sidelobe (dB)', 40.0))
    taylor_nbar = int(float(meta.get('Taylor nbar', 4)))
    taylor_sll  = cheby_at   # SLL 复用同一元数据字段
    N_pad  = Nf * zero_pad
    N_half = N_pad // 2

    profile_matrix = None
    range_axis     = None
    stats          = None
    if has_i_total and Nf >= 2:
        profile_matrix = np.zeros((N_angles, N_half))
        for i in range(N_angles):
            prof_db, r_ax, _, stats_i = compute_range_profile(
                I_total_matrix[i], freq_arr, window, zero_pad, cheby_at,
                taylor_nbar=taylor_nbar, taylor_sll=taylor_sll)
            profile_matrix[i] = prof_db[:N_half]
            if range_axis is None:
                range_axis = r_ax[:N_half]
                stats = stats_i

    # 统计量
    delta_f   = (freq_arr[-1] - freq_arr[0]) / (Nf - 1) if Nf > 1 else 1.0
    bandwidth = freq_arr[-1] - freq_arr[0]

    def _f(key, fallback):
        try:
            return float(meta[key]) if key in meta else fallback
        except (ValueError, TypeError):
            return fallback

    if stats is None:
        stats = {
            'range_resolution_m': _f('Range Resolution (m)', C0 / (2 * bandwidth) if bandwidth > 0 else 0),
            'max_range_m':        _f('Max Range (m)',        C0 / (2 * delta_f)   if delta_f  > 0 else 0),
            'bandwidth_mhz':      bandwidth / 1e6,
            'n_freqs':            Nf,
            'delta_f_mhz':        delta_f / 1e6,
        }

    scan_mode = meta.get('Scan Mode', '1d' if N_angles == 1 else '2d_angle_freq')

    if scan_mode == '1d':
        rcs_matrix     = rcs_matrix.squeeze(axis=0)
        I_total_matrix = I_total_matrix.squeeze(axis=0)
        if profile_matrix is not None:
            profile_matrix = profile_matrix.squeeze(axis=0)

    freq_sweep_params = {
        'f_start':      _f('Freq Start (MHz)', freq_arr[0] / 1e6),
        'f_end':        _f('Freq End (MHz)',   freq_arr[-1] / 1e6),
        'f_step':       _f('Freq Step (MHz)',  delta_f / 1e6),
        'window':       window,
        'zero_pad':     zero_pad,
        'polarization': meta.get('Polarization', 'VV'),
    }

    return {
        'mode':             'freq_sweep',
        'source':           'csv',
        'frequencies':      freq_arr,
        'theta_deg':        theta_arr,
        'phi_deg':          phi_arr,
        'scan_mode':        scan_mode,
        'I_po_matrix':      None,
        'I_ptd_matrix':     None,
        'I_total_matrix':   I_total_matrix,
        'rcs_matrix':       rcs_matrix,
        'profile_matrix':   profile_matrix,
        'range_axis':       range_axis,
        'stats':            stats,
        'elapsed_time':     0.0,
        'params': {
            'algorithm':   meta.get('Algorithm', ''),
            'ptd': {
                'enabled':      meta.get('PTD Enabled', 'False').lower() == 'true',
                'edges':        meta.get('PTD Edges', ''),
                'polarization': meta.get('Polarization', 'VV'),
            },
            'angles': {
                'theta_start': float(meta.get('Theta Start (deg)', theta_arr[0]  if len(theta_arr) else 0)),
                'theta_end':   float(meta.get('Theta End (deg)',   theta_arr[-1] if len(theta_arr) else 0)),
                'n_theta':     len(theta_arr),
                'phi_start':   float(meta.get('Phi Start (deg)',   phi_arr[0]    if len(phi_arr) else 0)),
                'phi_end':     float(meta.get('Phi End (deg)',     phi_arr[-1]   if len(phi_arr) else 0)),
                'n_phi':       len(phi_arr),
            },
        },
        'freq_sweep_params': freq_sweep_params,
        'timestamp':        0.0,
    }


def load_freq_sweep_csv(path):
    """
    读取频扫 CSV 文件，自动检测新/旧格式。

    新格式（export_freq_sweep_rcs_csv 生成）：
      长格式，列头含 Frequency (MHz)，从 I Total 重算距离像。

    旧格式（旧版 export_range_profile_csv 生成）：
      - 以 '# Key, Value' 表示元数据
      - '# === Frequency Domain ===' 后跟频域数据表
      - '# === Range Domain ===' 后跟距离域数据表（仅正距离半段）

    返回与 run_freq_sweep() 兼容的 result dict，额外带有 'source': 'csv'。
    profile_matrix / range_axis 已是正距离半段，plot_radar_imaging 不应再截半。

    异常：
      - ValueError：文件中没有有效的频域数据
      - 其他 IO/解析错误正常传播
    """
    if _is_new_format_csv(path):
        return _load_new_format_freq_sweep_csv(path)
    meta = {}
    freq_header = None
    range_header = None
    freq_rows = []
    range_rows = []
    section = None

    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.rstrip('\n').strip()
            if not line:
                continue

            # 段落标记
            if line.startswith('# ==='):
                if 'Frequency Domain' in line:
                    section = 'freq'
                elif 'Range Domain' in line:
                    section = 'range'
                else:
                    section = None
                continue

            # 元数据行
            if line.startswith('#'):
                content = line[1:].strip()
                if ',' in content:
                    key, _, val = content.partition(',')
                    meta[key.strip()] = val.strip()
                continue

            # 数据行
            parts = [p.strip() for p in line.split(',')]
            if section == 'freq':
                if freq_header is None:
                    freq_header = parts
                else:
                    freq_rows.append(parts)
            elif section == 'range':
                if range_header is None:
                    range_header = parts
                else:
                    range_rows.append(parts)

    if not freq_rows or freq_header is None:
        raise ValueError(f"No valid frequency domain data found in {path}")

    # ── 解析频域数据 ──
    freq_arr = np.array([float(r[0]) for r in freq_rows]) * 1e6   # Hz
    Nf = len(freq_arr)

    # 计算角度数（从列头中计数 RCS_i 列）
    n_angles = sum(1 for c in freq_header if c.startswith('RCS_') and '(dBsm)' in c)
    if n_angles == 0:
        n_angles = 1

    rcs_matrix    = np.zeros((n_angles, Nf))
    I_total_matrix = np.zeros((n_angles, Nf), dtype=np.complex128)

    for j, row in enumerate(freq_rows):
        for i in range(n_angles):
            try:
                idx_rcs = freq_header.index(f"RCS_{i} (dBsm)")
                rcs_matrix[i, j] = float(row[idx_rcs])
            except (ValueError, IndexError):
                pass
            try:
                idx_re = freq_header.index(f"I_re_{i}")
                idx_im = freq_header.index(f"I_im_{i}")
                I_total_matrix[i, j] = complex(float(row[idx_re]), float(row[idx_im]))
            except (ValueError, IndexError):
                pass

    # ── 解析距离域数据 ──
    range_axis = None
    profile_matrix = None
    if range_rows and range_header is not None:
        range_axis = np.array([float(r[0]) for r in range_rows])
        n_range = len(range_rows)
        n_prof = sum(1 for c in range_header if c.startswith('Profile_'))
        if n_prof == 0:
            n_prof = n_angles
        profile_matrix = np.zeros((n_prof, n_range))
        for j, row in enumerate(range_rows):
            for i in range(n_prof):
                try:
                    idx = range_header.index(f"Profile_{i} (dB)")
                    profile_matrix[i, j] = float(row[idx])
                except (ValueError, IndexError):
                    pass

    # ── 统计量（优先读 meta，回退到从数据推算）──
    delta_f = (freq_arr[-1] - freq_arr[0]) / (Nf - 1) if Nf > 1 else 1.0
    bandwidth = freq_arr[-1] - freq_arr[0]

    def _f(key, fallback):
        try:
            return float(meta[key]) if key in meta else fallback
        except (ValueError, TypeError):
            return fallback

    stats = {
        'range_resolution_m': _f('Range Resolution (m)', C0 / (2 * bandwidth) if bandwidth > 0 else 0),
        'max_range_m':        _f('Max Range (m)',        C0 / (2 * delta_f)   if delta_f  > 0 else 0),
        'bandwidth_mhz':      _f('Bandwidth (MHz)',      bandwidth / 1e6),
        'n_freqs':            Nf,
        'delta_f_mhz':        delta_f / 1e6,
    }

    # ── 角度信息 ──
    try:
        n_theta     = int(meta.get('N Theta', 1))
        n_phi       = int(meta.get('N Phi', 1))
        theta_start = float(meta.get('Theta Start (deg)', 0))
        theta_end   = float(meta.get('Theta End (deg)',   0))
        phi_start   = float(meta.get('Phi Start (deg)',   0))
        phi_end     = float(meta.get('Phi End (deg)',     0))
        theta_deg   = np.linspace(theta_start, theta_end, n_theta)
        phi_deg     = np.linspace(phi_start,   phi_end,   n_phi)
    except Exception:
        theta_deg = np.array([0.0])
        phi_deg   = np.array([0.0])

    scan_mode = meta.get('Scan Mode', '1d' if n_angles == 1 else '2d_angle_freq')

    # ── squeeze 1D ──
    if scan_mode == '1d':
        rcs_matrix     = rcs_matrix.squeeze(axis=0)
        I_total_matrix = I_total_matrix.squeeze(axis=0)
        if profile_matrix is not None:
            profile_matrix = profile_matrix.squeeze(axis=0)

    freq_sweep_params = {
        'f_start':      _f('Freq Start (MHz)', freq_arr[0] / 1e6),
        'f_end':        _f('Freq End (MHz)',   freq_arr[-1] / 1e6),
        'f_step':       _f('Freq Step (MHz)',  delta_f / 1e6),
        'window':       meta.get('Window',       'hamming'),
        'zero_pad':     int(_f('Zero Pad', 4)),
        'polarization': meta.get('Polarization', 'VV'),
    }

    return {
        'mode':           'freq_sweep',
        'source':         'csv',          # 标记来源，供 plot_radar_imaging 判断
        'frequencies':    freq_arr,
        'theta_deg':      theta_deg,
        'phi_deg':        phi_deg,
        'scan_mode':      scan_mode,
        'I_po_matrix':    None,
        'I_ptd_matrix':   None,
        'I_total_matrix': I_total_matrix,
        'rcs_matrix':     rcs_matrix,
        'profile_matrix': profile_matrix,  # 已是正距离半段，长度 = N_range
        'range_axis':     range_axis,      # 已是正距离半段
        'stats':          stats,
        'elapsed_time':   0.0,
        'params': {
            'algorithm':   meta.get('Algorithm', ''),
            'ptd': {
                'enabled':      meta.get('PTD Enabled', 'False').lower() == 'true',
                'edges':        meta.get('PTD Edges', ''),
                'polarization': meta.get('Polarization', 'VV'),
            },
            'angles': {
                'theta_start': float(meta.get('Theta Start (deg)', 0)),
                'theta_end':   float(meta.get('Theta End (deg)',   0)),
                'n_theta':     int(meta.get('N Theta', 1)),
                'phi_start':   float(meta.get('Phi Start (deg)',   0)),
                'phi_end':     float(meta.get('Phi End (deg)',     0)),
                'n_phi':       int(meta.get('N Phi', 1)),
            },
        },
        'freq_sweep_params': freq_sweep_params,
        'timestamp':      0.0,
    }
