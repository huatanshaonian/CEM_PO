import time
import numpy as np
import traceback
from physics.constants import C0
from physics.wave import IncidentWave
from solvers.api import get_integrator, AVAILABLE_ALGORITHMS
from solvers.rcs_analyzer import RCSAnalyzer
from core.mesh_data import merge_meshes
from core.freq_sweep import compute_po_freq_sweep, compute_ptd_freq_sweep, compute_range_profile

class SolverBridge:
    """
    桥接 UI 和 求解器核心的中间层。
    负责解析参数、管理网格缓存、调用求解器并返回标准化的结果。
    """
    def __init__(self):
        self.cached_mesh_data = None
        self.cached_mesh_params = {}

    def run_simulation(self, geo, params, progress_callback=None, abort_event=None,
                       prev_result=None):
        """
        运行仿真计算 (同步阻塞调用，建议在独立线程中运行)

        Args:
            geo: 几何对象 (Surface 列表或单个 Surface)
            params: 参数字典，包含所有计算配置
            progress_callback: 接受 (current, total, message) 的回调函数
            abort_event: threading.Event，设置后终止计算
            prev_result: 上次计算结果，用于 ptd_only 模式叠加已有 PO

        Returns:
            dict: 包含计算结果和元数据的字典
        """
        try:
            start_time = time.time()
            
            # --- 1. 参数解包 ---
            freq = params['frequency']
            algo_id = params['algorithm']
            mesh_params = params.get('mesh', {})
            compute_params = params.get('compute', {})
            ptd_params = params.get('ptd', {})
            angle_params = params.get('angles', {})
            
            samples = mesh_params.get('density', 10.0)
            min_points = mesh_params.get('min_points', 18)
            use_degen = mesh_params.get('use_degenerate', False)

            use_gpu = compute_params.get('gpu', False)
            parallel = compute_params.get('parallel', False)
            n_workers = compute_params.get('workers', 4)
            
            enable_ptd = ptd_params.get('enabled', False)
            ptd_only = ptd_params.get('ptd_only', False) and enable_ptd
            ptd_edges = ptd_params.get('edges', '')
            ptd_pol = ptd_params.get('polarization', 'VV')
            ptd_seg_angle = ptd_params.get('seg_angle_deg', 2.0)
            ptd_algorithm = ptd_params.get('algorithm', 'ufimtsev_eew')
            ptd_max_seg_lambda = ptd_params.get('max_seg_lambda', None)

            theta_start = angle_params.get('theta_start', -90)
            theta_end = angle_params.get('theta_end', 90)
            n_theta = angle_params.get('n_theta', 181)
            phi_start = angle_params.get('phi_start', 0)
            phi_end = angle_params.get('phi_end', 0)
            n_phi = angle_params.get('n_phi', 1)

            # --- 2. 角度生成 ---
            theta_deg = np.linspace(theta_start, theta_end, n_theta)
            theta_rad = np.radians(theta_deg)
            phi_deg = np.linspace(phi_start, phi_end, n_phi)
            phi_rad = np.radians(phi_deg)
            
            is_2d = n_phi > 1 and n_theta > 1
            is_phi_scan = n_phi > 1 and n_theta == 1  # 仅 phi 变化的 1D 扫描

            # --- 3. 网格缓存处理（ptd_only 时跳过，不需要 PO 网格） ---
            cached_mesh = None
            if not ptd_only:
                cached_mesh = self._resolve_mesh_cache(geo, freq, samples, use_degen, algo_id, use_gpu, progress_callback)

            # --- 4. 初始化求解器 ---
            # Only pass min_points to discrete_po algorithms
            if 'discrete_po' in algo_id:
                solver = get_integrator(algo_id, min_points=min_points)
            else:
                solver = get_integrator(algo_id)
            analyzer = RCSAnalyzer(solver)
            
            if progress_callback:
                progress_callback(0, 100, f"Using algorithm: {AVAILABLE_ALGORITHMS.get(algo_id, {}).get('name', algo_id)}")

            # --- 5. 执行计算 ---
            if is_2d:
                if progress_callback:
                    progress_callback(5, 100, f"Starting 2D scan: {n_theta}x{n_phi} angles...")
                    
                rcs_result_raw = analyzer.compute_monostatic_rcs_2d(
                    geo, freq, theta_rad, phi_rad,
                    samples_per_lambda=samples,
                    parallel=parallel, n_workers=n_workers,
                    show_progress=False, progress_callback=progress_callback,
                    enable_ptd=enable_ptd, ptd_edge_identifiers=ptd_edges,
                    cached_mesh_data=cached_mesh, polarization=ptd_pol,
                    gpu=use_gpu, use_degenerate_mesh=use_degen,
                    ptd_seg_angle_deg=ptd_seg_angle,
                    abort_event=abort_event, ptd_only=ptd_only,
                    ptd_algorithm=ptd_algorithm,
                    ptd_max_seg_lambda=ptd_max_seg_lambda,
                )

                # 结果标准化
                if isinstance(rcs_result_raw, dict):
                    rcs_total = rcs_result_raw['total']
                    rcs_po    = rcs_result_raw.get('po')
                    rcs_ptd   = rcs_result_raw.get('ptd')
                    I_total   = rcs_result_raw.get('total_c')
                    I_po      = rcs_result_raw.get('po_c')
                    I_ptd     = rcs_result_raw.get('ptd_c')
                else:
                    rcs_total = rcs_result_raw
                    rcs_po = rcs_ptd = I_total = I_po = I_ptd = None

                result_data = {
                    'mode': '2d',
                    'theta_deg': theta_deg, 'phi_deg': phi_deg,
                    'rcs_total': rcs_total, 'rcs_po': rcs_po, 'rcs_ptd': rcs_ptd,
                    'I_total': I_total, 'I_po': I_po, 'I_ptd': I_ptd,
                }
                
            elif is_phi_scan:
                # 1D phi scan: n_theta=1, n_phi>1 — scan along phi at fixed theta
                if progress_callback:
                    progress_callback(5, 100, f"Starting 1D phi scan: {n_phi} angles...")

                rcs_result_raw = analyzer.compute_monostatic_rcs_2d(
                    geo, freq, theta_rad, phi_rad,
                    samples_per_lambda=samples,
                    parallel=parallel, n_workers=n_workers,
                    show_progress=False, progress_callback=progress_callback,
                    enable_ptd=enable_ptd, ptd_edge_identifiers=ptd_edges,
                    cached_mesh_data=cached_mesh, polarization=ptd_pol,
                    gpu=use_gpu, use_degenerate_mesh=use_degen,
                    ptd_seg_angle_deg=ptd_seg_angle,
                    abort_event=abort_event, ptd_only=ptd_only,
                    ptd_algorithm=ptd_algorithm,
                    ptd_max_seg_lambda=ptd_max_seg_lambda,
                )

                if isinstance(rcs_result_raw, dict):
                    rcs_total = rcs_result_raw['total'][0]
                    rcs_po    = rcs_result_raw['po'][0]    if rcs_result_raw.get('po')    is not None else None
                    rcs_ptd   = rcs_result_raw['ptd'][0]   if rcs_result_raw.get('ptd')   is not None else None
                    I_total   = rcs_result_raw['total_c'][0] if rcs_result_raw.get('total_c') is not None else None
                    I_po      = rcs_result_raw['po_c'][0]    if rcs_result_raw.get('po_c')    is not None else None
                    I_ptd     = rcs_result_raw['ptd_c'][0]   if rcs_result_raw.get('ptd_c')   is not None else None
                else:
                    rcs_total = rcs_result_raw[0]
                    rcs_po = rcs_ptd = I_total = I_po = I_ptd = None

                result_data = {
                    'mode': '1d_phi',
                    'theta_deg': theta_deg[0],
                    'phi_deg': phi_deg,
                    'rcs_total': rcs_total, 'rcs_po': rcs_po, 'rcs_ptd': rcs_ptd,
                    'I_total': I_total, 'I_po': I_po, 'I_ptd': I_ptd,
                }

            else:
                # 1D Scan along theta
                if progress_callback:
                    progress_callback(5, 100, f"Starting 1D scan: {n_theta} angles...")

                rcs_result_raw = analyzer.compute_monostatic_rcs(
                    geo, {'frequency': freq, 'phi': phi_rad[0]}, theta_rad,
                    samples_per_lambda=samples,
                    parallel=parallel, n_workers=n_workers,
                    show_progress=False, progress_callback=progress_callback,
                    enable_ptd=enable_ptd, ptd_edge_identifiers=ptd_edges,
                    cached_mesh_data=cached_mesh, polarization=ptd_pol,
                    gpu=use_gpu, use_degenerate_mesh=use_degen,
                    ptd_seg_angle_deg=ptd_seg_angle,
                    abort_event=abort_event, ptd_only=ptd_only,
                    ptd_algorithm=ptd_algorithm,
                    ptd_max_seg_lambda=ptd_max_seg_lambda,
                )

                if isinstance(rcs_result_raw, dict):
                    rcs_total = rcs_result_raw['total']
                    rcs_po    = rcs_result_raw.get('po')
                    rcs_ptd   = rcs_result_raw.get('ptd')
                    I_total   = rcs_result_raw.get('total_c')
                    I_po      = rcs_result_raw.get('po_c')
                    I_ptd     = rcs_result_raw.get('ptd_c')
                else:
                    rcs_total = rcs_result_raw
                    rcs_po = rcs_ptd = I_total = I_po = I_ptd = None

                result_data = {
                    'mode': '1d',
                    'theta_deg': theta_deg,
                    'phi_deg': np.degrees(phi_rad[0]),
                    'rcs_total': rcs_total, 'rcs_po': rcs_po, 'rcs_ptd': rcs_ptd,
                    'I_total': I_total, 'I_po': I_po, 'I_ptd': I_ptd,
                }

            # --- 6. PTD Only 模式：叠加已有 PO 结果 ---
            if ptd_only and prev_result is not None:
                prev_I_po = prev_result.get('I_po')
                if prev_I_po is not None:
                    k_mag = 2 * np.pi * freq / C0
                    result_data['rcs_po'] = prev_result.get('rcs_po')
                    result_data['I_po'] = prev_I_po
                    I_ptd = result_data.get('I_ptd')
                    if I_ptd is not None:
                        I_combined = np.asarray(prev_I_po) + np.asarray(I_ptd)
                        result_data['I_total'] = I_combined
                        sigma = (k_mag**2 / np.pi) * np.abs(I_combined)**2
                        result_data['rcs_total'] = 10.0 * np.log10(
                            np.maximum(sigma, 1e-20))
                    if progress_callback:
                        progress_callback(100, 100, "PTD Only: 已与上次 PO 结果叠加")

            # --- 7. 结果打包 ---
            elapsed_time = time.time() - start_time
            result_data.update({
                'freq': freq,
                'elapsed_time': elapsed_time,
                'params': params,
                'timestamp': time.time(),
                'ptd_enabled': enable_ptd,
                'ptd_only': ptd_only,
            })
            
            return result_data

        except Exception as e:
            traceback.print_exc()
            raise e

    def _resolve_mesh_cache(self, geo, freq, samples, use_degen, algo_id, use_gpu, callback=None):
        """处理网格缓存和 GPU 合并逻辑"""
        # 只有离散类算法支持复用网格缓存
        if 'discrete_po' not in algo_id:
            return None
            
        check_params = {
            'freq': freq,
            'samples': samples,
            'use_degenerate_mesh': use_degen,
            'n_surfaces': len(geo) if isinstance(geo, list) else 1
        }
        
        # 简单检查：如果 self.cached_mesh_data 是由外部注入或通过某种方式保留的
        # 目前 SolverBridge 是每次 run_simulation 可能都是新的，除非它作为单例存在。
        # 建议：UI 层持有 SolverBridge 实例，从而保持缓存状态。
        
        cached_mesh = None
        if check_params == self.cached_mesh_params and self.cached_mesh_data is not None:
             cached_mesh = self.cached_mesh_data
             if callback: callback(0, 0, "Optimization: Using pre-calculated mesh.")
             
             # GPU 合并优化
             if use_gpu:
                 if isinstance(cached_mesh, list):
                     try:
                         if callback: callback(0, 0, "GPU Batching: Merging meshes...")
                         cached_mesh = merge_meshes(cached_mesh, to_gpu=True)
                     except Exception as e:
                         print(f"Merge failed: {e}")
        
        return cached_mesh

    def update_mesh_cache(self, mesh_data, params):
        """允许外部（如可视化模块）更新 Solver 的网格缓存"""
        self.cached_mesh_data = mesh_data
        self.cached_mesh_params = params

    def run_freq_sweep(self, geo, params, freq_sweep_params, progress_callback=None, abort_event=None):
        """
        频率扫描（相位旋转法）：固定角度扫频率，计算各频率的 PO/PTD 积分和距离像。

        Args:
            geo:               几何对象列表
            params:            solver 参数字典（与 run_simulation 相同结构）
            freq_sweep_params: 频扫专用参数 {'f_start','f_end','f_step'(MHz), 'window','zero_pad','polarization'}
            progress_callback: (current, total, msg) 回调
            abort_event:       threading.Event，设置后终止计算

        Returns:
            dict: 包含 freq_sweep 结果的字典
        """
        try:
            start_time = time.time()

            # --- 1. 解包频扫参数 ---
            f_start = freq_sweep_params['f_start'] * 1e6   # Hz
            f_end   = freq_sweep_params['f_end']   * 1e6
            f_step  = freq_sweep_params['f_step']  * 1e6
            window      = freq_sweep_params.get('window', 'hamming')
            zero_pad    = int(freq_sweep_params.get('zero_pad', 4))
            cheby_at    = float(freq_sweep_params.get('cheby_at', 40.0))
            taylor_nbar = int(freq_sweep_params.get('taylor_nbar', 4))
            taylor_sll  = float(freq_sweep_params.get('taylor_sll', 30.0))
            polarization = freq_sweep_params.get('polarization', 'VV')

            Nf = max(2, int(round((f_end - f_start) / f_step)) + 1)
            frequencies = np.linspace(f_start, f_end, Nf)
            f_max = frequencies[-1]

            # --- 2. 解包 solver 参数 ---
            algo_id    = params['algorithm']
            if 'discrete_po' not in algo_id:
                raise ValueError(f"频率扫描仅支持 discrete_po 算法，当前算法: {algo_id}")

            mesh_params    = params.get('mesh', {})
            compute_params = params.get('compute', {})
            ptd_params     = params.get('ptd', {})
            angle_params   = params.get('angles', {})

            samples    = mesh_params.get('density', 10.0)
            min_points = mesh_params.get('min_points', 18)
            use_degen  = mesh_params.get('use_degenerate', False)
            use_gpu    = compute_params.get('gpu', False)

            enable_ptd    = ptd_params.get('enabled', False)
            ptd_edges_str = ptd_params.get('edges', '')
            ptd_seg_angle = ptd_params.get('seg_angle_deg', 2.0)
            ptd_algorithm = ptd_params.get('algorithm', 'ufimtsev_eew')
            ptd_max_seg_lambda = ptd_params.get('max_seg_lambda', None)

            # sinc 模式从算法注册表读取
            sinc_mode = AVAILABLE_ALGORITHMS.get(algo_id, {}).get('kwargs', {}).get('sinc_mode', 'none')

            # --- 3. 建立网格（以 f_max 最高分辨率，尝试复用缓存）---
            wavelength_max = C0 / f_max
            surfaces = geo if isinstance(geo, list) else [geo]

            cached = self._resolve_mesh_cache(geo, f_max, samples, use_degen, algo_id, False)
            if cached is not None:
                mesh_list = cached if isinstance(cached, list) else [cached]
                if progress_callback:
                    progress_callback(0, 100, f"Using cached mesh at {f_max/1e6:.1f} MHz ({Nf} freqs)...")
            else:
                if progress_callback:
                    progress_callback(0, 100, f"Building mesh at {f_max/1e6:.1f} MHz ({Nf} freqs)...")
                solver = get_integrator(algo_id, min_points=min_points)
                mesh_list = []
                for surf in surfaces:
                    m = solver.precompute_mesh(surf, wavelength_max, samples, use_degenerate_mesh=use_degen)
                    mesh_list.append(m)
                # 写入缓存，供后续单频 run_simulation 或重复频扫复用
                check_params = {
                    'freq': f_max,
                    'samples': samples,
                    'use_degenerate_mesh': use_degen,
                    'n_surfaces': len(surfaces),
                }
                self.update_mesh_cache(mesh_list, check_params)

            # --- 4. 提取 PTD 边缘（若启用）---
            ptd_edges = []
            if enable_ptd and ptd_edges_str:
                from solvers.ptd import PTDProcessor
                try:
                    # 频扫: 用最高频对应 λ_min 算 max_seg_length, 保守上限
                    max_seg_length = None
                    if ptd_max_seg_lambda is not None and ptd_max_seg_lambda > 0:
                        max_seg_length = float(ptd_max_seg_lambda) * (C0 / f_max)
                    ptd_edges = PTDProcessor.extract_edges_from_face_pairs(
                        surfaces, ptd_edges_str, max_angle_deg=ptd_seg_angle,
                        max_seg_length=max_seg_length,
                    )
                    print(f"  [PTD] 已提取 {len(ptd_edges)} 条边缘 (频扫)")
                except Exception as e:
                    print(f"  [PTD] 边缘提取失败: {e}")

            # --- 5. 构建角度列表 ---
            theta_start = angle_params.get('theta_start', 0)
            theta_end   = angle_params.get('theta_end', 0)
            n_theta     = angle_params.get('n_theta', 1)
            phi_start   = angle_params.get('phi_start', 0)
            phi_end     = angle_params.get('phi_end', 0)
            n_phi       = angle_params.get('n_phi', 1)

            theta_deg = np.linspace(theta_start, theta_end, n_theta)
            phi_deg   = np.linspace(phi_start,   phi_end,   n_phi)

            angle_list = [(th, ph) for th in theta_deg for ph in phi_deg]
            N_angles = len(angle_list)

            # --- 6. 分配结果矩阵 ---
            N_pad = Nf * zero_pad
            I_po_matrix    = np.zeros((N_angles, Nf), dtype=np.complex128)
            I_ptd_matrix   = np.zeros((N_angles, Nf), dtype=np.complex128) if (enable_ptd and ptd_edges) else None
            I_total_matrix = np.zeros((N_angles, Nf), dtype=np.complex128)
            profile_matrix = np.zeros((N_angles, N_pad))
            range_axis = None
            stats = None

            # --- 7. 角度循环 ---
            for i, (th_deg, ph_deg) in enumerate(angle_list):
                if abort_event and abort_event.is_set():
                    from ui.workers import SimulationAborted
                    raise SimulationAborted("用户终止频扫")
                if progress_callback:
                    progress_callback(
                        int(i * 95 / N_angles), 100,
                        f"频扫角度 {i+1}/{N_angles}: θ={th_deg:.1f}°, φ={ph_deg:.1f}°"
                    )

                th_rad = np.radians(th_deg)
                ph_rad = np.radians(ph_deg)
                wave  = IncidentWave(f_max, th_rad, ph_rad)
                k_dir = wave.k_dir

                I_po = compute_po_freq_sweep(mesh_list, k_dir, frequencies, sinc_mode, use_gpu)
                I_po_matrix[i] = I_po

                if enable_ptd and ptd_edges:
                    I_ptd = compute_ptd_freq_sweep(
                        ptd_edges, k_dir, frequencies, polarization, use_gpu,
                        abort_event=abort_event, algorithm=ptd_algorithm)
                    I_ptd_matrix[i] = I_ptd
                    I_total = I_po + I_ptd
                else:
                    I_total = I_po

                I_total_matrix[i] = I_total

                prof_db, r_ax, _, stats_i = compute_range_profile(
                    I_total, frequencies, window, zero_pad, cheby_at,
                    taylor_nbar=taylor_nbar, taylor_sll=taylor_sll)
                profile_matrix[i] = prof_db
                if range_axis is None:
                    range_axis = r_ax
                    stats = stats_i

            if progress_callback:
                progress_callback(100, 100, "频扫完成")

            # --- 8. 计算 RCS 矩阵 ---
            k_arr = 2.0 * np.pi * frequencies / C0   # (Nf,)
            sigma_mat  = (k_arr ** 2 / np.pi) * np.abs(I_total_matrix) ** 2  # (N_angles, Nf)
            rcs_matrix = 10.0 * np.log10(np.maximum(sigma_mat, 1e-30))

            # --- 9. 判断扫描模式 ---
            scan_mode = '1d' if N_angles == 1 else '2d_angle_freq'

            if enable_ptd and ptd_edges:
                print(f"  [频扫] I_ptd 已计算，max|I_ptd|={np.max(np.abs(I_ptd_matrix)):.3e}")

            def _squeeze(arr):
                return arr.squeeze(axis=0) if (scan_mode == '1d' and arr is not None) else arr

            result = {
                'mode': 'freq_sweep',
                'frequencies': frequencies,
                'theta_deg': theta_deg,
                'phi_deg': phi_deg,
                'scan_mode': scan_mode,
                'I_po_matrix':    _squeeze(I_po_matrix),
                'I_ptd_matrix':   _squeeze(I_ptd_matrix) if I_ptd_matrix is not None else None,
                'I_total_matrix': _squeeze(I_total_matrix),
                'rcs_matrix':     _squeeze(rcs_matrix),
                'profile_matrix': _squeeze(profile_matrix),
                'range_axis':     range_axis,
                'stats':          stats,
                'elapsed_time':   time.time() - start_time,
                'params':         params,
                'freq_sweep_params': freq_sweep_params,
                'timestamp':      time.time(),
            }
            return result

        except Exception as e:
            traceback.print_exc()
            raise e

    def generate_mesh(self, geo, params):
        """
        生成网格数据用于预览，并更新缓存
        """
        try:
            freq = params['frequency']
            algo_id = params['algorithm']
            mesh_params = params.get('mesh', {})
            samples = mesh_params.get('density', 10.0)
            min_points = mesh_params.get('min_points', 18)
            use_degen = mesh_params.get('use_degenerate', False)

            # Check cache logic could be reused here but for simplicity we regenerate or overwrite
            # To ensure consistency, we just generate fresh and update cache.

            # Only pass min_points to discrete_po algorithms
            if 'discrete_po' in algo_id:
                solver = get_integrator(algo_id, min_points=min_points)
            else:
                solver = get_integrator(algo_id)
            if not hasattr(solver, 'precompute_mesh'):
                return None # Algorithm doesn't support discrete meshing

            wavelength = 299792458.0 / freq
            surfaces = geo if isinstance(geo, list) else [geo]
            meshes = []

            for surf in surfaces:
                m = solver.precompute_mesh(surf, wavelength, samples, use_degenerate_mesh=use_degen)
                meshes.append(m)
            
            # Update cache so subsequent run_simulation uses it
            check_params = {
                'freq': freq,
                'samples': samples,
                'use_degenerate_mesh': use_degen,
                'n_surfaces': len(surfaces)
            }
            self.update_mesh_cache(meshes, check_params)
            
            return meshes
            
        except Exception as e:
            traceback.print_exc()
            raise e

    def compute_surface_current(self, geo, params, theta_deg, phi_deg, polarization='V'):
        """
        计算指定入射角下每个 Surface 的 PO 表面电流分布。

        Args:
            geo: Surface 列表或单个 Surface
            params: 与 run_simulation 相同结构的参数字典（用到 frequency / algorithm / mesh）
            theta_deg, phi_deg: 入射方向（度）
            polarization: 'V' / 'H' / 'VV' / 'HH' / 'VH' / 'HV'（按首字母取入射极化）

        Returns:
            list[SurfaceCurrentField] —— 每个 Surface 一个，已在 CPU 端
        """
        from physics.wave import IncidentWave
        from physics.surface_current import compute_surface_current

        freq = params['frequency']
        algo_id = params['algorithm']
        mesh_params = params.get('mesh', {})
        samples = mesh_params.get('density', 10.0)
        min_points = mesh_params.get('min_points', 18)
        use_degen = mesh_params.get('use_degenerate', False)

        if 'discrete_po' not in algo_id:
            raise ValueError(f"表面电流可视化仅支持 discrete_po 类算法，当前: {algo_id}")

        # 取/生成网格（CPU 端，不合并、不上 GPU）
        cached = self._resolve_mesh_cache(geo, freq, samples, use_degen, algo_id, use_gpu=False)
        if cached is None:
            solver = get_integrator(algo_id, min_points=min_points)
            wavelength = C0 / freq
            surfaces = geo if isinstance(geo, list) else [geo]
            mesh_list = [solver.precompute_mesh(s, wavelength, samples, use_degenerate_mesh=use_degen)
                         for s in surfaces]
            check_params = {
                'freq': freq,
                'samples': samples,
                'use_degenerate_mesh': use_degen,
                'n_surfaces': len(surfaces),
            }
            self.update_mesh_cache(mesh_list, check_params)
        else:
            mesh_list = cached if isinstance(cached, list) else [cached]
            # 若缓存被推到 GPU，先拉回
            for m in mesh_list:
                if hasattr(m, 'to_cpu'):
                    m.to_cpu()

        wave = IncidentWave(freq, np.radians(theta_deg), np.radians(phi_deg))
        return [compute_surface_current(m, wave, polarization=polarization) for m in mesh_list]

    def generate_mesh_visualization(self, geo, params):
        """
        生成用于可视化的网格数据 (Structured Grid)
        Returns: list of dict {'points': (nu, nv, 3), 'dims': (nu, nv)}
        """
        try:
            freq = params['frequency']
            algo_id = params['algorithm']
            mesh_params = params.get('mesh', {})
            samples = mesh_params.get('density', 10.0)
            min_points = mesh_params.get('min_points', 18)

            # Only pass min_points to discrete_po algorithms
            if 'discrete_po' in algo_id:
                solver = get_integrator(algo_id, min_points=min_points)
            else:
                solver = get_integrator(algo_id)
            if not hasattr(solver, 'get_mesh_data'):
                return None
            
            # Mock Wave object
            class Wave:
                def __init__(self, f):
                    self.wavelength = 299792458.0 / f
            wave = Wave(freq)
            
            surfaces = geo if isinstance(geo, list) else [geo]
            vis_data = []
            
            for surf in surfaces:
                # get_mesh_data returns points, normals, (nu, nv)
                # points shape: (nu, nv, 3)
                points, normals, dims = solver.get_mesh_data(surf, wave, samples)
                vis_data.append({'points': points, 'dims': dims})
                
            return vis_data
            
        except Exception as e:
            traceback.print_exc()
            raise e
