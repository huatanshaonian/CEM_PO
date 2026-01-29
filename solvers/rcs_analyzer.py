import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from physics.constants import C0
from physics.wave import IncidentWave
from core.env import HAS_GPU
from core.mesh_data import MergedMeshData
from .ptd import PTDProcessor

class RCSAnalyzer:
    def __init__(self, solver):
        self.solver = solver

    def _make_wave(self, freq, theta, phi):
        return IncidentWave(freq, theta, phi)

    def _compute_single_angle_cached(self, args):
        """
        并行计算任务函数 (使用 CachedMeshData)
        """
        # 兼容性处理：支持 4, 5, 6 参数
        ptd_edges = None
        polarization = 'VV'
        
        if len(args) == 6:
            theta, cached_surfaces, wave_params, k_mag, ptd_edges, polarization = args
        elif len(args) == 5:
            theta, cached_surfaces, wave_params, k_mag, ptd_edges = args
        elif len(args) == 4:
            theta, cached_surfaces, wave_params, k_mag = args
        else:
            raise ValueError(f"Invalid args length: {len(args)}, expected 4, 5 or 6")
        
        wave = IncidentWave(wave_params['frequency'], theta, wave_params['phi'])

        total_I_po = 0j
        for mesh_data in cached_surfaces:
            total_I_po += self.solver.integrate_cached(mesh_data, wave)

        total_I_ptd = 0j
        if ptd_edges:
            for edge in ptd_edges:
                total_I_ptd += PTDProcessor.compute_contribution(edge, wave, polarization=polarization)

        total_I = total_I_po + total_I_ptd

        def to_rcs(val):
            sigma = (k_mag**2 / np.pi) * np.abs(val)**2
            return 10.0 * np.log10(max(sigma, 1e-20))

        return {
            'total': to_rcs(total_I),
            'po': to_rcs(total_I_po),
            'ptd': to_rcs(total_I_ptd)
        }

    def compute_monostatic_rcs(self, geometry, wave_params, angles,
                               samples_per_lambda=None,
                               parallel=False, n_workers=None,
                               show_progress=True,
                               progress_callback=None,
                               enable_ptd=False, ptd_edge_identifiers=None,
                               cached_mesh_data=None, polarization='VV',
                               gpu=False, use_degenerate_mesh=False):
        if gpu and not HAS_GPU:
            print("  [Warning] GPU 不可用 (未安装 cupy)，回退到 CPU 模式")
            gpu = False

        if isinstance(geometry, list):
            surfaces = geometry
        else:
            surfaces = [geometry]

        # 准备 PTD 边缘数据
        ptd_edges = []
        if enable_ptd and ptd_edge_identifiers:
            try:
                ptd_edges = PTDProcessor.extract_edges(surfaces, ptd_edge_identifiers)
                if show_progress:
                    print(f"  [PTD] 已启用，提取了 {len(ptd_edges)} 条手动边缘")
            except Exception as e:
                print(f"  [PTD] 边缘提取失败: {e}")

        k_mag = 2 * np.pi * wave_params['frequency'] / C0
        n_angles = len(angles)

        info_msg = (f"计算参数: {len(surfaces)} 个曲面, {n_angles} 个角度, "
                    f"f={wave_params['frequency']/1e9:.2f}GHz")
        if enable_ptd:
            info_msg += f" (PTD: {len(ptd_edges)} edges, {polarization})"
        if gpu:
            info_msg += " (GPU 加速模式)"

        if show_progress:
            print(info_msg)
        if progress_callback:
            progress_callback(0, n_angles, info_msg)

        # ---------------------------------------------------------------------
        # 优化: 几何预计算 (如果求解器支持)
        # ---------------------------------------------------------------------
        can_cache = hasattr(self.solver, 'precompute_mesh') and hasattr(self.solver, 'integrate_cached')
        geometry_data = surfaces
        is_cached = False

        try:
            if cached_mesh_data:
                # 外部已提供缓存
                if show_progress: print("  使用外部缓存的网格数据...")
                geometry_data = cached_mesh_data
                is_cached = True
            elif can_cache:
                # 尝试预计算
                wavelength = C0 / wave_params['frequency']
                if show_progress: print("  正在预计算几何网格 (加速模式)...")
                geometry_data = []
                n_surfs = len(surfaces)
                for i, s in enumerate(surfaces):
                    geometry_data.append(self.solver.precompute_mesh(
                        s, wavelength, samples_per_lambda, use_degenerate_mesh=use_degenerate_mesh))
                    if progress_callback and (i % 5 == 0 or i == n_surfs - 1):
                         progress_callback(0, n_angles, f"预计算几何: {i+1}/{n_surfs}")
                is_cached = True
        except Exception as e:
            print(f"  预计算失败: {e}, 回退到实时计算模式")
            can_cache = False
            geometry_data = surfaces
            is_cached = False

        # GPU 模式下的额外处理
        if gpu and is_cached:
            if show_progress: print("  正在将几何数据迁移到 GPU...")
            for mesh in geometry_data:
                mesh.to_gpu()

        if parallel and not gpu: # 并行与 GPU 互斥
            if not is_cached:
                if show_progress: print("  并行模式仅支持可缓存的求解器 (DiscretePO)，回退到串行模式。")
                # Fall through to serial logic
            else:
                return self._compute_parallel(
                    geometry_data, wave_params, angles, k_mag,
                    n_workers, show_progress, progress_callback, ptd_edges, polarization,
                    is_cached=is_cached
                )

        return self._compute_serial(
            geometry_data, wave_params, angles, samples_per_lambda,
            k_mag, show_progress, progress_callback, ptd_edges, polarization,
            is_cached=is_cached
        )

    def _compute_serial(self, geometry_data, wave_params, angles,
                        samples_per_lambda, k_mag, show_progress, progress_callback=None, ptd_edges=None, polarization='VV', is_cached=False):
        rcs_list = {'total': [], 'po': [], 'ptd': []}
        n_angles = len(angles)
        
        is_merged = isinstance(geometry_data, MergedMeshData)

        for i, theta in enumerate(angles):
            wave = self._make_wave(wave_params['frequency'], theta, wave_params['phi'])
            
            total_I_po = 0j
            
            if is_merged:
                total_I_po = self.solver.integrate_cached(geometry_data, wave)
            else:
                for obj in geometry_data:
                    if is_cached:
                        total_I_po += self.solver.integrate_cached(obj, wave)
                    else:
                        total_I_po += self.solver.integrate_surface(obj, wave, samples_per_lambda=samples_per_lambda)

            total_I_ptd = 0j
            if ptd_edges:
                for edge in ptd_edges:
                    total_I_ptd += PTDProcessor.compute_contribution(edge, wave, polarization=polarization)
            
            total_I = total_I_po + total_I_ptd

            def to_rcs(val):
                sigma = (k_mag**2 / np.pi) * np.abs(val)**2
                return 10.0 * np.log10(max(sigma, 1e-20))

            rcs_list['total'].append(to_rcs(total_I))
            rcs_list['po'].append(to_rcs(total_I_po))
            rcs_list['ptd'].append(to_rcs(total_I_ptd))

            if (i + 1) % max(1, n_angles // 20) == 0 or (i + 1) == n_angles:
                progress = (i + 1) / n_angles * 100
                msg = f"进度: {progress:.0f}% ({i+1}/{n_angles})"
                if show_progress: print(f"  {msg}")
                if progress_callback: progress_callback(i + 1, n_angles, msg)

        done_msg = "计算完成!"
        if show_progress: print(f"  {done_msg}")
        if progress_callback: progress_callback(n_angles, n_angles, done_msg)

        return {k: np.array(v) for k, v in rcs_list.items()}

    def _compute_parallel(self, cached_surfaces, wave_params, angles,
                          k_mag, n_workers, show_progress,
                          progress_callback=None, ptd_edges=None, polarization='VV', is_cached=True):
        
        if n_workers is None: n_workers = os.cpu_count() or 4

        parallel_msg = f"启用并行计算: {n_workers} 个进程 (Cached Mode)"
        if show_progress: print(f"  {parallel_msg}")
        if progress_callback: progress_callback(0, len(angles), parallel_msg)

        args_list = [
            (theta, cached_surfaces, wave_params, k_mag, ptd_edges, polarization)
            for theta in angles
        ]

        rcs_dict = {}
        n_angles = len(angles)

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(self._compute_single_angle_cached, args): i
                    for i, args in enumerate(args_list)
                }

                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    rcs_dict[idx] = future.result()
                    completed += 1

                    if completed % max(1, n_angles // 20) == 0 or completed == n_angles:
                        progress = completed / n_angles * 100
                        msg = f"进度: {progress:.0f}% ({completed}/{n_angles})"
                        if show_progress: print(f"  {msg}")
                        if progress_callback: progress_callback(completed, n_angles, msg)

            final_rcs = {'total': [], 'po': [], 'ptd': []}
            for i in range(n_angles):
                res = rcs_dict[i]
                final_rcs['total'].append(res['total'])
                final_rcs['po'].append(res['po'])
                final_rcs['ptd'].append(res['ptd'])

            done_msg = "并行计算完成!"
            if show_progress: print(f"  {done_msg}")
            if progress_callback: progress_callback(n_angles, n_angles, done_msg)
            
            return {k: np.array(v) for k, v in final_rcs.items()}

        except Exception as e:
            raise RuntimeError(f"并行计算致命错误: {e}")

    def compute_monostatic_rcs_2d(self, geometry, frequency, theta_array, phi_array,
                                   samples_per_lambda=None,
                                   parallel=False, n_workers=None,
                                   show_progress=True,
                                   progress_callback=None,
                                   enable_ptd=False, ptd_edge_identifiers=None,
                                   cached_mesh_data=None, polarization='VV',
                                   gpu=False, use_degenerate_mesh=False):
        # 2D 扫描逻辑 (此处略做简化，完整逻辑与 compute_monostatic_rcs 类似，但遍历 2D 数组)
        # 为节省篇幅，这里暂不展开所有代码，主要逻辑已在 compute_monostatic_rcs 中体现
        # 实际项目中应完整迁移 2D 逻辑，或将其与 1D 逻辑统一
        # 这里我先把原始文件中的 2D 逻辑复制过来
        
        if gpu and not HAS_GPU:
            print("  [Warning] GPU 不可用 (未安装 cupy)，回退到 CPU 模式")
            gpu = False

        if isinstance(geometry, list):
            surfaces = geometry
        else:
            surfaces = [geometry]

        ptd_edges = []
        if enable_ptd and ptd_edge_identifiers:
            try:
                ptd_edges = PTDProcessor.extract_edges(surfaces, ptd_edge_identifiers)
                if show_progress:
                    print(f"  [PTD] 2D扫描启用，提取了 {len(ptd_edges)} 条手动边缘")
            except Exception as e:
                print(f"  [PTD] 边缘提取失败: {e}")

        k_mag = 2 * np.pi * frequency / C0
        n_theta = len(theta_array)
        n_phi = len(phi_array)
        total_points = n_theta * n_phi

        info_msg = (f"2D扫描: {len(surfaces)} 个曲面, "
                    f"{n_theta}×{n_phi}={total_points} 个角度点, "
                    f"f={frequency/1e9:.2f}GHz")
        if enable_ptd: info_msg += f" (PTD: Enabled)"
        if gpu: info_msg += " (GPU 加速模式)"

        if show_progress: print(info_msg)
        if progress_callback: progress_callback(0, total_points, info_msg)

        can_cache = hasattr(self.solver, 'precompute_mesh') and hasattr(self.solver, 'integrate_cached')
        geometry_data = surfaces
        is_cached = False

        try:
            if cached_mesh_data:
                if show_progress: print("  使用外部缓存的网格数据...")
                geometry_data = cached_mesh_data
                is_cached = True
            elif can_cache:
                wavelength = C0 / frequency
                if show_progress: print("  正在预计算2D几何网格 (加速模式)...")
                geometry_data = []
                n_surfs = len(surfaces)
                for i, s in enumerate(surfaces):
                    geometry_data.append(self.solver.precompute_mesh(
                        s, wavelength, samples_per_lambda, use_degenerate_mesh=use_degenerate_mesh))
                    if progress_callback and (i % 5 == 0 or i == n_surfs - 1):
                         progress_callback(0, total_points, f"预计算几何: {i+1}/{n_surfs}")
                is_cached = True
        except Exception as e:
            print(f"  2D预计算失败: {e}, 回退到实时计算")
            can_cache = False
            geometry_data = surfaces
            is_cached = False

        if gpu and is_cached:
            if isinstance(geometry_data, MergedMeshData):
                if show_progress: print("  [GPU Batching] Merged mesh ready.")
            else:
                if show_progress: print("  正在将几何数据迁移到 GPU...")
                for mesh in geometry_data:
                    mesh.to_gpu()

        if parallel and not gpu:
            if not is_cached:
                if show_progress: print("  并行模式仅支持可缓存的求解器 (DiscretePO)，回退到串行模式。")
            else:
                return self._compute_parallel_2d(
                    geometry_data, frequency, theta_array, phi_array,
                    k_mag, n_workers, show_progress, progress_callback, ptd_edges, polarization,
                    is_cached=is_cached
                )

        rcs_2d = {
            'total': np.zeros((n_theta, n_phi)),
            'po': np.zeros((n_theta, n_phi)),
            'ptd': np.zeros((n_theta, n_phi))
        }
        
        is_merged = isinstance(geometry_data, MergedMeshData)

        computed = 0
        for i, theta in enumerate(theta_array):
            for j, phi in enumerate(phi_array):
                wave = self._make_wave(frequency, theta, phi)
                
                total_I_po = 0j
                
                if is_merged:
                    total_I_po = self.solver.integrate_cached(geometry_data, wave)
                else:
                    for obj in geometry_data:
                        if is_cached:
                            total_I_po += self.solver.integrate_cached(obj, wave)
                        else:
                            total_I_po += self.solver.integrate_surface(obj, wave, samples_per_lambda=samples_per_lambda)
                
                total_I_ptd = 0j
                if ptd_edges:
                    for edge in ptd_edges:
                        total_I_ptd += PTDProcessor.compute_contribution(edge, wave, polarization=polarization)

                total_I = total_I_po + total_I_ptd

                def to_rcs(val):
                    sigma = (k_mag**2 / np.pi) * np.abs(val)**2
                    return 10.0 * np.log10(max(sigma, 1e-20))

                rcs_2d['total'][i, j] = to_rcs(total_I)
                rcs_2d['po'][i, j] = to_rcs(total_I_po)
                rcs_2d['ptd'][i, j] = to_rcs(total_I_ptd)

                computed += 1
                if computed % max(1, total_points // 20) == 0 or computed == total_points:
                    progress = computed / total_points * 100
                    msg = f"进度: {progress:.0f}% ({computed}/{total_points})"
                    if show_progress: print(f"  {msg}")
                    if progress_callback: progress_callback(computed, total_points, msg)

        done_msg = "2D扫描完成!"
        if show_progress: print(f"  {done_msg}")
        if progress_callback: progress_callback(total_points, total_points, done_msg)
        return rcs_2d

    def _compute_parallel_2d(self, cached_surfaces, frequency, theta_array, phi_array,
                             k_mag, n_workers,
                             show_progress, progress_callback=None, ptd_edges=None, polarization='VV', is_cached=True):
        
        if n_workers is None: n_workers = os.cpu_count() or 4

        n_theta = len(theta_array)
        n_phi = len(phi_array)
        total_points = n_theta * n_phi

        parallel_msg = f"启用2D并行计算: {n_workers} 个进程 (Cached Mode)"
        if show_progress: print(f"  {parallel_msg}")
        if progress_callback: progress_callback(0, total_points, parallel_msg)

        args_list = []
        for i, theta in enumerate(theta_array):
            for j, phi in enumerate(phi_array):
                wave_params = {'frequency': frequency, 'phi': phi}
                args = (theta, cached_surfaces, wave_params, k_mag, ptd_edges, polarization)
                args_list.append(((i, j), args))

        rcs_2d = {
            'total': np.zeros((n_theta, n_phi)),
            'po': np.zeros((n_theta, n_phi)),
            'ptd': np.zeros((n_theta, n_phi))
        }

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(self._compute_single_angle_cached, args): idx_tuple
                    for idx_tuple, args in args_list
                }

                computed = 0
                for future in as_completed(future_to_idx):
                    i, j = future_to_idx[future]
                    res = future.result() # dict
                    rcs_2d['total'][i, j] = res['total']
                    rcs_2d['po'][i, j] = res['po']
                    rcs_2d['ptd'][i, j] = res['ptd']
                    
                    computed += 1

                    if computed % max(1, total_points // 20) == 0 or computed == total_points:
                        progress = computed / total_points * 100
                        msg = f"进度: {progress:.0f}% ({computed}/{total_points})"
                        if show_progress: print(f"  {msg}")
                        if progress_callback: progress_callback(computed, total_points, msg)

            done_msg = "2D并行计算完成!"
            if show_progress: print(f"  {done_msg}")
            if progress_callback: progress_callback(total_points, total_points, done_msg)
            return rcs_2d

        except Exception as e:
            raise RuntimeError(f"2D并行计算致命错误: {e}")
