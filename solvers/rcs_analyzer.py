import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import logging

from physics.constants import C0
from physics.wave import IncidentWave
from core.env import HAS_GPU
from core.mesh_data import MergedMeshData
from .ptd import PTDProcessor

logger = logging.getLogger("CEM-PO.Analyzer")


def _ptd_angle_task(args):
    """模块级 PTD 单角度计算任务，供 ProcessPoolExecutor 使用（返回复振幅）。"""
    theta, ptd_edges, frequency, phi, polarization = args
    wave = IncidentWave(frequency, theta, phi)
    total = 0j
    for edge in ptd_edges:
        total += PTDProcessor.compute_contribution(edge, wave, polarization)
    return total


def to_rcs_db(val, k_mag):
    """Convert a complex scatter integral to monostatic RCS in dBsm."""
    sigma = (k_mag**2 / np.pi) * np.abs(val)**2
    return 10.0 * np.log10(max(sigma, 1e-20))


class RCSAnalyzer:
    def __init__(self, solver):
        self.solver = solver

    def _make_wave(self, freq, theta, phi):
        return IncidentWave(freq, theta, phi)

    def _prepare_geometry(self, geometry, frequency, samples_per_lambda,
                          enable_ptd, ptd_edge_identifiers,
                          cached_mesh_data, gpu, use_degenerate_mesh,
                          show_progress, progress_callback, total_points,
                          ptd_seg_angle_deg=2.0, ptd_only=False):
        """共同前置逻辑：GPU检查、曲面归一化、PTD提取、网格预计算、GPU迁移。

        Returns (geometry_data, is_cached, ptd_edges, k_mag, gpu)
        """
        if gpu and not HAS_GPU:
            print("  [Warning] GPU 不可用 (未安装 cupy)，回退到 CPU 模式")
            gpu = False

        surfaces = geometry if isinstance(geometry, list) else [geometry]

        ptd_edges = []
        if enable_ptd and ptd_edge_identifiers:
            try:
                ptd_edges = PTDProcessor.extract_edges_from_face_pairs(
                    surfaces, ptd_edge_identifiers,
                    max_angle_deg=ptd_seg_angle_deg
                )
                if show_progress:
                    print(f"  [PTD] 已提取 {len(ptd_edges)} 条边缘")
            except Exception as e:
                print(f"  [PTD] 边缘提取失败: {e}")

        k_mag = 2 * np.pi * frequency / C0

        # ptd_only 模式：跳过网格预计算和 GPU 迁移，只需要 PTD 边和 k_mag
        if ptd_only:
            return surfaces, False, ptd_edges, k_mag, False

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
                if show_progress: print("  正在预计算几何网格 (加速模式)...")
                geometry_data = []
                n_surfs = len(surfaces)
                for i, s in enumerate(surfaces):
                    geometry_data.append(self.solver.precompute_mesh(
                        s, wavelength, samples_per_lambda, use_degenerate_mesh=use_degenerate_mesh))
                    if progress_callback and (i % 5 == 0 or i == n_surfs - 1):
                        progress_callback(0, total_points, f"预计算几何: {i+1}/{n_surfs}")
                is_cached = True
        except Exception as e:
            print(f"  预计算失败: {e}, 回退到实时计算模式")
            geometry_data = surfaces
            is_cached = False

        if gpu and is_cached:
            if isinstance(geometry_data, MergedMeshData):
                if show_progress: print("  [GPU Batching] Merged mesh ready.")
            else:
                if show_progress: print("  正在将几何数据迁移到 GPU...")
                for mesh in geometry_data:
                    mesh.to_gpu()

        return geometry_data, is_cached, ptd_edges, k_mag, gpu

    def _compute_single_angle_cached(self, args):
        """
        并行计算任务函数 (使用 CachedMeshData)
        args: (theta, cached_surfaces, wave_params, k_mag[, ptd_edges[, polarization[, ptd_only]]])
        """
        ptd_edges = None
        polarization = 'VV'
        ptd_only = False

        if len(args) == 7:
            theta, cached_surfaces, wave_params, k_mag, ptd_edges, polarization, ptd_only = args
        elif len(args) == 6:
            theta, cached_surfaces, wave_params, k_mag, ptd_edges, polarization = args
        elif len(args) == 5:
            theta, cached_surfaces, wave_params, k_mag, ptd_edges = args
        elif len(args) == 4:
            theta, cached_surfaces, wave_params, k_mag = args
        else:
            raise ValueError(f"Invalid args length: {len(args)}, expected 4-7")

        wave = IncidentWave(wave_params['frequency'], theta, wave_params['phi'])

        total_I_po = 0j
        if not ptd_only:
            for mesh_data in cached_surfaces:
                total_I_po += self.solver.integrate_cached(mesh_data, wave)

        total_I_ptd = 0j
        if ptd_edges:
            for edge in ptd_edges:
                total_I_ptd += PTDProcessor.compute_contribution(edge, wave, polarization=polarization)

        total_I = total_I_po + total_I_ptd

        return {
            'total': to_rcs_db(total_I, k_mag),
            'po':    to_rcs_db(total_I_po, k_mag),
            'ptd':   to_rcs_db(total_I_ptd, k_mag),
            'total_c': complex(total_I),
            'po_c':    complex(total_I_po),
            'ptd_c':   complex(total_I_ptd),
        }

    def compute_monostatic_rcs(self, geometry, wave_params, angles,
                               samples_per_lambda=None,
                               parallel=False, n_workers=None,
                               show_progress=True,
                               progress_callback=None,
                               enable_ptd=False, ptd_edge_identifiers=None,
                               cached_mesh_data=None, polarization='VV',
                               gpu=False, use_degenerate_mesh=False,
                               ptd_seg_angle_deg=2.0,
                               abort_event=None, ptd_only=False):

        frequency = wave_params['frequency']
        n_angles = len(angles)

        geometry_data, is_cached, ptd_edges, k_mag, gpu = self._prepare_geometry(
            geometry, frequency, samples_per_lambda,
            enable_ptd, ptd_edge_identifiers,
            cached_mesh_data, gpu, use_degenerate_mesh,
            show_progress, progress_callback, n_angles,
            ptd_seg_angle_deg=ptd_seg_angle_deg, ptd_only=ptd_only
        )

        info_msg = (f"计算参数: {len(geometry_data) if isinstance(geometry_data, list) else 1} 个曲面, "
                    f"{n_angles} 个角度, f={frequency/1e9:.2f}GHz")
        if enable_ptd:
            info_msg += f" (PTD: {len(ptd_edges)} edges, {polarization})"
        if gpu:
            info_msg += " (GPU 加速模式)"

        if show_progress: print(info_msg)
        if progress_callback: progress_callback(0, n_angles, info_msg)

        if parallel and not gpu:
            if not is_cached and not ptd_only:
                if show_progress: print("  并行模式仅支持可缓存的求解器 (DiscretePO)，回退到串行模式。")
            else:
                return self._compute_parallel(
                    geometry_data, wave_params, angles, k_mag,
                    n_workers, show_progress, progress_callback, ptd_edges, polarization,
                    is_cached=is_cached, abort_event=abort_event, ptd_only=ptd_only
                )

        # GPU 模式下预计算所有角度的 PTD（与 GPU PO 解耦）
        # parallel 勾选 → 多进程；未勾选 → 单线程串行预算
        ptd_workers = n_workers if (gpu and parallel and ptd_edges and not ptd_only) else None
        ptd_precompute = gpu and ptd_edges and not ptd_only

        return self._compute_serial(
            geometry_data, wave_params, angles, samples_per_lambda,
            k_mag, show_progress, progress_callback, ptd_edges, polarization,
            is_cached=is_cached, ptd_workers=ptd_workers,
            ptd_precompute=ptd_precompute, abort_event=abort_event,
            ptd_only=ptd_only
        )

    def _compute_serial(self, geometry_data, wave_params, angles,
                        samples_per_lambda, k_mag, show_progress, progress_callback=None,
                        ptd_edges=None, polarization='VV', is_cached=False, ptd_workers=None,
                        ptd_precompute=False, abort_event=None, ptd_only=False):
        rcs_list = {'total': [], 'po': [], 'ptd': [],
                    'total_c': [], 'po_c': [], 'ptd_c': []}
        n_angles = len(angles)

        is_merged = isinstance(geometry_data, MergedMeshData)

        # PTD 预计算：在 GPU PO 之前把所有角度的 PTD 算完
        ptd_precomputed = None
        if ptd_edges and ptd_precompute:
            freq = wave_params['frequency']
            phi  = wave_params['phi']
            args_list = [
                (theta, ptd_edges, freq, phi, polarization)
                for theta in angles
            ]
            if ptd_workers and ptd_workers > 1:
                msg = f"PTD 并行预计算: {ptd_workers} 个进程, {n_angles} 个角度..."
                if show_progress: logger.info(msg)
                if progress_callback: progress_callback(0, n_angles, msg)
                ptd_precomputed = [None] * n_angles
                with ProcessPoolExecutor(max_workers=ptd_workers) as ex:
                    future_to_idx = {ex.submit(_ptd_angle_task, a): i
                                     for i, a in enumerate(args_list)}
                    for f in as_completed(future_to_idx):
                        if abort_event and abort_event.is_set():
                            ex.shutdown(wait=False, cancel_futures=True)
                            from ui.workers import SimulationAborted
                            raise SimulationAborted("用户终止仿真")
                        ptd_precomputed[future_to_idx[f]] = f.result()
            else:
                msg = f"PTD 串行预计算: {n_angles} 个角度..."
                if show_progress: logger.info(msg)
                if progress_callback: progress_callback(0, n_angles, msg)
                ptd_precomputed = []
                for a in args_list:
                    if abort_event and abort_event.is_set():
                        from ui.workers import SimulationAborted
                        raise SimulationAborted("用户终止仿真")
                    ptd_precomputed.append(_ptd_angle_task(a))

        for i, theta in enumerate(angles):
            if abort_event and abort_event.is_set():
                from ui.workers import SimulationAborted
                raise SimulationAborted("用户终止仿真")

            wave = self._make_wave(wave_params['frequency'], theta, wave_params['phi'])

            total_I_po = 0j
            if not ptd_only:
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
                if ptd_precomputed is not None:
                    total_I_ptd = ptd_precomputed[i]
                else:
                    for edge in ptd_edges:
                        total_I_ptd += PTDProcessor.compute_contribution(edge, wave, polarization=polarization)

            total_I = total_I_po + total_I_ptd

            rcs_list['total'].append(to_rcs_db(total_I, k_mag))
            rcs_list['po'].append(to_rcs_db(total_I_po, k_mag))
            rcs_list['ptd'].append(to_rcs_db(total_I_ptd, k_mag))
            rcs_list['total_c'].append(complex(total_I))
            rcs_list['po_c'].append(complex(total_I_po))
            rcs_list['ptd_c'].append(complex(total_I_ptd))

            if (i + 1) % max(1, n_angles // 20) == 0 or (i + 1) == n_angles:
                progress = (i + 1) / n_angles * 100
                msg = f"进度: {progress:.0f}% ({i+1}/{n_angles})"
                if show_progress: logger.info(msg)
                if progress_callback: progress_callback(i + 1, n_angles, msg)

        done_msg = "计算完成!"
        if show_progress: logger.info(done_msg)
        if progress_callback: progress_callback(n_angles, n_angles, done_msg)

        return {k: np.array(v) for k, v in rcs_list.items()}

    def _compute_parallel(self, cached_surfaces, wave_params, angles,
                          k_mag, n_workers, show_progress,
                          progress_callback=None, ptd_edges=None, polarization='VV',
                          is_cached=True, abort_event=None, ptd_only=False):

        if n_workers is None: n_workers = os.cpu_count() or 4

        parallel_msg = f"启用并行计算: {n_workers} 个进程 (Cached Mode)"
        if show_progress: logger.info(parallel_msg)
        if progress_callback: progress_callback(0, len(angles), parallel_msg)

        args_list = [
            (theta, cached_surfaces, wave_params, k_mag, ptd_edges, polarization, ptd_only)
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
                    if abort_event and abort_event.is_set():
                        # 取消所有未完成的 future，立即退出
                        for f in future_to_idx:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        from ui.workers import SimulationAborted
                        raise SimulationAborted("用户终止仿真")

                    idx = future_to_idx[future]
                    rcs_dict[idx] = future.result()
                    completed += 1

                    if completed % max(1, n_angles // 20) == 0 or completed == n_angles:
                        progress = completed / n_angles * 100
                        msg = f"进度: {progress:.0f}% ({completed}/{n_angles})"
                        if show_progress: logger.info(msg)
                        if progress_callback: progress_callback(completed, n_angles, msg)

            final_rcs = {'total': [], 'po': [], 'ptd': [],
                         'total_c': [], 'po_c': [], 'ptd_c': []}
            for i in range(n_angles):
                res = rcs_dict[i]
                final_rcs['total'].append(res['total'])
                final_rcs['po'].append(res['po'])
                final_rcs['ptd'].append(res['ptd'])
                final_rcs['total_c'].append(res['total_c'])
                final_rcs['po_c'].append(res['po_c'])
                final_rcs['ptd_c'].append(res['ptd_c'])

            done_msg = "并行计算完成!"
            if show_progress: logger.info(done_msg)
            if progress_callback: progress_callback(n_angles, n_angles, done_msg)

            return {k: np.array(v) for k, v in final_rcs.items()}

        except Exception as e:
            if 'SimulationAborted' in type(e).__name__:
                raise
            raise RuntimeError(f"并行计算致命错误: {e}")

    def compute_monostatic_rcs_2d(self, geometry, frequency, theta_array, phi_array,
                                   samples_per_lambda=None,
                                   parallel=False, n_workers=None,
                                   show_progress=True,
                                   progress_callback=None,
                                   enable_ptd=False, ptd_edge_identifiers=None,
                                   cached_mesh_data=None, polarization='VV',
                                   gpu=False, use_degenerate_mesh=False,
                                   ptd_seg_angle_deg=2.0,
                                   abort_event=None, ptd_only=False):

        n_theta = len(theta_array)
        n_phi = len(phi_array)
        total_points = n_theta * n_phi

        geometry_data, is_cached, ptd_edges, k_mag, gpu = self._prepare_geometry(
            geometry, frequency, samples_per_lambda,
            enable_ptd, ptd_edge_identifiers,
            cached_mesh_data, gpu, use_degenerate_mesh,
            show_progress, progress_callback, total_points,
            ptd_seg_angle_deg=ptd_seg_angle_deg, ptd_only=ptd_only
        )

        info_msg = (f"2D扫描: {len(geometry_data) if isinstance(geometry_data, list) else 1} 个曲面, "
                    f"{n_theta}×{n_phi}={total_points} 个角度点, "
                    f"f={frequency/1e9:.2f}GHz")
        if enable_ptd: info_msg += f" (PTD: Enabled)"
        if gpu: info_msg += " (GPU 加速模式)"

        if show_progress: print(info_msg)
        if progress_callback: progress_callback(0, total_points, info_msg)

        if parallel and not gpu:
            if not is_cached and not ptd_only:
                if show_progress: print("  并行模式仅支持可缓存的求解器 (DiscretePO)，回退到串行模式。")
            else:
                return self._compute_parallel_2d(
                    geometry_data, frequency, theta_array, phi_array,
                    k_mag, n_workers, show_progress, progress_callback, ptd_edges, polarization,
                    is_cached=is_cached, abort_event=abort_event, ptd_only=ptd_only
                )

        # ── PTD 预计算（GPU 模式下先用 CPU 算完所有角度的 PTD） ──
        ptd_precomputed = None
        if gpu and ptd_edges and not ptd_only:
            angle_pairs = [(theta, phi)
                           for theta in theta_array for phi in phi_array]
            args_list = [
                (theta, ptd_edges, frequency, phi, polarization)
                for theta, phi in angle_pairs
            ]
            if parallel and n_workers and n_workers > 1:
                msg = f"PTD 并行预计算: {n_workers} 个进程, {total_points} 个角度..."
                if show_progress: logger.info(msg)
                if progress_callback: progress_callback(0, total_points, msg)
                ptd_precomputed = [None] * total_points
                with ProcessPoolExecutor(max_workers=n_workers) as ex:
                    future_to_idx = {ex.submit(_ptd_angle_task, a): i
                                     for i, a in enumerate(args_list)}
                    for f in as_completed(future_to_idx):
                        if abort_event and abort_event.is_set():
                            ex.shutdown(wait=False, cancel_futures=True)
                            from ui.workers import SimulationAborted
                            raise SimulationAborted("用户终止仿真")
                        ptd_precomputed[future_to_idx[f]] = f.result()
            else:
                msg = f"PTD 串行预计算: {total_points} 个角度..."
                if show_progress: logger.info(msg)
                if progress_callback: progress_callback(0, total_points, msg)
                ptd_precomputed = []
                for a in args_list:
                    if abort_event and abort_event.is_set():
                        from ui.workers import SimulationAborted
                        raise SimulationAborted("用户终止仿真")
                    ptd_precomputed.append(_ptd_angle_task(a))

        # ── 主循环（PO 在 GPU 上，PTD 已预算或在循环内串行计算） ──
        rcs_2d = {
            'total': np.zeros((n_theta, n_phi)),
            'po':    np.zeros((n_theta, n_phi)),
            'ptd':   np.zeros((n_theta, n_phi)),
            'total_c': np.zeros((n_theta, n_phi), dtype=complex),
            'po_c':    np.zeros((n_theta, n_phi), dtype=complex),
            'ptd_c':   np.zeros((n_theta, n_phi), dtype=complex),
        }

        is_merged = isinstance(geometry_data, MergedMeshData)
        computed = 0

        for i, theta in enumerate(theta_array):
            if abort_event and abort_event.is_set():
                from ui.workers import SimulationAborted
                raise SimulationAborted("用户终止仿真")
            for j, phi in enumerate(phi_array):
                wave = self._make_wave(frequency, theta, phi)

                total_I_po = 0j
                if not ptd_only:
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
                    if ptd_precomputed is not None:
                        total_I_ptd = ptd_precomputed[i * n_phi + j]
                    else:
                        for edge in ptd_edges:
                            total_I_ptd += PTDProcessor.compute_contribution(edge, wave, polarization=polarization)

                total_I = total_I_po + total_I_ptd

                rcs_2d['total'][i, j] = to_rcs_db(total_I, k_mag)
                rcs_2d['po'][i, j]    = to_rcs_db(total_I_po, k_mag)
                rcs_2d['ptd'][i, j]   = to_rcs_db(total_I_ptd, k_mag)
                rcs_2d['total_c'][i, j] = complex(total_I)
                rcs_2d['po_c'][i, j]    = complex(total_I_po)
                rcs_2d['ptd_c'][i, j]   = complex(total_I_ptd)

                computed += 1
                if computed % max(1, total_points // 20) == 0 or computed == total_points:
                    progress = computed / total_points * 100
                    msg = f"进度: {progress:.0f}% ({computed}/{total_points})"
                    if show_progress: logger.info(msg)
                    if progress_callback: progress_callback(computed, total_points, msg)

        done_msg = "2D扫描完成!"
        if show_progress: logger.info(done_msg)
        if progress_callback: progress_callback(total_points, total_points, done_msg)
        return rcs_2d

    def _compute_parallel_2d(self, cached_surfaces, frequency, theta_array, phi_array,
                             k_mag, n_workers,
                             show_progress, progress_callback=None, ptd_edges=None,
                             polarization='VV', is_cached=True, abort_event=None,
                             ptd_only=False):

        if n_workers is None: n_workers = os.cpu_count() or 4

        n_theta = len(theta_array)
        n_phi = len(phi_array)
        total_points = n_theta * n_phi

        parallel_msg = f"启用2D并行计算: {n_workers} 个进程 (Cached Mode)"
        if show_progress: logger.info(parallel_msg)
        if progress_callback: progress_callback(0, total_points, parallel_msg)

        args_list = []
        for i, theta in enumerate(theta_array):
            for j, phi in enumerate(phi_array):
                wave_params = {'frequency': frequency, 'phi': phi}
                args = (theta, cached_surfaces, wave_params, k_mag, ptd_edges, polarization, ptd_only)
                args_list.append(((i, j), args))

        rcs_2d = {
            'total': np.zeros((n_theta, n_phi)),
            'po':    np.zeros((n_theta, n_phi)),
            'ptd':   np.zeros((n_theta, n_phi)),
            'total_c': np.zeros((n_theta, n_phi), dtype=complex),
            'po_c':    np.zeros((n_theta, n_phi), dtype=complex),
            'ptd_c':   np.zeros((n_theta, n_phi), dtype=complex),
        }

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(self._compute_single_angle_cached, args): idx_tuple
                    for idx_tuple, args in args_list
                }

                computed = 0
                for future in as_completed(future_to_idx):
                    if abort_event and abort_event.is_set():
                        for f in future_to_idx:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        from ui.workers import SimulationAborted
                        raise SimulationAborted("用户终止仿真")

                    i, j = future_to_idx[future]
                    res = future.result()
                    rcs_2d['total'][i, j] = res['total']
                    rcs_2d['po'][i, j]    = res['po']
                    rcs_2d['ptd'][i, j]   = res['ptd']
                    rcs_2d['total_c'][i, j] = res['total_c']
                    rcs_2d['po_c'][i, j]    = res['po_c']
                    rcs_2d['ptd_c'][i, j]   = res['ptd_c']

                    computed += 1
                    if computed % max(1, total_points // 20) == 0 or computed == total_points:
                        progress = computed / total_points * 100
                        msg = f"进度: {progress:.0f}% ({computed}/{total_points})"
                        if show_progress: logger.info(msg)
                        if progress_callback: progress_callback(computed, total_points, msg)

            done_msg = "2D并行计算完成!"
            if show_progress: logger.info(done_msg)
            if progress_callback: progress_callback(total_points, total_points, done_msg)
            return rcs_2d

        except Exception as e:
            if 'SimulationAborted' in type(e).__name__:
                raise
            raise RuntimeError(f"2D并行计算致命错误: {e}")
