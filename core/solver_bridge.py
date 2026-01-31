import time
import numpy as np
import traceback
from solvers.api import get_integrator, AVAILABLE_ALGORITHMS
from solvers.rcs_analyzer import RCSAnalyzer
from core.mesh_data import merge_meshes

class SolverBridge:
    """
    桥接 UI 和 求解器核心的中间层。
    负责解析参数、管理网格缓存、调用求解器并返回标准化的结果。
    """
    def __init__(self):
        self.cached_mesh_data = None
        self.cached_mesh_params = {}

    def run_simulation(self, geo, params, progress_callback=None):
        """
        运行仿真计算 (同步阻塞调用，建议在独立线程中运行)
        
        Args:
            geo: 几何对象 (Surface 列表或单个 Surface)
            params: 参数字典，包含所有计算配置
            progress_callback: 接受 (current, total, message) 的回调函数
            
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
            ptd_edges = ptd_params.get('edges', [])
            ptd_pol = ptd_params.get('polarization', 'VV')

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
            
            is_2d = n_phi > 1

            # --- 3. 网格缓存处理 ---
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
                    gpu=use_gpu, use_degenerate_mesh=use_degen
                )
                
                # 结果标准化
                if isinstance(rcs_result_raw, dict):
                    rcs_total = rcs_result_raw['total']
                    rcs_po = rcs_result_raw.get('po')
                    rcs_ptd = rcs_result_raw.get('ptd')
                else:
                    rcs_total = rcs_result_raw
                    rcs_po, rcs_ptd = None, None
                    
                result_data = {
                    'mode': '2d',
                    'theta_deg': theta_deg, 'phi_deg': phi_deg,
                    'rcs_total': rcs_total, 'rcs_po': rcs_po, 'rcs_ptd': rcs_ptd
                }
                
            else:
                # 1D Scan
                if progress_callback:
                    progress_callback(5, 100, f"Starting 1D scan: {n_theta} angles...")
                
                # Fix: compute_monostatic_rcs signature uses dict for wave_params or separate args?
                # Looking at gui.py: compute_monostatic_rcs(geo, {'frequency': freq, 'phi': phi_rad}, ...)
                
                rcs_result_raw = analyzer.compute_monostatic_rcs(
                    geo, {'frequency': freq, 'phi': phi_rad[0]}, theta_rad,
                    samples_per_lambda=samples,
                    parallel=parallel, n_workers=n_workers,
                    show_progress=False, progress_callback=progress_callback,
                    enable_ptd=enable_ptd, ptd_edge_identifiers=ptd_edges,
                    cached_mesh_data=cached_mesh, polarization=ptd_pol,
                    gpu=use_gpu, use_degenerate_mesh=use_degen
                )
                
                if isinstance(rcs_result_raw, dict):
                    rcs_total = rcs_result_raw['total']
                    rcs_po = rcs_result_raw.get('po')
                    rcs_ptd = rcs_result_raw.get('ptd')
                else:
                    rcs_total = rcs_result_raw
                    rcs_po, rcs_ptd = None, None

                result_data = {
                    'mode': '1d',
                    'theta_deg': theta_deg,
                    'phi_deg': np.degrees(phi_rad[0]),
                    'rcs_total': rcs_total, 'rcs_po': rcs_po, 'rcs_ptd': rcs_ptd
                }

            # --- 6. 结果打包 ---
            elapsed_time = time.time() - start_time
            result_data.update({
                'freq': freq,
                'elapsed_time': elapsed_time,
                'params': params, # 回传输入参数以便记录
                'timestamp': time.time()
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
