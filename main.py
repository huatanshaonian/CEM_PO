import argparse
import json
import os
import time
import sys
import logging
import numpy as np
import pandas as pd

# 确保在导入 pyplot 之前设置后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.solver_bridge import SolverBridge
from geometry.factory import GeometryFactory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("CEM-PO-CLI")

def parse_args():
    parser = argparse.ArgumentParser(description="CEM PO Solver - Batch Execution CLI")
    parser.add_argument("config_file", help="Path to the JSON configuration file for batch tasks.")
    parser.add_argument("--gpu", action="store_true", help="Force enable GPU acceleration (overrides config).")
    parser.add_argument("--no-gpu", action="store_true", help="Force disable GPU acceleration.")
    parser.add_argument("--workers", type=int, help="Override number of CPU workers.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running simulations.")
    return parser.parse_args()

def load_config(path):
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        sys.exit(1)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse JSON config: {e}")
        sys.exit(1)

def ensure_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created output directory: {path}")

def build_sim_params(task_cfg, global_overrides):
    """
    Construct standardized simulation parameter dictionary for SolverBridge.
    """
    solver_cfg = task_cfg.get('solver', {})
    scan_cfg = task_cfg.get('scan', {})
    ptd_cfg = solver_cfg.get('ptd', {})
    
    # 1. Frequency
    freq_mhz = solver_cfg.get('frequency_mhz', 1000.0)
    
    # 2. Angles
    theta_scan = scan_cfg.get('theta', [-90, 90, 181]) # [start, end, n]
    phi_scan = scan_cfg.get('phi', [0, 0, 1])
    
    # 3. Compute Settings (Global Overrides)
    use_gpu = solver_cfg.get('use_gpu', False)
    if global_overrides.get('force_gpu') is not None:
        use_gpu = global_overrides['force_gpu']
        
    n_workers = solver_cfg.get('workers', 4)
    if global_overrides.get('workers') is not None:
        n_workers = global_overrides['workers']

    # 4. Polarization (Can be at solver level or ptd level)
    polarization = solver_cfg.get('polarization', ptd_cfg.get('polarization', 'VV'))

    # 5. Construct Dict
    params = {
        'frequency': freq_mhz * 1e6, # Convert to Hz
        'algorithm': solver_cfg.get('algorithm', 'discrete_po_sinc_dual'),
        'polarization': polarization,
        'angles': {
            'theta_start': theta_scan[0],
            'theta_end': theta_scan[1],
            'n_theta': int(theta_scan[2]),
            'phi_start': phi_scan[0],
            'phi_end': phi_scan[1],
            'n_phi': int(phi_scan[2])
        },
        'mesh': {
            'density': solver_cfg.get('mesh_density', 10.0),
            'min_points': solver_cfg.get('min_points', 18),
            'use_degenerate': solver_cfg.get('use_degenerate', True)
        },
        'ptd': {
            'enabled': ptd_cfg.get('enabled', False),
            'edges': ptd_cfg.get('edges', []),
            'polarization': polarization # Sync with solver level
        },
        'compute': {
            'gpu': use_gpu,
            'parallel': not use_gpu, # Typically disable parallel if GPU is on
            'workers': n_workers
        }
    }
    return params

def save_plot(result, output_path, title_suffix=""):
    """
    Generate and save a plot for the result using matplotlib (Agg backend).
    """
    try:
        mode = result.get('mode')
        freq_mhz = result.get('freq', 0) / 1e6
        
        fig = plt.figure(figsize=(10, 6))
        
        if mode == '2d':
            ax = fig.add_subplot(111)
            rcs = result['rcs_total']
            theta = result['theta_deg']
            phi = result['phi_deg']
            
            X, Y = np.meshgrid(phi, theta)
            # Handle NaN for better visualization
            Z = np.nan_to_num(rcs, nan=-100.0)
            
            c = ax.pcolormesh(X, Y, Z, cmap='jet', shading='auto')
            fig.colorbar(c, ax=ax, label='RCS (dBsm)')
            ax.set_xlabel("Phi (deg)")
            ax.set_ylabel("Theta (deg)")
            ax.set_title(f"2D RCS Pattern - {title_suffix} @ {freq_mhz:.1f} MHz")
            
        else: # 1d
            ax = fig.add_subplot(111)
            angles = result['theta_deg']
            ax.plot(angles, result['rcs_total'], 'b-', linewidth=2, label='Total RCS')
            
            if result.get('rcs_po') is not None:
                ax.plot(angles, result['rcs_po'], 'g--', linewidth=1.5, alpha=0.7, label='PO Component')
            if result.get('rcs_ptd') is not None:
                 # Check if PTD has valid data
                if np.max(result['rcs_ptd']) > -150:
                    ax.plot(angles, result['rcs_ptd'], 'r:', linewidth=1.5, alpha=0.7, label='PTD Component')
            
            ax.set_xlabel("Theta (deg)")
            ax.set_ylabel("RCS (dBsm)")
            ax.set_title(f"1D RCS Cut - {title_suffix} @ {freq_mhz:.1f} MHz")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved plot to: {output_path}")
        
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

def save_csv(result, output_path):
    """
    Save results to CSV.
    """
    try:
        mode = result.get('mode')
        
        if mode == '2d':
            # Flat list format: Theta, Phi, RCS(dB), RCS(m2)
            theta = result['theta_deg']
            phi = result['phi_deg']
            rcs = result['rcs_total']
            
            data = []
            for i, t in enumerate(theta):
                for j, p in enumerate(phi):
                    val_db = rcs[i, j]
                    data.append({
                        'Theta': t,
                        'Phi': p,
                        'RCS_dBsm': val_db,
                        'RCS_m2': 10**(val_db/10)
                    })
            df = pd.DataFrame(data)
            
        else:
            df = pd.DataFrame({
                'Theta': result['theta_deg'],
                'RCS_Total_dBsm': result['rcs_total'],
                'RCS_Total_m2': 10**(result['rcs_total']/10)
            })
            if result.get('rcs_po') is not None:
                df['RCS_PO_dBsm'] = result['rcs_po']
            if result.get('rcs_ptd') is not None:
                df['RCS_PTD_dBsm'] = result['rcs_ptd']
                
        # Add metadata as comments in the first few lines? 
        # Pandas doesn't support writing comments easily without custom logic.
        # We will write standard CSV.
        df.to_csv(output_path, index=False)
        logger.info(f"Saved data to: {output_path}")
        
    except Exception as e:
        logger.error(f"CSV export failed: {e}")

def main():
    args = parse_args()
    cfg = load_config(args.config_file)
    
    global_settings = cfg.get('global_settings', {})
    base_output_dir = global_settings.get('output_dir', 'results/batch_run')
    save_plot_flag = global_settings.get('save_plot', True)
    
    # Prepare overrides
    overrides = {}
    if args.gpu: overrides['force_gpu'] = True
    if args.no_gpu: overrides['force_gpu'] = False
    if args.workers: overrides['workers'] = args.workers
    
    if not args.dry_run:
        ensure_output_dir(base_output_dir)
        bridge = SolverBridge()
    
    tasks = cfg.get('tasks', [])
    expanded_tasks = []
    
    # --- Task Expansion Logic ---
    for task in tasks:
        geo_cfg = task.get('geometry', {})
        params = geo_cfg.get('params', {})
        fpath = params.get('file_path', "")
        
        if ";" in fpath:
            paths = [p.strip() for p in fpath.split(";") if p.strip()]
            logger.info(f"Task '{task.get('name')}' contains {len(paths)} files. Expanding...")
            for p in paths:
                new_task = json.loads(json.dumps(task)) # Deep copy
                new_task['geometry']['params']['file_path'] = p
                # Update name to include filename
                fname = os.path.splitext(os.path.basename(p))[0]
                new_task['name'] = f"{task.get('name', 'task')}_{fname}"
                expanded_tasks.append(new_task)
        else:
            expanded_tasks.append(task)
            
    tasks = expanded_tasks
    logger.info(f"Total tasks after expansion: {len(tasks)}")
    
    total_start = time.time()
    
    for i, task in enumerate(tasks):
        task_name = task.get('name', f"task_{i}")
        logger.info(f"=== Processing Task [{i+1}/{len(tasks)}]: {task_name} ===")
        
        try:
            # 1. Geometry
            geo_cfg = task.get('geometry', {})
            geo_type = geo_cfg.get('type')
            geo_params = geo_cfg.get('params', {})
            
            if not geo_type:
                logger.error("Skipping task: Geometry type missing.")
                continue

            # Special logging for IGES advanced features
            if geo_type == "IGES File":
                logger.info(f"  IGES Config: Unit={geo_params.get('unit','mm')}, "
                            f"Mirror={geo_params.get('mirror_plane','None')}, "
                            f"Rotation={geo_params.get('rotation','None')}")
                if geo_params.get('delete_indices'):
                    logger.info(f"  Deleting faces: {geo_params['delete_indices']}")
                if geo_params.get('invert_indices'):
                    logger.info(f"  Inverting faces: {geo_params['invert_indices']}")

            # 2. Solver Params
            sim_params = build_sim_params(task, overrides)
            
            if args.dry_run:
                logger.info(f"Dry Run: Would run '{geo_type}' with {sim_params['algorithm']} on {sim_params['compute']}")
                continue
                
            # 3. Build Geometry
            logger.info(f"Building geometry: {geo_type}")
            geo_obj = GeometryFactory.create_geometry(geo_type, geo_params)
            
            if isinstance(geo_obj, tuple): # Handle Wedge/Brick PTD return
                geo_obj = geo_obj[0] 
                
            if not geo_obj:
                logger.error("Geometry creation failed (empty result).")
                continue

            # 4. Run Simulation
            logger.info("Starting simulation...")
            def progress(curr, total, msg):
                # Simple progress log, maybe throttle?
                if total > 0 and curr % 20 == 0:
                    print(f"  Progress: {curr/total*100:.0f}% - {msg}", end='\r')

            result = bridge.run_simulation(geo_obj, sim_params, progress_callback=progress)
            print("") # Newline after progress
            logger.info(f"Simulation completed in {result['elapsed_time']:.2f}s")
            
            # 5. Save Results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_name = "".join([c if c.isalnum() else "_" for c in task_name])
            
            csv_path = os.path.join(base_output_dir, f"{safe_name}_{timestamp}.csv")
            save_csv(result, csv_path)
            
            if save_plot_flag:
                img_path = os.path.join(base_output_dir, f"{safe_name}_{timestamp}.png")
                save_plot(result, img_path, title_suffix=task_name)
                
        except Exception as e:
            logger.error(f"Task '{task_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            
    if not args.dry_run:
        total_time = time.time() - total_start
        logger.info(f"=== All tasks finished in {total_time:.2f}s ===")

if __name__ == "__main__":
    main()
