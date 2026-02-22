import argparse
import json
import os
import time
import sys
import logging

from core.solver_bridge import SolverBridge
from geometry.factory import GeometryFactory
from tools.export import save_plot, save_csv  # also sets matplotlib Agg backend

# 配置日志
def setup_logging(log_dir, task_name=None):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = f"batch_run_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有的处理器，防止重复
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # 文件处理器
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)
    
    return logging.getLogger("CEM-PO-CLI"), log_path

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
        print(f"Error: Config file not found: {path}")
        sys.exit(1)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: Failed to parse JSON config: {e}")
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

def main():
    args = parse_args()
    cfg = load_config(args.config_file)
    
    global_settings = cfg.get('global_settings', {})
    base_output_dir = global_settings.get('output_dir', 'results/batch_run')
    log_dir = global_settings.get('log_dir', os.path.join(base_output_dir, 'logs'))
    save_plot_flag = global_settings.get('save_plot', True)
    
    # 命名格式，例如 "{task_name}" 或 "{task_name}_{timestamp}"
    filename_format = global_settings.get('filename_format', "{task_name}")
    
    # 设置日志
    global logger
    logger, log_path = setup_logging(log_dir)
    logger.info(f"Logging initialized. Log file: {log_path}")
    
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
        geo_type = geo_cfg.get('type', "")
        params = geo_cfg.get('params', {})
        
        # 收集所有待处理的文件路径
        all_files = []
        
        # 1. 处理显式指定的文件 (支持分号隔离)
        fpath = params.get('file_path', "")
        if fpath:
            paths = [p.strip() for p in fpath.split(";") if p.strip()]
            all_files.extend(paths)
            
        # 2. 处理文件夹自动遍历 (新功能)
        folder_path = params.get('folder_path', "")
        if folder_path and os.path.isdir(folder_path):
            logger.info(f"Scanning folder: {folder_path}")
            
            # 根据几何类型决定允许的后缀
            if geo_type == "IGES File":
                valid_exts = ('.igs', '.iges')
            elif geo_type == "STEP File":
                valid_exts = ('.stp', '.step')
            else:
                valid_exts = ('.igs', '.iges', '.stp', '.step')
                
            found_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(valid_exts)
            ]
            logger.info(f"  Found {len(found_files)} matching CAD models in folder.")
            all_files.extend(found_files)

        # 3. 如果有多个文件，进行任务展开
        if len(all_files) > 1:
            logger.info(f"Task '{task.get('name')}' expanded into {len(all_files)} sub-tasks.")
            for p in all_files:
                new_task = json.loads(json.dumps(task))
                new_task['geometry']['params']['file_path'] = p
                # 移除 folder_path 防止重复触发逻辑
                if 'folder_path' in new_task['geometry']['params']:
                    del new_task['geometry']['params']['folder_path']
                
                fname = os.path.splitext(os.path.basename(p))[0]
                
                # 智能命名逻辑：如果原名为 "task" 或空，则直接使用文件名
                orig_name = task.get('name', 'task')
                if orig_name.lower() == "task" or not orig_name.strip():
                    new_task['name'] = fname
                else:
                    new_task['name'] = f"{orig_name}_{fname}"
                
                expanded_tasks.append(new_task)
        elif len(all_files) == 1:
            p = all_files[0]
            task['geometry']['params']['file_path'] = p
            fname = os.path.splitext(os.path.basename(p))[0]
            
            orig_name = task.get('name', 'task')
            if orig_name.lower() == "task" or not orig_name.strip():
                task['name'] = fname
            else:
                task['name'] = f"{orig_name}_{fname}"
                
            expanded_tasks.append(task)
        else:
            expanded_tasks.append(task)
            
    tasks = expanded_tasks
    logger.info(f"Total tasks after expansion: {len(tasks)}")
    
    total_tasks = len(tasks)
    total_start = time.time()
    
    for i, task in enumerate(tasks):
        task_name = task.get('name', f"task_{i}")
        logger.info(f"=== Processing Task [{i+1}/{total_tasks}]: {task_name} ===")
        
        try:
            # 1. Geometry
            geo_cfg = task.get('geometry', {})
            geo_type = geo_cfg.get('type')
            geo_params = geo_cfg.get('params', {})
            
            if not geo_type:
                logger.error("Skipping task: Geometry type missing.")
                continue

            sim_params = build_sim_params(task, overrides)
            
            if args.dry_run:
                logger.info(f"Dry Run: Would run '{geo_type}' with {sim_params['algorithm']} on {sim_params['compute']}")
                continue
                
            # 3. Build Geometry
            logger.info(f"Building geometry: {geo_type}")
            geo_obj = GeometryFactory.create_geometry(geo_type, geo_params)
            
            if isinstance(geo_obj, tuple):
                geo_obj = geo_obj[0] 
                
            if not geo_obj:
                logger.error("Geometry creation failed (empty result).")
                continue

            # 4. Run Simulation
            logger.info(f"Starting simulation for '{task_name}'...")
            
            current_task_idx = i
            def progress(curr, total, msg):
                # --- 双进度条渲染 ---
                bar_len = 20
                
                # A. 子任务进度 (Sub-task)
                sub_percent = curr / total if total > 0 else 0
                sub_filled = int(bar_len * sub_percent)
                sub_bar = "█" * sub_filled + "-" * (bar_len - sub_filled)
                
                # B. 总任务进度 (Total Tasks)
                # 总进度 = (已完成任务数 + 当前任务完成百分比) / 总任务数
                total_percent = (current_task_idx + sub_percent) / total_tasks
                total_filled = int(bar_len * total_percent)
                total_bar = "█" * total_filled + "-" * (bar_len - total_filled)
                
                # 仅在控制台显示 (合并为一行)
                status_line = f"\r  Task [{current_task_idx+1}/{total_tasks}] [{sub_bar}] {sub_percent*100:3.0f}% | Total [{total_bar}] {total_percent*100:3.0f}% | {msg[:30]:<30}"
                sys.stdout.write(status_line)
                sys.stdout.flush()
                
                # 仅向日志文件记录进度 (避免刷屏)
                if total > 0 and (curr % 20 == 0 or curr == total):
                    for handler in logging.getLogger().handlers:
                        if isinstance(handler, logging.FileHandler):
                            record = logger.makeRecord(logger.name, logging.INFO, None, 0, 
                                                     f"Task {current_task_idx+1}/{total_tasks} - Progress: {sub_percent*100:.0f}% - {msg}", None, None)
                            handler.handle(record)

            result = bridge.run_simulation(geo_obj, sim_params, progress_callback=progress)
            print("") # 每个任务结束后换行
            logger.info(f"Simulation completed in {result['elapsed_time']:.2f}s")
            
            # 5. Save Results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_name = "".join([c if c.isalnum() else "_" for c in task_name])
            
            # 使用命名模版生成文件名
            try:
                final_filename = filename_format.format(task_name=safe_name, timestamp=timestamp)
            except Exception as e:
                logger.warning(f"Naming format error: {e}. Falling back to task_name.")
                final_filename = safe_name

            csv_path = os.path.join(base_output_dir, f"{final_filename}.csv")
            save_csv(result, csv_path)
            
            if save_plot_flag:
                img_path = os.path.join(base_output_dir, f"{final_filename}.png")
                save_plot(result, img_path, title_suffix=task_name)
                
        except Exception as e:
            logger.error(f"Task '{task_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            
    if not args.dry_run:
        total_time = time.time() - total_start
        logger.info(f"=== All tasks finished in {total_time:.2f}s ===")
        logger.info(f"Total results saved in: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    main()
