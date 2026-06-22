import os
import json

from PySide6.QtWidgets import QComboBox

# 固定在项目根目录，避免因运行目录不同而丢失配置
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cem_po_qt_config.json")


def cached_val(window, attr, default=""):
    """Read widget value if it exists, otherwise fall back to _input_cache."""
    w = getattr(window, attr, None)
    if w is not None:
        try:
            return w.currentText() if isinstance(w, QComboBox) else w.text()
        except RuntimeError:
            pass  # C++ object already deleted by deleteLater()
    return window._input_cache.get(attr, default)


def save_config(window):
    try:
        window._cache_dynamic_inputs()

        geo_params_vals = {}
        for k, v in window.geo_inputs.items():
            geo_params_vals[k] = v.text()

        step_unit        = cached_val(window, 'step_unit_combo')
        invert_indices   = cached_val(window, 'invert_indices_input')

        cfg = {
            "geo_type":   window.geo_type_combo.currentText(),
            "geo_params": geo_params_vals,

            "step_file_path": window.step_file_path,
            "step_unit":      step_unit,
            "invert_indices": invert_indices,

            # IGES 多文件：直接序列化 self.iges_files 列表
            "iges_files":           list(getattr(window, 'iges_files', [])),
            "iges_selected_idx":    getattr(window, '_iges_selected_idx', -1),

            "freq":          window.freq_input.text(),
            "mesh_density":  window.mesh_density.text(),
            "min_points":    window.min_points.text(),
            "vis_subsample": window.vis_subsample.text(),
            "use_degen":     window.degen_mesh.isChecked(),

            "algorithm":   window.algo_combo.currentData(),
            "theta_start": window.theta_start.text(),
            "theta_end":   window.theta_end.text(),
            "theta_n":     window.theta_n.text(),
            "phi_start":   window.phi_start.text(),
            "phi_end":     window.phi_end.text(),
            "phi_n":       window.phi_n.text(),

            "ptd_enabled":      window.chk_ptd_enabled.isChecked(),
            "ptd_edges":        window._get_ptd_pairs_str(),
            "ptd_pol":          window.ptd_pol.currentText(),

            "use_gpu":      window.use_gpu.isChecked(),
            "use_parallel": window.use_parallel.isChecked(),
            "cpu_workers":  window.cpu_workers.text(),

            "freq_sweep_enabled": window.chk_freq_sweep_enabled.isChecked(),
            "fsweep_start":       window.fsweep_start.text(),
            "fsweep_end":         window.fsweep_end.text(),
            "fsweep_step":        window.fsweep_step.text(),

            "img_window":       window.img_window.currentText(),
            "img_cheby_at":     window.img_cheby_at.text(),
            "img_taylor_nbar":  window.img_taylor_nbar.text(),
            "img_zeropad":      window.img_zeropad.text(),
            "img_db_min":      window.img_db_min.text(),
            "img_db_max":      window.img_db_max.text(),
            "img_range_limit": window.img_range_limit.text(),
        }

        with open(CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=4)

    except Exception as e:
        print(f"Failed to save config: {e}")


def load_config(window):
    if not os.path.exists(CONFIG_FILE):
        return

    try:
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)

        # Geometry
        gtype = cfg.get("geo_type", "Cylinder")
        window.geo_type_combo.setCurrentText(gtype)

        saved_params = cfg.get("geo_params", {})
        for k, v in saved_params.items():
            if k in window.geo_inputs:
                window.geo_inputs[k].setText(str(v))

        # STEP — 无论控件是否存在都先写入 _input_cache，确保切换页面时能正确恢复
        window.step_file_path = cfg.get("step_file_path", "")
        window._input_cache['step_unit_combo']    = cfg.get("step_unit", "mm")
        window._input_cache['invert_indices_input'] = cfg.get("invert_indices", "")
        if window.step_file_path and hasattr(window, 'lbl_step'):
            window.lbl_step.setText(os.path.basename(window.step_file_path))
        if hasattr(window, 'step_unit_combo'):
            window.step_unit_combo.setCurrentText(cfg.get("step_unit", "mm"))
        if hasattr(window, 'invert_indices_input'):
            window.invert_indices_input.setText(cfg.get("invert_indices", ""))

        # IGES 多文件：优先读 iges_files 列表；若没有则从老版单文件字段迁移
        iges_files = cfg.get("iges_files")
        if iges_files is None:
            old_path = cfg.get("iges_file_path", "")
            if old_path:
                iges_files = [{
                    'path': old_path,
                    'unit': cfg.get("iges_unit", "mm"),
                    'invert_indices': cfg.get("iges_invert_indices", ""),
                    'delete_indices': cfg.get("iges_delete_indices", ""),
                    'mirror_plane': cfg.get("iges_mirror_plane", "None"),
                    'rotation': cfg.get("iges_rotation", ""),
                }]
            else:
                iges_files = []
        window.iges_files = iges_files
        window._iges_selected_idx = cfg.get("iges_selected_idx", 0 if iges_files else -1)
        if hasattr(window, 'iges_file_list'):
            window._rebuild_iges_list_widget()

        # Physics
        window.freq_input.setText(str(cfg.get("freq", "3000.0")))
        window.mesh_density.setText(str(cfg.get("mesh_density", "10.0")))
        window.min_points.setText(str(cfg.get("min_points", "18")))
        window.vis_subsample.setText(str(cfg.get("vis_subsample", "1")))
        window.degen_mesh.setChecked(cfg.get("use_degen", True))

        # Solver
        algo = cfg.get("algorithm")
        idx = window.algo_combo.findData(algo)
        if idx >= 0:
            window.algo_combo.setCurrentIndex(idx)

        window.theta_start.setText(str(cfg.get("theta_start", "-90")))
        window.theta_end.setText(str(cfg.get("theta_end", "90")))
        window.theta_n.setText(str(cfg.get("theta_n", "181")))
        window.phi_start.setText(str(cfg.get("phi_start", "0")))
        window.phi_end.setText(str(cfg.get("phi_end", "0")))
        window.phi_n.setText(str(cfg.get("phi_n", "1")))

        window.chk_ptd_enabled.setChecked(cfg.get("ptd_enabled", False))
        import re as _re
        saved_pairs = cfg.get("ptd_edges", "")
        window.ptd_pairs_list.clear()
        if saved_pairs:
            pairs = [(int(a), int(b)) for a, b in _re.findall(r'(\d+)\s*,\s*(\d+)', saved_pairs)]
            window._ptd_add_pairs(pairs)
        window.ptd_pol.setCurrentText(cfg.get("ptd_pol", "VV"))

        window.use_gpu.setChecked(cfg.get("use_gpu", False))
        window.use_parallel.setChecked(cfg.get("use_parallel", False))
        window.cpu_workers.setText(str(cfg.get("cpu_workers", "4")))

        window.chk_freq_sweep_enabled.setChecked(cfg.get("freq_sweep_enabled", False))
        window.fsweep_start.setText(str(cfg.get("fsweep_start", "1000")))
        window.fsweep_end.setText(str(cfg.get("fsweep_end", "5000")))
        window.fsweep_step.setText(str(cfg.get("fsweep_step", "10")))

        window.img_window.setCurrentText(cfg.get("img_window", "hamming"))
        window.img_cheby_at.setText(str(cfg.get("img_cheby_at", "40")))
        window.img_taylor_nbar.setText(str(cfg.get("img_taylor_nbar", "4")))
        window.img_zeropad.setText(str(cfg.get("img_zeropad", "4")))
        window.img_db_min.setText(str(cfg.get("img_db_min", "-60")))
        window.img_db_max.setText(str(cfg.get("img_db_max", "5")))
        window.img_range_limit.setText(cfg.get("img_range_limit", ""))

        window.log("Configuration loaded.")

    except Exception as e:
        window.log(f"Failed to load config: {e}")
