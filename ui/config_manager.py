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
        iges_unit        = cached_val(window, 'iges_unit_combo')
        iges_invert_indices = cached_val(window, 'iges_invert_indices_input')
        iges_delete_indices = cached_val(window, 'iges_delete_indices_input')
        iges_mirror_plane   = cached_val(window, 'iges_mirror_plane_combo', 'None')
        iges_rotation    = cached_val(window, 'iges_rotation_input')

        cfg = {
            "geo_type":   window.geo_type_combo.currentText(),
            "geo_params": geo_params_vals,

            "step_file_path": window.step_file_path,
            "step_unit":      step_unit,
            "invert_indices": invert_indices,

            "iges_file_path":        window.iges_file_path,
            "iges_unit":             iges_unit,
            "iges_invert_indices":   iges_invert_indices,
            "iges_delete_indices":   iges_delete_indices,
            "iges_mirror_plane":     iges_mirror_plane,
            "iges_rotation":         iges_rotation,

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
            "ptd_edges":        window.ptd_edges.text(),
            "ptd_pol":          window.ptd_pol.currentText(),
            "ptd_wedge_angle":  window.ptd_wedge_angle.value(),

            "use_gpu":      window.use_gpu.isChecked(),
            "use_parallel": window.use_parallel.isChecked(),
            "cpu_workers":  window.cpu_workers.text()
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

        # STEP
        window.step_file_path = cfg.get("step_file_path", "")
        if window.step_file_path and hasattr(window, 'lbl_step'):
            window.lbl_step.setText(os.path.basename(window.step_file_path))
        if hasattr(window, 'step_unit_combo'):
            window.step_unit_combo.setCurrentText(cfg.get("step_unit", "mm"))
        if hasattr(window, 'invert_indices_input'):
            window.invert_indices_input.setText(cfg.get("invert_indices", ""))

        # IGES
        window.iges_file_path = cfg.get("iges_file_path", "")
        if window.iges_file_path and hasattr(window, 'lbl_iges'):
            window.lbl_iges.setText(os.path.basename(window.iges_file_path))
        if hasattr(window, 'iges_unit_combo'):
            window.iges_unit_combo.setCurrentText(cfg.get("iges_unit", "mm"))
        if hasattr(window, 'iges_invert_indices_input'):
            window.iges_invert_indices_input.setText(cfg.get("iges_invert_indices", ""))
        if hasattr(window, 'iges_delete_indices_input'):
            window.iges_delete_indices_input.setText(cfg.get("iges_delete_indices", ""))
        if hasattr(window, 'iges_mirror_plane_combo'):
            window.iges_mirror_plane_combo.setCurrentText(cfg.get("iges_mirror_plane", "None"))
        if hasattr(window, 'iges_rotation_input'):
            window.iges_rotation_input.setText(cfg.get("iges_rotation", ""))

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
        window.ptd_edges.setText(cfg.get("ptd_edges", ""))
        window.ptd_pol.setCurrentText(cfg.get("ptd_pol", "VV"))
        window.ptd_wedge_angle.setValue(float(cfg.get("ptd_wedge_angle", 90.0)))

        window.use_gpu.setChecked(cfg.get("use_gpu", False))
        window.use_parallel.setChecked(cfg.get("use_parallel", False))
        window.cpu_workers.setText(str(cfg.get("cpu_workers", "4")))

        window.log("Configuration loaded.")

    except Exception as e:
        window.log(f"Failed to load config: {e}")
