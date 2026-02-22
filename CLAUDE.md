# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# GUI mode
python gui_qt.py

# Batch CLI (JSON config required)
python main.py my_tasks.json
python main.py my_tasks.json --gpu          # force GPU
python main.py my_tasks.json --no-gpu       # force CPU
python main.py my_tasks.json --workers 8    # override CPU worker count
python main.py my_tasks.json --dry-run      # validate config only
```

Config JSON files (`*_config.json`) are gitignored — they contain local paths and are user-specific. Use `batch_tasks_example.json` as the template.

## Architecture

### Geometry Layer (`geometry/`)

All geometry types implement `Surface` ABC (`geometry/surface.py`), which requires:
- `u_domain` / `v_domain` properties (parameter range)
- `evaluate(u, v)` → 3D point
- `get_normal(u, v)` → unit normal
- `get_jacobian(u, v)` → area element

`GeometryFactory.create_geometry(geo_type, params)` is the single entry point. For Wedge and Brick it returns a tuple `(surfaces, ptd_edge_ids)` — callers must handle both return shapes. CAD files (STEP/IGES) go through PythonOCC via `geometry/step_loader.py`; if OCC parsing fails, a manual IGES B-Spline parser acts as fallback.

### Solver Layer (`solvers/`)

Algorithm registry lives in `solvers/api.py` as `AVAILABLE_ALGORITHMS` dict. Adding a new algorithm means adding an entry there — no other dispatch code to change. `get_integrator(algo_id, **kwargs)` is the factory.

All `discrete_po_*` variants use `DiscretePOIntegrator` with `sinc_mode` kwarg. Only these algorithms support mesh caching (`precompute_mesh` / `get_mesh_data` interface). Other algorithms (ribbon, analytic) do not.

### SolverBridge (`core/solver_bridge.py`)

`SolverBridge` decouples UI from computation. The GUI/CLI holds a **single persistent instance** so that mesh cache (`cached_mesh_data`) survives between runs. Key methods:
- `run_simulation(geo, params, progress_callback)` — blocking, run in a thread from GUI
- `generate_mesh(geo, params)` — builds and caches mesh for preview
- `update_mesh_cache(mesh_data, params)` — allows external injection of precomputed meshes

The `params` dict passed to `run_simulation` has a fixed schema:
```python
{
  'frequency': float,           # Hz
  'algorithm': str,             # key from AVAILABLE_ALGORITHMS
  'angles': {'theta_start', 'theta_end', 'n_theta', 'phi_start', 'phi_end', 'n_phi'},
  'mesh': {'density', 'min_points', 'use_degenerate'},
  'ptd': {'enabled', 'edges', 'polarization'},
  'compute': {'gpu', 'parallel', 'workers'}
}
```

### Mesh Data (`core/mesh_data.py`)

- `CachedMeshData` — per-surface precomputed mesh (numpy arrays). Supports `.to_gpu()` / `.to_cpu()` via CuPy.
- `MergedMeshData` — GPU optimization: merges a list of `CachedMeshData` into one giant concatenated array for a single GPU kernel launch. Created by `merge_meshes(mesh_list, to_gpu=True)`.

GPU availability is detected at import time (`HAS_GPU`). All GPU paths are guarded with `if HAS_GPU`.

### Data Flow

```
JSON config / GUI params
    → GeometryFactory.create_geometry()  →  list[Surface]
    → SolverBridge.run_simulation()
        → precompute_mesh() per surface  →  list[CachedMeshData]
        → (GPU) merge_meshes()           →  MergedMeshData
        → RCSAnalyzer.compute_monostatic_rcs[_2d]()
        → result dict {mode, theta_deg, phi_deg, rcs_total, rcs_po, rcs_ptd, ...}
```

## Key Conventions

- **Branch**: develop on `Qt-dev`, merge to `master` for releases.
- **File organization**: keep functions grouped by responsibility — do not accumulate unrelated logic in one file.
- **`*_config.json`** files are gitignored and must not be committed.
- **Test/validation scripts** (`test_*.py`, `verify_*.py`, `main_*_validation.py`) are gitignored and not tracked.
- **`results/`** directory is gitignored — all run outputs stay local.
- **matplotlib backend**: `main.py` sets `matplotlib.use('Agg')` before any pyplot import for headless batch runs. Do not import pyplot at module level in shared code.
