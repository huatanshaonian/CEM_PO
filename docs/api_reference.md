# CEM PO 项目 API 调用参考指南

本项目采用 **逻辑与 GUI 完全解耦** 的架构设计。所有计算功能、几何建模和结果处理均可通过 Python API 直接调用，无需启动图形界面。

---

## 1. 核心流程概览

一个典型的仿真流程包含以下步骤：
1. **定义几何**：使用 `GeometryFactory` 创建或加载几何体。
2. **选择算法**：使用 `get_integrator` 获取指定的求解器实例。
3. **设置分析器**：初始化 `RCSAnalyzer` 并调用扫描函数。
4. **处理结果**：获取 RCS 数组并进行绘图或保存。

---

## 2. 几何建模接口 (`geometry/factory.py`)

使用 `GeometryFactory.create_geometry(geo_type, params)` 创建几何体。

### 支持的类型与参数
| 类型 (`geo_type`) | 参数字典 (`params`) | 说明 |
| :--- | :--- | :--- |
| `Plate` | `width`, `length` | 平板 |
| `Sphere` | `radius` | 球体 |
| `Cylinder` | `radius`, `height` | 圆柱体（侧面） |
| `Wedge` | `width`, `length`, `height` | 楔形体 |
| `Brick` | `width`, `length`, `height` | 长方体 |
| `STEP File` | `file_path`, `unit`('mm'/'m'), `scale` | 加载 CAD 文件 (STEP) |
| `IGES File` | `file_path`, `unit`, `scale` | 加载 CAD 文件 (IGES) |

**示例：**
```python
from geometry.factory import GeometryFactory
surfaces = GeometryFactory.create_geometry("Plate", {"width": 5.0, "length": 10.0})
```

---

## 3. 算法工厂接口 (`solvers/api.py`)

使用 `get_integrator(algorithm_id, **kwargs)` 获取求解器。

### 可用算法 ID
* `discrete_po_sinc_dual`: **(推荐)** 离散 PO + 双向 Sinc 校正。
* `gauss_ribbon`: Gauss-Ribbon 自适应积分。
* `analytic_ribbon`: 经典解析 Ribbon 算法（1995）。
* `discrete_po_none`: 纯离散 PO（无校正，需极细网格）。

**示例：**
```python
from solvers.api import get_integrator
solver = get_integrator('discrete_po_sinc_dual')
```

---

## 4. RCS 分析器接口 (`solvers/rcs_analyzer.py`)

`RCSAnalyzer` 是驱动角度扫描的核心类。

### 主要方法
* **`compute_monostatic_rcs(...)`**: 1D 角度扫描（通常是单站 Theta 扫描）。
* **`compute_monostatic_rcs_2d(...)`**: 2D 全空间扫描 (Theta x Phi)。

### 关键参数
* `geometry`: 几何对象列表（由 Factory 生成）。
* `wave_params`: `{'frequency': f, 'phi': p}`。
* `angles`: Theta 角度数组（弧度）。
* `samples_per_lambda`: 网格密度（建议 10.0 以上）。
* `parallel`: 是否启用多进程并行 (`True/False`)。
* `gpu`: 是否启用 GPU 加速 (`True/False`)。
* `enable_ptd`: 是否启用 PTD 边缘修正。

---

## 5. 快速调用示例 (CLI 脚本模式)

以下脚本演示了如何不启动 GUI 直接运行一个平板的 RCS 计算：

```python
import numpy as np
from geometry.factory import GeometryFactory
from solvers.api import get_integrator
from solvers.rcs_analyzer import RCSAnalyzer

# 1. 准备几何
surfaces = GeometryFactory.create_geometry("Plate", {"width": 1.0, "length": 1.0})

# 2. 初始化求解器
solver = get_integrator('discrete_po_sinc_dual')
analyzer = RCSAnalyzer(solver)

# 3. 设置参数
freq = 3e9  # 3 GHz
theta_deg = np.linspace(-90, 90, 181)
theta_rad = np.radians(theta_deg)
wave_params = {'frequency': freq, 'phi': 0.0}

# 4. 运行计算
results = analyzer.compute_monostatic_rcs(
    surfaces, 
    wave_params, 
    theta_rad, 
    samples_per_lambda=10.0,
    parallel=True  # 启用并行
)

# 5. 获取结果 (dBsm)
rcs_po = results['po']
print(f"Max RCS: {np.max(rcs_po):.2f} dBsm")
```

---

## 6. 高级桥接接口 (`core/solver_bridge.py`)

如果你希望使用与 GUI 完全一致的逻辑（包括网格缓存管理和自动参数解析），可以使用 `SolverBridge`。

```python
from core.solver_bridge import SolverBridge

bridge = SolverBridge()
params = {
    'frequency': 1e9,
    'algorithm': 'discrete_po_sinc_dual',
    'mesh': {'density': 10.0},
    'angles': {'theta_start': -90, 'theta_end': 90, 'n_theta': 181},
    'compute': {'parallel': True, 'gpu': False}
}

# 这里的 geo 是 GeometryFactory 生成的对象
result_dict = bridge.run_simulation(surfaces, params)
print(f"Elapsed time: {result_dict['elapsed_time']:.2f}s")
```

---

## 7. 辅助工具

### 物理常数与入射波 (`physics/`)
* `physics.constants.C0`: 光速。
* `physics.wave.IncidentWave(freq, theta, phi)`: 入射平面波对象。

### 结果读取 (`read-compare/rcs_data_reader.py`)
* `RCSDataReader.load_csv(path)`: 加载本项目保存的 CSV。
* `RCSDataReader.load_feko_out(path)`: 加载 Feko 的 `.out` 文件进行对比。
