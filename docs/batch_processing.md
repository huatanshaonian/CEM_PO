# 批量计算说明文档 (Batch Processing Guide)

本文档介绍如何使用 `main.py` 进行电磁散射计算的批量处理任务。

## 1. 快速开始

使用命令行运行指定配置文件：

```bash
python main.py batch_tasks.json
```

### 命令行参数
- `config_file`: 必需。JSON 格式的任务配置文件路径。
- `--gpu`: 强制启用 GPU 加速（覆盖配置文件中的设置）。
- `--no-gpu`: 强制禁用 GPU 加速。
- `--workers N`: 覆盖 CPU 并行计算的进程数。
- `--dry-run`: 仅校验配置文件和几何模型，不执行实际计算。

---

## 2. 配置文件架构 (JSON)

配置文件包含全局设置 (`global_settings`) 和任务列表 (`tasks`)。

### 2.1 全局设置 (global_settings)
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `output_dir` | string | `results/batch_run` | 结果保存的根目录 |
| `log_dir` | string | `results/batch_run/logs` | 日志保存目录 |
| `save_plot` | boolean | `true` | 是否生成并保存 RCS 曲线图 (PNG) |
| `filename_format`| string | `"{task_name}"` | 文件命名格式。支持 `{task_name}` 和 `{timestamp}` 占位符。 |

**命名格式示例：**
- `"{task_name}"`: (推荐) 仅使用任务名，不带时间戳，方便覆盖或版本管理。
- `"{task_name}_{timestamp}"`: 每次运行生成带时间戳的唯一文件。

### 2.2 任务配置 (tasks)
每个任务包含 `geometry` (几何)、`solver` (求解器) 和 `scan` (扫描范围) 三部分。

#### Geometry (几何体定义)
| 类型 (type) | 参数 (params) | 说明 |
| :--- | :--- | :--- |
| `Cylinder` | `radius`, `height` | 圆柱体 |
| `Sphere` | `radius` | 球体 |
| `Plate` | `width`, `length` | 平板 (XY 平面) |
| `Brick` | `width`, `length`, `height` | 长方体 |
| `Wedge` | `width`, `length`, `height` | 劈尖 |
| `IGES File` | `file_path`, `unit`, `mirror_plane`, `rotation`, `delete_indices`, `invert_indices` | 加载并修正 IGES 文件 |
| `STEP File` | `file_path`, `unit`, `invert_indices` | 加载并修正 STEP 文件 |

**IGES 高级参数说明：**
- `unit`: "mm", "cm" 或 "m" (默认 "mm")。
- `mirror_plane`: 对称镜像平面，可选 "XY", "YZ", "XZ", "None"。
- `rotation`: 绕 [X, Y, Z] 轴的旋转角度 (度)。
- `delete_indices`: 列表，指定要删除的面索引。
- `invert_indices`: 列表，指定需要翻转法向的面索引。
- **多模型处理**：在 `file_path` 中使用分号 `;` 分隔多个文件路径（如 `"a.igs; b.igs"`），程序将自动拆分为多个子任务。

#### Solver (求解器设置)
| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `frequency_mhz` | 1000.0 | 仿真频率 (MHz) |
| `algorithm` | `discrete_po_sinc_dual` | 核心算法（见下方算法列表） |
| `polarization` | "VV" | 极化方式 ("VV", "HH", "VH", "HV") |
| `mesh_density` | 10.0 | 网格密度 (每波长采样点数) |
| `use_gpu` | false | 是否使用 GPU 加速 |
| `ptd.enabled` | false | 是否开启 PTD 修正 |
| `ptd.edges` | [] | 手动指定的 PTD 边索引列表 |

**支持算法列表 (`algorithm`):**
- `discrete_po_sinc_dual`: **(推荐)** 双向 Sinc 校正 PO。
- `discrete_po_sinc_u`: 单向 Sinc 校正 PO。
- `gauss_ribbon`: 自适应 Gauss-Ribbon 积分。
- `analytic_ribbon`: 解析 Ribbon 积分 (1995 论文复现)。
- `discrete_po_none`: 纯离散 PO（需极高网格密度）。

#### Scan (扫描角度)
- `theta`: `[起始角度, 终止角度, 点数]`
- `phi`: `[起始角度, 终止角度, 点数]`

---

## 3. 输出结果

计算完成后，结果将保存在 `output_dir` 中：
1. **CSV 文件**: 包含 Theta, Phi, RCS(dBsm), RCS(m^2) 的数据。
2. **PNG 图片**: RCS 随角度变化的曲线图。
    - **1D 扫描**: 常规曲线图。
    - **2D 扫描**: 角度 1:1 比例热力图，**小角度在上、大角度在下**展示。
3. **Log 文件**: 保存在 `log_dir` 下，包含任务运行过程中的所有输出，文件名带时间戳。

---

## 4. 示例配置

参考项目根目录下的 `batch_tasks_example.json` 文件。
