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
| `IGES File` | `file_path`, `folder_path`, `unit`, `mirror_plane`, `rotation`, `delete_indices`, `invert_indices` | 加载并修正 IGES 文件 |
| `STEP File` | `file_path`, `folder_path`, `unit`, `invert_indices` | 加载并修正 STEP 文件 |

**多模型批量处理 (Multi-Model Batching):**
- **手动列表**：在 `file_path` 中使用分号 `;` 分隔多个文件路径（如 `"a.igs; b.igs"`）。
- **文件夹自动扫描**：指定 `folder_path` 为一个目录，程序将自动扫描其中的模型。
    - **智能过滤**：若类型为 `IGES File`，则仅提取 `.igs`/`.iges`；若为 `STEP File`，则仅提取 `.stp`/`.step`。
- **智能命名逻辑**：
    - 若任务 `name` 为 `""` 或 `"task"`，则自动使用 **模型文件名** 作为任务名。
    - 若指定了具体 `name`（如 `"Validation"`），则命名为 **`Validation_文件名`**。

#### Solver (求解器设置)
| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `frequency_mhz` | 1000.0 | 仿真频率 (MHz) |
| `algorithm` | `discrete_po_sinc_dual` | 核心算法（见下方算法列表） |
| `polarization` | "VV" | 极化方式 ("VV", "HH", "VH", "HV") |
| `mesh_density` | 10.0 | 网格密度 (每波长采样点数) |
| `min_points` | - | 最小网格点数（可选） |
| `use_degenerate`| false | 是否使用退化处理（如去除退化面） |
| `use_gpu` | false | 是否使用 GPU 加速 |
| `workers` | - | CPU 并行计算的进程数（覆盖全局设置） |
| `ptd.enabled` | false | 是否开启 PTD 修正 |
| `ptd.edges` | `""` | 手动指定的 PTD 边索引（字符串，如 `"0,1,2,3"` 或 `"(0,4);(5,9)"`） |
| `ptd.seg_angle_deg` | 2.0 | PTD 边缘分段角度 (度) |
| `ptd.use_parallel_ptd`| false | 是否开启 PTD 并行计算 |

#### Scan (扫描角度)
- `theta`: `[起始角度, 终止角度, 点数]`
- `phi`: `[起始角度, 终止角度, 点数]`

#### Freq Sweep (频率扫描设置，可选)
如果在任务配置中包含此模块，将会进行宽带频率扫描，而不是单频点计算。
| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `enabled` | false | 是否开启频率扫描 |
| `f_start` | - | 起始频率 (MHz) |
| `f_end` | - | 终止频率 (MHz) |
| `f_step` | - | 频率步长 (MHz) |
| `window` | `"none"` | 窗函数类型，如 `"none"`, `"hamming"`, `"hanning"`, `"blackman"` |
| `zero_pad`| 1 | 补零倍数，用于提高时域/距离域插值分辨率 |
| `polarization` | `"VV"` | 频率扫描使用的极化方式 |

---

## 3. 输出结果与日志

### 3.1 结果文件 (Results)
计算完成后，结果将保存在 `output_dir` 中：
1. **CSV 文件**: 包含 Theta, Phi, RCS(dBsm), RCS(m^2) 的数据。
2. **PNG 图片**: RCS 随角度变化的曲线图。
    - **2D 扫描**: 角度 1:1 比例热力图，**小角度在上、大角度在下**展示。

### 3.2 增强型日志系统 (Logging)
系统会自动将整个仿真的“System Log”全数保存到 `log_dir` 下的 `.log` 文件中：
- **全量记录**：包括 PTD 边缘提取详情、网格预计算状态、GPU 迁移信息、仿真参数概览。
- **进度跟踪**：自动记录关键节点的计算进度（20%, 40%, ... 100%）。
- **多任务追溯**：日志文件名带时间戳（如 `batch_run_20260219_1030.log`），便于回溯不同批次的运行过程。

---

## 4. 示例配置

参考项目根目录下的 `batch_tasks_example.json` 文件。
