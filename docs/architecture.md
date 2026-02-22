# CEM PO Solver — 项目架构文档

## 完整目录结构

```
CEM_PO/
├── main.py                     # CLI 入口，驱动批量计算任务
├── gui_qt.py                   # PySide6 GUI 主程序入口
├── batch_tasks_example.json    # 批量任务配置模板（示例）
├── environment.yml             # Conda 环境配置
├── requirements.txt            # pip 依赖列表
│
├── core/                       # 核心架构层
│   ├── env.py                  # 运行环境检测（GPU/CPU 可用性）
│   ├── mesh_data.py            # 网格数据结构（MeshData）
│   └── solver_bridge.py        # UI 与算法解耦的桥接层（SolverBridge）
│
├── geometry/                   # 几何建模层
│   ├── factory.py              # 几何工厂，统一创建各类几何体
│   ├── surface.py              # 参数曲面基类
│   ├── occ_surface.py          # PythonOCC 曲面封装（B-Spline/NURBS）
│   ├── step_loader.py          # STEP / IGES 文件加载（含手动 IGES 解析）
│   ├── iges_fix.py             # IGES 文件非标准格式修复工具
│   ├── sphere.py               # 球体参数曲面
│   ├── cylinder.py             # 柱体参数曲面
│   ├── plate.py                # 平板参数曲面
│   ├── brick.py                # 长方体参数曲面
│   └── wedge.py                # 楔形参数曲面
│
├── solvers/                    # 电磁算法层
│   ├── api.py                  # 算法工厂，统一调度各求解器
│   ├── po.py                   # 离散 PO 积分器（CPU / CUDA GPU）
│   ├── ptd.py                  # 物理绕射理论（PTD）修正
│   ├── ptd_structures.py       # PTD 数据结构定义
│   ├── rcs_analyzer.py         # RCS 角度扫描控制器
│   ├── ribbon.py               # Strip Mesh (条带网格) 生成
│   ├── ribbon_analytic.py      # Analytic-Ribbon 解析算法（学术对照）
│   └── ribbon_polynomials.py   # Ribbon 解析算法多项式系数表
│
├── physics/                    # 物理/数学基础层
│   ├── constants.py            # 物理常数（光速、波阻抗等）
│   ├── wave.py                 # 入射波定义（平面波）
│   ├── analytical_rcs.py       # 解析 RCS 公式（球、平板等标准体）
│   └── ptd_core.py             # PTD 绕射系数核心计算
│
├── ui/                         # GUI 组件层
│   └── __init__.py             # 组件导出
│
├── tools/                      # 独立工具脚本
│   └── visualize_mesh.py       # 网格可视化（离线调试用）
│
├── read-compare/               # 数据读取与对比工具
│   ├── rcs_data_reader.py      # 读取 Feko / CSV 格式 RCS 数据
│   ├── rcs_visual.py           # RCS 曲线对比可视化
│   └── RCS可视化函数参考.md    # 可视化函数接口参考
│
└── docs/                       # 文档
    ├── architecture.md         # 本文件：项目架构说明
    ├── batch_processing.md     # 批量计算配置详细说明
    └── true_ribbon_implementation.md  # Ribbon 算法实现说明
```

## 模块依赖关系

```
gui_qt.py / main.py
    │
    ├── core/solver_bridge.py   ← 隔离 UI 与计算
    │       │
    │       ├── solvers/api.py  ← 算法调度
    │       │       ├── solvers/po.py        (PO 积分)
    │       │       ├── solvers/ptd.py       (PTD 修正)
    │       │       ├── solvers/rcs_analyzer.py
    │       │       └── solvers/ribbon.py    (Strip Mesh)
    │       │
    │       └── geometry/factory.py  ← 几何创建
    │               ├── geometry/surface.py / sphere.py / ...  (解析体)
    │               └── geometry/step_loader.py                (CAD 文件)
    │
    ├── core/mesh_data.py       ← 网格数据流转
    ├── physics/                ← 被 solvers/ 调用
    └── core/env.py             ← GPU 环境检测
```

## 关键数据流

1. **几何输入** → `geometry/factory.py` 生成参数曲面或加载 CAD 文件
2. **网格化** → `core/mesh_data.py` 的 `MeshData` 存储面元中心、法线、面积
3. **求解** → `solvers/rcs_analyzer.py` 驱动角度扫描，调用 `solvers/po.py` 计算每个方向的 RCS
4. **输出** → 图表（matplotlib）或 CSV 文件写入 `results/`

## 主要算法

| 算法 | 文件 | 说明 |
|------|------|------|
| Discrete PO (Dual Sinc) | `solvers/po.py` | 旗舰算法，双向 Sinc 相位误差修正 |
| Strip Mesh | `solvers/ribbon.py` | 旋转体极点简并网格特殊拓扑 |
| PTD 边缘修正 | `solvers/ptd.py` + `physics/ptd_core.py` | Michaeli/Ufimtsev 绕射公式 |
| Analytic Ribbon | `solvers/ribbon_analytic.py` | 1995 经典解析算法（学术对照） |
| 解析 RCS | `physics/analytical_rcs.py` | 标准体精确解（用于验证） |
