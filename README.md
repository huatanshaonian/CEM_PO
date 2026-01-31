# 电磁散射 PO 求解器 (CEM PO Solver)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Qt](https://img.shields.io/badge/GUI-PySide6-green)](https://www.qt.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 项目简介

**CEM PO Solver** 是一个高性能的高频电磁散射仿真软件，基于物理光学 (Physical Optics, PO) 近似与物理绕射理论 (PTD) 边缘修正。它专为 RCS (雷达散射截面) 的快速预测、工程验证及算法研究而设计。

本项目目前正处于架构升级阶段，提供了两个版本的图形界面：
1.  **Professional Edition (New)**: 基于 **PySide6 (Qt)** 和 **PyVista** 的全新现代化界面，支持 GPU 流畅渲染百万级网格、多线程后台计算及更丰富的数据对比功能。
2.  **Legacy Standard Edition**: 基于 **Tkinter** 的经典界面，轻量级，适合快速测试。

### ✨ 核心特性

*   **🖥️ 双模 GUI**:
    *   **Qt 版**: 现代暗色/亮色主题，Docker 布局，交互式 3D 旋转/缩放/拾取，支持从 CSV 导入多组数据对比。
    *   **Tkinter 版**: 简单直接，无需复杂依赖。
*   **⚡ 高性能计算**:
    *   **GPU 加速**: 利用 CUDA/CuPy 进行大规模面元积分，计算速度提升 10x-50x。
    *   **并行计算**: 多核 CPU 并行扫描引擎。
*   **📐 高级网格技术**:
    *   **Sinc 插值修正**: 独创的 Per-cell Sinc 修正技术，在粗糙网格下仍能保持高精度（RMSE < 0.2 dB）。
    *   **Strip Mesh (条带网格)**: 针对旋转体极点（如球顶）的简并几何进行特殊拓扑优化，消除物理奇点。
    *   **CAD 支持**: 集成 `PythonOCC`，直接解析 **STEP (.stp)** 格式工业模型。
*   **🧮 算法库**:
    *   **Discrete PO (Dual Sinc)**: 旗舰算法，双向相位误差修正，鲁棒性最强。
    *   **PTD 修正**: 支持指定边缘的物理绕射电流修正 (Michaeli/Ufimtsev 公式)。
    *   **Analytic-Ribbon**: 1995 年经典解析算法复现（学术对照）。

## 🛠️ 环境搭建

本项目依赖 Python 3.9+。推荐使用 Conda 管理环境，因为 `pythonocc-core` 包含二进制依赖。

### 1. 创建环境

```bash
conda create -n cem python=3.13 -y
conda activate cem
```

### 2. 安装依赖

```bash
# 基础科学计算
pip install numpy scipy matplotlib pandas

# GPU 加速 (可选，需 NVIDIA 显卡)
pip install cupy-cuda12x  # 请根据您的 CUDA 版本选择 (如 cuda11x)

# Qt GUI & 3D 可视化 (新版 GUI 必需)
pip install PySide6 pyvista pyvistaqt

# CAD 内核 (OpenCascade)
conda install -c conda-forge pythonocc-core=7.9.0
```

## 🚀 运行指南

### 启动专业版 (Qt GUI) —— **推荐**

提供更流畅的 3D 体验和完整功能：

```bash
python gui_qt.py
```

### 启动标准版 (Tkinter GUI)

轻量级启动：

```bash
python gui.py
```

## 📂 项目结构

```
CEM_PO/
├── gui_qt.py               # [New] 基于 PySide6 的主程序入口
├── gui.py                  # [Legacy] 基于 Tkinter 的旧版入口
├── core/                   # 核心架构层
│   ├── solver_bridge.py    # UI 与算法的解耦桥接器
│   └── mesh_data.py        # 网格数据结构定义
├── solvers/                # 电磁算法实现
│   ├── api.py              # 算法工厂
│   ├── po.py               # 离散 PO 积分器 (CPU/GPU)
│   ├── ptd.py              # 物理绕射理论核心
│   └── rcs_analyzer.py     # RCS 扫描控制器
├── geometry/               # 几何建模
│   ├── factory.py          # [New] 几何工厂模式
│   ├── step_loader.py      # STEP 文件加载器
│   └── occ_surface.py      # OCC 曲面封装
├── gui_managers/           # (旧版) Tkinter 的逻辑管理器
└── results/                # 计算结果输出目录
```

## 📅 更新日志 (Changelog)

### v1.3.0 (Dev) - Qt Architecture Overhaul
*   **GUI 重构**: 引入 PySide6 + PyVistaQt，彻底解决 Tkinter 在大模型下的渲染卡顿问题。
*   **架构解耦**: 实现 `SolverBridge` 中间层，分离 UI 逻辑与计算核心，支持多线程任务管理。
*   **交互升级**: 3D 视图支持显示法线、PTD 边缘、入射波方向；图表支持多文件拖入对比。

### v1.2.1 (2026-01-30)
*   **精度提升**: 引入 Per-cell Sinc 步长修正，显著降低网格离散误差。
*   **网格优化**: 为 `DiscretePOIntegrator` 添加 `min_points` 参数，防止极细长面元导致的数值不稳定。

### v1.2 (2026-01-29)
*   **新特性**: 支持退化曲面（Degenerate Surface）的条带网格生成与缓存优化。
*   **性能**: 修复了 Strip Mesh 重复计算的 Bug，实现了真正的网格缓存复用。

## 🧠 核心算法简介

### 1. Discrete PO with Dual Sinc
传统的离散 PO 假设面元上相位线性变化，这在曲面上会引入 $\text{sinc}$ 形式的误差。
本项目不仅在积分中引入 $\text{sinc}(k \cdot \Delta r)$ 修正，还针对非均匀网格（如条带网格）计算每个面元独立的 $\Delta u / \Delta v$ 步长，从而在极低网格密度下也能获得高精度 RCS。

### 2. Strip Mesh (条带网格)
针对球体、纺锤体等旋转体，传统的矩形网格在极点会退化为点，导致法线奇异和积分发散。
Strip Mesh 技术通过在极点附近自适应调整网格拓扑，结合简并点处理，保证了全角度的物理正确性。

## 🤝 贡献

欢迎通过 Issue 提交 Bug 或建议。对于新功能开发，请基于 `Qt-dev` 分支提交 PR。

## 📄 许可证

MIT License
