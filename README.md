# 电磁散射 PO 求解器 (CEM PO Solver)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 项目简介

**CEM PO Solver** 是一个基于物理光学 (Physical Optics, PO) 近似的高频电磁散射求解器。它提供了一个现代化的图形用户界面 (GUI)，专为工程验证和学术研究设计。

本项目采用 **Ribbon Method**（带状积分法）作为核心算法，将二维面积分转化为一维解析积分（Sinc 函数）与一维数值求和，在保证精度的同时大幅提高了计算效率。

### ✨ 核心特性

*   **🖥️ 现代化 GUI**: 基于 Tkinter 的图形界面，支持实时日志、参数配置和结果可视化。
*   **📐 CAD 支持**: 集成 `PythonOCC`，支持直接导入 **STEP (.stp/.step)** 格式的复杂 CAD 模型。
*   **🚀 并行计算**: 内置多进程并行加速引擎，利用多核 CPU 快速完成海量角度的 RCS 扫描。
*   **📊 交互式可视化**:
    *   **3D 预览**: 交互式查看 STEP 模型网格、法线方向及入射波示意图。
    *   **2D 扫描**: 支持生成高质量的 RCS 热力图 (Heatmap) 和等高线图。
    *   **数据对比**: 内置对比工具，可加载 CSV/参考数据计算 RMSE 误差并生成差异图。
*   **💾 数据持久化**: 自动保存计算结果为 CSV，支持从界面直接加载历史数据。

## 🛠️ 环境搭建

本项目依赖 Python 3.9+ 以及科学计算和 CAD 处理库。推荐使用 Conda 管理环境。

### 1. 创建环境

```bash
conda create -n cem python=3.9 -y
conda activate cem
```

### 2. 安装依赖

请确保安装了以下核心库：

```bash
# 科学计算基础
pip install numpy scipy matplotlib pandas

# CAD 内核 (OpenCascade)
conda install -c conda-forge pythonocc-core

# GUI 增强 (可选)
pip install sv_ttk
```

> **注意**: `pythonocc-core` 必须通过 Conda 安装，因为它包含大量二进制依赖。

## 🚀 快速开始

### 启动 GUI

项目的入口是 `gui.py`。在终端中运行：

```bash
python gui.py
```

### 使用流程

1.  **配置参数**: 在左侧面板设置频率 (Frequency)、网格密度 (Mesh Density) 和扫描角度范围。
2.  **选择几何**:
    *   选择 "STEP File" 并加载您的 `.stp` 模型。
    *   点击 "预览全部 (Preview)" 检查模型导入是否正确及法线方向（红色箭头应指向外部）。
    *   *可选*: 使用 "反转法线索引" 修复法线反向的面。
3.  **运行计算**:
    *   勾选 "并行计算 (Parallel)" 以加速。
    *   点击 "计算 RCS (Run 1D/2D)"。
4.  **查看结果**:
    *   计算完成后，结果将自动在 "RCS 结果" 标签页显示。
    *   切换到 "对比 (Comparison)" 标签页，加载参考数据进行精度验证。

## 📂 项目结构

```
CEM_PO/
├── gui.py                  # [核心] GUI 入口与主控制器
├── geometry/
│   ├── step_loader.py      # STEP 文件解析与网格化 (基于 PythonOCC)
│   ├── occ_surface.py      # OpenCascade 曲面封装
│   └── surface.py          # 几何基类
├── solver/
│   └── ribbon_solver.py    # Ribbon Method 核心算法与并行计算引擎
├── ui/
│   └── plotting.py         # Matplotlib 嵌入式绘图与可视化管理
├── read-compare/
│   ├── rcs_data_reader.py  # 参考数据读取与清洗
│   └── rcs_visual.py       # 独立的可视化脚本
├── physics/
├── cem_po_config.json      # 用户配置持久化文件 (自动生成)
└── results/                # 计算结果默认保存目录
```

## 🧠 算法原理

求解器基于 **Ribbon Method**：

1.  **几何离散**: 利用 OpenCascade 的参数化能力，将任意 CAD 曲面沿 $v$ 方向切分为细长的带状区域。
2.  **相位线性化**: 在每个带状微元内，假设相位沿 $u$ 方向线性变化。
3.  **解析积分**: 沿 $u$ 方向的积分被解析为 Sinc 函数形式：
    $$ I_u \propto L_u \cdot \text{sinc}\left(\frac{k \cdot L_u \cdot \partial \Phi / \partial u}{2\pi}\right) $$
4.  **并行求和**:
    *   **预计算**: 主进程将几何数据转化为纯 NumPy 数组 (`CachedMeshData`)。
    *   **分发**: 通过 `ProcessPoolExecutor` 将角度任务分发至各 CPU 核心。
    *   **归约**: 汇总各核心计算的复数散射场，得到总 RCS。

## 🤝 贡献

欢迎提交 Issue 或 Pull Request 来改进本项目！

## 📄 许可证

本项目基于 MIT 许可证开源。详见 LICENSE 文件。
