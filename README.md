# 电磁散射 PO 求解器 (CEM PO Solver)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 项目简介

**CEM PO Solver** 是一个基于物理光学 (Physical Optics, PO) 近似的高频电磁散射求解器。它提供了一个现代化的图形用户界面 (GUI)，专为工程验证和学术研究设计。

本项目实现了多种高频积分算法，包括传统的离散 PO 以及基于 1995 年经典论文实现的 **Ribbon Method**（带状积分法）及其改进版本。

### ✨ 核心特性

*   **🖥️ 现代化 GUI**: 基于 Tkinter 的图形界面，支持实时日志、参数配置和结果可视化。
*   **📐 CAD 支持**: 集成 `PythonOCC`，支持直接导入 **STEP (.stp/.step)** 格式的复杂 CAD 模型。
*   **🧮 多种核心算法**:
    *   **Discrete PO (Dual Sinc)**: 推荐算法。结合双向 Sinc 相位校正的离散积分，精度高且鲁棒性最强。
    *   **Gauss-Ribbon**: 自适应分段 Gauss 积分，适用于光滑曲面的高精度求解。
    *   **Analytic-Ribbon**: 严格基于多项式拟合的解析算法 (CADDSCAT, 1995)，具有精确的阴影边界检测（学术研究用）。
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
conda create -n cem python=3.13 -y
conda activate cem
```

### 2. 安装依赖

```bash
# 科学计算基础
pip install numpy scipy matplotlib pandas

# CAD 内核 (OpenCascade)
conda install -c conda-forge pythonocc-core=7.9.0

# 单元测试 (可选)
pip install pytest
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
2.  **选择算法**:
    *   推荐使用默认的 **Discrete PO (Dual Sinc)**，它在大多数情况下提供最佳的速度与精度平衡。
    *   对于非有理双三次曲面 (Non-rational Bi-cubic) 的学术研究，可尝试 **Analytic-Ribbon**。
3.  **选择几何**:
    *   选择 "STEP File" 并加载您的 `.stp` 模型。
    *   点击 "预览全部 (Preview)" 检查模型导入是否正确及法线方向（红色箭头应指向外部）。
4.  **运行计算**:
    *   勾选 "并行计算 (Parallel)" 以加速。
    *   点击 "计算 RCS (Run 1D/2D)"。
5.  **查看结果**:
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
│   ├── ribbon_solver.py    # 积分算法工厂与并行计算引擎
│   ├── ribbon_analytic.py  # 解析/Gauss 积分核心实现
│   └── ribbon_polynomials.py # 多项式拟合与阴影边界检测
├── ui/
│   └── plotting.py         # Matplotlib 嵌入式绘图与可视化管理
├── tests/
│   └── test_algorithms.py  # 算法精度对比测试脚本
└── results/                # 计算结果默认保存目录
```

## 🧠 算法原理对比

本项目包含三类主要的积分策略，各有优劣：

1.  **Discrete PO (Dual Sinc)**:
    *   **原理**: 将曲面离散为微小平面元，并在 U/V 两个方向都应用 $\text{sinc}(k \Delta x)$ 因子来修正线性相位误差。
    *   **优势**: **鲁棒性最强**。V 方向的 Sinc 修正使其在处理多维曲率（如球体）时比 Ribbon 方法更准。
    *   **劣势**: 需要较密的网格。

2.  **Gauss-Ribbon**:
    *   **原理**: 将曲面沿 V 方向切条。U 方向采用**自适应分段 Gauss 积分**，V 方向离散。
    *   **优势**: U 方向精度极高，适合单向曲率物体（如圆柱）。
    *   **劣势**: V 方向缺乏相位修正，处理双向曲率物体时不如 Dual Sinc。

3.  **Analytic-Ribbon**:
    *   **原理**: 严格遵循 1995 年 CADDSCAT 论文。使用 5 阶多项式拟合几何，通过求根精确确定阴影边界。
    *   **优势**: 理论上的“解析”解，阴影边界精度极高 ($10^{-6}$)。
    *   **限制**: 仅适用于 Bi-cubic 曲面。对于非多项式曲面（如球体、NURBS），多项式拟合误差会导致精度下降。

## 🤝 贡献

欢迎提交 Issue 或 Pull Request 来改进本项目！

## 📄 许可证

本项目基于 MIT 许可证开源。详见 LICENSE 文件。