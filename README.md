# 高频物理光学 (PO) 电磁散射求解器 (原型)

## 项目简介

本项目是一个基于 Python 的高频电磁散射求解器原型，采用物理光学 (Physical Optics, PO) 近似方法。
其核心算法实现了“Ribbon Method”（带状积分法），通过将二维面积分转化为一维数值求和与一维解析积分（Sinc 函数），提高了计算效率。

当前版本（Phase 1）主要关注算法的正确性验证，已实现了针对解析圆柱体的单站 RCS (雷达散射截面) 计算，并与解析解进行了对比验证。

## 环境搭建

本项目依赖 Python 3.9+ 以及 numpy, scipy, matplotlib 等科学计算库。建议使用 Conda 进行环境管理。

### 1. 创建 Conda 环境

```bash
conda create -n cem python -y
conda activate cem
```

### 2. 安装依赖

```bash
pip install numpy scipy matplotlib
```

## 快速开始

项目包含一个主要的验证脚本 `main.py`，用于计算圆柱体的单站 RCS 并与理论公式进行对比。

### 运行验证

```bash
python main.py
```

运行成功后，控制台将输出主瓣附近的误差（应小于 0.5 dB），并生成一张对比图 `rcs_verification.png`。

## 项目结构

```
F:\data\CEM_PO\
├── geometry/           # 几何定义层
│   ├── surface.py      # Surface 抽象基类
│   └── cylinder.py     # 解析圆柱体实现
├── physics/            # 物理定义层
│   ├── constants.py    # 物理常数 (光速, 阻抗等)
│   └── wave.py         # 入射波定义
├── solver/             # 求解器核心层
│   └── ribbon_solver.py # Ribbon 积分器与 RCS 分析器
├── main.py             # 验证脚本入口
├── instruction.txt     # 项目需求说明
└── README.md           # 项目说明文档
```

## 核心算法说明

求解器使用了 **Ribbon Method**：
1.  **几何离散**：将曲面沿 $v$ 方向切分为多个带状区域 (ribbons)。
2.  **相位近似**：在每个带状区域内，假设相位沿 $u$ 方向线性变化。
3.  **解析积分**：利用 Sinc 函数解析计算 $u$ 方向的积分贡献。
4.  **数值求和**：将所有带状区域的贡献累加，得到总散射场。

公式：
$$ I \approx \sum_{ribbons} (\hat{n} \cdot \hat{k}_{inc}) \cdot e^{j \Phi_0} \cdot L_u L_v \cdot \text{sinc}\left(\frac{\alpha L_u}{2\pi}\right) $$

## 扩展指南

要添加新的几何形状（例如球体或 NURBS 曲面）：
1.  继承 `geometry.surface.Surface` 类。
2.  实现 `evaluate(u, v)`, `get_normal(u, v)`, 和 `get_jacobian(u, v)` 方法。
3.  确保这些方法支持 numpy 的向量化操作。

```