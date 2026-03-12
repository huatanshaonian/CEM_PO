# PTD (Physical Theory of Diffraction) MATLAB 代码说明文档

本仓库包含了源自《Fundamentals of the Physical Theory of Diffraction》附录中的 MATLAB 仿真程序。这些工具主要用于研究无限长楔形、圆锥、抛物面等几何体的边缘衍射效应，并对比 **物理光学 (PO)** 与 **物理衍射理论 (PTD)** 的计算结果。

---

## 1. 文件列表及详细说明

### 核心仿真脚本 (Directly Runnable)
| 文件名 | 功能描述 | 核心原理/公式 |
| :--- | :--- | :--- |
| **`main_fringe.m`** | **主对比程序**：计算并对比无限长楔形的**精确边缘场**（数值积分）与 **PTD 渐近场**。生成极坐标方向图，验证 PTD 在边界附近的准确性。 | 公式 4.16-4.21 |
| **`Fig_610.m`** | 计算**圆锥体 (Cone)** 的轴向后向散射截面 (RCS) 随长度 $kl$ 变化的曲线，对比 PO、PTD Soft/Hard 条件。 | 第 6 章仿真 |
| **`Fig_614.m`** | 计算**共形抛物面 (Paraboloid)** 的轴向后向散射曲线。 | 第 6 章仿真 |
| **`directivity_patternEEW_Section77.m`** | 计算并绘制**声学初等边缘波 (Acoustic EEW)** 的方向图。支持用户输入楔角和入射角。 | 第 7.7 节 |
| **`electromagnetic_EEW_Section78.m`** | 计算并绘制**电磁初等边缘波 (Electromagnetic EEW)** 的矢量分量方向图。 | 第 7.8 节 |

### 计算函数 (Dependency Functions)
| 文件名 | 功能描述 | 被调用关系 |
| :--- | :--- | :--- |
| **`fun_fg.m`** | 计算 Ufimtsev 渐近衍射系数 $f^{(1)}$ 和 $g^{(1)}$，处理了阴影/反射边界的奇异性。 | `main_fringe.m`, `FG.m`, `Fsh.m` |
| **`Int_calcFringe.m`** | 实现边缘场积分的**数值计算**（精确解），作为渐近公式的参考基准。 | `main_fringe.m` |
| **`Fsh.m`** | 计算声学初等边缘波的衍射函数 $F_s^{(1)}$ 和 $F_h^{(1)}$。 | `directivity_patternEEW_Section77.m` |
| **`FG.m`** | 计算电磁初等边缘波的矢量衍射系数分量 $F^{(1)}$ 和 $G^{(1)}$。 | `electromagnetic_EEW_Section78.m` |
| **`SCSn_general.m`** | 通用散射截面计算函数，支持抛物面 (Paraboloid) 和球冠 (Spherical) 形状。 | `Fig_614.m` |
| **`sigma12.m`** | 计算用于边缘波定义的中间复变量 $\sigma_{1,2}$。 | `FG.m`, `Fsh.m` |
| **`eps_x.m`** | 阶跃辅助函数 $\epsilon(x)$，用于处理照明区与阴影区的逻辑切换。 | `FG.m`, `Fsh.m` |

---

## 2. 操作指南

### 运行环境
*   **软件要求**：MATLAB R2014a 或更高版本。
*   **路径设置**：请确保所有 `.m` 文件位于同一个文件夹内。

### 执行步骤
1.  **验证 PTD 理论**：
    *   运行 `main_fringe.m`。
    *   程序会计算精确解（Exact）与渐近解（Asymptotic）的对比，并弹出两个极坐标图（Soft/Hard 边界条件）。
2.  **观察几何体 RCS**：
    *   直接运行 `Fig_610.m` (圆锥) 或 `Fig_614.m` (抛物面)。
    *   图形窗口将展示 PO 与 PTD 随频率/尺寸变化的修正效果。
3.  **自定义参数计算**：
    *   运行 `directivity_patternEEW_Section77.m`。
    *   **注意**：程序会提示输入参数（如楔角 `wedge angle`、入射角 `incident angle` 等）。请输入**角度制 (Degrees)**，例如输入 `300` 代表 $300^\circ$ 的外楔角（即内楔角为 $60^\circ$）。

### 输入参数说明
*   **Wedge angle (alfa)**：通常在 $\pi < \alpha \le 2\pi$ 之间。
*   **Incident angle (angle0)**：入射波方向，需小于楔角。
*   **Observation angle**：观察点位置。

---

## 3. 常见问题 (FAQ)
*   **为什么运行速度较慢？**
    `main_fringe.m` 调用了 `Int_calcFringe.m` 进行数值积分，为了保证精度积分步长较细，计算“精确场”可能需要几秒到十几秒时间。
*   **结果图中显示 "Inf" 或奇异点？**
    PTD 的核心价值在于消除了传统几何衍射理论（GTD）在阴影边界的奇异性，但如果观察角直接设在边界点，数值计算仍可能存在极小的震荡，这是正常现象。
