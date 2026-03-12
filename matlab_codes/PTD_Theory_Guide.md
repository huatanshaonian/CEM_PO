# 物理衍射理论 (PTD) 公式体系技术指南

这份文档系统地梳理了物理衍射理论（Physical Theory of Diffraction, PTD）的核心数学框架，旨在帮助理解从 Ufimtsev 的基础理论到三维电磁仿真的演进过程。

---

## 1. 核心思想：场的分层叠加 (Field Decomposition)

在 PTD 框架下，任何观察点处的总电磁场 $\vec{E}_{total}$ 被分解为两个部分：

$$\vec{E}_{total} = \vec{E}_{PO} + \vec{E}_{fringe}$$

*   **$\vec{E}_{PO}$ (Physical Optics)**：物理光学场。基于感应电流 $\vec{J} = 2\hat{n} 	imes \vec{H}^i$ 在物体表面产生的贡献。它能很好地处理大型平滑表面，但在边缘处会出现不连续的“阶跃”错误。
*   **$\vec{E}_{fringe}$ (Fringe Field)**：边缘修正场（或称边缘波）。这是 PTD 的精髓，用于抵消 PO 在边缘处产生的错误并补全真实的衍射贡献。

---

## 2. 衍射系数层：Ufimtsev 系数 ($f^{(1)}$ 与 $g^{(1)}$)

这是代码中 `fun_fg.m` 的数学基础。对于一个无限长楔形，其修正系数定义为：

### 标量形式 (2D 垂直入射)
对于两种基本极化，修正衍射系数 $f^{(1)}$ (Soft/TM) 和 $g^{(1)}$ (Hard/TE) 定义为：

$$f^{(1)} = f_{GTD} - f_{PO}$$
$$g^{(1)} = g_{GTD} - g_{PO}$$

其中，$f_{GTD}$ 是经典的几何衍射系数（Sommerfeld 精确解的渐近形式），而 $f_{PO}$ 是物理光学在边缘处产生的伪贡献。

### Ufimtsev 核心公式 (对应 `fun_fg.m`)
公式主要由一系列余切函数组成：
$$f,g = \frac{1}{n} \left[ \frac{\sin(\frac{\pi}{n})}{\cos(\frac{\pi}{n}) - \cos(\frac{\psi - \phi_0}{n})} \mp \frac{\sin(\frac{\pi}{n})}{\cos(\frac{\pi}{n}) - \cos(\frac{\psi + \phi_0}{n})} ight]$$
*   $n = \alpha/\pi$ 是楔角参数（$\alpha$ 为外楔角）。
*   $\phi_0$ 为入射角，$\psi$ 为观察角。

---

## 3. 矢量扩展层：初等边缘波 (EEW)

在三维空间和斜入射（对应 `electromagnetic_EEW_Section78.m`）情况下，标量系数需要扩展为矢量分量。

### 斜入射几何 (Oblique Incidence)
引入斜入射角 $\gamma_0$（入射波与棱边的夹角）。此时，衍射场分布在一个以棱边为轴线的 **衍射锥**（Keller Cone）上。

### 3.1 衍射系数分量 (对应 `FG.m`)
在球坐标系 $(\vartheta, \phi)$ 下，散射场被分解为：
*   **$F^{(1)}$**：描述 $E_\vartheta$ 分量的修正系数。
*   **$G^{(1)}$**：描述 $E_\phi$ 分量的修正系数。

### 3.2 传播算子 (Propagation Factor)
最终的边缘场强通过下式计算：
$$E_{fringe} \approx \frac{e^{j(kr + \pi/4)}}{\sqrt{2\pi kr} \sin \gamma_0} \cdot [	ext{Diffraction Coefficients}]$$

---

## 4. 复杂几何集成：增量长度衍射 (ILDC)

要将 PTD 应用于复杂几何体（如曲边缘或复杂零件），需要将无限长边缘公式转化为 **积分形式**：

### 4.1 边缘微元化
将边缘切割成长度为 $dl$ 的微元，每一段产生一个微小的电场贡献 $dE_{fringe}$。

### 4.2 边缘积分 (Edge Integral)
对于一条有限长度（或弯曲）的边缘 $C$，总衍射场为：
$$\vec{E}_{fringe} = \int_{C} \vec{f}_{ILDC}(l) \cdot \frac{e^{jkr(l)}}{r(l)} dl$$

*   **$\vec{f}_{ILDC}$**：增量长度衍射系数。它使得 PTD 可以处理观察点不在衍射锥上的情况（Off-cone calculation）。

---

## 5. PTD 计算标准流程

1.  **几何离散化**：将 3D 模型识别为 **棱边 (Edges)** 和 **面板 (Facets)**。
2.  **PO 项计算**：对所有可见面板进行面积分，得到 $\vec{E}_{PO}$。
3.  **局部坐标变换**：为每一条棱边建立局部坐标系（$z$ 轴沿切线）。
4.  **系数解算**：调用 `fun_fg` 计算 Ufimtsev 修正系数。
5.  **矢量场叠加**：对所有棱边进行线积分（对应 `FG.m` 的逻辑）。
6.  **总场合成**：将 PO 场与 Fringe 场矢量相加，取模得到 RCS。

---

## 6. PTD 的意义

*   **修正性**：解决了物理光学（PO）在边缘处不连续、不准确的问题。
*   **连续性**：消除了几何衍射理论（GTD）在阴影边界处的无穷大奇异值。
*   **高效性**：作为高频近似算法，其计算速度远超矩量法（MoM）等全波算法，是大型目标（如隐身飞机）仿真的首选。

---
*参考资料：Pyotr Ufimtsev, "Fundamentals of the Physical Theory of Diffraction".*
