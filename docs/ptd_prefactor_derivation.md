# PTD 边缘积分预因子推导

本文档记录 `ptd_core.py` 和 `freq_sweep.py` 中 PTD 边缘积分预因子的完整推导过程。

---

## 1. 代码的 RCS 约定

### 1.1 时谐因子与传播方向

- 时间谐和约定：`exp(+jωt)`
- 外向传播球面波：`exp(-jkR)/R`

### 1.2 PO 面积分定义

代码中 PO 积分（`solvers/po.py`）计算的标量积分量为：

```
I_PO = ∫∫_S (-n̂·k̂) · exp(+j·2k·k̂·r') dS
```

**注意**：物理上单站往返相位为 `exp(-j·2k·k̂·r')`，代码取正号是因为
计算了物理积分的复共轭 `I_code = I_phys*`。由于 RCS 只取模方，
`|I_code|² = |I_phys|²`，结果正确。

### 1.3 散射场与 I 的关系

```
E_s / E_i = (jk / 2πR) · exp(-jkR) · I
```

### 1.4 RCS 公式

```
σ = 4πR² |E_s/E_i|² = (k²/π) |I|²
```

**验证**：平板面积 A，法向入射时 `I_PO = A`：

```
σ = k²A²/π = 4πA²/λ²  ✓  （标准 PO 平板公式）
```

### 1.5 量纲约束

`I_PO` 的量纲为面积 (m²)。为使 `I_total = I_PO + I_PTD` 合法，
`I_PTD` 的量纲也必须为面积 (m²)。

---

## 2. Ufimtsev 2D fringe 渐近场

### 2.1 原始公式（exp(-jωt) 约定）

来自 Ufimtsev 教材 "Fundamentals of the Physical Theory of Diffraction"，
MATLAB 源码 `main_fringe.m` 第 32 行确认：

```
u^(1)(ρ) = f₁ · exp(j(kρ + π/4)) / √(2πkρ)
```

### 2.2 转换到 exp(+jωt) 约定

对整个表达式取复共轭（f₁ 为实数，f₁* = f₁）：

```
u^(1)(ρ) = f₁ · exp(-j(kρ + π/4)) / √(2πkρ)
          = f₁ · exp(-jkρ) · exp(-jπ/4) / √(2πkρ)
```

其中：
- `ρ`：截面平面内观察点到棱边的距离
- `f₁`（soft/TE）或 `g₁`（hard/TM）：Ufimtsev 无量纲 fringe 系数
- `exp(-jkρ)`：外向柱面波传播因子（exp(+jωt) 约定下）
- `exp(-jπ/4)`：鞍点渐近展开产生的固有相位
- `1/√(2πkρ)`：柱面扩展因子

---

## 3. 2D → 3D 转换（稳相法 SPA 反演）

### 3.1 问题描述

对有限长直棱边（长度 L，沿 ŷ 方向），需要找到每单位长度的 3D 等效源
幅度 A，使其产生的远场经 SPA 积分后恰好给出已知的 2D 场。

### 3.2 3D 线源的 SPA 积分

沿 z 轴分布的均匀线源 A，在远场投影到 2D 平面时：

```
E^{3D}(ρ) = ∫_{-∞}^{∞} A · exp(-jkR)/R · dz,  R = √(ρ² + z²)
```

稳相点 z₀ = 0，相位 φ(z) = -kR：
- φ(0) = -kρ
- φ''(0) = -k/ρ < 0

SPA 结果：

```
E^{2D}(ρ) = A · √(2π/(kρ)) · exp(-jkρ) · exp(-jπ/4)
```

（φ'' < 0 时 SPA 给出 exp(-jπ/4) 相位因子。）

### 3.3 匹配：提取 3D 源幅度

令 SPA 结果 = Ufimtsev 2D 场：

```
A · √(2π/(kρ)) · exp(-jkρ) · exp(-jπ/4) = f₁ · exp(-jkρ) · exp(-jπ/4) / √(2πkρ)
```

**关键**：两边的 `exp(-jkρ)` 和 `exp(-jπ/4)` **完全对消**。剩下：

```
A · √(2π/(kρ)) = f₁ / √(2πkρ)
```

即：

```
A = f₁ / (2π)
```

**结论**：3D 等效源幅度 A 中不含 √k，也不含残余相位。
分母恰好是 2π。

---

## 4. 3D PTD 边缘积分方程

### 4.1 每个 3D 边缘微元的散射场

引入斜入射角 γ₀ 修正：

```
dE_s = E_i · [D / (2π sinγ₀)] · (exp(-jkR) / R) · exp(-j·phase_rt) · dζ
```

其中 `phase_rt = 2k·k̂·r'` 为单站往返相位。

### 4.2 映射到代码积分量 I

代码约定：

```
E_s / E_i = (jk / 2πR) · exp(-jkR) · I
```

消去公共传播因子 `exp(-jkR)/R`：

```
(jk / 2π) · dI = [D / (2π sinγ₀)] · exp(-j·phase_rt) · dζ
```

解出：

```
dI_physical = [1/(jk sinγ₀)] · D · exp(-j·phase_rt) · dζ
            = [-j/(k sinγ₀)] · D · exp(-j·phase_rt) · dζ
```

### 4.3 量纲校验

`pre_factor = j/(k sinγ₀)` 的量纲为 `1/k → m`。
乘以 `dζ`（m）后，`dI_PTD` 量纲为 `m²`，与 `I_PO` 完美对齐。 ✓

---

## 5. 代码相位共轭对齐

### 5.1 PO 的相位共轭

代码 PO 积分使用 `exp(+j·2k·k̂·r')`（正号），即物理积分的复共轭：

```
I_PO_code = I_PO_phys*
```

### 5.2 PTD 必须采用相同共轭

为使 `I_total = I_PO + I_PTD` 产生正确的交叉项干涉：

```
|I_PO_code + I_PTD_code|² = |I_PO_phys* + I_PTD_phys*|²
                           = |I_PO_phys + I_PTD_phys|²
```

所以 PTD 预因子也取共轭：

```
pre_factor_code = (-j / (k sinγ₀))* = +j / (k sinγ₀)
```

### 5.3 最终结果

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                     j                                       │
│  dI  =  ─────────────────  · D · exp(+j·phase) · dζ        │
│          k · sin(γ₀)                                       │
│                                                             │
│  pre_factor = j / (k · sinγ₀)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 代码实现

### 6.1 对应代码

`physics/ptd_core.py`：

```python
pre_factor = 1j / (k * sin_gamma0)
```

`core/freq_sweep.py`（频率扫描向量化版本）：

```python
pre_arr = 1j / (k_arr_xp * sin_gamma0)
```

### 6.2 边缘段积分结构

对每个边缘段（segment），总贡献为：

```python
seg_contrib = pre_factor * D * seg.length * sinc_val * exp(j * phase_mid)
```

其中：
- `D`：衍射系数，由 `FG_monostatic()` 计算（VV: `-G1_phi + G1_Vt`，HH: `-F1_Vt`）
- `seg.length`：段长度 dζ
- `sinc_val`：`sinc(k·L·cos(γ₀)/π)` 棱边延伸方向干涉
- `exp(j * phase_mid)`：`exp(+j·2k·k̂·r_mid)` 单站往返相位（共轭约定）

---

## 7. RCS 验证

### 7.1 解析验证

正入射棱边（γ₀ = π/2, sinγ₀ = 1），边长 L：

```
|pre_factor|² = 1/k²

σ = (k²/π) · |pre_factor|² · |D|² · L²
  = (k²/π) · (1/k²) · |D|² · L²
  = |D|²L²/π
```

### 7.2 从 Keller GTD 独立推导

Keller 2D 衍射系数 D_K（有量纲）与 Ufimtsev 无量纲 D 的关系：

```
D_K = D · exp(-jπ/4) / √(2πk)     （量纲: √m）
```

3D → 2D SPA 给出等效源 = D_K · √(k/(2π)) · exp(jπ/4)

3D 远场: u_s = D_K · √(k/(2π)) · exp(jπ/4) · L · exp(-jkR)/R

3D RCS:
```
σ = 4πR² |u_s/u_i|²
  = 4π · |D_K|² · k/(2π) · L²
  = 2k · L² · |D_K|²
  = 2k · L² · |D|²/(2πk)
  = |D|²L²/π                       ✓
```

两种推导给出完全一致的结果。

### 7.3 斜入射情况

一般 γ₀：

```
σ = |D|² L² / (π sin²γ₀)
```

---

## 8. 旧版本的错误分析

### 8.1 旧预因子

```
pre_factor_old = √(2π/k) · exp(-jπ/4) / (k · sinγ₀)
              = √(2π) · exp(-jπ/4) / (k^{3/2} · sinγ₀)
```

### 8.2 错误根源

旧推导在 SPA 反演时未正确对消 Ufimtsev 2D 场中的
`exp(-jπ/4)/√(2πkρ)` 与 SPA 给出的 `√(2π/(kρ))·exp(-jπ/4)`，
导致残留了 `√(2π/k)` 幅度因子和 `exp(-jπ/4)` 相位。

### 8.3 错误量级

旧预因子幅度 = `√(2π)/(k^{3/2}·sinγ₀)`
正确预因子幅度 = `1/(k·sinγ₀)`

比值 = `√(2π/k) = √λ`

- σ 偏差 = λ 倍（线性域）= `10·log10(λ)` dB
- λ = 1 m 时无误差（巧合）
- λ = 0.1 m (3 GHz): PTD σ 偏大 10 dB
- λ = 0.03 m (10 GHz): PTD σ 偏大 15 dB

### 8.4 为何旧验证通过

旧版同时使用了错误的解析公式 `σ = 2L²|D|²/k`，
而正确公式为 `σ = |D|²L²/π`。两者之比恰好也是 λ，
与错误的预因子互相补偿，导致 "RMSE = 0.000 dB" 的假象。

| 项目 | 旧（错误） | 新（正确） |
|------|-----------|-----------|
| pre_factor | `√(2π/k)·exp(-jπ/4)/(k·sinγ₀)` | `j/(k·sinγ₀)` |
| \|pre\|² (γ₀=π/2) | `2π/k³` | `1/k²` |
| σ 公式 | `2\|D\|²L²/k` | `\|D\|²L²/π` |
| σ 比值 | 2π/k = λ | 1 |

---

## 9. 补充说明

### 9.1 相位对 RCS 的影响

预因子的相位影响 PO+PTD 总和的干涉模式：

```
σ_total = (k²/π) |I_po + I_ptd|²
```

旧相位 `exp(-jπ/4)` vs 新相位 `j = exp(+jπ/2)` 相差 `exp(j3π/4)`，
即 135°。这改变了 PO-PTD 交叉干涉项 `2·Re(I_po · I_ptd*)`。

### 9.2 一阶 PTD 的局限性

对于正方体等复杂几何体，一阶 PTD 不包含：
- 顶点衍射（vertex diffraction）
- 边缘-边缘多次衍射（edge-edge interaction）
- 爬行波（creeping waves，仅对曲面体有关）

这些高阶效应是 PO+PTD 与全波算法之间残余误差的主要来源，
与预因子公式无关。

### 9.3 参考来源

- Ufimtsev, P.Ya., "Fundamentals of the Physical Theory of Diffraction", 2nd ed., Wiley, 2014
- Ufimtsev, P.Ya., "Theory of Edge Diffraction in Electromagnetics", 2nd ed., SciTech, 2014
- Keller, J.B., "Geometrical Theory of Diffraction", JOSA, 1962
- MATLAB 源码 `main_fringe.m` 第 32 行：`coeff = exp(1i*(kr+pi/4))/sqrt(2*pi*kr)`
- MATLAB 源码 `FG.m`：3D 矢量衍射系数（Eq. 7.137）
