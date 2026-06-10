# PTD 边缘积分预因子推导

本文档记录 `ptd_core.py` 和 `freq_sweep.py` 中 PTD 边缘积分预因子的完整推导过程。

---

## 1. 代码的 RCS 约定

### 1.1 时谐因子与传播方向

- 时间谐和约定：`exp(-jωt)`（Ufimtsev 约定）
- 外向传播球面波：`exp(+jkR)/R`
- 入射平面波在 r' 处：`exp(+jk·k̂·r')`（k̂ 为入射方向单位向量）

### 1.2 PO 面积分定义

代码中 PO 积分（`solvers/po.py`）计算的标量积分量为：

```
I_PO = ∫∫_S (-n̂·k̂) · exp(+j·2k·k̂·r') dS
```

其中 `exp(+j·2k·k̂·r')` 是 Ufimtsev `exp(-jωt)` 约定下的单站往返相位：
- 入射场在 r' 处贡献 `exp(+jk·k̂·r')`
- 反向散射场从 r' 出发贡献 `exp(+jk·k̂·r')`
- 合计：`exp(+j·2k·k̂·r')`

**这不是复共轭，而是 Ufimtsev 约定的自然结果。** PTD 积分使用完全相同的相位约定。

### 1.3 散射场与 I 的关系

```
E_s / E_i = (jk / 2πR) · exp(+jkR) · I
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

其中 `exp(+jkρ)` 是 `exp(-jωt)` 约定下的外向柱面波，不需要做共轭变换。

### 2.2 各因子含义

- `ρ`：截面平面内观察点到棱边的距离
- `f₁`（soft/TE）或 `g₁`（hard/TM）：Ufimtsev 无量纲 fringe 系数
- `exp(+jkρ)`：外向柱面波传播因子（exp(-jωt) 约定）
- `exp(+jπ/4)`：鞍点渐近展开产生的固有相位
- `1/√(2πkρ)`：柱面扩展因子

---

## 3. 2D → 3D 转换（稳相法 SPA 反演）

### 3.1 问题描述

对有限长直棱边（长度 L，沿 ŷ 方向），需要找到每单位长度的 3D 等效源
幅度 A，使其产生的远场经 SPA 积分后恰好给出已知的 2D fringe 场。

### 3.2 斜入射 SPA

棱边沿 ẑ 方向，入射方向与棱边夹角为 γ₀。
入射场在棱边点 z' 处引入相位梯度 `exp(+jk·cosγ₀·z')`。

3D 线源在截面平面（⊥ẑ）内距离 ρ 处产生的场：

```
u(ρ) = ∫_{-∞}^{∞} A · exp(+jk·cosγ₀·z') · exp(+jkR) / R · dz'
```

其中 `R = √(ρ² + z'²)`。

SPA 分析：

```
φ(z') = k·cosγ₀·z' + k√(ρ² + z'²)

φ'(z') = k·cosγ₀ + k·z'/R = 0
→ z₀ = -ρ·cosγ₀/sinγ₀

R₀ = ρ/sinγ₀
φ(z₀) = k·ρ·sinγ₀

φ''(z₀) = k·sin³γ₀/ρ    ← 关键：含 sin³γ₀ 因子！
```

**注**：正入射 (γ₀=π/2) 时 φ''= k/ρ；斜入射时 φ''= k·sin³γ₀/ρ。

SPA 结果（φ'' > 0 → exp(+jπ/4)）：

```
u(ρ) = A · (sinγ₀/ρ) · exp(+jkρsinγ₀) · √(2πρ/(k·sin³γ₀)) · exp(+jπ/4)
     = A · √(2π/(kρsinγ₀)) · exp(+jkρsinγ₀ + jπ/4)
```

### 3.3 匹配：提取 3D 源幅度

2D fringe 场使用截面波数 k⊥ = k·sinγ₀：

```
u^(1)(ρ) = f₁ · exp(+jk⊥ρ + jπ/4) / √(2πk⊥ρ)
         = f₁ · exp(+jkρsinγ₀ + jπ/4) / √(2πkρsinγ₀)
```

令 SPA 结果 = 2D fringe 场：

```
A · √(2π/(kρsinγ₀)) = f₁ / √(2πkρsinγ₀)
```

两边的 `exp(+jkρsinγ₀)` 和 `exp(+jπ/4)` 以及 `1/√(kρsinγ₀)` **完全对消**。剩下：

```
A · √(2π) = f₁ / √(2π)
```

即：

```
A = f₁ / (2π)
```

**结论**：3D 等效源幅度 A = D₂D/(2π)。

注意：sinγ₀ 对 φ'' 的影响恰好被 SPA 的 √(1/φ'') 和 1/R₀ 项完全吸收，
最终 A 与 γ₀ 无关。

---

## 4. 3D PTD 边缘积分方程

### 4.1 每个 3D 边缘微元的散射场

边缘微元 dζ 在远场产生的散射场：

```
dE_s = E_i · A · (exp(+jkR) / R) · exp(+j·2k·k̂·r') · dζ
     = E_i · [D₂D / (2π)] · (exp(+jkR) / R) · exp(+j·2k·k̂·r') · dζ
```

### 4.2 映射到代码积分量 I

代码约定：

```
E_s / E_i = (jk / 2πR) · exp(+jkR) · I
```

消去公共传播因子 `exp(+jkR)/R`：

```
(jk / 2π) · dI = [D₂D / (2π)] · exp(+j·2k·k̂·r') · dζ
```

解出：

```
dI = [2π / (jk)] · [D₂D / (2π)] · exp(+j·2k·k̂·r') · dζ
   = [-j/k] · D₂D · exp(+j·2k·k̂·r') · dζ
```

### 4.3 引入 3D 衍射系数

代码使用 `FG_monostatic` 返回的 3D 衍射系数 D₃D = D₂D/sinγ₀
（例如 VV 极化时 D₃D = g₁/sinγ₀）。重写：

```
dI = [-j/k] · D₂D · exp(+j·phase) · dζ
   = [-j·sinγ₀/k] · [D₂D/sinγ₀] · exp(+j·phase) · dζ
   = [-j·sinγ₀/k] · D₃D · exp(+j·phase) · dζ
```

因此：

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│              -j·sinγ₀                                       │
│  dI  =  ─────────────  · D₃D · exp(+j·phase) · dζ         │
│               k                                             │
│                                                             │
│  pre_factor = -j·sinγ₀/k                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 量纲校验

`pre_factor = -j·sinγ₀/k` 的量纲为 `1/k → m`。
乘以 `dζ`（m）后，`dI_PTD` 量纲为 `m²`，与 `I_PO` 对齐。 ✓

### 4.5 符号验证

pre_factor 中的 `-j` 来自 `2π/(jk) = -j·2π/k` 中 `1/j = -j` 的转换。
通过有限条形板（strip）单边缘 PO+PTD 干涉验证：

- `-j` 符号：PO+PTD 在旁瓣处表现为平滑减小（物理正确）
- `+j` 符号：PO+PTD 在旁瓣处异常增大（物理不正确）

验证脚本：`verify_ptd_sign.py`

---

## 5. 代码实现

### 5.1 对应代码

`physics/ptd_core.py`：

```python
pre_factor = -1j * sin_gamma0 / k
```

`core/freq_sweep.py`（频率扫描向量化版本）：

```python
pre_arr = -1j * sin_gamma0 / k_arr_xp
```

### 5.2 边缘段积分结构

对每个边缘段（segment），总贡献为：

```python
seg_contrib = pre_factor * D * seg.length * sinc_val * exp(+j * phase_mid)
```

其中：
- `D`：衍射系数，由 `FG_monostatic()` 返回 (F1_Vt, G1_Vt, G1_phi)，按极化组合
  - `D = F1_Vt·cos²χ + G1_phi·sin²χ + G1_Vt·sinχ·cosχ`
- `seg.length`：段长度 dζ
- `sinc_val`：`sinc(k·L·cos(γ₀)/π)` 棱边延伸方向干涉
- `exp(+j * phase_mid)`：`exp(+j·2k·k̂·r_mid)` 单站往返相位（Ufimtsev 约定）

---

## 6. RCS 验证

### 6.1 解析验证（正入射）

正入射棱边（γ₀ = π/2, sinγ₀ = 1），边长 L，VV 极化：

```
|pre_factor|² = sin²γ₀/k² = 1/k²
D₃D = g₁/sinγ₀ = g₁

σ = (k²/π) · |pre_factor|² · |D₃D|² · L²
  = (k²/π) · (1/k²) · |g₁|² · L²
  = |g₁|²L²/π
  = |D₂D|²L²/π    ✓
```

### 6.2 一般 γ₀ 验证

一般斜入射角 γ₀，VV 极化：

```
|pre_factor|² = sin²γ₀/k²
D₃D = g₁/sinγ₀

σ = (k²/π) · (sin²γ₀/k²) · |g₁/sinγ₀|² · L²
  = (sin²γ₀/π) · |g₁|²/sin²γ₀ · L²
  = |g₁|²L²/π = |D₂D|²L²/π
```

sinγ₀ 在 pre_factor 与 D₃D 之间完全对消。
最终 RCS 仅通过 g₁ 的角度依赖来反映几何关系。

**数值验证**：`verify_ptd_prefactor.py` 确认在 γ₀ = 30°~90° 范围内，
代码公式与解析公式的相对误差为 0%。

### 6.3 从 Keller GTD 独立推导

Keller 2D 衍射系数 D_K（有量纲）与 Ufimtsev 无量纲 D₂D 的关系：

```
D_K = D₂D · exp(+jπ/4) / √(2πk)     （量纲: √m）
```

3D → 2D SPA 给出等效源 = D_K · √(k/(2π)) · exp(-jπ/4)

3D 远场: u_s = D_K · √(k/(2π)) · exp(-jπ/4) · L · exp(+jkR)/R

3D RCS:
```
σ = 4πR² |u_s/u_i|²
  = 4π · |D_K|² · k/(2π) · L²
  = 2k · L² · |D_K|²
  = 2k · L² · |D₂D|²/(2πk)
  = |D₂D|²L²/π                       ✓
```

两种推导给出完全一致的结果。

---

## 7. 旧版本的错误分析

### 7.1 旧预因子（v1：含多余幅度和相位）

```
pre_factor_old = √(2π/k) · exp(-jπ/4) / (k · sinγ₀)
```

错误根源：SPA 反演时未正确对消 2D 场中的 `exp(+jπ/4)/√(2πkρ)` 因子。

### 7.2 旧预因子（v2：本文档早期版本推导的错误公式）

```
pre_factor_doc = +j / (k · sinγ₀)
```

此公式有两处错误：

**错误 1：SPA 中 φ'' 遗漏 sin³γ₀**

本文档早期版本将 φ''(z₀) 错误地写为 k/ρ。
正确值为 k·sin³γ₀/ρ（见 §3.2）。
此误差在 γ₀=π/2 时消失（sin³(π/2)=1），但导致斜入射时：

- 文档推导出 A = D₂D/(2πsinγ₀)（错误），正确为 A = D₂D/(2π)
- 预因子得到 j/(k·sinγ₀)（错误），正确为 -j·sinγ₀/k

幅度偏差 = sin²γ₀ 倍：

| γ₀ | 偏差 (dB) |
|----|----------|
| 90° | 0 |
| 60° | -2.5 |
| 45° | -6.0 |
| 30° | -12.0 |

**错误 2：不必要的共轭变换**

早期版本认为代码 PO 使用 `exp(+j·2k·k̂·r')` 是物理积分的共轭，
因此对 PTD 预因子做了共轭（`-j → +j`）。
实际上 `exp(+j·2k·k̂·r')` 就是 Ufimtsev exp(-jωt) 约定下的自然相位，
PO 和 PTD 使用完全相同的约定，无需共轭。

### 7.3 正确公式与错误公式对比

| 项目 | 正确（当前代码） | 错误（文档 v2） |
|------|-----------------|----------------|
| pre_factor | `-j·sinγ₀/k` | `+j/(k·sinγ₀)` |
| |pre|² (γ₀=π/2) | 1/k² | 1/k² |
| |pre|² (γ₀=45°) | 1/(2k²) | 2/k² |
| σ (γ₀=π/2) | |D₂D|²L²/π ✓ | |D₂D|²L²/π ✓ |
| σ (γ₀=45°) | |D₂D|²L²/π ✓ | 4·|D₂D|²L²/π ✗ |

---

## 8. 补充说明

### 8.1 相位对 RCS 的影响

预因子的相位影响 PO+PTD 总和的干涉模式：

```
σ_total = (k²/π) |I_po + I_ptd|²
```

`-j` 相位使 PTD 在 PO 旁瓣处产生部分相消，平滑 RCS 曲线——
这是 PTD 修正的预期物理行为。

### 8.2 一阶 PTD 的局限性

对于正方体等复杂几何体，一阶 PTD 不包含：
- 顶点衍射（vertex diffraction）
- 边缘-边缘多次衍射（edge-edge interaction）
- 爬行波（creeping waves，仅对曲面体有关）

这些高阶效应是 PO+PTD 与全波算法之间残余误差的主要来源，
与预因子公式无关。

### 8.3 参考来源

- Ufimtsev, P.Ya., "Fundamentals of the Physical Theory of Diffraction", 2nd ed., Wiley, 2014
- Ufimtsev, P.Ya., "Theory of Edge Diffraction in Electromagnetics", 2nd ed., SciTech, 2014
- Keller, J.B., "Geometrical Theory of Diffraction", JOSA, 1962
- MATLAB 源码 `main_fringe.m` 第 32 行：`coeff = exp(1i*(kr+pi/4))/sqrt(2*pi*kr)`
- MATLAB 源码 `FG.m`：3D 矢量衍射系数（Eq. 7.137）
- 验证脚本 `verify_ptd_prefactor.py`：pre_factor 幅度验证
- 验证脚本 `verify_ptd_sign.py`：pre_factor 符号验证
