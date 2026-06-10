# 截断 MEC (`michaeli_mec_truncated`) 实现说明

Johansen 1996 截断 PTD 等效边电流在本仓库的代码实现详解。覆盖文件、数据流、所有计算公式、奇点处理、时谐约定一致性这五块。

## 1. 算法目标

Michaeli 1986 非截断 EEC (`michaeli_mec`) 在所谓 **Ufimtsev 奇点** (`μ + cosφ_0 = 0`) 处发散。物理上这些奇点出现在 Keller 锥反射边界附近，对应有限边长 strip 的尾部"虚源"贡献被错误外推到无穷远。

Johansen 1996 (IEEE TAP vol.44 no.7) 给出**修正项** `M_cor, I_cor`，把虚源尾部从非截断闭式中减掉，得到对有限边长 strip / wedge 物理正确的截断 EEC：

$$\boxed{\; I_f^T = I_f^{UT} - I_f^{cor}, \quad M_f^T = M_f^{UT} - M_f^{cor} \;}$$

`I_f^{cor}` 和 `M_f^{cor}` 各自发散，但 Johansen §III.B 证明二者在奇点处发散项**精确同型**，差分后有限。

## 2. 文件分工

| 文件 | 作用 | 是否接入 |
|---|---|---|
| `physics/mec_truncated_core.py` | PTD 主入口（同 `mec_core` 接口）；逐 segment 累加 | 是 |
| `physics/mec_truncated_coefficients.py` | **N=2 闭式**（半平面 Eq.26/27，面 A+B 合并） | 是 |
| `physics/mec_truncated_general_coefficients.py` | **通用 N 闭式**（Eq.21/22，Face A only + B 镜像） | **否**（已实现但 core 未调用）|
| `physics/mec_truncated_geometry.py` | 截断长度 `l_A` 几何辅助（ray-cast） | 是 |
| `physics/uniform_transition.py` | 改进 Fresnel `F(x)` | 是 |
| `solvers/ptd.py:73-75` | 在边提取后预算 `seg.l_A` | 是 |
| `physics/ptd_algorithms.py` | 注册算法 ID `michaeli_mec_truncated` | 是 |

## 3. 主入口 `compute_mec_truncated_contribution(edge, wave, polarization)`

文件 `physics/mec_truncated_core.py`，函数签名与 `mec_core.compute_mec_contribution` 完全一致 → 是 PTD 算法注册表 (`ptd_algorithms.PTD_ALGORITHMS['michaeli_mec_truncated']`) 的平替项。

### 3.1 接口

```python
def compute_mec_truncated_contribution(edge, wave, polarization='VV') -> complex
```

- `edge`：`PTDEdge` 对象，每个 `segment` 需有 `.l_A` 属性
- `wave`：`IncidentWave`，提供 `k_dir, k_vector, k, theta_hat, phi_hat`
- `polarization`：`VV / HH / VH / HV` （后两者支持交叉极化）

返回值是该边对单站散射场积分量 `I` 的复贡献。

### 3.2 极化基

```python
_POL_BASIS = {
    'VV': ('theta_hat', 'theta_hat'),
    'HH': ('phi_hat',   'phi_hat'),
    'VH': ('theta_hat', 'phi_hat'),
    'HV': ('phi_hat',   'theta_hat'),
}
```

`e_t = getattr(wave, et_name)`（发射极化），`e_r`（接收极化）。

### 3.3 逐 segment 流程

| 步骤 | 公式 / 行为 | 行号 |
|---|---|---|
| 1. 局部三轴 `(x̂₁, ŷ₁, t̂)` | `ŷ₁ = (n̂_lit − (n̂·t̂)t̂)/‖…‖`，`x̂₁ = ŷ₁×t̂`；用 `seg.inward` 矫正 x̂₁ 朝向面内 | 70–95 |
| 2. Michaeli 角 `(β′, φ′)` | `cos β′ = -k̂·t̂`；`s_⊥ = -k̂ − (-k̂·t̂)t̂`；`φ′ = atan2(s_⊥·ŷ₁, s_⊥·x̂₁)`，绕到 `[0, 2π)` | 97–110 |
| 3. 阴影判定 | `N = α/π`；若 `φ′ > Nπ` 跳过（不照亮该面对） | 112–114 |
| 4. 入射切向激励 | `E_z0 = ê_t·t̂`，`H_z0 = (k̂×ê_t)·t̂/Z` | 117–118 |
| 5. UT 部分 + conj | `If_UT = conj(If_m), Mf_UT = conj(Mf_m)`，调 `mec_coefficients.compute_total_fringe_currents` | 122–125 |
| 6. cor 部分 + conj | 调 `mec_truncated_coefficients.compute_correction_currents`；`l_A` 为 None/inf/<1e-12 时退化为 0 | 134–145 |
| 7. T = UT − cor | `If_T = If_UT − I_cor`，`Mf_T = Mf_UT − M_cor` | 148–149 |
| 8. 接收侧投影 | `amp = −Z·If_T·(t̂·ê_r) + Mf_T·((ŝ×t̂)·ê_r)`，`ŝ = -k̂` | 151–156 |
| 9. 段积分累加 | `contrib = 0.5·L·sinc(k·L·k̂·t̂/π)·exp(2j·k_vec·mid)·amp` | 159–165 |

**关键细节**：
- 单站化简体现在 `compute_correction_currents` 调用处显式传 `β_obs = π−β′, φ_obs = φ′`（行 139–140），把双站接口收成单站
- 装配前因子 `0.5` 与 `mec_core` 同，conj 电流后的 Ufimtsev e^{-iωt} 约定下的固定值
- 退化判据：`sin γ_0 < 1e-3` 或 `‖ŷ₁‖ < 1e-10` 或 `‖x̂₁‖ < 1e-10` 或 `‖s_⊥‖ < 1e-10`，遇到就 `continue`

## 4. N=2 修正电流 `compute_correction_currents`

文件 `physics/mec_truncated_coefficients.py`。**实现范围**：半平面 (N=2, 外角 α=2π)，薄板刀刃边/strip，**面 A+B 已合并**。

### 4.1 核心参量

$$L = k\, l_A\, \sin^2\beta_0$$

$$\mu = \frac{\sin\beta\,\sin\beta_0\,\cos\varphi + \cos\beta_0(\cos\beta - \cos\beta_0)}{\sin^2\beta_0}\quad \text{(Michaeli Eq.6)}$$

$$F_1 = F\!\bigl(\sqrt{2L}\,\bigl|\cos\tfrac{\varphi_0}{2}\bigr|\bigr),\qquad F_2 = F\!\bigl(\sqrt{L(1-\mu)}\bigr)$$

### 4.2 Eq.26 — `M_cor`

$$M_{cor} = \frac{4ZH_{z0}\sin\varphi\, e^{jL(\mu-1)}}{jk\sin\beta\sin\beta_0(\mu+\cos\varphi_0)} \Bigl\{ -\text{sgn}\bigl(\cos\tfrac{\varphi_0}{2}\bigr) F_1 + \frac{\sqrt{2}\cos\frac{\varphi_0}{2}}{\sqrt{1-\mu}}\, F_2 \Bigr\}$$

### 4.3 Eq.27 — `I_cor = I_E + I_H`

$$I_{E} = \frac{4E_{z0}\sin\frac{\varphi_0}{2}\,e^{jL(\mu-1)}}{jkZ\sin^2\beta_0(\mu+\cos\varphi_0)} \Bigl\{ 2\cos\tfrac{\varphi_0}{2}\,F_1 - \sqrt{2(1-\mu)}\,F_2 \Bigr\}$$

$$I_{H} = \frac{4H_{z0}\,e^{jL(\mu-1)}}{jk\sin\beta_0(\mu+\cos\varphi_0)} \Bigl\{ -\text{sgn}(\cos\tfrac{\varphi_0}{2})(\cot\beta_0\cos\varphi_0 + \cot\beta\cos\varphi) F_1 + \tfrac{\sqrt{2}\cos\frac{\varphi_0}{2}}{\sqrt{1-\mu}}(\cot\beta\cos\varphi - \mu\cot\beta_0) F_2 \Bigr\}$$

### 4.4 数值边界

| 情形 | 处理 | 物理意义 |
|---|---|---|
| `sin β_0 < 1e-12` | 返 `(0, 0)` | 入射方向与边几乎平行，β_0 → 0 |
| `|μ + cosφ_0| < _DENOM_EPS (1e-9)` | 返 `(0, 0)`，让上层退回 `M_UT` | Ufimtsev 奇点；M_UT 和 M_cor 各自发散，强行计算会产生 NaN |
| `|1 − μ| < _MU1_EPS (1e-9)` | 返 `(0, 0)` | Keller 锥反射边界，系数有限极限但数值上分母太小 |
| `μ > 1` | `np.lib.scimath.sqrt(1-μ)` → 纯虚；`modified_fresnel` 接复参数延拓 | Keller 锥外部 |
| `l_A → 0` | `L → 0, F(0)=0.5 → M_cor → M_UT`，故 `M_T = I_T = 0` | 零长度边无电流，物理正确 |

奇点处选择"返 0 让 M_T = M_UT 退回"是 first-pass 策略；Johansen §III.B 用 Taylor 展开给出严格有限极限，后续如需要可在此处加 ε 展开。

## 5. 通用 N 修正电流（待接入）

文件 `physics/mec_truncated_general_coefficients.py`。实现范围：**任意外角参数 N**（论文假设 `1 < N ≤ 2`），任意双站，**Face A only**（Face B 由调用方坐标替换得到）。

### 5.1 接口

```python
compute_correction_face_A_general(β_0, φ_0, β, φ, N, l_A, E_z0, H_z0, k, Z)
    → (M_cor^A, I_cor^A)                         # 带 ±SING_OFFSET 平均

compute_total_correction_general(β_0, φ_0, β, φ, N, l_A, l_B, E_z0, H_z0, k, Z)
    → (M_cor, I_cor)                             # Face A + Face B

compute_total_correction_general_raw(...)        # 跳过 ±SING_OFFSET 平均
```

Face B 通过 Johansen 替换得到：

$$\{ \beta_0 \to \pi - \beta_0,\ \beta \to \pi - \beta,\ \varphi_0 \to N\pi - \varphi_0,\ \varphi \to N\pi - \varphi,\ l^A \to l^B \}$$

### 5.2 Eq.21 — `M_cor^A`

$$M_{cor}^A = \frac{2ZH_{z0}\sin\varphi\, e^{jL(\mu-1)}}{jk\sin\beta\sin\beta_0}\Bigl[ \tfrac{-\text{sgn}(\cos\frac{\varphi_0}{2})}{\mu+\cos\varphi_0}\, F_1 + \Bigl(\tfrac{\sqrt{1-\mu}}{\sqrt{2}(\mu+\cos\varphi_0)\cos\frac{\varphi_0}{2}} - \tfrac{\sqrt{2}\sin\frac{\pi}{N}}{N\sqrt{1-\mu}\bigl(\cos\frac{\pi}{N}-\cos\frac{\varphi_0}{N}\bigr)} \Bigr)\, F_2 \Bigr]$$

### 5.3 Eq.22 — `I_cor^A`

$$I_{cor}^A = \frac{2 e^{jL(\mu-1)}}{jk\sin\beta_0(\mu+\cos\varphi_0)}\bigl[\text{sgn}(\cos\tfrac{\varphi_0}{2})\,B_1\,F_1 + \sqrt{2(1-\mu)}\,B_2\,F_2\bigr]$$

$$B_1 = \tfrac{E_{z0}\sin\varphi_0}{Z\sin\beta_0} - H_{z0}(\cot\beta_0\cos\varphi_0+\cot\beta\cos\varphi)$$

$$B_2 = -\tfrac{E_{z0}\sin\frac{\varphi_0}{2}}{Z\sin\beta_0} + \tfrac{H_{z0}(\cot\beta_0\cos\varphi_0+\cot\beta\cos\varphi)}{2\cos\frac{\varphi_0}{2}} + \tfrac{H_{z0}\sin\frac{\pi}{N}(\mu+\cos\varphi_0)(\cot\beta_0-\cot\beta\cos\varphi)}{N(\cos\frac{\pi}{N}-\cos\frac{\varphi_0}{N})(1-\mu)}$$

N=2 化简：`sin(π/N)=1, cos(π/N)=0`，代入 Face A+B 求和后即得 Eq.26/27。

### 5.4 三类奇点（`_is_singular`）

与 `mec_coefficients._is_singular_bistatic` 同口径：

1. **Ufimtsev**：`|μ + cosφ_0| < 1e-9`
2. **Keller 锥反射**：`|1 − μ| < 1e-9`
3. **sub-Keller**：`|cos(π/N) − cos(φ_0/N)| < 1e-9`

### 5.5 奇点处理 — `±_SING_OFFSET` 平均

`_SING_OFFSET = 3e-5`（与 `mec_coefficients._SING_OFFSET` 一致，**关键约束**）。

奇点处 wrapper 沿 `φ` 做 `±_SING_OFFSET` 偏移并取均值：

```python
Mc_m, Ic_m = _raw(..., φ - _SING_OFFSET, ...)
Mc_p, Ic_p = _raw(..., φ + _SING_OFFSET, ...)
return 0.5 * (Mc_m + Mc_p), 0.5 * (Ic_m + Ic_p)
```

因为 `M_UT` 在同一 `_SING_OFFSET` 下用同样方式平滑，两侧 `1/(μ+cosφ_0)` 反号 → 平均后各自发散项 → 0，二者有限残量在 `M_T = M_UT − M_cor` 中正确相减。

### 5.6 硬保护

平移后仍可能踩 0（例如 `dμ/dφ=0` 处偏移不能避开），用 `_SAFE = 1e-15` 给安全分母避免 NaN：

```python
if abs(one_minus_mu) < _SAFE: one_minus_mu = _SAFE if ≥0 else -_SAFE
if abs(denom_a)     < _SAFE: denom_a     = _SAFE if ≥0 else -_SAFE
if abs(denom_e)     < _SAFE: denom_e     = _SAFE if ≥0 else -_SAFE
if abs(cos_half)    < _SAFE: return (0, 0)   # φ_0=π 是真物理奇点不能 escape
```

## 6. 截断长度 `compute_truncation_length`

文件 `physics/mec_truncated_geometry.py`。

### 6.1 简化前提

当前实现假定 **β_0 = π/2**（入射 ⊥ 棱边）。此时 Keller 锥退化为 ⊥t̂ 平面，`û^A = seg.inward`（亮面内、由边指向面内，由 PTD 边提取器 `ptd_edge_finder` 算出）。

适用范围：**主平面单站后向散射**（如平板 φ=0 任意 θ）。对斜入射 / 双站，严格 `û^A` 应在 face A 内、与 t̂ 成 β_0 角度旋转向 -k_dir 投影；此版本未实现，留接口 `method='geodesic'` 给 NURBS 曲面（Task #13 远期）。

### 6.2 `ray_cast` 算法

```python
def _ray_cast_l_A(seg, all_edges):
    u_A = seg.inward / ‖seg.inward‖
    origin = seg.midpoint
    min_dist = +∞
    for other_seg in all_edges (跳过 seg 自己):
        dist = _ray_segment_intersection(origin, u_A, other_seg)
        if 0 < dist < min_dist: min_dist = dist
    return min_dist  # 找不到返 +∞
```

`_ray_segment_intersection` 解 3D 射线 (`origin + s·dir, s≥0`) 与线段 `start→end` 的相交：

1. 计算 `n = direction × target_dir_unit`，`n_norm < 1e-12` → 平行，返 -1
2. 共面性：`|delta · n_unit| > _INTERSECTION_EPS (1e-9)` → 出平面，返 -1
3. 解 2×2 线性方程组 `s·direction − t·target_dir = delta`
4. 验收：`s > _RAY_FORWARD_EPS (1e-9)` 且 `0 ≤ t ≤ target_len`

### 6.3 预计算时机

`solvers/ptd.py:73-75`，在 `PTDProcessor.extract_edges_from_face_pairs` 边提取后立即：

```python
for edge in ptd_edges:
    for seg in edge.segments:
        seg.l_A = compute_truncation_length(seg, None, ptd_edges)
```

仅几何，与 `k_dir` 无关 → 一次性预算。其它 PTD 算法不消费 `l_A` 但也不报错（`mec_core` 不查这个属性）。

## 7. 改进 Fresnel `modified_fresnel`

文件 `physics/uniform_transition.py`。Pauli-Clemmow 鞍点-极点合并通用工具，不止 MEC 截断专用，也出现在 UTD (Kouyoumjian-Pathak 1974) 和 strip 掠射过渡 (Tiberio-Kouyoumjian 1982) 中。

### 7.1 定义

$$F(x) = \sqrt{\tfrac{j}{\pi}}\, e^{jx^2}\int_x^\infty e^{-jt^2}\,dt$$

### 7.2 数值实现

```python
def modified_fresnel(x):
    if |x| ≤ 10:
        F(x) = 0.5 · exp(j·x²) · erfc(x · exp(jπ/4))    # 精确，scipy.special.erfc 接复参数
    else:
        F(x) ≈ 1 / (2x · √(jπ))                          # Michaeli Eq.40 渐近
```

阈值 `_ASYMPTOTIC_THRESHOLD = 10`（论文建议 3），因为渐近 O(1/x³) 修正在 `x=3` 时 ~7% 相对误差，`x=10` 时 <0.1%。

### 7.3 特殊值

- `F(0) = 0.5` (erfc(0)=1)
- `F(x → ∞) → 0`
- `x=10` 处精确与渐近一致（<0.1% 相对误差）

### 7.4 备用 `modified_fresnel_uf`

文件同时提供 e^{-iωt} 版本，实参等于 `conj(F(x))`。**当前未被 core 调用** —— core 走"原版 F + 整支 conj"路径。

## 8. 时谐约定全栈一致性

这是 truncated 实现里最容易翻车的地方，专门强调。

| 模块 | 时谐约定 | conj 责任 |
|---|---|---|
| `mec_coefficients.compute_total_fringe_currents` | e^{+jωt} (Michaeli 原文) | 调用方 conj |
| `mec_truncated_coefficients.compute_correction_currents` | e^{+jωt} (Johansen 原文，含 `exp(+jL(μ-1))`、原版 F) | 调用方 conj |
| `mec_truncated_general_coefficients.*` | e^{+jωt} (Johansen 原文) | 调用方 conj |
| `mec_truncated_core` 调用层 | e^{-iωt} (Ufimtsev / 本仓库) | 对 UT 和 cor 都整支 `np.conj` |

```python
If_UT_m, Mf_UT_m = compute_total_fringe_currents(...)
If_UT = np.conj(If_UT_m)
Mf_UT = np.conj(Mf_UT_m)

M_cor_m, I_cor_m = compute_correction_currents(...)
M_cor = np.conj(M_cor_m)
I_cor = np.conj(I_cor_m)

If_T = If_UT - I_cor
Mf_T = Mf_UT - M_cor
```

**关键不变量**：UT 和 cor 必须用**同一原始约定**、**同一复分支**求值、**同一 conj 翻转**。这样 Ufimtsev 奇点 (`μ→1, φ_0→π`) 处两个发散项才会精确抵消（Johansen §III.B）。

旧的 `_consistent` 半翻译版（相位翻、代数 j 不翻）已被实测否决（见 `git log` 中 c52c0cf 之前的清理）。

## 9. 装配总览

```
PTDProcessor.extract_edges_from_face_pairs
    ├─ find_shared_edges (自适应采样)
    ├─ 每个 segment 预存 .l_A
    └─ PTD_ALGORITHMS['michaeli_mec_truncated'].func = compute_mec_truncated_contribution

compute_mec_truncated_contribution(edge, wave, pol)
  for seg in edge.segments:
    ├─ 局部三轴 (x̂₁, ŷ₁, t̂)
    ├─ (β', φ', N, E_z0, H_z0)
    ├─ If_UT, Mf_UT = conj(compute_total_fringe_currents)      ← mec_coefficients
    ├─ If_cor, Mf_cor = conj(compute_correction_currents)       ← N=2 闭式
    ├─ If_T = If_UT - If_cor;  Mf_T = Mf_UT - Mf_cor
    └─ contrib = 0.5·L·sinc(k·L·k̂·t̂/π)·exp(2j·k_vec·mid)
                 · (-Z·If_T·(t̂·ê_r) + Mf_T·((ŝ×t̂)·ê_r))
  return Σ contrib
```

## 10. 已知限制与未接入项

1. **核心走 N=2 路径**：`mec_truncated_core` 调 `compute_correction_currents`（半平面）。要支持任意外角 Wedge 截断需要替换为 `compute_total_correction_general` 并补 `l_B` 计算（当前 `seg.l_A` 只有一个，缺 Face B 的 ray-cast）
2. **斜入射 / 双站 l_A**：当前 ray-cast 在 β_0=π/2 简化下进行，斜入射严格 `û^A` 未实现
3. **NURBS 曲面 `geodesic` 方法**：`method='geodesic'` stub 抛 `NotImplementedError`，未来用 `potpourri3d` 离散测地线
4. **奇点 ε 展开**：N=2 路径在三类奇点处直接返 0 让上层退回非截断；通用 N 路径用 `±_SING_OFFSET` 平均。Johansen §III.B 严格 Taylor 展开有限极限均未实现
5. **GUI 接入**：`gui_qt.py` 写死 `algorithm='ufimtsev_eew'`，用户不能从下拉切到截断 MEC（CLI/JSON 配置可以）

## 参考文献

- P. M. Johansen, "Uniform Physical Theory of Diffraction Equivalent Edge Currents for Truncated Wedge Strips," *IEEE TAP* vol.44 no.7, July 1996, pp.989–995.
- A. Michaeli, "Equivalent Edge Currents for Arbitrary Aspects of Observation," *IEEE TAP* vol.AP-34 no.12, Dec 1986.
- A. Michaeli, "Elimination of Infinities in Equivalent Edge Currents, Part I/II," *IEEE TAP* 1986/1987.
- L. B. Felsen, N. Marcuvitz, *Radiation and Scattering of Waves*, 1973, ch.4 (Pauli-Clemmow).
