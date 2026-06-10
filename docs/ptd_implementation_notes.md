# PTD 实现技术文档

本文档完整说明本项目中 PTD（Physical Theory of Diffraction）边缘修正的每一步实现细节，
包括所有关键公式、符号约定和代码对应关系，供审核和讨论。

---

## 1. 总体架构

### 1.1 RCS 计算公式

单站（monostatic）RCS 的最终公式：

$$
\sigma = \frac{k^2}{\pi} \left| I_{\text{PO}} + I_{\text{PTD}} \right|^2
$$

其中：
- $k = 2\pi / \lambda$ 为波数
- $I_{\text{PO}}$ 为 PO（物理光学）面积分
- $I_{\text{PTD}} = \sum_{\text{edges}} I_{\text{edge}}$ 为所有边缘的 PTD 修正之和

代码位置：`solvers/rcs_analyzer.py:27`
```python
sigma = (k_mag**2 / np.pi) * np.abs(val)**2
```

### 1.2 PO 积分

$$
I_{\text{PO}} = \iint_{\text{lit}} \cos\theta_i \cdot e^{j 2k \hat{k} \cdot \mathbf{r}'} \, dS
$$

其中：
- $\cos\theta_i = \max(-\hat{n} \cdot \hat{k}, 0)$ 为入射余弦（仅照亮面）
- $\hat{k} = $ `k_dir`：入射方向单位向量（从雷达指向目标）
- $\mathbf{r}'$ 为面元位置
- **注意：被积函数中无因子 2**

代码位置：`solvers/po.py:207, 239-243`
```python
illumination_factor = xp.maximum(-n_dot_k, 0.0)       # cos(theta_i)
phase_local = 2.0 * xp.sum((points - ref_point) * k_vec_xp, axis=-1)  # 2k·k_dir·r'
contributions = illumination_factor * weights * xp.exp(1j * phase_local) * sinc_factor
```

#### 关于因子 2 的说明

PO 面电流 $\mathbf{J}_{\text{PO}} = 2(\hat{n} \times \mathbf{H}_i)$ 中含因子 2。
推导远场时，这个因子 2 被吸收进了 $k^2/\pi$ 的系数中：

$$
\sigma = \frac{k^2}{\pi} |I|^2 \quad \text{等价于} \quad \sigma = \frac{k^2}{4\pi} |2I|^2
$$

两种写法完全等价。本项目使用前者（$k^2/\pi$，被积函数无因子 2）。

---

## 2. PTD 边缘修正

### 2.1 单条边缘的贡献

每条边缘被分成 $N$ 个线段，对每个线段 $m$ 求和：

$$
I_{\text{edge}} = \sum_{m=1}^{N} \underbrace{\left( \frac{-j \sin\gamma_0}{k} \right)}_{\text{pre\_factor}} \cdot D_m \cdot L_m \cdot \text{sinc}\!\left(\frac{k L_m \cos\gamma_0}{\pi}\right) \cdot e^{j \, 2k \hat{k} \cdot \mathbf{r}_{\text{mid},m}}
$$

代码位置：`physics/ptd_core.py:156-158`
```python
pre_factor = -1j * sin_gamma0 / k
seg_contrib = pre_factor * D * seg.length * sinc_val * np.exp(1j * phase_mid)
```

各符号含义：

| 符号 | 含义 | 代码 |
|------|------|------|
| $\gamma_0$ | 入射方向与棱边的夹角（$0$时平行，$\pi/2$时垂直） | `gamma0 = arcsin(sin_gamma0)` |
| $\sin\gamma_0$ | $= \sqrt{1 - (\hat{k}\cdot\hat{t})^2}$ | `sin_gamma0 = sqrt(1 - k_dot_t**2)` |
| $D_m$ | 组合衍射系数（见 §2.3） | `D` |
| $L_m$ | 线段长度 | `seg.length` |
| $\text{sinc}(x)$ | $= \sin(\pi x)/(\pi x)$，即 `np.sinc` | `np.sinc(sinc_arg)` |
| $\cos\gamma_0$ | $= \hat{k}\cdot\hat{t}$ | `k_dot_t` |
| $\mathbf{r}_{\text{mid},m}$ | 线段中点 | `seg.midpoint` |

#### 关于 pre_factor 的推导

Ufimtsev EEW（Eq. 7.137）给出边缘衍射对远场散射振幅的贡献。
PO 和 PTD 共享相同的远场归一化（都乘以 $jk/(2\pi r) \cdot e^{jkr} |E_i|$），因此：

$$
I_{\text{total}} = I_{\text{PO}} + I_{\text{edge}}, \qquad \sigma = \frac{k^2}{\pi} |I_{\text{total}}|^2
$$

EEW 的边缘积分直接给出 $I_{\text{edge}}$，pre_factor 为 $-j\sin\gamma_0/k$。

**已确认**：pre_factor 无需 $/2$。SPA 反演推导和数值验证均确认
$\text{pre\_factor} = -j\sin\gamma_0/k$（详见 `docs/ptd_prefactor_derivation.md`）。

---

### 2.2 楔形截面局部坐标系

对每个边缘线段，在垂直于棱边切线 $\hat{t}$ 的截面平面内，建立二维坐标系 $(e_1, e_2)$：

**Step 1**: 将 Face A 法向量 $\hat{n}_{\text{lit}}$ 投影到 $\perp\hat{t}$ 平面：
$$
\hat{e}_1 = \text{normalize}\!\left(\hat{n}_{\text{lit}} - (\hat{n}_{\text{lit}}\cdot\hat{t})\hat{t}\right)
$$

**Step 2**:
$$
\hat{e}_2 = \hat{t} \times \hat{e}_1
$$

**Step 3**: 消歧义 — 若 Face B 法向量 $\hat{n}_b$ 存在且 $\hat{e}_2 \cdot \hat{n}_b > 0$，则翻转 $\hat{e}_2 \to -\hat{e}_2$。

含义：
- $\hat{e}_1$ = Face A 外法向在截面内的方向 → 楔形坐标 $\varphi = \pi/2$
- $\hat{e}_2$ = 沿 Face A 表面方向 → 楔形坐标 $\varphi = 0$
- 楔形外角 $\alpha$ 由 Face A ($\varphi=0$) 到 Face B ($\varphi=\alpha$) 逆时针度量

代码位置：`physics/ptd_core.py:52-70`

---

### 2.3 入射角 $\varphi_0$ 和观察角 $\varphi$ 的计算

**入射角 $\varphi_0$ (angle0)**：

将入射方向 $\hat{k}$ 投影到 $\perp\hat{t}$ 平面，取反向（从棱边看向光源的方向）：

$$
\hat{k}_\perp = \hat{k} - (\hat{k}\cdot\hat{t})\hat{t}, \qquad \hat{d}_{\text{inc}} = -\hat{k}_\perp / |\hat{k}_\perp|
$$

$$
\varphi_0 = \text{atan2}(\hat{d}_{\text{inc}} \cdot \hat{e}_1, \; \hat{d}_{\text{inc}} \cdot \hat{e}_2)
$$

若 $\varphi_0 < 0$，加 $2\pi$；然后 clip 到 $[0, \alpha]$。

**观察角 $\varphi$ (angle_obs)**：

$$
\hat{s}_\perp = \hat{s} - (\hat{s}\cdot\hat{t})\hat{t}, \qquad \hat{s} = -\hat{k} \text{（单站反向）}
$$

$$
\varphi = \text{atan2}(\hat{s}_\perp \cdot \hat{e}_1, \; \hat{s}_\perp \cdot \hat{e}_2)
$$

**单站特性**：由于 $\hat{s} = -\hat{k}$，投影后 $\hat{s}_\perp = -\hat{k}_\perp$，
因此 $\hat{s}_{\perp,\text{unit}} = -\hat{k}_{\perp,\text{unit}} = \hat{d}_{\text{inc}}$，
故 **$\varphi = \varphi_0$**（单站时观察角等于入射角）。

代码位置：`physics/ptd_core.py:72-102`

---

### 2.4 衍射系数 $D$ 的组合

组合公式（Eq. 7.137 单站情形）：

$$
D = F_1^{V\theta} \cos^2\chi + G_1^\varphi \sin^2\chi + G_1^{V\theta} \sin\chi \cos\chi
$$

其中 $\chi$ 为极化投影角（见 §2.5），三个系数由 `FG_monostatic` 给出。

代码位置：`physics/ptd_core.py:140-143`

---

### 2.5 Keller 锥极化投影

在散射方向 $\hat{s}$ 处建立 Keller 锥局部基向量：

$$
\hat{\beta} = \text{normalize}\!\left(\hat{t} - (\hat{t}\cdot\hat{s})\hat{s}\right), \qquad \hat{\alpha} = \hat{s} \times \hat{\beta}
$$

- $\hat{\beta}$：棱边方向在垂直于 $\hat{s}$ 的平面上的投影（类似远场 $\hat{\theta}_{\text{local}}$）
- $\hat{\alpha}$：对应远场 $\hat{\varphi}_{\text{local}}$

极化投影：
$$
\cos\chi = \hat{e}_{\text{pol}} \cdot \hat{\beta}, \qquad \sin\chi = \hat{e}_{\text{pol}} \cdot \hat{\alpha}
$$

其中：
- VV 极化：$\hat{e}_{\text{pol}} = \hat{\theta}$（远场球坐标 $\theta$ 方向）
- HH 极化：$\hat{e}_{\text{pol}} = \hat{\varphi}$（远场球坐标 $\varphi$ 方向）

代码位置：`physics/ptd_core.py:104-116`

---

## 3. 衍射系数的计算

### 3.1 二维修正系数 $f_1, g_1$（fun_fg）

对应 Ufimtsev Eq. (4.20)-(4.21)，即 MATLAB `fun_fg.m`。

定义：$n = \alpha/\pi$，$\psi_1 = \varphi - \varphi_0$，$\psi_2 = \varphi + \varphi_0$。

四个中间角：
$$
a_1 = \frac{\pi - \psi_2}{2n}, \quad a_2 = \frac{\pi - \psi_1}{2n}, \quad a_3 = \frac{\pi + \psi_2}{2n}, \quad a_4 = \frac{\pi + \psi_1}{2n}
$$

#### SSI（单侧照射）：$\varphi_0 \in [0, \alpha - \pi]$

一般情况（无奇点）：

$$
f = \frac{1}{2n}\left[\cot a_1 - \cot a_2 + \cot a_3 - \cot a_4\right]
$$
$$
g = -\frac{1}{2n}\left[\cot a_1 + \cot a_2 + \cot a_3 + \cot a_4\right]
$$
$$
f_0 = \frac{1}{2}\left[\cot\frac{\pi-\psi_2}{2} - \cot\frac{\pi-\psi_1}{2}\right]
$$
$$
g_0 = -\frac{1}{2}\left[\cot\frac{\pi-\psi_2}{2} + \cot\frac{\pi-\psi_1}{2}\right]
$$
$$
f_1 = f - f_0, \qquad g_1 = g - g_0
$$

物理含义：$f, g$ 是 GTD 全衍射系数；$f_0, g_0$ 是 PO 等效衍射系数；$f_1, g_1$ 是修正量（fringe wave）。

#### DSI（双侧照射）：$\varphi_0 \in (\alpha - \pi, \pi)$

一般情况：

$f, g$ 公式同 SSI。PO 项多出 Face B 的贡献（映射角 $\phi_1 = -\psi_1$，$\phi_2 = 2\alpha - \psi_2$）：

$$
f_0 = \frac{1}{2}\left[\cot\frac{\pi-\psi_2}{2} - \cot\frac{\pi-\psi_1}{2} + \cot\frac{\pi-2\alpha+\psi_2}{2} - \cot\frac{\pi+\psi_1}{2}\right]
$$
$$
g_0 = -\frac{1}{2}\left[\cot\frac{\pi-\psi_2}{2} + \cot\frac{\pi-\psi_1}{2} + \cot\frac{\pi-2\alpha+\psi_2}{2} + \cot\frac{\pi+\psi_1}{2}\right]
$$

#### 奇点处理

当 $\psi_1 \approx \pi$、$\psi_2 \approx \pi$ 或 $2\alpha - \psi_2 \approx \pi$ 时，
相关的 $\cot$ 项发散。代码中使用解析极限：将发散项对消后，写出不含奇异项的等价公式。

阈值：`_EPS_ANGLE = 1e-2`（约 0.57°），在此范围内使用奇点处理分支。

代码位置：`physics/ptd_coefficients.py:55-144`

**验证**：fun_fg 已对半平面（$\alpha = 2\pi$）与 Sommerfeld 精确解对比，比值 = 1.000000。

---

### 3.2 三维单站系数（FG_monostatic）

对应 Ufimtsev Eq. (7.137) 的单站分支（Keller 锥上 $V_\theta = \pi - \gamma_0$）。

$$
F_1^{V\theta} = -f_1 / \sin\gamma_0
$$

$$
G_1^{V\theta} = \left[\varepsilon(\varphi_0) - \varepsilon(\alpha - \varphi_0)\right] \cdot \frac{\cos\gamma_0}{\sin\gamma_0}
$$

$$
G_1^\varphi = g_1 / \sin\gamma_0
$$

其中阶跃函数：
$$
\varepsilon(x) = \begin{cases} 1 & \text{if } 0 < x \leq \pi \\ 0 & \text{otherwise} \end{cases}
$$

**物理含义**：
- $F_1^{V\theta}$：soft 边界（TE）修正的 $\theta$ 分量
- $G_1^\varphi$：hard 边界（TM）修正的 $\varphi$ 分量
- $G_1^{V\theta}$：斜入射极化耦合项（$\gamma_0 = \pi/2$ 正入射时为零）

**关于 $G_1^{V\theta}$ 的符号**：

Ufimtsev Eq.(7.153) 在 Keller 锥 $V_\theta = \pi - \gamma_0$ 上给出
$G_1^{V\theta} = [\varepsilon(\varphi_0) - \varepsilon(\alpha-\varphi_0)] \cdot \cos\gamma_0/\sin\gamma_0$。
MATLAB 代码 `FG.m` line 11 与此一致（`cot(gamma0)` = $\cos\gamma_0/\sin\gamma_0$）。
**已确认正确**。

代码位置：`physics/ptd_coefficients.py:147-168`

---

## 4. 相位约定

### 4.1 入射波

源位于远场 $(\theta, \phi)$ 方向，波向原点（目标）传播：

$$
\hat{k}_{\text{dir}} = -(\sin\theta\cos\phi, \; \sin\theta\sin\phi, \; \cos\theta)
$$

即 `k_dir` 指向目标（从源到目标）。

代码位置：`physics/wave.py:55-59`

### 4.2 极化基向量

$$
\hat{\theta} = (\cos\theta\cos\phi, \; \cos\theta\sin\phi, \; -\sin\theta)
$$
$$
\hat{\varphi} = (-\sin\phi, \; \cos\phi, \; 0)
$$

代码位置：`physics/wave.py:69-70`

### 4.3 PO 和 PTD 的相位

两者使用完全相同的相位约定：

$$
\text{phase} = 2k \, \hat{k}_{\text{dir}} \cdot \mathbf{r}'
$$

- PO：`phase = 2.0 * sum(points * k_vec)`
- PTD：`phase_mid = 2.0 * dot(seg.midpoint, k_vec)`

其中 `k_vec = k * k_dir`。

---

## 5. 边缘几何提取

### 5.1 外部二面角 $\alpha$

从两个相邻面的法向量计算：

$$
\alpha = \pi + \arccos(\hat{n}_a \cdot \hat{n}_b)
$$

其中 $\hat{n}_a, \hat{n}_b$ 是边缘两侧面在边缘上各采样点的外法向量。

验证：
| 几何形状 | $\hat{n}_a \cdot \hat{n}_b$ | $\alpha$ | 含义 |
|---------|---------------------------|---------|------|
| 平面（无棱边） | $+1$ | $\pi$ | $f_1 = g_1 = 0$，无修正 |
| 直角棱（立方体边） | $0$ | $3\pi/2$ | 标准凸楔 |
| 刀刃（半平面） | $-1$ | $2\pi$ | 最强衍射 |

代码位置：`solvers/ptd_edge_finder.py:143-146`

### 5.2 曲面边缘的逐段 $\alpha$

对于曲面棱边，每段独立计算 $\alpha$：将两面法向投影到 $\perp\hat{t}$ 截面后再算夹角：

$$
\hat{n}_{a,\perp} = \text{normalize}(\hat{n}_a - (\hat{n}_a \cdot \hat{t})\hat{t})
$$
$$
\alpha_{\text{seg}} = \pi + \arccos(\hat{n}_{a,\perp} \cdot \hat{n}_{b,\perp})
$$

代码位置：`solvers/ptd_structures.py:134-145`

---

## 6. Sinc 因子（线段积分）

对长度为 $L$ 的直线段，沿棱边的相位积分：

$$
\int_{-L/2}^{L/2} e^{j \cdot 2k (\hat{k}\cdot\hat{t}) s} \, ds = L \cdot \text{sinc}\!\left(\frac{k L (\hat{k}\cdot\hat{t})}{\pi}\right)
$$

代码位置：`physics/ptd_core.py:146-147`
```python
sinc_arg = k * seg.length * k_dot_t / np.pi
sinc_val = np.sinc(sinc_arg)    # np.sinc(x) = sin(pi*x) / (pi*x)
```

---

## 7. 奇点邻域处理（ptd_core 层）

在 `compute_ptd_contribution` 中，除了 `fun_fg` 内部的奇点处理外，
还有一层外部保护：当 $\psi_1 \approx \pi$、$\psi_2 \approx \pi$ 或 $2\alpha - \psi_2 \approx \pi$ 时，
对 `angle_obs` 做 $\pm\delta$ 偏移后取平均：

$$
D \approx \frac{1}{2}\left[D(\varphi - \delta, \varphi_0) + D(\varphi + \delta, \varphi_0)\right], \quad \delta = 3\times10^{-4} \text{ rad}
$$

代码位置：`physics/ptd_core.py:121-137`

---

## 8. 完整计算流程总结

对每个观测角度：

```
1. 计算 I_PO = Σ (面元) cos(θ_i) · exp(j·2k·k̂·r') · w · J
   → 面积分，仅照亮面

2. 对每条 PTD 边缘的每个线段 m：
   a. 计算 sin(γ₀) = √(1 - (k̂·t̂)²)
   b. 建立截面坐标系 (e₁, e₂) → 计算 φ₀ = angle0
   c. 单站：φ = φ₀
   d. 建立 Keller 锥基 (β̂, α̂) → 计算 cos(χ), sin(χ)
   e. 调用 fun_fg(φ, φ₀, α) → f₁, g₁
   f. 调用 FG_monostatic → F₁ᵛᶿ, G₁ᵛᶿ, G₁ᵠ
   g. D = F₁ᵛᶿ·cos²χ + G₁ᵠ·sin²χ + G₁ᵛᶿ·sinχ·cosχ
   h. I_seg = (-j·sinγ₀/k) · D · L · sinc · exp(j·2k·k̂·r_mid)

3. I_total = I_PO + Σ I_seg

4. σ = (k²/π) · |I_total|²
```

---

## 9. 已解决问题清单

| # | 问题 | 位置 | 结论 |
|---|------|------|------|
| 1 | pre_factor 是否需要 $/2$ | ptd_core.py:156 | ✅ 无需 $/2$。SPA 推导和数值验证均确认 |
| 2 | pre_factor 符号 $-j$ 还是 $+j$ | ptd_core.py:156 | ✅ $-j$ 正确。PO+PTD 干涉验证确认（`verify_ptd_sign.py`） |
| 3 | $G_1^{V\theta}$ 用 $+\cos\gamma_0/\sin\gamma_0$ | ptd_coefficients.py:214 | ✅ 正号正确。与 MATLAB `FG.m` 和原书 Eq.(7.153) 一致 |
| 4 | $D$ 的耦合项系数 | ptd_core.py:143 | ✅ $1\times$ 正确。矩阵推导确认无需 $2\times$ |

## 9.1 未验证问题

| # | 问题 | 位置 | 备注 |
|---|------|------|------|
| 5 | $e_2$ 翻转逻辑对凹楔是否正确 | ptd_core.py:69 | 凸楔（$\alpha > \pi$）已验证正确 |

---

## 10. 参考文献

- Ufimtsev, P. Ya., *Fundamentals of the Physical Theory of Diffraction*, 2nd ed., Wiley, 2014.
  - Eq. (4.20)-(4.21): $f_1, g_1$ 定义
  - Eq. (7.48): $\varepsilon(x)$ 阶跃函数
  - Eq. (7.137): 3D EEW 衍射系数（含单站分支）
- 对应 MATLAB 文件：`eps_x.m`, `fun_fg.m`, `sigma12.m`, `FG.m`
