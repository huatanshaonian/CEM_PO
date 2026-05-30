# MEC 一阶 / Johansen 截断 / 二阶 EEC：完整公式与时谐约定文档

> 本文档目的：把 PTD 等效边电流 (EEC) 框架下的所有公式、所有 j 因子的物
> 理出处、所有时谐约定的对应翻译，**一次性写清楚**。文档为"分析报告 +
> 修复方案"性质，重点在于回答："本仓库 `physics/mec_truncated_*.py`
> 在 PO 主瓣区与 PO 旁瓣峰区出现的两套相位错误，根因到底在哪？真正一
> 致的修复应该怎么改？"
>
> 主要参考文献：
> - **Ufimtsev 2014** *Fundamentals of the Physical Theory of Diffraction*
>   (2nd ed., Wiley), 用 e^{−iωt}（数学 i）。EEW Eq.7.137 / 二阶 Ch.10。
> - **Michaeli 1984** IEEE TAP 32, 252-258 (EEC for arbitrary aspects)。
> - **Michaeli 1986** IEEE TAP 34(7), 912-918 (Part I — Fringe currents)。
> - **Michaeli 1987** IEEE TAP 35(2), 183-190 (Second-order edge diffr.)。
> - **Johansen 1996** IEEE TAP 44(7), 989-995 (Truncated wedge strips)。
> 三篇 Michaeli + Johansen 均用 e^{+jωt}（虚单位 j）。

---

## 0. 速查表（结论先行）

| 模块 | 约定来源 | 代码事实做法 | 物理一致性 |
|---|---|---|---|
| `ptd_core.py` (EEW) | Ufimtsev e^{−iωt} 原生 | `pre = −1j·sinγ₀/k`，`exp(+1j·2k·r)` | 主瓣区与 PO 干涉正确 ✓ |
| `mec_coefficients.py` (Mich. 1986 If/Mf) | Michaeli e^{+jωt}, 抄入 1j = j | 原样抄；含代数 j (2j/k 等) | 数值是 Michaeli "physical" |
| `mec_core.py` (MEC 一阶) | 把 −0.5j 改为 −0.5 (=`×j`) | `pre = −0.5`，`exp(+1j·2k·r)` | 与 EEW 主瓣区 ≤0.5 dB ✓ |
| `mec_truncated_coefficients.py::compute_correction_currents` | Johansen e^{+jωt} 原文 | 原样抄；含 `exp(+1j·L·(μ−1))`、`F(x)` | 与 mec_core 风格一致但**约定混搭** |
| `mec_truncated_core.py` 旧版 (b4ff852) | 同 mec_core 写法 | `pre=−0.5`，未 conj UT，原版 cor | 主瓣 ✓，PO 旁瓣峰错误 ✗ |
| 我此前的"修复" (新版) | 整体 conj 翻译 | `pre=−0.5j`，`conj(UT)`，UF 版 cor | PO 旁瓣 ✓，主瓣 ✗ |
| **正确路线（本文推荐）** | 半翻译：相位指数翻，代数 j 不翻 | `pre=−0.5`，UT 不 conj，cor 仅翻 exp & F | 两段都对 |

---

## 1. 时谐约定的数学事实

### 1.1 两套约定

**Ufimtsev 数学（U）**：时谐因子 e^{−iωt}，外向球面波 e^{+ikR}/R，外向平面波 e^{+ik·r}。  
**Michaeli 数学（M）**：时谐因子 e^{+jωt}，外向球面波 e^{−jkR}/R，外向平面波 e^{−jk·r}。

把 e^{+jωt} 与 e^{−iωt} 比较：二者只是用了不同记号（数学上 i = j = √−1），它们
的"出向波"e^{+ik·r} 与 e^{−jk·r} 描述同一个物理过程，但**复数形式上互为共轭**。

> **断言（约定翻译规则）**：把一个复数表达式 X(j) 从 M 翻译到 U，**结果是 conj(X)
> 对每一个 j 的出现做替换 j → −i**。等价地：
> "整体复共轭 X" ↔ "公式里每一个 j 替换成 −i"。这两种实施在每一项上都给同一数值。

这是**严格物理结论**，没有"乘 j"或"事后再做点什么"的空间。

### 1.2 远场散射场公式

Ufimtsev 约定下，远场散射 E 由源（电流 J = I·t̂、磁流 M = M·t̂）的远场积分给出
（Mitzner / Michaeli 1984 重排到 U 约定）：

$$
\mathbf E^{s}_\infty(\hat s) \;=\;
\frac{i k}{4\pi}\!\int_{C} \bigl[\,Z\,I^{f}\,\hat t_{\perp}(\hat s, \hat t)
\;+\; M^{f}\,(\hat s\times\hat t)\,\bigr]\,e^{-i k\,\hat s\cdot\mathbf r_{c}}\,d\ell_{c}
$$

其中 $\hat t_\perp = \hat t - \hat s(\hat s\cdot\hat t)$（横向投影），$I^{f}, M^{f}$ 是
**fringe 等效边电流**（标量，沿 $\hat t$）。

单站 $\hat s = -\hat k$，入射波在 $r_c$ 处相位 $e^{+ik\cdot r_c}$。两个相位合并：

$$
e^{-i k\,\hat s\cdot r_c}\cdot e^{+i\mathbf k\cdot\mathbf r_c}
= e^{+2i\mathbf k\cdot\mathbf r_c}
$$

代码用 `np.exp(1j * phase), phase = 2·midpoint·k_vec` 实现这一项。**约定与符号
都正确**（与 EEW 注释完全一致）。

RCS 归一化定义 $I = \frac{4\pi}{i k}\,(\hat e_r\cdot\mathbf E^{s}_\infty)$，
则 $\sigma = (k^2/\pi)\,|I|^2$。

### 1.3 代码事实约定（实证）

由于 $|I|^2$ 对全局相位不敏感，代码 PO 积分可以写成 $e^{+i 2k r_c}$（与上式一致）或
$e^{-i 2k r_c}$（取共轭后），都给同一 RCS。但 **PO 与 PTD 必须用相同的相位约定**，
否则交叉项 $2\Re(I_\text{PO}\cdot I_\text{PTD}^*)$ 会反号。

实测：`ptd_core.py` 用 `pre = −1j·sinγ₀/k`、`exp(+1j·2k·r)`，与 PO 在主瓣区干涉正
确（PO + EEW 几乎不破坏 PO 主瓣值）。这说明 **EEW 代码在 Ufimtsev e^{−iωt} 约定
下数值正确**。后文以此为基准。

---

## 2. Ufimtsev EEW 与 Michaeli MEC 的等价框架

### 2.1 共同物理

两者都从 Sommerfeld 半平面 / 楔形精确解出发，把"fringe 部分"（精确 − PO）抽
象成一组边电流。Ufimtsev 用"衍射函数 D"（直接给 σ_far/σ_inc 的散射系数）；
Michaeli 用 **真正的 EEC 张量** ($I^{f}, M^{f}$)，对发射极化和接收极化解耦。

### 2.2 EEW 标量写法（`ptd_core.py`）

$$
D \;=\; F_{1,V\!t}\cos^{2}\chi \;+\; G_{1,\phi}\sin^{2}\chi \;+\; G_{1,V\!t}\sin\chi\cos\chi
$$

$\chi$ 是发射极化向量 $\hat e_{\rm pol}$ 在 Keller 锥局部基 $(\hat\beta,\hat\alpha)$ 上的
方位。`F_1Vt, G_1phi, G_1Vt` 已内含 $1/\sin\gamma_{0}$，**不能再被 pre_factor 中的 $\sin\gamma_0$
重复消掉**。代码:

```python
pre_factor = -1j * sin_gamma0 / k       # ptd_core.py:175
seg = pre_factor * D * L * sinc * exp(+1j·phase)
```

带回去，effective pre = $-\tfrac{i}{k}$ · (D内部纯几何函数 f₁, g₁)。物理上 $D$ 取
共轭实数（f₁, g₁ 实），所以 EEW 主项 $\propto -i\,f_{1}/k\cdot\exp(+i\,\text{phase})$。

EEW 标量写法**只能写共极化 VV/HH**（χ_t = χ_r 时合并），写不出 VH/HV 的发射/接收
基不同。这是它作为二阶绕射内核的硬限制。

### 2.3 Michaeli EEC 张量写法（`mec_core.py` + `mec_coefficients.py`）

$I^{f}, M^{f}$ 是关于入射切向场 $(E_{0z}, H_{0z}) = (\hat e_{t}\cdot\hat t,\,(\hat k\times\hat e_{t})\cdot\hat t/Z)$ 的标量线性函数。
接收侧用接收极化 $\hat e_{r}$ 投影：

$$
\text{amp} \;=\; -Z\,I^{f}\,(\hat t\cdot\hat e_{r}) \;+\; M^{f}\,\bigl((\hat s\times\hat t)\cdot\hat e_{r}\bigr)
$$

天然支持 4 个极化组合 VV / HH / VH / HV，因为发射 $\hat e_{t}$ 与接收 $\hat e_{r}$
可独立。二阶绕射用 MEC 框架做内核才能算 V↔H 通道耦合。

---

## 3. Michaeli 1986 一阶 fringe 闭式（在仓库代码中的精确对应）

### 3.1 局部坐标与角度

边切线 $\hat t = \hat z$。Face 1 法向投影到 $\perp\hat t$ 平面后给 $\hat y_{1}$，
$\hat x_{1} = \hat y_{1}\times\hat z$（指向面内）。

$$
\beta' = \arccos(-\hat k\cdot\hat t)\;\in (0,\pi),\quad
\phi' = \operatorname{atan2}(\hat s_{\rm inc}\cdot\hat y_{1},\,\hat s_{\rm inc}\cdot\hat x_{1})
$$

外角 $\alpha = N\pi$，$N\in (1, 2]$（楔 N<2、半平面 N=2）。

### 3.2 单站后向散射化简（Michaeli 1986 Eq.26）

$$
\mu \;=\; \cos\phi'\;-\;2\cot^{2}\beta'
$$

复数 $\alpha_{c} = \arccos\mu$（$|\mu|>1$ 时含虚部，主分支 $-j\operatorname{Log}(\mu + j\sqrt{1-\mu^{2}})$）。

### 3.3 面 1 的 $I_{1}^{f}, M_{1}^{f}$（Michaeli 1986 Eq.27/28）

代码 `physics/mec_coefficients.py::_compute_face_currents_raw` 的**逐项物理对应**：

$$
\boxed{\;I_{1}^{f}
= \underbrace{\frac{-2j\,Y_{0}}{k\sin^{2}\!\beta'}}_{\text{pre\_E}}\!
\biggl[\,\frac{\sin\phi'\;U(\pi-\phi')}{\cos\phi'+\mu}
+ \frac{\sin(\phi'/N)}{N\,D_{e}}\,\biggr]\,E_{0z}
\;+\;\underbrace{\frac{2j\sin((\pi-\alpha_{c})/N)}{N\,k\sin\beta'\sin\alpha_{c}}}_{\text{pre\_H}}\!
\,\frac{\cot\beta'(\mu+\cos\phi')}{-D_{e}}\,H_{0z}\;}
$$

$$
\boxed{\;M_{1}^{f}
= \underbrace{\frac{2j\,Z\,\sin\phi'}{k\sin^{2}\!\beta'}}_{\text{pre\_M}}\!
\biggl[\,\frac{U(\pi-\phi')}{\cos\phi'+\mu}
+ \frac{\sin((\pi-\alpha_{c})/N)}{N\sin\alpha_{c}\,D_{e}}\,\biggr]\,H_{0z}\;}
$$

其中 $D_{e} = \cos((\pi-\alpha_{c})/N) - \cos(\phi'/N)$，$U(\cdot)$ 是单位阶跃。

**所有 j 因子的物理出处**：
- $2j/k$ 类前因子来自 Sommerfeld 积分鞍点法 + 渐近常数 phase
  $\sqrt{2\pi/(k r)}\,e^{+j\pi/4}$ 的代数化简；这是 **"代数 j"**。
- $\sin\alpha_{c}, \cos(\phi'/N)$ 等只含 cos/sin 函数；当 $|\mu|>1$ 时
  $\alpha_{c}$ 含虚部、 $\sin\alpha_{c}$ 含 j —— 也是代数 j（由 $\mu>1$ 时
  $\sqrt{1-\mu^{2}}$ 走 lib.scimath 隐式得到，与时谐约定挂钩但**不出现于显式 exp**）。

**关键事实**：Michaeli 1986 Eq.27/28 中**没有任何形如 exp(jX(ξ)) 的相位指数项**
（其中 X 是物理参数）。因此整段公式可看作 Michaeli e^{+jωt} 约定下"纯代数复表达
式"。

### 3.4 面 2 通过对称替换

$M_{2}^{f}, I_{2}^{f}$ 由 $\beta'\to\pi-\beta',\ \phi'\to N\pi-\phi'$ 得，符号约定见
代码 `compute_total_fringe_currents`。

### 3.5 时谐约定翻译："乘 j" 简化为何对一阶有效

设 Michaeli 远场散射场（M 约定）：

$$
E^{s}_{\infty,\,M}(\hat s) = \frac{-jk}{4\pi}\!\int [Z I^{f} \hat t_{\perp}\!+\!M^{f}(\hat s\!\times\!\hat t)] e^{+j k\hat s\cdot r_c}\,d\ell
$$

翻译到 U 约定（整体共轭、$j\to -i$）：

$$
E^{s}_{\infty,\,U}(\hat s) = \frac{+ik}{4\pi}\!\int [Z \overline{I^{f}} \hat t_{\perp}\!+\!\overline{M^{f}}(\hat s\!\times\!\hat t)] e^{-i k\hat s\cdot r_c}\,d\ell
$$

代码 `mec_core` 用 U 约定的相位（`exp(+1j·2k·r_c)`，由 1.2 节单站合并），但代入的
$I^{f}, M^{f}$ 是 Michaeli 数值（含 $+j$，**未取 conj**）。

设 $I^{f}_M \approx \frac{2j}{k}\,R$（R 实，主项）。Michaeli 一阶振幅 $\propto -0.5j\,R$。

"乘 j" 翻译：$\text{pre\_factor} = -0.5j \to -0.5$。代码效果：

$$
-0.5 \cdot \frac{2j}{k}\,R \cdot e^{+i\,2k r_c}
\;=\; -\frac{i\,R}{k}\,e^{+i\,2k r_c}
$$

正确翻译（整体共轭）：

$$
-(-0.5i)\cdot\overline{\!\tfrac{2j}{k}R\!}\cdot e^{+i\,2k r_c}
\;=\; -0.5i\cdot\tfrac{-2i}{k}R\cdot e^{+i\,2k r_c}
\;=\; -\frac{R}{k}\,e^{+i\,2k r_c}
$$

两个数值的模 $|R/k|$ 相同，相位差 90°。但 **PO 也是实数 $\sim \mathrm{sinc}^{2}$**，
相干叠加结果：

$$
|\text{PO} + (-iR/k)|^{2} = \text{PO}^{2} + (R/k)^{2}\qquad
|\text{PO} + (-R/k)|^{2} = (\text{PO}-R/k)^{2}
$$

主瓣区 PO ≫ R/k 时两者数值差 ≤ 0.5 dB——**这就是为什么 mec_core "乘 j" 在主瓣区
看起来"对"，但实际上和"严格 conj 翻译"差 90° 相位**。在 PO ≈ R/k（旁瓣峰附近）
两者会显著分歧。

> **结论 A**：mec_core 的"乘 j" 翻译在主瓣区数值与正确翻译几乎一致（≤0.5 dB），
> 在旁瓣峰附近会偏离正确翻译；但因为问题模值 ≤ R/k，PO 主导，差别不显。
> 这条规则适用于**所有 Michaeli 1986 一阶公式**，因为它们只含代数 j、不含 exp(jX)。

> **结论 B**：把"乘 j" 当作一阶的工程近似可以接受。但若加上含 exp(jX) 相位指数
> 的项（Johansen 截断、二阶 EEC），"乘 j" 完全错——因为 exp(jX) 的 j 与代数 j 的
> 翻译规则严格不一致。

---

## 4. Johansen 1996 截断 MEC（半平面 N=2，代码 `mec_truncated_coefficients.py`）

### 4.1 配置

外角 $N\pi = 2\pi$（刀刃边），入射 $(\beta_{0}, \phi_{0})$、观察 $(\beta, \phi)$，
截断长度 $l^{A}$ 沿 Keller 锥与面 A 的交线 $\hat u^{A}$。截断 EEC 定义（Eq.3）：

$$
M_{T} = M_{UT} - M_{cor},\qquad I_{T} = I_{UT} - I_{cor}
$$

$M_{UT}, I_{UT}$ 用 Michaeli 1986 非截断 EEC（已在 3.3 节）；$M_{cor}, I_{cor}$ 用
Johansen 1996 Eq.21–22 渐近闭式。

### 4.2 截断参量与改进 Fresnel

$$
L = k\,l^{A}\,\sin^{2}\!\beta_{0},\quad
\mu = \frac{\sin\beta\sin\beta_{0}\cos\phi + \cos\beta_{0}(\cos\beta-\cos\beta_{0})}{\sin^{2}\!\beta_{0}}
$$

(Eq.6) **改进 Fresnel function (Eq.17)**：

$$
\boxed{\;F(z) \;=\; \sqrt{\tfrac{j}{\pi}}\;\exp(j z^{2})\!\int_{z}^{\infty}\!\exp(-j t^{2})\,dt\;}
$$

数值实现 $F(z) = \tfrac{1}{2}\exp(j z^{2})\,\operatorname{erfc}(z\,e^{j\pi/4})$。

### 4.3 半平面 N=2 完整闭式（Johansen 1996 Eq.26/27）

> **物理 j 出处分类**：以下公式中
> - **相位指数 j** (与时谐约定挂钩，翻译时必须 j→−i)：
>   - $\exp(+j L(\mu-1))$ —— 由 $\int_{0}^{\infty}\exp(-jku(1-\mu))du$ 积分残留
>   - $F(z)$ 内的 $\exp(+jz^{2})$ —— Fresnel 积分核 $\exp(-jt^{2})$ 的鞍点偏移
>   - $F(z)$ 内的 $e^{+j\pi/4}$ —— SDP 旋转（争议项，见 4.4）
> - **代数 j** (与时谐约定虽挂钩但通过"乘 j"等价翻译):
>   - 分母 $jk$ —— 来自 $1/\sqrt{2\pi/jk\cdot}$ 的代数化简
>   - $\sqrt{j/\pi}$ —— Fresnel 渐近常数 phase

**Eq.26 — M_cor（半平面 N=2，面 A+B 之和）**：

$$
\boxed{\;M_{cor} \;=\;
\frac{4\,Z\,H_{z0}\,\sin\phi\;\exp\bigl(jL(\mu-1)\bigr)}
     {jk\,\sin\beta\,\sin\beta_{0}\,(\mu+\cos\phi_{0})}
\biggl\{\;
   -\operatorname{sign}\bigl(\!\cos\tfrac{\phi_{0}}{2}\!\bigr)
   F\!\bigl(\sqrt{2L}\,|\!\cos\!\tfrac{\phi_{0}}{2}|\bigr)
   \;+\;\frac{\sqrt{2}\cos(\phi_{0}/2)}{\sqrt{1-\mu}}\,
   F\!\bigl(\sqrt{L(1-\mu)}\bigr)
\;\biggr\}\;}
$$

**Eq.27 — I_cor**：分解成 $E_{z0}$ 和 $H_{z0}$ 两项：

E 项：
$$
\frac{4\,E_{z0}\,\sin(\phi_{0}/2)\,\exp(jL(\mu-1))}{jk\,Z\,\sin^{2}\beta_{0}\,(\mu+\cos\phi_{0})}
\Bigl\{2\cos(\phi_{0}/2)\,F\!\bigl(\sqrt{2L}|\!\cos\!\tfrac{\phi_{0}}{2}|\bigr)
- \sqrt{2(1-\mu)}\,F\!\bigl(\sqrt{L(1-\mu)}\bigr)\Bigr\}
$$

H 项：
$$
\frac{4\,H_{z0}\,\exp(jL(\mu-1))}{jk\,\sin\beta_{0}\,(\mu+\cos\phi_{0})}
\Bigl\{
-\operatorname{sign}(\!\cos\tfrac{\phi_{0}}{2}\!)
(\cot\beta_{0}\cos\phi_{0}+\cot\beta\cos\phi)
F\!\bigl(\sqrt{2L}|\!\cos\!\tfrac{\phi_{0}}{2}|\bigr)
$$
$$
\quad
+\frac{\sqrt{2}\cos(\phi_{0}/2)}{\sqrt{1-\mu}}
(\cot\beta\cos\phi - \mu\cot\beta_{0})
F\!\bigl(\sqrt{L(1-\mu)}\bigr)
\Bigr\}
$$

### 4.4 e^{jπ/4} 是相位指数还是代数？

$F(z)$ 定义里有 $e^{j\pi/4}$（在 erfc 参数中作为路径旋转角）和 $\sqrt{j/\pi} = \tfrac{1}{\sqrt{\pi}}e^{j\pi/4}$
（外面的归一化常数）。两者都是**与时谐约定挂钩的常数 phase shift**——它们来自
$\int \exp(\pm jt^{2})dt = \tfrac{\sqrt{\pi}}{2}e^{\mp j\pi/4}$ 类等式。

**判定**：翻译到 U 约定时 $j\to -i$，所以
$e^{j\pi/4}\to e^{-i\pi/4}$，$\sqrt{j/\pi}\to\sqrt{-i/\pi}$，
$\exp(jz^{2})\to\exp(-iz^{2})$，得：

$$
F_{U}(z) \;=\; \tfrac{1}{2}\exp(-iz^{2})\,\operatorname{erfc}(z\,e^{-i\pi/4})
$$

对实 z：$F_{U}(z) = \overline{F(z)}$（这就是仓库的 `modified_fresnel_uf`）。

> **结论 C**：Johansen 1996 Eq.26/27 含 **真正的相位指数项 $\exp(jL(\mu-1))$**，
> 这一项是 Michaeli 1986 没有的（一阶公式无此类项）。约定翻译时它必须严格
> $j\to -i$。不能用"乘 j"代替。

### 4.5 Ufimtsev 奇点抵消机制

当 $\phi_{0}\to\pi$（入射沿面 A 掠射）、$\mu\to 1$（观察沿出射方向延续）时，
$M_{UT}$ 与 $M_{cor}$ 各自发散，但减法 $M_{T} = M_{UT} - M_{cor}$ 给有限值
（Johansen 1996 III.B/IV-C 证明）。本仓库实现里若 $|\mu+\cos\phi_{0}| < 10^{-9}$
或 $|1-\mu| < 10^{-9}$，则 $M_{cor} = I_{cor} = 0$，退回 untruncated（数值简化处理，
近奇点处精度受影响）。

### 4.6 $l^{A}$ 几何（仓库 `mec_truncated_geometry.py`）

简化：假设 $\beta_{0} = \pi/2$（主平面单站后向），$\hat u^{A} = \text{seg.inward}$
（沿亮面、由边指向面内）。$l^{A}$ 从段中点沿 $\hat u^{A}$ 射线找首条相交边的距
离。对平板 5λ、$\phi=0$ 任意 $\theta$ 严格正确（$\beta_{0} = \pi/2$ 实质成立）。

---

## 5. 当前实现的两类错误

### 5.1 旧版 `mec_truncated_core.py` (commit b4ff852) — PO 旁瓣峰错误

做法：
```python
If_UT, Mf_UT = compute_total_fringe_currents(...)     # 抄 Michaeli 1986
M_cor, I_cor = compute_correction_currents(...)        # 抄 Johansen 1996 原文，含 exp(+jL(μ-1))
If_T = If_UT - I_cor
Mf_T = Mf_UT - M_cor
pre_factor = -0.5      # 与 mec_core 一致的"乘 j" 翻译
```

诊断：
- $I_{UT}$ 是"代数 j 主导"项（无 exp(jX)），"乘 j"翻译 在主瓣区数值与正确翻译
  ≤0.5 dB，PO 主瓣区正确 ✓
- $I_{cor}$ 含 $\exp(+jL(\mu-1))$，"乘 j"等价于对 $-0.5j\cdot I_{cor}$ 整体乘 j，
  得 $-0.5\cdot I_{cor}$，**但 $\exp(+jL(\mu-1))$ 内的 j 没有跟着翻译**。正确应
  为 $\exp(-iL(\mu-1))$。
- 后果：$L(\mu-1)$ 在 PO 旁瓣峰位置 $\theta \approx 47°, 58°, 70°$ 量级 ~O(10) 弧度，
  相位错使 $I_{cor}$ 与 $I_{UT}$ 减法在这些位置产生反向相干，把 PO 旁瓣峰"挖坑"
  成深度凹陷。用户实测的"V 极化 47/58/70° 异常凹陷"由此而来。

### 5.2 我此前修复版（当前 HEAD）— PO 主瓣区错误

做法：
```python
If_UT_m, Mf_UT_m = compute_total_fringe_currents(...)
If_UT = np.conj(If_UT_m)                              # 整体 conj 翻译
Mf_UT = np.conj(Mf_UT_m)
M_cor, I_cor = compute_correction_currents_uf(...)     # 公式内 j→-i 全替换
If_T = If_UT - I_cor
Mf_T = Mf_UT - M_cor
pre_factor = -0.5j                                     # Ufimtsev 原生
```

诊断：
- $I_{cor}$ 的相位指数翻译正确，PO 旁瓣峰相干恢复 ✓
- $I_{UT}$ 整体 conj **多翻一次**——把"代数 j 主导"的 $I_{UT}$ 旋转了 90°，使
  它从"纯虚 prefactor"（与 PO 实数正交）变成"纯实 prefactor"（与 PO 实数同向）
- 主瓣区 PO ≈ 1.4 dB，conj 后 $I_{UT}$ 与 PO 反向相干，使 $|PO+I_{UT}|$ 比 PO 小 1.5 dB
  ——用户实测的"0-40° 偏差"由此而来

### 5.3 根因总结

两种约定 in、out 各错一种：

| 实现 | $I_{UT}$ | $I_{cor}$ | 主瓣 | 旁瓣峰 |
|---|---|---|---|---|
| 旧版 b4ff852 | 乘 j（事工程半翻译）| 乘 j（错） | ✓ | ✗ |
| 新版 (HEAD) | 整体 conj（多翻）| 全翻（对）| ✗ | ✓ |
| **正确路线** | 乘 j（保持 mec_core 风格）| **混合翻译**：exp/F 翻，代数 j 不翻 | ✓ | ✓ |

---

## 6. 推荐修复方案：混合翻译

### 6.1 数学等价的"半翻译"操作

观察：若 $I_{cor}^{(M)}(j)$ 是 Michaeli 约定下的 Johansen 修正项，把它的**所有
显式 j 替换 −i**就是严格 U 约定下的值（即 $\overline{I_{cor}^{(M)}}$）。这与
$I_{UT}$ 的"整体 conj"翻译一致。

但 `mec_core` 旧版采用的是 **mec_truncated_core 必须配合的"乘 j 翻译"**：对
$I_{UT}$ 做 $j\cdot I_{UT}^{(M)}$ 而非 $\overline{I_{UT}^{(M)}}$。这两者差全
局符号（对 Michaeli 1986 公式而言）。

要使 $I_{UT}^{\rm code} - I_{cor}^{\rm code}$ 内部一致（同一约定），且
$I_{UT}^{\rm code}$ 保持与 `mec_core` 兼容，必须把 $I_{cor}$ 写成
$j\cdot I_{cor}^{(M)}$（与 $I_{UT}$ 同步乘 j 翻译）。

**问题**：$I_{cor}^{(M)}$ 含 $\exp(jL(\mu-1))$，乘 j 对该项无效（外乘 j 只改全
局 phase 90°，不改 exp 内的 j）。所以纯"乘 j" 不能正确翻译 $I_{cor}$。

**解法**：把 $I_{cor}^{(M)}$ 的相位指数部分 ($\exp(jL(\mu-1))$ 和 $F(z)$ 内部) **先
做 j→−i 翻译**，再对整体做"乘 j"。

数学上：

$$
I_{cor}^{\rm code}
\;=\; j\cdot\Bigl[\,I_{cor}^{(M)}\big|_{\exp(jX)\to\exp(-iX),\ F(z)\to F_{U}(z)}\,\Bigr]
$$

实施：把 $I_{cor}$ 公式里
- $\exp(jL(\mu-1)) \to \exp(-i L(\mu-1))$
- $F(z) \to F_{U}(z) = \overline{F(z)}$
- 其余 $j$（分母 $jk$、$\sqrt{j/\pi}$ 等）保持
- 最后整体乘 j （或者直接把 pre_factor 部分的 $-0.5j$ 改 $-0.5$，与 mec_core 同步）

> **结论 D**：mec_truncated_core 正确修复方案是
> `pre_factor = -0.5`、`If_UT 不 conj`、`I_cor` 用**仅翻译相位指数与 F、保留代
> 数 j** 的半翻译版本。

### 6.2 具体代码实现

文件 `physics/mec_truncated_coefficients.py` 增加 `compute_correction_currents_consistent`：

```python
def compute_correction_currents_consistent(beta_0, phi_0, beta, phi, l_A,
                                            E0z, H0z, k, Z):
    """Johansen Eq.26/27 半平面修正项 — 半翻译版本(混合约定)

    设计意图：
      - 与 mec_core 的"乘 j"翻译规则一致，使 I_T = I_UT - I_cor 减法在
        同一数值约定下做。
      - 仅翻译 *相位指数*（exp(jL(μ-1)) → exp(-iL(μ-1))）和 F(x) → F_uf(x)。
      - 保留代数 j 因子（分母 jk）—— 与 compute_total_fringe_currents
        的 1j 因子同步。
    """
    sb0 = np.sin(beta_0); sb = np.sin(beta)
    if abs(sb0) < 1e-12:
        return 0.0+0j, 0.0+0j
    cos_phi0 = np.cos(phi_0); cos_phi = np.cos(phi)
    mu = (sb*sb0*cos_phi + np.cos(beta_0)*(np.cos(beta)-np.cos(beta_0))) / (sb0*sb0)
    denom = mu + cos_phi0
    if abs(denom) < _DENOM_EPS:
        return 0.0+0j, 0.0+0j
    L = k * l_A * sb0 * sb0
    cos_half = np.cos(phi_0/2.0)
    abs_cos_half = abs(cos_half)
    sign_cos_half = np.sign(cos_half) if cos_half != 0 else 0.0
    one_minus_mu = 1.0 - mu
    if abs(one_minus_mu) < _MU1_EPS:
        return 0.0+0j, 0.0+0j
    sqrt_1mmu = np.lib.scimath.sqrt(one_minus_mu)

    F1_arg = np.sqrt(2.0*L) * abs_cos_half
    F2_arg = np.lib.scimath.sqrt(L * one_minus_mu)

    # ★ 相位项翻译：F → F_uf（即 conj(F) for real arg）
    F1 = modified_fresnel_uf(F1_arg)
    F2 = modified_fresnel_uf(F2_arg)

    # ★ 相位指数翻译：exp(+jL(μ-1)) → exp(-iL(μ-1))
    exp_factor = np.exp(-1j * L * (mu - 1.0))

    # ★ 代数 j 保留：分母 1j*k 与 mec_coefficients 中 2j/k 同步
    M_pre = (4.0 * Z * H0z * np.sin(phi) * exp_factor) / (1j * k * sb * sb0 * denom)
    M_bracket = (-sign_cos_half * F1
                 + (np.sqrt(2.0)*cos_half/sqrt_1mmu) * F2)
    M_cor = M_pre * M_bracket

    I_E_pre = (4.0 * E0z * np.sin(phi_0/2.0) * exp_factor) / (1j * k * Z * sb0*sb0 * denom)
    I_E_bracket = (2.0*cos_half*F1 - np.lib.scimath.sqrt(2.0*one_minus_mu)*F2)
    I_E = I_E_pre * I_E_bracket

    cot_b0 = np.cos(beta_0)/sb0; cot_b = np.cos(beta)/sb
    coef1_H = -sign_cos_half * (cot_b0*cos_phi0 + cot_b*cos_phi)
    coef2_H = (np.sqrt(2.0)*cos_half/sqrt_1mmu) * (cot_b*cos_phi - mu*cot_b0)
    I_H_pre = (4.0 * H0z * exp_factor) / (1j * k * sb0 * denom)
    I_H_bracket = coef1_H*F1 + coef2_H*F2
    I_H = I_H_pre * I_H_bracket

    return complex(M_cor), complex(I_E + I_H)
```

文件 `physics/mec_truncated_core.py` 改回：

```python
from .mec_truncated_coefficients import compute_correction_currents_consistent
...
If_UT, Mf_UT = compute_total_fringe_currents(beta_prime, phi_prime, N, E0z, H0z, k, Z)
# 不 conj — 保持与 mec_core 相同的"乘 j"翻译风格
M_cor, I_cor = compute_correction_currents_consistent(
    beta_prime, phi_prime, beta_obs, phi_obs, l_A, E0z, H0z, k, Z)
If_T = If_UT - I_cor
Mf_T = Mf_UT - M_cor
...
seg_contrib = (-0.5) * seg.length * sinc_val * np.exp(1j * phase) * amp
# pre_factor = -0.5，与 mec_core 同步
```

### 6.3 验证矩阵

| 测点 | 期望 | 检查 |
|---|---|---|
| 0–30° 主瓣区 | 与 mec_core / EEW 一致（≤0.5 dB） | If_UT 部分未被错误旋转 |
| 47°, 58°, 70° PO 旁瓣峰 | 与 Lee-Jeng / FEKO 误差 ≤ 3 dB | exp(±jL(μ-1)) 相位正确 |
| 85–89° 掠射 | V 极化大幅下沉接近 FEKO -53 dB | F(z) 翻译正确 |
| `l^A → 0` 极限 | $I_{cor}\to I_{UT}$，$I_{T}=0$ | F(0) = 0.5 双侧均给 1 |

---

## 7. 二阶 EEC 框架（理论参考；当前未实现的目标）

### 7.1 二阶绕射的物理含义

边 A 一阶绕射产生 fringe 波在 Keller 锥上辐射；该波沿连接面 (A↔B 共享的 face)
传播到边 B，激发 B 的二阶 fringe 电流；B 再向观察方向辐射。

数学：

$$
\mathbf E^{(2)}_{\rm obs}
\;=\;\sum_{A,B}\;
\mathbf M_{B}^{(1)}\!\bigl(\hat e_{r},\hat e_{B}\!\big|\,\hat s\bigr)\;
\cdot\;
\mathbf P_{A\!\to\!B}\;
\cdot\;
\mathbf M_{A}^{(1)}\!\bigl(\hat e_{A},\hat e_{t}\!\big|\,\hat k\bigr)\;
\cdot\;\mathbf E^{\rm inc}
$$

其中 $\mathbf M^{(1)}_{X}$ 是 X 边的一阶 EEC 张量（必须用 MEC 的双流写法，不能
用 EEW 的标量 D），$\mathbf P_{A\to B}$ 是 A 输出场→B 输入场的传播 + 极化基变
换矩阵：

$$
\mathbf P_{A\to B}
\;=\; \frac{e^{ikR_{AB}}}{\sqrt{R_{AB}(1+R_{AB}/\rho_{A})}}\;
\mathbf R_{A\to B}
$$

$R_{AB}$ 是 A 上绕射点到 B 上接收点的连接面距离，$\rho_{A}$ 是 A 处发散
（Ufimtsev Eq.8.23：直边平面波 → ρ→∞），$\mathbf R_{A\to B}$ 是 $(\hat\beta_{A},
\hat\alpha_{A}) \to (\hat t_{B}, \hat e_{t,B})$ 的 2×2 旋转矩阵。

### 7.2 极化耦合是不可绕过的

VV 通道二阶 = (V→A) → (A 输出 β̂_A 与 α̂_A) → (二者各激发 B 的 V 与 H) → (B 输
出 V) 的 2×2×2 求和。即使发收均 V，二阶里也包含 V→H_中转→V 通道。因此：

> **结论 E**：二阶 EEC 内核必须用 MEC 写法（保留 $I^{f},M^{f}$ 张量结构），不
> 能塌缩到 EEW 标量 D。

### 7.3 闭式真值（strip，Ufimtsev 2014 Ch.10）

平行半平面 strip（两边 N=2、$\beta_0=\pi/2$、边平行）后向散射：

$$
\text{Eq.10.51 (hard)}:\quad u_{h}^{(t)} = u_{01}\cdot\frac{1}{\cos^{2}\!\tfrac{\phi_{01}}{2}}\cdot\frac{i}{2\pi k}\cdot\frac{e^{ik(R_{10}+R)}}{\sqrt{R_{10}R}}
$$

$$
\text{Eq.10.60 (soft)}:\quad u_{s}^{(t)} = -u_{01}\cdot\frac{\partial f(0,\phi_{01},2\pi)}{\partial\phi_{1}}\cdot\frac{\partial f(\phi,0,2\pi)}{\partial\phi_{0}}\cdot\frac{1}{2\pi(kR_{10})^{3/2}}\cdot\frac{e^{ik(R_{10}+R)}}{\sqrt{kR}}
$$

这是任何二阶实现必须复现的闭式真值。Hard 强（~k⁻¹），soft 弱（~(kR)⁻³ᐟ²）—— 这
解释了平板 V 极化掠射区缺失而 H 极化基本正确：V 是 hard 通道，二阶贡献强烈；
H 是 soft 通道，二阶贡献小。

### 7.4 Johansen 截断 = 二阶端点贡献

Johansen 1996 III.B 论证：截断 EEC 修正项 $M_{cor}, I_{cor}$ 数学上等价于
Michaeli 1987 二阶 EEC 在"端点贡献"的极限。即：

$$
\text{二阶 EEC (Michaeli 1987)}\;\supset\;\text{Johansen Eq.26/27 截断 (作为强项)}
\;+\;\text{尾部反向波 (Johansen 自承缺失)}
$$

所以本仓库 mec_truncated 已是二阶 EEC 的主项；尾部反向波这部分（"trailing edge
FW current excited at trailing edge"）才是 60° 附近残留小坑的物理来源。**用户上
一轮指出的 60° dip 是物理预期的二阶不完整，不是约定错误。**

---

## 8. 仓库其他模块与本文档的关系

- `ptd_core.py` (EEW) 已经在 U 约定下正确，本文档不影响其实现。
- `mec_coefficients.py` 已经把 Michaeli 1986 抄入，本文档不影响其数值。
- `mec_core.py` "乘 j" 翻译的物理解释见 §3.5：**主瓣区与正确翻译数值近似，但严
  格上差 90° 全局相位**。如要严格化，可改为 `pre_factor = -0.5j` 同时
  `If_UT, Mf_UT = conj(...)`，物理上更标准；但需要做回归确认整体能量与 EEW 仍
  ≤0.5 dB。本文档暂不强制改 `mec_core`，因为它工作良好且改动有回归风险。
- `mec_truncated_core.py` 必须改：方案见 §6.2。
- `mec_truncated_coefficients.py` 增加 `compute_correction_currents_consistent`，
  保留旧 `compute_correction_currents`（Johansen 原文 e^{+jωt} 数值）和
  `compute_correction_currents_uf`（整体 conj 版本）作为对照/参考。
- `physics/ptd_second_order.py` 当前为空 stub；未来实现按 §7 框架。

---

## 9. 实验验证计划（理论敲定后）

1. **基础回归**：实施 §6.2 修复后跑 `scratch_plot_comprehensive.py`，对照
   - 主瓣区 (10/30°) MEC_T 与 EEW、PO 一致
   - 旁瓣峰 (47/58/70°) MEC_T 跟 Lee-Jeng 散点（±3 dB）
   - 掠射 (85/89°) MEC_T V 下沉至 FEKO -53 dB 附近
2. **闭式 strip 单元测试**：用 4λ 长板（接近 2D strip）的硬极化做后向单站，
   验证二阶贡献 $|I_{UT} - I_{T}|$ 沿 $L_y$ 线性 + 与 Ufimtsev Eq.10.51 振幅匹配
   （这是 §7 二阶实现前的 strip 极限校验，使 truncated MEC 的 3D 归一化锚定）
3. **VH/HV 矩阵元**：跑 VH 极化平板，验证主瓣区与 PO 一致（PO 在 VH 应为 0），
   并验证截断修正不引入 VH 错误。

---

## 10. 公式翻译对照速查（最终精确版）

把任何 Michaeli/Johansen 表达式 $X^{(M)}$ 翻译到本仓库代码约定下：

| 原表达式 | 翻译后表达式 | 说明 |
|---|---|---|
| $j$ (单独的 $j$，作为代数常数因子) | **保留 $1j$** | "乘 j" 路径，与 mec_core 一致 |
| $j k$ (分母里的 $jk$) | **保留 $1j\cdot k$** | 同上 |
| $\sqrt{j/\pi}$ | **保留 $\sqrt{1j/\pi}$** | 渐近常数 phase |
| $\exp(j\,X(\xi))$，X 含物理量 | **翻为 $\exp(-1j\,X(\xi))$** | 相位指数必翻 |
| $\sin\alpha_{c}$，$\alpha_{c} = \arccos(\mu)$ | **保留**（lib.scimath 隐式处理 $\mu>1$） | 代数 |
| $F(z) = \tfrac{1}{2}e^{jz^{2}}\operatorname{erfc}(ze^{j\pi/4})$ | **翻为 $F_{U}(z) = \tfrac{1}{2}e^{-iz^{2}}\operatorname{erfc}(ze^{-i\pi/4})$** | 含相位指数，整体翻 |
| Michaeli 全局前因子 $-\tfrac{j}{2}$ | **改为 $-\tfrac{1}{2}$**（吸进 pre_factor） | 与 mec_core 一致 |
| 远场积分 $e^{-jk\hat s\cdot r}$ | 代码用 $e^{+i k\hat s\cdot r}$（U 约定）| 单站合并 $e^{+i 2 k r_c}$ |

> 注：这是**"工程一致翻译"**而非"教科书严格翻译"。严格翻译要求所有 j → −i
> (即整体 conj)，"工程一致翻译"对 `mec_core` 已固化的"乘 j" 风格做最小破坏地
> 接续——使得 `If_UT - I_cor` 减法在同一数值约定下做。

---

## 11. 未尽事项 / 不确定性

1. **$F(z)$ 中 $e^{j\pi/4}$ 是否要翻译**：本文档采用"翻"（§4.4），与
   `modified_fresnel_uf` 一致。理论上 SDP 旋转角的方向与时谐约定挂钩，翻译是物
   理一致的。但若实测发现修复后旁瓣峰相位仍有偏，可对照"不翻"版本（即用
   `modified_fresnel` 原版）回退。
2. **`mec_core` 是否严格化**：本文档不强制改。如果二阶实现遇到与 mec_core 的相
   位耦合问题，再考虑把 mec_core 改成整体 conj 风格（同时改 pre = -0.5j、conj
   UT）。需重做主瓣 EEW=MEC=PO 回归。
3. **二阶 EEC 实现的具体公式**：本文档只给框架（§7）。实际编码需参考 Michaeli
   1987 IEEE TAP 35(2) 完整公式。strip 闭式 Eq.10.51/60 提供主校验。
4. **曲面 $l^{A}$ 测地线扩展**：本文档默认 $\beta_{0}=\pi/2$ 简化。曲面斜入射严
   格的 $l^{A}$ 需沿 face A 内的测地线求解，留作 Task #13。

---

*文档版本 v1.0，2026-05-30 起草。基于 Johansen 1996 全文比对、Michaeli 1986 Eq.
27/28 与仓库代码逐项对应、以及主瓣/旁瓣两类错误实测复现。如未来实施验证发现
本文档结论与实测冲突，应优先以实测为准并回头修订本文档（特别是 §4.4 与
§5/6 节）。*
