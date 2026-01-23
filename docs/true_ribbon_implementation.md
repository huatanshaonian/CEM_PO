# 真正 Ribbon 方法实现指南

## 1. 背景

### 1.1 当前实现 (Discrete PO + Sinc)

当前代码 (`DiscretePOIntegrator`) 的做法：
- 将曲面离散为 `nv × nu` 的网格
- 每个小格子内假设相位是 u 的**线性函数**
- 使用 `sinc(α·du/2π)` 因子修正相位误差

```python
# 当前实现 (ribbon_solver.py)
sinc_term = np.sinc(alpha * du / (2.0 * np.pi))
contributions = illumination * jacobians * exp(i*phase) * sinc_term * du * dv
```

**问题**：这本质上是离散方法，只是加了一阶修正。需要较多网格点才能收敛。

### 1.2 真正的 Ribbon 方法

论文 (CADDSCAT, 1995) 中描述的方法：
- v 方向：离散为 `nv` 条 ribbon
- u 方向：**每条 ribbon 整体解析积分**，不离散

**优势**：
- 计算量从 O(N²) 降为 O(N)
- 相同网格数下精度大幅提升
- 或者说：相同精度下网格数大幅减少

---

## 2. 数学基础

### 2.1 PO 积分公式

单站 RCS 的 PO 积分：

```
I = ∫∫_S (n̂·k̂) · exp(i·2k·P) dS
```

其中：
- `n̂` 是表面法向量
- `k̂` 是入射波方向
- `P` 是表面上的点
- `k = 2π/λ` 是波数

### 2.2 参数曲面表示

对于参数曲面 `P(u,v)`：
```
dS = |∂P/∂u × ∂P/∂v| du dv = J(u,v) du dv
```

积分变为：
```
I = ∫∫ G(u,v) · exp(i·φ(u,v)) du dv
```

其中：
- `G(u,v) = (n̂·k̂) · J(u,v)` 是振幅函数
- `φ(u,v) = 2k·P(u,v)` 是相位函数

### 2.3 Ribbon 分解

将 2D 积分分解为：
```
I = ∫_v [ ∫_u G(u,v) · exp(i·φ(u,v)) du ] dv
     ↑        ↑
   数值积分   解析积分
```

关键：对于固定的 `v`，内层 u 积分可以解析计算。

---

## 3. Bi-cubic 曲面的解析积分

### 3.1 Bi-cubic 曲面定义

Bi-cubic 参数曲面：
```
P(u,v) = Σᵢ₌₀³ Σⱼ₌₀³ aᵢⱼ uⁱ vʲ
```

对于固定的 `v = v₀`：
```
P(u, v₀) = Σᵢ₌₀³ bᵢ(v₀) uⁱ    （u 的三次多项式）
```

### 3.2 相位函数

```
φ(u) = 2k · P(u, v₀) = 2k · Σᵢ₌₀³ bᵢ uⁱ = c₀ + c₁u + c₂u² + c₃u³
```

相位是 u 的**三次多项式**。

### 3.3 振幅函数

对于 bi-cubic 曲面，论文指出：
```
G(u) = (n̂·k̂) · J(u,v₀)
```

这是 u 的**五阶多项式**（因为法向量和 Jacobian 都涉及导数的叉乘）。

### 3.4 核心积分

每条 ribbon 的积分形式：
```
I_ribbon(v₀) = ∫₀¹ G(u) · exp(i·φ(u)) du
```

其中 G(u) 是五阶多项式，φ(u) 是三阶多项式。

---

## 4. 解析积分方法

### 4.1 方法一：驻点法 (Stationary Phase)

当相位变化快时（高频），使用驻点近似：
```
∫ G(u) exp(i·φ(u)) du ≈ Σ G(uₛ) √(2π/|φ''(uₛ)|) exp(i·φ(uₛ) ± iπ/4)
```

其中 `uₛ` 是驻点（满足 `φ'(uₛ) = 0`）。

**优点**：高频时非常准确
**缺点**：低频或无驻点时不适用

### 4.2 方法二：Ludwig 积分

将积分分解为边界项和递归项：
```
∫ uⁿ exp(i·φ(u)) du = [uⁿ exp(i·φ)]/(i·φ') - n/(i·φ') ∫ uⁿ⁻¹ exp(i·φ) du
```

对于多项式相位，可以递归求解。

### 4.3 方法三：分段 + Fresnel 积分

将 u 区间分成小段，每段内将相位近似为二次：
```
φ(u) ≈ φ₀ + φ₁(u-u₀) + φ₂(u-u₀)²
```

二次相位的积分涉及 Fresnel 积分 C(x) 和 S(x)。

### 4.4 推荐方法：混合策略 (CADDSCAT 使用)

1. **判断相位类型**：
   - 计算 φ'(u) 在 [0,1] 上的变化
   - 如果 |φ'| 始终很大：使用驻点法
   - 如果 |φ'| 有零点：需要处理阴影边界

2. **阴影边界处理**：
   - G(u) 的零点对应阴影边界
   - 分段积分：[0, u_shadow] 被照亮，[u_shadow, 1] 阴影

3. **数值稳定性**：
   - 对于接近掠入射（grazing incidence），需要特殊处理

---

## 5. 实现步骤

### 5.1 数据结构

```python
class TrueRibbonIntegrator:
    """真正的 Ribbon 积分器"""

    def __init__(self, nv=None, samples_per_lambda=8):
        """
        参数:
            nv: v 方向的 ribbon 数（如果不指定则自动计算）
            samples_per_lambda: 每波长 ribbon 数
        """
        self.nv_manual = nv
        self.default_samples_per_lambda = samples_per_lambda
```

### 5.2 预计算 ribbon 数据

```python
def precompute_ribbons(self, surface, wavelength):
    """
    预计算每条 ribbon 的多项式系数

    返回:
        list of RibbonData，每个包含：
        - v_center: ribbon 的 v 坐标
        - dv: ribbon 宽度
        - G_coeffs: G(u) 的多项式系数 (6个，0-5阶)
        - phi_coeffs: φ(u) 的多项式系数 (4个，0-3阶)
    """
    nv = self._estimate_ribbon_count(surface, wavelength)

    v_min, v_max = surface.v_domain
    dv = (v_max - v_min) / nv
    v_centers = np.linspace(v_min + dv/2, v_max - dv/2, nv)

    ribbons = []
    for v in v_centers:
        # 计算该 ribbon 的多项式系数
        G_coeffs, phi_coeffs = self._compute_ribbon_polynomials(surface, v)
        ribbons.append(RibbonData(v, dv, G_coeffs, phi_coeffs))

    return ribbons
```

### 5.3 计算多项式系数

```python
def _compute_ribbon_polynomials(self, surface, v_center):
    """
    对于固定 v，计算 G(u) 和 φ(u) 的多项式系数

    对于 bi-cubic 曲面：
    - P(u) 是 u 的三次多项式
    - ∂P/∂u 是二次多项式
    - n̂(u) 涉及叉乘，是四阶多项式
    - J(u) = |∂P/∂u × ∂P/∂v| 是四阶多项式
    - G(u) = (n̂·k̂) · J 是五阶多项式
    - φ(u) = 2k·P(u) 是三次多项式

    方法：在 u 方向采样 6+ 个点，拟合多项式
    """
    u_min, u_max = surface.u_domain

    # 采样足够多的点来拟合多项式
    n_samples = 10  # 对于五阶多项式，需要至少 6 个点
    u_samples = np.linspace(u_min, u_max, n_samples)

    # 计算每个采样点的 G 和 φ 值
    G_values = []
    phi_values = []

    for u in u_samples:
        P = surface.evaluate(u, v_center)
        n = surface.get_normal(u, v_center)
        J = surface.get_jacobian(u, v_center)

        # G = (n · k_dir) * J * (-1)  # -1 因为只积分被照亮区域
        n_dot_k = np.dot(n, k_dir)
        G = -n_dot_k * J if n_dot_k < 0 else 0.0

        # φ = 2 * P · k_vec
        phi = 2.0 * np.dot(P, k_vec)

        G_values.append(G)
        phi_values.append(phi)

    # 多项式拟合
    # G(u) 五阶多项式
    G_coeffs = np.polyfit(u_samples, G_values, 5)

    # φ(u) 三阶多项式
    phi_coeffs = np.polyfit(u_samples, phi_values, 3)

    return G_coeffs, phi_coeffs
```

### 5.4 解析积分核心

```python
def _integrate_ribbon_analytic(self, G_coeffs, phi_coeffs, u_min, u_max):
    """
    解析计算单条 ribbon 的积分

    ∫ G(u) exp(i·φ(u)) du

    其中 G(u) = g₅u⁵ + g₄u⁴ + ... + g₀
         φ(u) = p₃u³ + p₂u² + p₁u + p₀
    """
    # 方法：数值 Gauss 积分 + 解析优化
    #
    # 对于高频情况，使用驻点法
    # 对于低频情况，使用高阶 Gauss 积分

    # 简化实现：使用自适应 Gauss 积分
    # （真正的 CADDSCAT 使用更复杂的解析公式）

    from scipy.integrate import quad

    def integrand_real(u):
        G = np.polyval(G_coeffs, u)
        phi = np.polyval(phi_coeffs, u)
        return G * np.cos(phi)

    def integrand_imag(u):
        G = np.polyval(G_coeffs, u)
        phi = np.polyval(phi_coeffs, u)
        return G * np.sin(phi)

    real_part, _ = quad(integrand_real, u_min, u_max)
    imag_part, _ = quad(integrand_imag, u_min, u_max)

    return real_part + 1j * imag_part
```

### 5.5 阴影边界处理

```python
def _find_shadow_boundaries(self, G_coeffs, u_min, u_max):
    """
    找到 G(u) = 0 的根（阴影边界）

    G(u) = 0 对应 n·k = 0，即阴影边界
    """
    # G 是五阶多项式，最多 5 个实根
    roots = np.roots(G_coeffs)

    # 只保留 [u_min, u_max] 区间内的实根
    real_roots = []
    for r in roots:
        if np.isreal(r):
            r = r.real
            if u_min < r < u_max:
                real_roots.append(r)

    return sorted(real_roots)
```

---

## 6. 与 OCC 曲面的适配

### 6.1 问题

OCC (OpenCascade) 曲面不一定是 bi-cubic 的，可能是：
- NURBS (有理 B-spline)
- 圆柱、球等解析曲面
- 任意参数曲面

### 6.2 解决方案

**方案 A：多项式拟合（推荐）**
- 对任意曲面，在每条 ribbon 上采样足够多的点
- 用最小二乘拟合 G(u) 和 φ(u) 的多项式
- 优点：通用，适用于任何曲面
- 缺点：拟合可能有误差

**方案 B：分段线性/二次近似**
- 将每条 ribbon 分成小段
- 每段内用低阶多项式近似
- 优点：简单
- 缺点：需要更多段数

**方案 C：自适应数值积分**
- 对复杂曲面，使用自适应 Gauss 积分
- 根据被积函数的变化自动调整采样点
- 优点：稳健
- 缺点：计算量可能较大

---

## 7. 实现建议

### 7.1 分阶段实现

**Phase 1：基础框架**
- 实现 `TrueRibbonIntegrator` 类
- v 方向离散，u 方向使用高阶 Gauss 积分（而非解析）
- 这已经比当前方法好，因为不需要在 u 方向离散

**Phase 2：多项式拟合**
- 为 bi-cubic 曲面实现多项式系数计算
- 使用 scipy 的解析积分

**Phase 3：优化**
- 实现驻点法处理高频情况
- 实现阴影边界检测和分段积分
- 添加缓存机制

### 7.2 验证测试

1. **圆柱体**：与解析解对比
2. **平板**：与解析解对比
3. **球体**：与光学区公式对比
4. **复杂 STEP 模型**：与当前离散方法对比

---

## 8. 代码模板

```python
class TrueRibbonIntegrator:
    """
    真正的 Ribbon 积分器

    v 方向离散为 ribbon，u 方向整体积分（高阶 Gauss 或解析）
    """

    def __init__(self, samples_per_lambda=8, u_gauss_order=16):
        self.default_samples_per_lambda = samples_per_lambda
        self.u_gauss_order = u_gauss_order  # u 方向 Gauss 积分阶数

        # 预计算 Gauss 点和权重
        self.gauss_points, self.gauss_weights = np.polynomial.legendre.leggauss(u_gauss_order)

    def integrate_surface(self, surface, wave, samples_per_lambda=None):
        """计算单个曲面的 PO 积分"""
        spl = samples_per_lambda or self.default_samples_per_lambda
        nv = self._estimate_nv(surface, wave.wavelength, spl)

        v_min, v_max = surface.v_domain
        u_min, u_max = surface.u_domain
        dv = (v_max - v_min) / nv

        # 将 Gauss 点映射到 [u_min, u_max]
        u_gauss = 0.5 * (u_max - u_min) * (self.gauss_points + 1) + u_min
        u_scale = 0.5 * (u_max - u_min)

        total_I = 0j

        for iv in range(nv):
            v_center = v_min + (iv + 0.5) * dv

            # 对这条 ribbon，在 u 方向用 Gauss 积分
            ribbon_I = 0j
            for ig, (u_g, w_g) in enumerate(zip(u_gauss, self.gauss_weights)):
                # 计算该点的贡献
                P = surface.evaluate(u_g, v_center)
                n = surface.get_normal(u_g, v_center)
                J = surface.get_jacobian(u_g, v_center)

                n_dot_k = np.dot(n, wave.k_dir)
                if n_dot_k >= 0:  # 阴影区
                    continue

                illumination = -n_dot_k
                phase = 2.0 * np.dot(P, wave.k_vector)

                ribbon_I += illumination * J * np.exp(1j * phase) * w_g

            ribbon_I *= u_scale  # Gauss 积分的缩放因子
            total_I += ribbon_I * dv

        return total_I
```

---

## 9. 性能对比预期

| 方法 | 网格数 (相同精度) | 相对计算量 |
|------|-----------------|-----------|
| 当前离散 PO | nv × nu | 1.0 |
| 真 Ribbon (Gauss u) | nv × n_gauss | ~0.3 |
| 真 Ribbon (解析 u) | nv | ~0.1 |

---

## 10. 参考资料

1. Elking et al., "A Review of High-Frequency RCS Analysis Capabilities at MDA", IEEE AP Magazine, 1995
2. Gordon, "Far-field approximations to the Kirchoff-Helmholtz representations of scattered fields", IEEE TAP, 1975
3. Ludwig, "Computation of radiation patterns involving numerical double integration", IEEE TAP, 1968
