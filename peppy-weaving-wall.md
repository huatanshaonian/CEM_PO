# Ribbon 积分算法开发计划

## 1. 背景与现状分析

### 1.1 PDF论文核心要点（CADDSCAT, 1995）

**Ribbon技术定义**（Section 3.2.3）：
- v方向：Gaussian数值积分
- u方向：**解析积分**（对于非有理bi-cubic曲面）
- G(u)是五阶实多项式，φ(u)是三阶多项式
- 精度：8个Gaussian采样 ≈ 16个梯形采样，收敛误差 0.5dB
- **限制**：快速解析技术仅限于**非有理双三次参数曲面(non-rational PBC)**

### 1.2 当前项目实现状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | TrueRibbonIntegrator (自适应分段Gauss积分) | ✅ 已实现 |
| Phase 2 | 多项式拟合 + 精确阴影边界检测 | ❌ 未实现 |
| Phase 3 | 驻点法优化（高频场景） | ❌ 未实现 |

**当前 TrueRibbonIntegrator 特点**：
- ✅ v方向离散为nv条ribbon
- ✅ u方向自适应分段Gauss积分（基于相位变化）
- ✅ 相位稳定化（参考点技术）
- ❌ 阴影边界检测在每个Gauss点单独判断，非精确
- ❌ 无多项式系数计算

### 1.3 true_ribbon_implementation.md 文档评估

**正确的部分**：
- 核心数学公式正确
- G(u)五阶、φ(u)三阶多项式
- 实现建议的分阶段策略

**需补充的点**：
- 解析技术限于non-rational bi-cubic曲面
- OCC任意曲面需用数值方法或多项式拟合

---

## 2. 开发计划

### 核心目标：严格按照论文实现真正的Ribbon积分

**当前问题**：现有 `TrueRibbonIntegrator` 使用Gauss数值积分，这不是论文描述的方法。
**解决方案**：重写为基于多项式的解析积分。

### 论文中的真正Ribbon算法（Section 3.2.3）

```
I_ribbon(v₀) = ∫ G(u) × exp(i·φ(u)) du

其中：
- G(u) = (n̂·k̂) × J(u) 是五阶实多项式
- φ(u) = 2k·P(u) 是三阶多项式
```

**论文明确的关键步骤**：
1. 将曲面在v方向离散为ribbon
2. 对每条ribbon，计算G(u)的多项式系数
3. 找G(u)=0的根（阴影边界），精度达到**百万分之一**
4. 只在被照亮区间[u_a, u_b]进行解析积分

---

### Phase 2A: 多项式系数计算模块

**文件**: `solver/ribbon_polynomials.py` (新建)

```python
class RibbonPolynomialCalculator:
    """计算单条ribbon的G(u)和φ(u)多项式系数"""

    def compute_coefficients(surface, v_center, wave) -> (G_coeffs, phi_coeffs):
        """
        对于bi-cubic曲面，G(u)和φ(u)都是u的多项式。

        方法：在u方向采样n_samples个点，最小二乘拟合多项式

        返回：
        - G_coeffs: [g5, g4, g3, g2, g1, g0] (五阶，numpy convention)
        - phi_coeffs: [p3, p2, p1, p0] (三阶)
        """

    def find_shadow_boundaries(G_coeffs, u_min, u_max, tol=1e-6) -> List[float]:
        """
        找G(u)=0在[u_min, u_max]内的实根

        论文要求：精度达到 one part per million (1e-6)
        """
```

**实现要点**：
1. G(u) = -(n̂·k̂) × J(u)，注意负号（n̂·k̂ < 0时被照亮）
2. 采样点数 ≥ 10（对五阶多项式，最小6个点）
3. 使用 `np.polyfit` 拟合
4. 根求解后用Newton-Raphson精化到1e-6

### Phase 2B: 解析积分核心（关键！）

**文件**: `solver/ribbon_analytic.py` (新建)

论文提到多种方法，推荐**Ludwig积分**（分部积分递归）：

```python
def ludwig_integral(G_coeffs, phi_coeffs, u_a, u_b) -> complex:
    """
    Ludwig积分方法（分部积分递归）

    核心公式：
    ∫ u^n × exp(i·φ(u)) du = [u^n × exp(i·φ)] / (i·φ')
                             - n/(i·φ') × ∫ u^(n-1) × exp(i·φ) du

    对于G(u) = Σ gₙ u^n，逐项应用上述公式。

    边界条件：∫ u^0 × exp(i·φ) du 需要特殊处理（Fresnel或数值）
    """

def integrate_illuminated_segment(G_coeffs, phi_coeffs, u_a, u_b) -> complex:
    """
    在被照亮区间[u_a, u_b]上积分

    选择策略：
    - 如果相位变化 |Δφ| < 2π：高阶Gauss积分
    - 如果相位变化 |Δφ| > 2π 且有驻点：驻点法
    - 否则：Ludwig积分或分段Fresnel
    """
```

### Phase 3: 驻点法（高频优化）

**文件**: `solver/ribbon_stationary_phase.py` (新建)

```python
def stationary_phase_integral(G_coeffs, phi_coeffs, u_a, u_b) -> complex:
    """
    驻点近似（高频有效）：

    ∫ G(u) exp(i·φ(u)) du ≈ Σ G(uₛ) × √(2π/|φ''(uₛ)|) × exp(i·φ(uₛ) ± iπ/4)

    其中 uₛ 是驻点：φ'(uₛ) = 0
    符号：φ''(uₛ) > 0 用 +iπ/4，否则 -iπ/4

    注意：
    - 只对[u_a, u_b]区间内的驻点求和
    - 边界贡献另外处理
    """

def fresnel_functions(x) -> (C, S):
    """
    Fresnel积分：C(x) = ∫cos(πt²/2)dt, S(x) = ∫sin(πt²/2)dt

    使用scipy.special.fresnel实现
    """
```

---

## 3. 具体实现步骤

### Step 1: 新建 `AnalyticRibbonIntegrator` 类

**不修改现有 TrueRibbonIntegrator**（保留作为对比基准），新建严格按照论文的实现。

**文件**: `solver/ribbon_solver.py` (新增类)

```python
class AnalyticRibbonIntegrator:
    """
    严格按照论文实现的Ribbon积分器

    与 TrueRibbonIntegrator 的区别：
    - TrueRibbonIntegrator: 使用自适应Gauss数值积分
    - AnalyticRibbonIntegrator: 使用多项式拟合 + 解析积分

    参考: Elking et al., "A Review of High-Frequency RCS Analysis
          Capabilities at MDA", IEEE AP Magazine, 1995
    """

    def __init__(self, nv=None, samples_per_lambda=8,
                 n_fit_samples=10, shadow_tol=1e-6):
        """
        参数:
            nv: v方向ribbon数（自动估算若不指定）
            samples_per_lambda: 每波长采样数
            n_fit_samples: 多项式拟合采样点数（≥10）
            shadow_tol: 阴影边界精度（论文要求1e-6）
        """
        self.nv_manual = nv
        self.samples_per_lambda = samples_per_lambda
        self.n_fit_samples = max(10, n_fit_samples)
        self.shadow_tol = shadow_tol

    def integrate_surface(self, surface, wave, samples_per_lambda=None):
        """
        主入口：对曲面进行Ribbon积分

        流程：
        1. 估算ribbon数nv
        2. 对每条ribbon:
           a. 计算G(u), φ(u)多项式系数
           b. 找阴影边界（G(u)=0的根）
           c. 在被照亮区间进行解析积分
        3. 累加所有ribbon贡献
        """
        pass  # 具体实现见下文
```

### Step 2: 实现多项式系数计算

```python
def _compute_polynomial_coeffs(self, surface, wave, v_center):
    """
    计算固定v时，G(u)和φ(u)的多项式系数

    G(u) = -(n̂·k̂) × J(u)   (五阶多项式)
    φ(u) = 2k·P(u)·k̂       (三阶多项式)

    注意：论文定义 G = (n̂·k̂) × J，负值表示被照亮
    这里我们用 G = -(n̂·k̂) × J，正值表示被照亮
    """
    u_min, u_max = surface.u_domain
    u_samples = np.linspace(u_min, u_max, self.n_fit_samples)
    v_arr = np.full_like(u_samples, v_center)

    # 获取几何数据
    points, normals, jacobians = surface.get_data(u_samples, v_arr)

    k_vec = wave.k_vector
    k_dir = wave.k_dir

    # G(u) = -(n·k_dir) * J
    n_dot_k = np.sum(normals * k_dir, axis=-1)
    G_values = -n_dot_k * jacobians

    # φ(u) = 2 * P · k_vec
    phi_values = 2.0 * np.sum(points * k_vec, axis=-1)

    # 多项式拟合
    G_coeffs = np.polyfit(u_samples, G_values, 5)    # 五阶
    phi_coeffs = np.polyfit(u_samples, phi_values, 3)  # 三阶

    return G_coeffs, phi_coeffs
```

### Step 3: 实现阴影边界检测（论文核心！）

```python
def _find_shadow_boundaries(self, G_coeffs, u_min, u_max):
    """
    找G(u)=0在[u_min, u_max]内的实根

    论文要求：精度达到 one part per million (1e-6)

    步骤：
    1. np.roots() 获取所有根（可能有复根）
    2. 过滤：只保留区间内的实根
    3. Newton-Raphson 精化到指定精度
    """
    roots = np.roots(G_coeffs)
    real_roots = []

    # 计算G(u)的导数多项式系数（用于Newton-Raphson）
    dG_coeffs = np.polyder(G_coeffs)

    for r in roots:
        # 只考虑近似实数的根
        if np.abs(r.imag) < 0.01:
            r_real = r.real
            # 只考虑区间内的根
            if u_min - 0.01 < r_real < u_max + 0.01:
                # Newton-Raphson 精化
                r_refined = self._newton_raphson(G_coeffs, dG_coeffs,
                                                  r_real, u_min, u_max)
                if r_refined is not None:
                    real_roots.append(r_refined)

    return sorted(real_roots)

def _newton_raphson(self, poly_coeffs, dpoly_coeffs, x0, u_min, u_max,
                    max_iter=20):
    """Newton-Raphson迭代精化根"""
    x = x0
    for _ in range(max_iter):
        f = np.polyval(poly_coeffs, x)
        df = np.polyval(dpoly_coeffs, x)
        if abs(df) < 1e-12:
            break
        x_new = x - f / df
        if abs(x_new - x) < self.shadow_tol:
            if u_min <= x_new <= u_max:
                return x_new
            break
        x = x_new
    return None  # 收敛失败或超出区间
```

### Step 4: 确定被照亮区间

```python
def _get_illuminated_intervals(self, G_coeffs, u_min, u_max, shadow_bounds):
    """
    根据阴影边界确定被照亮的区间

    判断方法：检查区间中点的G值
    G > 0 表示 -(n·k) > 0，即 n·k < 0，被照亮
    """
    all_bounds = [u_min] + shadow_bounds + [u_max]
    lit_intervals = []

    for i in range(len(all_bounds) - 1):
        u_start, u_end = all_bounds[i], all_bounds[i+1]
        u_mid = (u_start + u_end) / 2
        G_mid = np.polyval(G_coeffs, u_mid)

        if G_mid > 0:  # 被照亮
            lit_intervals.append((u_start, u_end))

    return lit_intervals
```

### Step 5: 解析积分（可先用Gauss作为初步实现）

```python
def _integrate_segment_analytic(self, G_coeffs, phi_coeffs, u_a, u_b,
                                  ref_point, k_vec):
    """
    在被照亮区间[u_a, u_b]上进行解析积分

    初步实现：使用高阶Gauss积分（因为多项式已知）
    后续优化：实现Ludwig积分或驻点法
    """
    # 使用16阶Gauss积分（对多项式×振荡函数足够精确）
    nodes, weights = np.polynomial.legendre.leggauss(16)

    # 映射到[u_a, u_b]
    u_scale = (u_b - u_a) / 2
    u_shift = (u_b + u_a) / 2
    u_arr = nodes * u_scale + u_shift

    # 评估多项式
    G_values = np.polyval(G_coeffs, u_arr)
    phi_values = np.polyval(phi_coeffs, u_arr)

    # 相位稳定化
    phi_ref = 2.0 * np.dot(ref_point.flatten(), k_vec)
    phi_local = phi_values - phi_ref

    # 积分
    integrand = G_values * np.exp(1j * phi_local)
    return np.sum(weights * integrand) * u_scale * np.exp(1j * phi_ref)
```

### Step 6: 验证测试

**关键测试用例**：

```python
def test_shadow_boundary_cylinder():
    """圆柱阴影边界测试"""
    cylinder = AnalyticCylinder(radius=0.5, height=2.0)
    wave = IncidentWave(frequency=3e9, theta=np.pi/4)  # 45度入射

    integrator = AnalyticRibbonIntegrator()

    # 对于圆柱，阴影边界应该在 n·k = 0 处
    # 即入射方向与表面相切的位置
    v_center = 0.0
    G_coeffs, phi_coeffs = integrator._compute_polynomial_coeffs(
        cylinder, wave, v_center)
    shadow_bounds = integrator._find_shadow_boundaries(
        G_coeffs, 0, 1)

    # 验证：对于45度入射，阴影边界应该在u=0.25和u=0.75附近
    # （对应圆周角 π/2 和 3π/2）
    print(f"检测到的阴影边界: {shadow_bounds}")
    assert len(shadow_bounds) >= 1

def test_compare_with_gauss_integrator():
    """与Gauss数值积分对比"""
    plate = AnalyticPlate(width=1.0, height=1.0)
    wave = IncidentWave(frequency=3e9, theta=0)  # 正入射

    analytic = AnalyticRibbonIntegrator()
    gauss = TrueRibbonIntegrator()

    I_analytic = analytic.integrate_surface(plate, wave)
    I_gauss = gauss.integrate_surface(plate, wave)

    # 应该非常接近
    rel_error = abs(I_analytic - I_gauss) / abs(I_gauss)
    print(f"相对误差: {rel_error:.2e}")
    assert rel_error < 0.01  # 1%误差以内
```

---

## 4. 关键文件修改清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `solver/ribbon_solver.py` | 修改 | 新增 `AnalyticRibbonIntegrator` 类 |
| `solver/ribbon_polynomials.py` | 新建 | 多项式系数计算工具函数 |
| `tests/test_analytic_ribbon.py` | 新建 | 专项测试 |
| `docs/true_ribbon_implementation.md` | 更新 | 补充非有理bi-cubic限制说明 |

---

## 5. 验证方案

### 5.1 数值验证

1. **阴影边界验证**：
   - 圆柱45度入射：验证边界位置是否在预期角度
   - 球体：验证边界数量和位置
   - 平板正入射：应无阴影边界

2. **与解析解对比**：
   - 平板：PO解析解已知
   - 圆柱：有参考公式
   - 球体：光学区公式 σ = πR²

3. **与现有Gauss积分对比**：相同条件下结果应接近（误差<1%）

### 5.2 算法对比矩阵

| 方法 | 复杂度 | 阴影边界处理 | 适用场景 |
|------|--------|-------------|----------|
| 离散PO (sinc) | O(nv × nu) | 逐点检测 | 通用基准 |
| TrueRibbon (Gauss) | O(nv × n_seg × n_gauss) | 逐点检测 | 快速计算 |
| **AnalyticRibbon** | O(nv × n_fit) | **精确求根** | 高精度需求 |

### 5.3 测试命令

```bash
# 运行专项测试
python -m pytest tests/test_analytic_ribbon.py -v

# 阴影边界可视化
python tools/visualize_shadow_boundary.py

# 算法对比测试
python test_algorithms.py --compare discrete_po gauss_ribbon analytic_ribbon

# GUI验证
python gui.py
```

---

## 6. 风险与注意事项

1. **论文限制**：解析技术仅适用于**非有理双三次参数曲面(non-rational PBC)**
   - 对于OCC的NURBS曲面，多项式拟合可能有误差
   - 建议：对非bi-cubic曲面，回退到Gauss数值积分

2. **数值稳定性**：
   - 高阶多项式求根可能不稳定（五阶多项式）
   - 解决：使用Newton-Raphson精化 + 区间检查

3. **边界情况**：
   - 掠入射(grazing incidence)时G(u)接近0，阴影边界检测困难
   - 解决：增加采样点，使用更严格的容差

4. **向后兼容**：
   - 新增 `AnalyticRibbonIntegrator` 类，不修改现有接口
   - 在 `AVAILABLE_ALGORITHMS` 中添加新选项

---

## 7. 开发顺序

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1 | 实现 `_compute_polynomial_coeffs` | 多项式系数计算 |
| 2 | 实现 `_find_shadow_boundaries` | 阴影边界检测（论文核心） |
| 3 | 实现 `_get_illuminated_intervals` | 被照亮区间确定 |
| 4 | 实现 `_integrate_segment_analytic` | 区间积分（初步用Gauss） |
| 5 | 组装 `integrate_surface` 主流程 | 完整积分器 |
| 6 | 编写测试并验证 | 测试通过 |
| 7 | （后续）实现真正的Ludwig/驻点法积分 | 性能优化 |

---

## 8. 总结

**核心改进**：
- 从"逐点检测阴影"改为"多项式求根精确定位阴影边界"
- 严格按照论文Section 3.2.3的算法实现
- 阴影边界精度达到**百万分之一**（论文要求）

**新增算法**：`analytic_ribbon`（将添加到 `AVAILABLE_ALGORITHMS`）

**保留现有**：`TrueRibbonIntegrator` 作为对比基准
