# PTD算法90dB亮斑问题修复计划

## 问题根源

当前代码使用的绕射系数公式在法向入射时产生**伪奇异性**：

```python
# 当前代码 (physics/ptd_core.py:70-79)
denom_utd = np.cos(np.pi/n) - np.cos(2*phi/n)  # 法向入射时=0
if abs(denom_utd) < 1e-6: denom_utd = 1e-6
D_utd = (1.0/n) * np.sin(np.pi/n) / denom_utd  # → 577,000

D_po_approx = -0.5 * np.tan(alpha)  # alpha=0时为0，无法抵消
D_coeff = D_utd - D_po_approx       # → 577,000 (导致90dB)
```

**关键问题**：使用sin/cos形式的分母在法向入射时为0，而PO修正项公式错误无法抵消。

---

## 修复方案

### 核心修改：使用余切公式替换sin/cos公式

**文件**: `physics/ptd_core.py`

将第66-79行替换为：

```python
# 3. PTD绕射系数 (使用余切公式，数值稳定)
n = edge.n_param  # 1.5

# 对于后向散射，使用余切公式计算精确解
cot_arg = np.pi / (2 * n)  # pi/3 for n=1.5
sin_cot = np.sin(cot_arg)

if abs(sin_cot) > 1e-10:
    cot_val = np.cos(cot_arg) / sin_cot  # 0.577 for n=1.5
    D_exact = -(1.0 / n) * cot_val       # -0.385 for n=1.5
else:
    D_exact = 0.0

# PO绕射系数
cos_alpha_val = np.cos(alpha)
if abs(cos_alpha_val) > 0.01:  # 避免掠射奇异
    D_po = -1.0 / (2.0 * cos_alpha_val)  # -0.5 at normal
else:
    D_po = 0.0  # 掠射时PO无贡献

# PTD = Exact - PO
D_coeff = D_exact - D_po  # 0.115 at normal (合理值)

# 安全限制
D_coeff = np.clip(D_coeff, -50.0, 50.0)
```

### 验证数值

| 参数 | 当前代码 | 修复后 |
|------|---------|--------|
| n | 1.5 | 1.5 |
| 法向入射时D_exact | 577,000 | -0.385 |
| 法向入射时D_po | 0 | -0.5 |
| 法向入射时D_coeff | 577,000 | **0.115** |

---

## 遮挡检测（可选）

在`physics/ptd_core.py`的循环开始处添加：

```python
for seg in edge.segments:
    # 遮挡检测：检查入射方向是否照亮该面
    # k_dir指向入射方向，n_lit是照亮面法向
    # 如果入射波从背面照射，跳过
    if np.dot(-k_dir, n_lit) < 0:
        continue
```

---

## 修改文件

| 文件 | 修改内容 |
|------|----------|
| `physics/ptd_core.py` | 替换D系数计算公式（第66-79行）|
| `physics/ptd_core.py` | 添加遮挡检测（可选）|

---

## 验证方法

1. **修改后检查D_coeff值**：法向入射时应约为0.1-0.5范围
2. **RCS曲线检查**：不应出现90dB亮斑
3. **对比验证**：与已知半平面/楔形散射解析解对比

---

## 物理理论依据

根据Ufimtsev PTD理论：
- PTD绕射系数 = 精确解 - PO贡献
- 对于后向散射：`D_exact = -(1/n) * cot(π/(2n))`
- PO贡献：`D_po = -1/(2*cos(α))`
- 两者在阴影边界/反射边界的奇异性相互抵消
