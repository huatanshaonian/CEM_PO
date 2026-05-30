"""
一致渐近过渡函数 (UTD / Pauli-Clemmow 鞍点-极点合并处理)

Modified Fresnel function F(x)  (Johansen 1996 Eq.17, Michaeli 1987 Eq.37):

    F(x) = √(j/π) · exp(j·x²) · ∫_x^∞ exp(-j·t²) dt

数值实现:
    - 精确: F(x) = (1/2) · exp(j·x²) · erfc(x · exp(jπ/4))
            (用 scipy.special.erfc, 支持复参数)
    - 大宗渐近 (|x| > 3, Michaeli 1987 Eq.40 / Johansen Eq.23):
            F(x) ≈ 1 / (2x · √(jπ))

此函数是 Pauli-Clemmow 鞍点-极点合并方法的通用工具, 不只是 MEC 截断专用:
也出现在 Kouyoumjian-Pathak UTD 1974 (阴影/反射边界过渡, F_KP(X) =
2√(jπX)·F(√X)) 和 Tiberio-Kouyoumjian 1982 (strip 掠射过渡区) 中。

参考:
    [1] P. M. Johansen, IEEE TAP 1996, Eq.(17)
    [2] A. Michaeli, IEEE TAP 1987, Eq.(37, 40)
    [3] L. B. Felsen & N. Marcuvitz, Radiation and Scattering of Waves, 1973, ch.4
"""
import numpy as np
from scipy.special import erfc

_ASYMPTOTIC_THRESHOLD = 10.0   # 仅 |x|>10 走渐近 (避免 scipy.erfc 复参数潜在数值噪声)


def modified_fresnel_uf(x):
    """
    Ufimtsev 约定 (e^{-iωt}) 版本的改进 Fresnel 函数.

    把原 Johansen/Michaeli 公式 F(x) = √(j/π) e^{jx²} ∫_x^∞ e^{-jt²} dt 中的 j 替换为 -i:
        F_uf(x) = √(-i/π) e^{-i·x²} ∫_x^∞ e^{+i·t²} dt
                = (1/2) · exp(-i·x²) · erfc(x · exp(-iπ/4))

    实数参数下: F_uf(x) = conj(F(x))。

    这是 mec_truncated_coefficients 用的版本; modified_fresnel (原版) 留作其他可能用途。
    """
    x_arr = np.asarray(x, dtype=complex)
    abs_x = np.abs(x_arr)

    sqrt_mjpi = np.sqrt(-1j * np.pi)   # √(-iπ)
    rot = np.exp(-1j * np.pi / 4.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        exact = 0.5 * np.exp(-1j * x_arr * x_arr) * erfc(x_arr * rot)
        asymp = 1.0 / (2.0 * x_arr * sqrt_mjpi)

    result = np.where(abs_x > _ASYMPTOTIC_THRESHOLD, asymp, exact)

    if np.ndim(x) == 0:
        return complex(result)
    return result


def modified_fresnel(x):
    """
    计算 Johansen / Michaeli 改进 Fresnel 函数 F(x).

    参数:
        x: 实数或复数标量, 或同形 ndarray (实/复均可)。

    返回:
        complex 标量或 ndarray, 与 x 同形。

    数值策略:
        |x| ≤ 10: F(x) = 0.5 · exp(j x²) · erfc(x · exp(jπ/4))  (精确, scipy.erfc 支持复参数)
        |x| > 10: F(x) ≈ (2x · √(jπ))⁻¹                          (Michaeli Eq.40 渐近)

    阈值选 10 而非 Michaeli 建议的 3, 因为 Eq.40 是领头项渐近, 次阶 O(1/x³) 修正
    在 x=3 时给 ~7% 相对误差; 提到 x=10 时误差降到 <0.1%。精确式在 x∈[0,10] 范围
    可证不溢出 (rot 系数 e^{jπ/4} 让 z²=jx² 实部=0, |exp(jx²·erfc·rot)| 有界)。

    单测要点:
        F(0) = 0.5  (erfc(0)=1)
        F(x→∞) → 0
        x=10 处精确与渐近一致 (<0.1% 相对误差)
    """
    x_arr = np.asarray(x, dtype=complex)
    abs_x = np.abs(x_arr)

    sqrt_jpi = np.sqrt(1j * np.pi)   # √(jπ) = √π · exp(jπ/4)
    rot = np.exp(1j * np.pi / 4.0)

    # 精确分支: 用 np.errstate 抑制 |x|>阈值时的 1/x 警告 (那些点最后被 where 抛弃)
    with np.errstate(divide='ignore', invalid='ignore'):
        exact = 0.5 * np.exp(1j * x_arr * x_arr) * erfc(x_arr * rot)
        asymp = 1.0 / (2.0 * x_arr * sqrt_jpi)

    result = np.where(abs_x > _ASYMPTOTIC_THRESHOLD, asymp, exact)

    # 标量输入返回标量
    if np.ndim(x) == 0:
        return complex(result)
    return result
