"""
二阶边缘绕射（Ufimtsev Ch.9/10：掠射 + 斜率绕射）

边 A 的一阶边波沿连接面掠射到边 B → B 再绕射 → 观察者。按极化分两支：
  VV (hard / H_z, 掠射绕射, 强 ~k⁻¹)   : 用全 GTD 系数 g(φ,0,α)         (Eq.10.40/10.34)
  HH (soft / E_z, 斜率绕射, 弱 ~(kR)⁻³ᐟ²): 用斜率系数 ∂f(φ,0,α)/∂φ₀       (Eq.10.41/10.35)

复用一阶的截面坐标系 (e1,e2,angle0,angle_obs,γ₀) 与辐射归一化 (sinγ₀/k·L·sinc·相位)，
故 3D "I" 量与一阶自洽，直接相加到 total_I_ptd。

系数已对书闭式校验：g_full(0,φ₀)=−1/cos(φ₀/2)；∂f 积 = Eq.10.61；
hard 2D 组装 = Eq.10.51（见 scratch 验证）。

接口：compute_second_order_contribution(edges, wave, polarization) -> complex
默认仅 VV/HH（与 Ufimtsev EEW 一致；VH/HV 暂不支持）。
"""
import numpy as np
from .ptd_coefficients import _cot

_DPHI = 1e-6           # 斜率系数有限差分步长
_CONNECT_TOL = 1e-3    # 连接面判据容差
_GAMMA_MIN = 1e-3      # γ₀ 下限（入射近平行棱边则无贡献）


# ───────────────── 全 GTD 系数 + 斜率导数 ─────────────────

def _fg_full(phi, phi0, alfa):
    """全 GTD 系数 (f=soft, g=hard)，书 Eq.2.64 / fun_fg 的 GTD 项（未减 PO）。"""
    n = alfa / np.pi
    psi1 = phi - phi0
    psi2 = phi + phi0
    a1 = (np.pi - psi2) / (2 * n)
    a2 = (np.pi - psi1) / (2 * n)
    a3 = (np.pi + psi2) / (2 * n)
    a4 = (np.pi + psi1) / (2 * n)
    f = (1.0 / (2 * n)) * (_cot(a1) - _cot(a2) + _cot(a3) - _cot(a4))
    g = -(1.0 / (2 * n)) * (_cot(a1) + _cot(a2) + _cot(a3) + _cot(a4))
    return f, g


def _df_dphi0(phi, phi0, alfa):
    """∂f/∂φ₀（对入射角求导，斜率绕射系数），有限差分。"""
    fp = _fg_full(phi, phi0 + _DPHI, alfa)[0]
    fm = _fg_full(phi, phi0 - _DPHI, alfa)[0]
    return (fp - fm) / (2 * _DPHI)


def _df_dphi_obs(phi, phi0, alfa):
    """∂f/∂φ（对观察角求导，主波斜率发射），有限差分。"""
    fp = _fg_full(phi + _DPHI, phi0, alfa)[0]
    fm = _fg_full(phi - _DPHI, phi0, alfa)[0]
    return (fp - fm) / (2 * _DPHI)


# ───────────────── 截面几何（镜像一阶 ptd_core） ─────────────────

def _seg_frame(seg, k_dir, s_dir):
    """返回截面坐标系与角度：(ok, gamma0, angle0, angle_obs, beta_hat, alpha_hat)。
    复用 ptd_core 的 e1/e2/inward/angle0/angle_obs 逻辑。ok=False 表示跳过。"""
    t = seg.tangent
    n_lit = seg.normal if seg.normal is not None else None
    if n_lit is None:
        return (False, 0, 0, 0, None, None)

    k_dot_t = float(np.clip(np.dot(k_dir, t), -1.0, 1.0))
    sin_g = np.sqrt(max(0.0, 1.0 - k_dot_t ** 2))
    if sin_g < _GAMMA_MIN:
        return (False, 0, 0, 0, None, None)

    n_dot_t = np.dot(n_lit, t)
    e1 = n_lit - n_dot_t * t
    e1n = np.linalg.norm(e1)
    if e1n < 1e-10:
        return (False, 0, 0, 0, None, None)
    e1 = e1 / e1n
    e2 = np.cross(t, e1)
    e2n = np.linalg.norm(e2)
    if e2n < 1e-10:
        return (False, 0, 0, 0, None, None)
    e2 = e2 / e2n
    inward = getattr(seg, 'inward', None)
    if inward is not None:
        if np.dot(e2, inward) < 0:
            e2 = -e2
    else:
        nb = seg.normal_b if hasattr(seg, 'normal_b') else None
        if nb is not None and np.dot(e2, nb) > 0:
            e2 = -e2

    alfa = seg.alpha
    # 入射角 angle0
    k_perp = k_dir - k_dot_t * t
    kpl = np.linalg.norm(k_perp)
    if kpl < 1e-10:
        return (False, 0, 0, 0, None, None)
    inc = -k_perp / kpl
    a0 = np.arctan2(np.dot(inc, e1), np.dot(inc, e2))
    if a0 < -1e-6:
        a0 += 2 * np.pi
    a0 = float(np.clip(a0, 0.0, alfa))
    # 观察角 angle_obs
    s_dot_t = np.dot(s_dir, t)
    s_perp = s_dir - s_dot_t * t
    spl = np.linalg.norm(s_perp)
    if spl < 1e-10:
        return (False, 0, 0, 0, None, None)
    su = s_perp / spl
    aobs = np.arctan2(np.dot(su, e1), np.dot(su, e2))
    if aobs < -1e-6:
        aobs += 2 * np.pi
    aobs = float(np.clip(aobs, 0.0, alfa))

    beta = t - np.dot(t, s_dir) * s_dir
    bl = np.linalg.norm(beta)
    if bl < 1e-10:
        return (False, 0, 0, 0, None, None)
    beta = beta / bl
    alpha_hat = np.cross(s_dir, beta)
    return (True, float(np.arcsin(sin_g)), a0, aobs, beta, alpha_hat)


# ───────────────── 连接面判据 ─────────────────

def _connection(edge_a, edge_b):
    """判断 B 是否经"沿 face 掠射"连到 A：从 B 中点沿 inward_B 的射线能否打到 A。
    返回 (connected, R10, src_point_on_A, srcseg_a) 或 (False, ...)。"""
    # 用整条边的代表中点 + inward 做粗判（手选边、N 小）
    bsegs = edge_b.segments
    asegs = edge_a.segments
    if not bsegs or not asegs:
        return (False, 0.0, None, None)
    b_mid = np.mean([s.midpoint for s in bsegs], axis=0)
    b_in = bsegs[len(bsegs) // 2].inward
    if b_in is None:
        return (False, 0.0, None, None)
    a_mid = np.mean([s.midpoint for s in asegs], axis=0)
    d = a_mid - b_mid
    R10 = np.linalg.norm(d)
    if R10 < 1e-9:
        return (False, 0.0, None, None)
    # inward_B 必须大致指向 A（沿连接面横穿）
    if np.dot(b_in, d / R10) < (1.0 - _CONNECT_TOL):
        return (False, 0.0, None, None)
    return (True, float(R10), a_mid, asegs[len(asegs) // 2])


# ───────────────── 主入口 ─────────────────

def compute_second_order_contribution(edges, wave, polarization='VV'):
    """所有连通边对 (A→B) 的二阶绕射对 I 的贡献之和。"""
    if polarization not in ('VV', 'HH'):
        return 0j  # 暂仅支持共极化

    k = wave.k
    k_dir = wave.k_dir
    k_vec = wave.k_vector
    s_dir = -k_dir
    e_pol = wave.theta_hat if polarization == 'VV' else wave.phi_hat
    hard = (polarization == 'VV')

    total = 0j
    for edge_a in edges:
        for edge_b in edges:
            if edge_a is edge_b:
                continue
            ok, R10, a_pt, a_seg = _connection(edge_a, edge_b)
            if not ok:
                continue

            # A 处入射角 angle0_A（用 A 代表段）
            fa = _seg_frame(a_seg, k_dir, s_dir)
            if not fa[0]:
                continue
            angle0_A = fa[2]
            alfa_A = a_seg.alpha

            # 主波从 A 朝面内掠射(φ1=0)的发射系数
            if hard:
                pri_coeff = _fg_full(0.0, angle0_A, alfa_A)[1]          # g(0,φ01)
            else:
                pri_coeff = _df_dphi_obs(0.0, angle0_A, alfa_A)         # ∂f/∂φ1 |_{φ1=0}

            # 沿 B 逐段积分再绕射
            for seg in edge_b.segments:
                fb = _seg_frame(seg, k_dir, s_dir)
                if not fb[0]:
                    continue
                gamma0_B, _, angle_obs_B, beta_B, alpha_B = (
                    fb[1], fb[2], fb[3], fb[4], fb[5])
                sin_gB = np.sin(gamma0_B)
                alfa_B = seg.alpha

                # B 处掠射(φ0=0)再绕射系数
                if hard:
                    sec_coeff = _fg_full(angle_obs_B, 0.0, alfa_B)[1]   # g(φ,0)
                else:
                    sec_coeff = _df_dphi0(angle_obs_B, 0.0, alfa_B)     # ∂f/∂φ0 |_{φ0=0}

                # 入射(平面波)在 A 棱切向激励：VV→H0t, HH→E0t（单位幅值）
                t_a = a_seg.tangent
                if hard:
                    H0 = np.cross(k_dir, e_pol)        # 归一化(Z0=1)的 H
                    exc = float(np.dot(H0, t_a))
                else:
                    exc = float(np.dot(e_pol, t_a))

                # 主波传播 A→B（射线渐近, 直边 ρ=∞ → 1/√(2πkR10)）
                prim = (exc * pri_coeff
                        * np.exp(1j * (k * R10) + 1j * np.pi / 4.0)
                        / np.sqrt(2.0 * np.pi * k * R10))

                # 再绕射方向图 → 接收极化投影（co-pol）
                proj = float(np.dot(e_pol, alpha_B)) if hard else float(np.dot(e_pol, beta_B))
                directivity = prim * sec_coeff * proj

                # 与一阶共享辐射归一化：sinγ₀/k · L · sinc · 相位
                k_dot_tB = float(np.dot(k_dir, seg.tangent))
                sinc_val = np.sinc(k * seg.length * k_dot_tB / np.pi)
                # 双次路径单站相位：入射到A + A→B + B到观察者
                phase = (k * float(np.dot(k_dir, a_pt))
                         + k * R10
                         + k * float(np.dot(k_dir, seg.midpoint)))
                pre = sin_gB / k
                total += (pre * directivity * seg.length * sinc_val
                          * np.exp(1j * phase))

    return total
