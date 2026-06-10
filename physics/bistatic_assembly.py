"""
双站散射场装配例程 (bistatic radiation-integral assembly) —— 验证专用, 暂不用于业务计算

================================================================================
用途与边界 (务必先读)
================================================================================
本模块是一个 **验证 (verification) 工具**, 不是主计算管线的一部分:

  * 主管线的单站计算仍然只走 physics/ptd_algorithms.py 注册的三个算法
    (ufimtsev_eew / michaeli_mec / michaeli_mec_truncated), 经
    solvers/rcs_analyzer.py 调用。本模块不在该注册表中, 不被任何业务代码引用。
  * 引入本模块的唯一原因: 主管线的装配层把观察方向写死为 s_obs = -k_dir
    (单站后向), 无法做双站。要拿代码结果去对 Ufimtsev/Johansen 等
    **双站** 解析/参考数据, 必须有一个能独立指定观察方向的装配例程。
  * 因此本模块 **只供 scratch_* 验证脚本调用**。在它通过充分的外部真值
    校验、并明确决定纳入业务前, 不要在 rcs_analyzer / GUI / freq_sweep
    中调用它。

为什么这是"装配层" (assembly layer), 不是新物理:
  等效边电流的闭式系数 (mec_coefficients / mec_truncated_coefficients) 是
  **双站通式** —— 入射 (β0,φ0) 与观察 (β,φ) 本就是独立的 4 个角。真正把
  观察方向写死成单站的, 是 mec_core / mec_truncated_core 里"把几何+入射波
  翻译成系数参数、再把电流组装成散射积分"的那一层 (assembly layer)。本模块
  把同一套装配逻辑重写为"观察方向 s_obs 独立输入", 系数层一行不改。

自检契约 (self-test contract):
  当 s_obs = -wave.k_dir 时, 本模块对每条边的复数贡献 **必须逐字复现**
  主管线 (mec_truncated_core / mec_core / ptd_core) 的单站结果。
  run_self_test() 实现该校验; CI/手动验证时应先确认它返回的最坏偏差 < 1e-9。
  (历史教训: 装配层若使用自定义波数常数而非 wave.k, 两条相距数 λ 的边
   相位会有微差, 经干涉放大成百分级误差 —— 必须始终用 wave.k。)

================================================================================
接口
================================================================================
assemble_bistatic(edges, wave, s_obs, polarization, algorithm='michaeli_mec_truncated')
    -> complex
    返回所有边在观察方向 s_obs 上的复数散射积分 I (与主管线同一 I 量纲,
    可经 σ = (k²/π)|I|² 转 3D RCS, 或自行做 2D echo-width 归一化)。

    参数:
      edges        : list[PTDEdge], 每条边的 segment 需含 .l_A (截断算法用);
                     非截断算法可不含。
      wave         : IncidentWave, 提供 k_dir / k / k_vector / theta_hat / phi_hat。
      s_obs        : (3,) 单位向量, 指向远场观察点的方向。单站时取 -wave.k_dir。
      polarization : 'VV' | 'HH' | 'VH' | 'HV' (发射/接收极化基组合)。
      algorithm    : 'ufimtsev_eew' 暂不支持 (EEW 是标量 D 写法, 双站需另接);
                     'michaeli_mec'            一阶 MEC (非截断);
                     'michaeli_mec_truncated'  截断 MEC (Michaeli/Johansen 二阶主项)。

run_self_test(verbose=True) -> float
    返回壳(s_obs=-k_dir) 对主管线的最坏复数偏差; <1e-9 视为通过。

时谐约定: e^{-iωt}, 与全仓库一致 (Michaeli/Johansen 闭式经整支 conj 翻入)。
"""
import numpy as np

from .constants import ETA0
from .mec_coefficients import (
    compute_total_fringe_currents,
    compute_total_fringe_currents_bistatic,
    compute_total_fringe_currents_bistatic_raw,
)
from .mec_truncated_coefficients import compute_correction_currents
from .mec_truncated_general_coefficients import (
    compute_total_correction_general,
    compute_total_correction_general_raw,
)

_POL_BASIS = {
    'VV': ('theta_hat', 'theta_hat'),
    'HH': ('phi_hat',   'phi_hat'),
    'VH': ('theta_hat', 'phi_hat'),
    'HV': ('phi_hat',   'theta_hat'),
}

# 算法清单:
#   michaeli_mec               — Michaeli 1986 非截断 EEC, 含 ±_SING_OFFSET 平滑
#                                 (Ufimtsev 奇点被平均掉, 适合常规 RCS)
#   michaeli_mec_raw           — Michaeli 1986 非截断 EEC, 无平滑
#                                 (奇点保留 → Johansen Fig.4 复现)
#   michaeli_mec_truncated     — N=2 (半平面) 截断, Eq.26/27 + 平滑
#   michaeli_mec_truncated_general    — 任意 N 截断 (Eq.21/22), 平滑版
#   michaeli_mec_truncated_general_raw — 任意 N 截断, 无平滑
#                                         (M_UT_raw - M_cor_raw 自然抵消奇点
#                                          → Johansen Fig.6 复现)
_SUPPORTED = ('michaeli_mec',
              'michaeli_mec_raw',
              'michaeli_mec_truncated',
              'michaeli_mec_truncated_general',
              'michaeli_mec_truncated_general_raw')


def _edge_frame(seg, k_dir):
    """段局部坐标 (与 mec_core/mec_truncated_core 完全一致)。

    返回 (t, y1, x1, k_dot_t):
      t      边切线
      y1     面 1 法向投影到 ⊥t 平面并归一 (= Michaeli 的 ŷ1)
      x1     y1×t, 经 inward 定向到"由边指向面内"(= Michaeli 的 x̂1)
      k_dot_t  k_dir·t
    返回 None 表示该段退化 (近端入射 / 法向投影为零), 调用方应跳过。
    """
    t = seg.tangent
    n_lit = seg.normal
    if n_lit is None:
        return None
    k_dot_t = float(np.clip(np.dot(k_dir, t), -1.0, 1.0))
    if np.sqrt(max(0.0, 1.0 - k_dot_t * k_dot_t)) < 1e-3:
        return None
    y1 = n_lit - np.dot(n_lit, t) * t
    yl = np.linalg.norm(y1)
    if yl < 1e-10:
        return None
    y1 = y1 / yl
    x1 = np.cross(y1, t)
    xl = np.linalg.norm(x1)
    if xl < 1e-10:
        return None
    x1 = x1 / xl
    inward = getattr(seg, 'inward', None)
    if inward is not None:
        if float(np.dot(x1, inward)) < 0:
            x1 = -x1
    else:
        n_b = getattr(seg, 'normal_b', None)
        if n_b is not None and float(np.dot(x1, n_b)) > 0:
            x1 = -x1
    return t, y1, x1, k_dot_t


def _incidence_angles(t, y1, x1, k_dir):
    """入射 (β', φ') —— 与主管线一致: 源方向 ŝ' = -k_dir。"""
    k_dot_t = float(np.clip(np.dot(k_dir, t), -1.0, 1.0))
    beta_p = np.arccos(np.clip(-k_dot_t, -1.0, 1.0))
    s_inc = -k_dir
    sp = s_inc - np.dot(s_inc, t) * t
    phi_p = np.arctan2(float(np.dot(sp, y1)), float(np.dot(sp, x1)))
    if phi_p < 0:
        phi_p += 2 * np.pi
    return beta_p, phi_p


def _observation_angles(t, y1, x1, s_obs, beta_p, phi_p, mono):
    """观察 (β_obs, φ_obs)。

    mono=True : 单站后向特例 β_obs=π-β', φ_obs=φ' —— 与主管线逐字一致。
    mono=False: 由 s_obs 独立投影 (真双站)。注意 s_obs 是"指向观察者"的方向,
                与入射用 -k_dir(指向源) 对称, 故同样直接投影。
    """
    if mono:
        return np.pi - beta_p, phi_p
    beta_o = np.arccos(np.clip(float(np.dot(s_obs, t)), -1.0, 1.0))
    so = s_obs - np.dot(s_obs, t) * t
    if np.linalg.norm(so) < 1e-12:
        return beta_o, phi_p
    phi_o = np.arctan2(float(np.dot(s_obs, y1)), float(np.dot(s_obs, x1)))
    if phi_o < 0:
        phi_o += 2 * np.pi
    return beta_o, phi_o


def assemble_bistatic(edges, wave, s_obs, polarization='VV',
                      algorithm='michaeli_mec_truncated'):
    """双站散射场装配。详见模块 docstring。返回复数 I。"""
    if algorithm not in _SUPPORTED:
        raise ValueError(
            f"assemble_bistatic 暂支持 {_SUPPORTED}; 收到 {algorithm!r}. "
            f"(ufimtsev_eew 为标量 D 写法, 双站需另行实现。)")
    if polarization not in _POL_BASIS:
        raise ValueError(f"未知极化 {polarization!r}; 可用 {list(_POL_BASIS)}")

    truncated = (algorithm == 'michaeli_mec_truncated')
    truncated_general = (algorithm == 'michaeli_mec_truncated_general')
    truncated_general_raw = (algorithm == 'michaeli_mec_truncated_general_raw')
    raw_mode = algorithm in ('michaeli_mec_raw', 'michaeli_mec_truncated_general_raw')
    s_obs = np.asarray(s_obs, dtype=float)
    s_obs = s_obs / np.linalg.norm(s_obs)
    mono = bool(np.allclose(s_obs, -wave.k_dir, atol=1e-12))

    et_name, er_name = _POL_BASIS[polarization]
    e_t = getattr(wave, et_name)
    e_r = getattr(wave, er_name)
    k_dir = wave.k_dir
    k = wave.k                      # 必须用 wave.k (见模块 docstring 历史教训)
    k_vec = wave.k_vector
    Z = ETA0
    H0_vec = np.cross(k_dir, e_t) / Z

    total = 0.0 + 0.0j
    for edge in edges:
        for seg in edge.segments:
            fr = _edge_frame(seg, k_dir)
            if fr is None:
                continue
            t, y1, x1, k_dot_t = fr
            beta_p, phi_p = _incidence_angles(t, y1, x1, k_dir)
            N = seg.alpha / np.pi
            if phi_p > N * np.pi:
                # 入射落在 (Face A 自身投影的) 阴影区, 即源方向在导体楔内 (局部).
                # 旧逻辑直接跳过该段; 但对 Johansen truncated_general / raw, 当几何
                # 不是单边受照的简单情形 (例如三角柱顶点, 两邻面都局部 shadow),
                # 跳过会丢失整条棱的贡献. Johansen 1996 公式本身允许 φ_0 > Nπ 时
                # U(π-φ_0)=0 (Face A PO 自动为 0), 公式给出截断+二阶差等贡献.
                if algorithm not in ('michaeli_mec_truncated_general',
                                     'michaeli_mec_truncated_general_raw',
                                     'michaeli_mec_raw'):
                    continue

            E0z = complex(np.dot(e_t, t))
            H0z = complex(np.dot(H0_vec, t))

            # --- 观察角 (mono 时 = π-β'/φ', 双站时由 s_obs 独立投影) ---
            beta_o, phi_o = _observation_angles(
                t, y1, x1, s_obs, beta_p, phi_p, mono)

            # --- 非截断 EEC: 双站通式 Michaeli 1986 Eq.4-7, 整支 conj 到 e^{-iωt} ---
            if raw_mode:
                # raw: 无奇点平滑, 保留 Ufimtsev 奇点 (用于 Johansen Fig.4 复现,
                # 以及与 _raw 截断修正配对让 M_UT - M_cor 自然抵消奇点)
                If_m, Mf_m = compute_total_fringe_currents_bistatic_raw(
                    beta_p, phi_p, beta_o, phi_o, N, E0z, H0z, k, Z)
            elif mono:
                # 单站特化 (生产路径), ~1.15x 加速
                If_m, Mf_m = compute_total_fringe_currents(
                    beta_p, phi_p, N, E0z, H0z, k, Z)
            else:
                If_m, Mf_m = compute_total_fringe_currents_bistatic(
                    beta_p, phi_p, beta_o, phi_o, N, E0z, H0z, k, Z)
            If = np.conj(If_m)
            Mf = np.conj(Mf_m)

            # --- 截断修正 (Johansen), 同样整支 conj ---
            if truncated:
                # N=2 半平面闭式 (Eq.26/27); 仅对薄板有效, 任意 N 用 general 分支
                l_A = getattr(seg, 'l_A', None)
                if l_A is not None and np.isfinite(l_A) and l_A > 1e-12:
                    Mc_m, Ic_m = compute_correction_currents(
                        beta_p, phi_p, beta_o, phi_o, l_A, E0z, H0z, k, Z)
                    If = If - np.conj(Ic_m)
                    Mf = Mf - np.conj(Mc_m)
            elif truncated_general or truncated_general_raw:
                # 任意 N (Johansen Eq.21/22 + Face B 替换), Face A 与 B 用不同截断长度
                l_A = getattr(seg, 'l_A', None)
                l_B = getattr(seg, 'l_B', l_A)   # 缺省: l_B = l_A
                if (l_A is not None and np.isfinite(l_A) and l_A > 1e-12
                        and l_B is not None and np.isfinite(l_B) and l_B > 1e-12):
                    if truncated_general_raw:
                        # 与 raw 模式 M_UT 同口径, 让 M_UT - M_cor 在奇点处自然抵消
                        Mc_m, Ic_m = compute_total_correction_general_raw(
                            beta_p, phi_p, beta_o, phi_o, N,
                            l_A, l_B, E0z, H0z, k, Z)
                    else:
                        Mc_m, Ic_m = compute_total_correction_general(
                            beta_p, phi_p, beta_o, phi_o, N,
                            l_A, l_B, E0z, H0z, k, Z)
                    If = If - np.conj(Ic_m)
                    Mf = Mf - np.conj(Mc_m)

            # --- 接收投影 (用独立 s_obs) ---
            s_cross_t = np.cross(s_obs, t)
            amp = -Z * If * float(np.dot(t, e_r)) + Mf * float(np.dot(s_cross_t, e_r))

            # --- 双站相位 e^{i k (k_dir - s_obs)·r_c}; 单站时 = e^{i 2 k_vec·r_c} ---
            phase = float(np.dot(seg.midpoint, k_vec)) \
                - k * float(np.dot(seg.midpoint, s_obs))
            sinc_arg = k * seg.length * float(np.dot(k_dir - s_obs, t)) / (2 * np.pi)
            sinc = np.sinc(sinc_arg)

            total += 0.5 * seg.length * sinc * np.exp(1j * phase) * amp
    return total


def run_self_test(verbose=True):
    """壳(s_obs=-k_dir) vs 主管线单站, 返回最坏复数偏差 (<1e-9 视为通过)。

    仅依赖主管线 mec_truncated_core / mec_core, 不引外部数据, 可随时重跑。
    """
    from .wave import IncidentWave
    from .mec_core import compute_mec_contribution
    from .mec_truncated_core import compute_mec_truncated_contribution
    from solvers.ptd_structures import PTDEdge

    lam = 1.0
    freq = 3e8 / lam

    def strip(a, Ly=lam):
        pL = np.array([[0.0, -Ly / 2, 0.0], [0.0, Ly / 2, 0.0]])
        pR = np.array([[a, -Ly / 2, 0.0], [a, Ly / 2, 0.0]])
        nm = np.array([[0, 0, 1.0], [0, 0, 1.0]])
        eL = PTDEdge("L", pL, [0, 0, 1.0], exterior_angle_rad=2 * np.pi,
                     point_normals=nm,
                     point_inwards=np.array([[1.0, 0, 0], [1.0, 0, 0]]))
        eR = PTDEdge("R", pR, [0, 0, 1.0], exterior_angle_rad=2 * np.pi,
                     point_normals=nm,
                     point_inwards=np.array([[-1.0, 0, 0], [-1.0, 0, 0]]))
        for e in (eL, eR):
            for s in e.segments:
                s.l_A = a
        return [eL, eR]

    worst = 0.0
    cases = [('michaeli_mec', compute_mec_contribution),
             ('michaeli_mec_truncated', compute_mec_truncated_contribution)]
    for algo, main_fn in cases:
        for th in (20, 50, 75):
            for pol in ('VV', 'HH'):
                w = IncidentWave(freq, np.radians(th), 0.0)
                edges = strip(5 * lam)
                s_obs = -w.k_dir
                I_main = sum(main_fn(e, w, pol) for e in edges)
                I_shell = assemble_bistatic(edges, w, s_obs, pol, algorithm=algo)
                if abs(I_main) > 1e-30:
                    r = I_shell / I_main
                    dev = abs(abs(r) - 1) + abs(np.angle(r))
                    worst = max(worst, dev)
                    if verbose:
                        print(f"  [{algo}] th={th} {pol}: "
                              f"|shell/main|={abs(r):.6f} "
                              f"arg={np.degrees(np.angle(r)):+.3f}")
    if verbose:
        status = "PASS" if worst < 1e-9 else "FAIL"
        print(f">>> assemble_bistatic self-test worst dev = {worst:.2e}  [{status}]")
    return worst


if __name__ == "__main__":
    run_self_test()
