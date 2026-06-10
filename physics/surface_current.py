"""
物理光学 (PO) 表面电流计算。

公式：
    亮区 (n̂ · k̂_inc < 0):  J_s(r) = 2 n̂ × H_inc(r)
    暗区:                  J_s(r) = 0

其中入射平面波单位振幅:
    E_inc(r) = ê_pol * exp(j k · r)
    H_inc(r) = (1/η0) k̂_inc × E_inc(r)

极化基向量来自 IncidentWave.theta_hat / phi_hat:
    'V': ê_pol = θ̂   (VV 通道入射)
    'H': ê_pol = φ̂   (HH 通道入射)

PTD 习惯用 'VV','HH','VH','HV'，本模块只关心入射极化，按首字母取 V 或 H。
"""
from dataclasses import dataclass
import numpy as np

from .constants import ETA0


@dataclass
class SurfaceCurrentField:
    """单个 Surface 上的表面电流场。所有数组对齐到面元中心。"""
    points: np.ndarray       # (N, 3) 面元中心 xyz
    normals: np.ndarray      # (N, 3) 单位法向
    J: np.ndarray            # (N, 3) 复电流向量 (A/m)
    J_mag: np.ndarray        # (N,) |J| (A/m)
    lit_mask: np.ndarray     # (N,) bool，亮区 True
    n_dot_k: np.ndarray      # (N,) 法向 · 入射传播方向，用于诊断
    grid_shape: tuple | None = None  # 规则网格时为 (nv, nu)；退化网格 None


def _incident_pol_vector(wave, polarization: str) -> np.ndarray:
    """返回入射极化基向量 (3,)。"""
    code = (polarization or 'V').upper()
    # PTD 风格 'VV'/'HH'/'VH'/'HV' 取首字母作为入射极化
    if len(code) >= 1 and code[0] in ('V', 'H'):
        c = code[0]
    else:
        raise ValueError(f"未知极化: {polarization!r}")
    if c == 'V':
        return wave.theta_hat
    return wave.phi_hat


def compute_surface_current(cached_mesh, wave, polarization: str = 'V') -> SurfaceCurrentField:
    """
    给定一个预计算网格和入射波，返回每个面元上的 PO 表面电流。

    Args:
        cached_mesh: CachedMeshData 实例（必须在 CPU 端；调用者负责 to_cpu）
        wave: physics.wave.IncidentWave 实例
        polarization: 'V'/'H' 或 'VV'/'HH'/'VH'/'HV'，按入射极化取首字母

    Returns:
        SurfaceCurrentField
    """
    pts = np.asarray(cached_mesh.points)
    nrm = np.asarray(cached_mesh.normals)
    if pts.ndim == 3:
        grid_shape = pts.shape[:2]  # (nv, nu)
        pts_flat = pts.reshape(-1, 3)
        nrm_flat = nrm.reshape(-1, 3)
    else:
        grid_shape = None
        pts_flat = pts
        nrm_flat = nrm

    k_dir = np.asarray(wave.k_dir)        # 传播方向单位向量
    k_vec = np.asarray(wave.k_vector)     # k * k_dir
    e_pol = _incident_pol_vector(wave, polarization)

    # H_inc(r) = (1/η0) k̂ × E_inc(r), E_inc(r) = ê * exp(j k·r)
    h_pol = np.cross(k_dir, e_pol)        # (3,)，与 r 无关的极化部分

    # J_s = 2 n̂ × H_inc * lit  (亮区)
    # 先算 n̂ × ĥ_pol 的常数部分，再乘相位
    n_cross_h = np.cross(nrm_flat, h_pol[None, :])  # (N, 3)
    phase = np.exp(1j * (pts_flat @ k_vec))         # (N,)
    J = (2.0 / ETA0) * n_cross_h.astype(np.complex128) * phase[:, None]

    n_dot_k = nrm_flat @ k_dir              # (N,)
    lit_mask = n_dot_k < 0.0
    J[~lit_mask] = 0.0

    J_mag = np.linalg.norm(J, axis=1).real

    return SurfaceCurrentField(
        points=pts_flat,
        normals=nrm_flat,
        J=J,
        J_mag=J_mag,
        lit_mask=lit_mask,
        n_dot_k=n_dot_k,
        grid_shape=grid_shape,
    )
