"""
解析 RCS 公式模块

提供基本几何体的物理光学 (PO) 解析解，用于验证数值计算结果。
"""

import numpy as np
from .constants import C0


def cylinder_rcs(radius, height, frequency, theta_rad):
    """
    圆柱体侧向散射的 PO 解析解

    参数:
    radius: 圆柱半径 (m)
    height: 圆柱高度 (m)
    frequency: 频率 (Hz)
    theta_rad: 入射角数组 (弧度)，theta=90° 为侧向入射

    返回:
    RCS 数组 (dBsm)

    公式来源: Ruck et al., "Radar Cross Section Handbook"
    σ ≈ (2πRL²/λ) × cos²(φ) × sinc²(kL·sinφ)
    其中 φ = theta - 90° (相对于侧向的偏角)
    """
    wavelength = C0 / frequency
    k = 2 * np.pi / wavelength

    # φ 是相对于侧向 (theta=90°) 的偏角
    phi_rad = theta_rad - np.pi / 2

    # sinc 参数
    sinc_arg = (k * height * np.sin(phi_rad)) / np.pi

    # PO 解析公式
    sigma = (2 * np.pi * radius * height**2 / wavelength) * \
            (np.cos(phi_rad)**2) * (np.sinc(sinc_arg)**2)

    # 避免 log(0)
    sigma = np.maximum(sigma, 1e-20)

    return 10 * np.log10(sigma)


def plate_rcs(width, height, frequency, theta_rad, phi_rad=0.0):
    """
    矩形平板的 PO 解析解

    参数:
    width: 平板宽度 (m)，沿 x 轴
    height: 平板高度 (m)，沿 y 轴
    frequency: 频率 (Hz)
    theta_rad: 入射角数组 (弧度)，theta=0 为法向入射
    phi_rad: 方位角 (弧度)，默认 0 (在 xz 平面扫描)

    返回:
    RCS 数组 (dBsm)

    公式:
    σ = (4πA²/λ²) × cos²(θ) × sinc²(kW·sinθ·cosφ/π) × sinc²(kH·sinθ·sinφ/π)
    """
    wavelength = C0 / frequency
    k = 2 * np.pi / wavelength
    area = width * height

    theta_rad = np.asarray(theta_rad)

    # 对于 phi=0 的情况 (在 xz 平面扫描)
    sinc_arg_w = (k * width * np.sin(theta_rad) * np.cos(phi_rad)) / np.pi
    sinc_arg_h = (k * height * np.sin(theta_rad) * np.sin(phi_rad)) / np.pi

    # 当 phi=0 时，sinc_arg_h = 0，sinc(0) = 1
    sigma = (4 * np.pi * area**2 / wavelength**2) * \
            (np.cos(theta_rad)**2) * \
            (np.sinc(sinc_arg_w)**2) * (np.sinc(sinc_arg_h)**2)

    sigma = np.maximum(sigma, 1e-20)

    return 10 * np.log10(sigma)


def sphere_rcs(radius, frequency=None):
    """
    球体光学区的 RCS (高频近似)

    参数:
    radius: 球体半径 (m)
    frequency: 频率 (Hz)，可选，仅用于验证光学区条件

    返回:
    RCS 值 (dBsm)，光学区为常数 πR²

    注意: 仅在 ka >> 1 (光学区) 时有效
    """
    sigma = np.pi * radius**2

    if frequency is not None:
        wavelength = C0 / frequency
        ka = 2 * np.pi * radius / wavelength
        if ka < 5:
            import warnings
            warnings.warn(f"ka={ka:.2f} < 5，可能不在光学区，解析解精度降低")

    return 10 * np.log10(sigma)


def wedge_gtd_rcs(length, phi_deg_array, frequency, polarization='VV'):
    """
    直角楔形 (90度) 的 GTD 解析解 (仅用于非奇点区域参考)
    假设侧面入射 (beta=90)，单站 RCS。
    
    参数:
        length: 棱边长度 (m)
        phi_deg_array: 观察角度数组 (度)
            定义: 0度=Face1法向, 90度=Face1切向(Face2法向?), 
            Check consistency: 
            In code: Face 1 Normal is at 90 deg (pi/2).
            User inputs: theta. theta=0 is Face 1 Normal.
            So phi_local = theta + 90.
    
    返回:
        RCS (dBsm)
    """
    wavelength = C0 / frequency
    k = 2 * np.pi / wavelength
    n = 1.5 # 90 deg wedge
    
    phi_rad = np.radians(phi_deg_array)
    # 转换: 用户输入 theta (0=Face1 Normal).
    # Ufimtsev phi: Face 1 is at phi=0. Normal is at phi=90 (pi/2).
    # So phi_ufimtsev = theta + pi/2.
    phi = phi_rad + np.pi/2
    
    sin_pi_n = np.sin(np.pi/n)
    cos_pi_n = np.cos(np.pi/n)
    
    # 1. Incident term (f)
    denom_f = cos_pi_n - 1.0 # cos(pi/n) - cos(0)
    term_f = (1.0/n) * sin_pi_n / denom_f
    
    # 2. Reflection term (g)
    # cos(pi/n) - cos(2*phi/n)
    denom_g = cos_pi_n - np.cos(2*phi/n)
    
    # Avoid singularity for plotting
    denom_g[np.abs(denom_g) < 1e-6] = 1e-6
    
    term_g = (1.0/n) * sin_pi_n / denom_g
    
    if polarization == 'VV':
        # Hard: D = f + g
        D = term_f + term_g
    else:
        # Soft: D = f - g
        D = term_f - term_g
        
    # GTD RCS Formula for finite length edge (Normal incidence)
    # sigma = L^2 / pi * |D|^2 * (something?)
    # Standard formula for finite cylinder: 2pi R L^2 / lambda.
    # For edge:
    # E_s = E_i * D * exp(-jkr)/sqrt(r)
    # This D is 2D diffraction coeff (units 1/sqrt(k)? No, Ufimtsev D is dimensionless? No).
    # Ufimtsev D as defined in code is Dimensionless?
    # Let's check code: pre_factor = 2pi / (jk).
    # So Total Field ~ (2pi/k) * D * L.
    # Sigma = 4pi * |Total|^2 = 4pi * (4pi^2/k^2) * |D|^2 * L^2
    #       = 16 pi^3 / k^2 * L^2 * |D|^2
    #       = 4 pi * lambda^2 * L^2 * |D|^2 ?
    # Let's re-verify unit.
    # Code D is dimensionless (pure trig).
    # Result is Area (Sigma).
    # Code uses: sigma = k^2/pi * |I|^2.
    # I = (2pi/k) * D * L.
    # |I|^2 = 4pi^2/k^2 * |D|^2 * L^2.
    # Sigma = (k^2/pi) * (4pi^2/k^2 * |D|^2 * L^2)
    #       = 4 * pi * L^2 * |D|^2.
    
    sigma = 4 * np.pi * length**2 * np.abs(D)**2
    
    return 10 * np.log10(np.maximum(sigma, 1e-20))


def get_analytical_solution(geometry_type, geometry_params, frequency, theta_rad, polarization='VV'):
    """
    统一接口：根据几何类型获取解析解

    参数:
    geometry_type: 几何类型字符串 ('cylinder', 'plate', 'sphere')
    geometry_params: 几何参数字典
    frequency: 频率 (Hz)
    theta_rad: 角度数组 (弧度)
    polarization: 'VV' or 'HH'

    返回:
    (rcs_analytical, label) 或 (None, None) 如果不支持
    """
    geometry_type = geometry_type.lower()

    if geometry_type == 'cylinder':
        rcs = cylinder_rcs(
            geometry_params['radius'],
            geometry_params['height'],
            frequency,
            theta_rad
        )
        label = f"解析解 (圆柱 R={geometry_params['radius']}m, H={geometry_params['height']}m)"
        return rcs, label

    elif geometry_type == 'plate':
        rcs = plate_rcs(
            geometry_params['width'],
            geometry_params['height'],
            frequency,
            theta_rad
        )
        label = f"解析解 (平板 PO)"
        return rcs, label

    elif geometry_type == 'sphere':
        # 球体 RCS 是常数，扩展为数组
        rcs_val = sphere_rcs(geometry_params['radius'], frequency)
        rcs = np.full_like(theta_rad, rcs_val, dtype=float)
        label = f"解析解 (球体 R={geometry_params['radius']}m)"
        return rcs, label
        
    elif 'wedge' in geometry_type:
        # Standard Wedge GTD
        length = geometry_params.get('length', 10.0)
        phi_deg = np.degrees(theta_rad)
        rcs = wedge_gtd_rcs(length, phi_deg, frequency, polarization)
        label = f"参考解 (GTD Edge Only, {polarization})"
        return rcs, label

    else:
        return None, None


def compute_error_stats(rcs_numerical, rcs_analytical):
    """
    计算数值解与解析解的误差统计

    参数:
    rcs_numerical: 数值解 RCS 数组 (dBsm)
    rcs_analytical: 解析解 RCS 数组 (dBsm)

    返回:
    字典包含:
        - max_error: 最大误差 (dB)
        - mean_error: 平均误差 (dB)
        - rms_error: RMS 误差 (dB)
        - max_error_idx: 最大误差的索引
    """
    error = np.abs(rcs_numerical - rcs_analytical)

    return {
        'max_error': np.max(error),
        'mean_error': np.mean(error),
        'rms_error': np.sqrt(np.mean(error**2)),
        'max_error_idx': np.argmax(error),
        'error_array': error
    }
