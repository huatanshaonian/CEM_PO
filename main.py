import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from geometry.cylinder import AnalyticCylinder
from physics.wave import IncidentWave
from physics.constants import C0
from solvers.po import DiscretePOIntegrator as RibbonIntegrator
from solvers.rcs_analyzer import RCSAnalyzer

def main():
    # 参数设置
    radius = 0.5  # 半径 0.5m
    height = 2.0  # 高度 2.0m
    freq = 1.0e9  # 1 GHz
    wavelength = C0 / freq
    k = 2 * np.pi / wavelength
    
    print(f"频率: {freq/1e9:.2f} GHz")
    print(f"波长: {wavelength:.4f} m")
    
    # 初始化几何和求解器
    cylinder = AnalyticCylinder(radius, height)
    # 增加网格密度以确保高频下的准确性
    solver = RibbonIntegrator(nu=400, nv=200)
    analyzer = RCSAnalyzer(solver)
    
    # 定义角度范围 (theta 从 0 到 180 度，绕 z 轴)
    # 90 度是正入射 (broadside)
    theta_deg = np.linspace(60, 120, 100)
    theta_rad = np.radians(theta_deg)
    
    # 计算数值解
    print("正在计算数值解...")
    rcs_numerical = analyzer.compute_monostatic_rcs(
        cylinder, 
        {'frequency': freq, 'phi': 0.0},
        theta_rad
    )
    
    # 计算解析参考值 (根据 instruction.txt)
    # 公式: Sigma \approx (2 * pi * R * L^2 / lambda) * cos(phi)^2 * sinc(k * L * sin(phi))^2
    # 注意：这里的 phi 是相对于法线的偏角。在我们的坐标系中，phi = theta - 90 deg
    phi_rad = theta_rad - np.pi/2
    
    # np.sinc(x) 是 sin(pi*x)/(pi*x)
    # 我们需要 sin(k*height*np.sin(phi_rad)) / (k*height*np.sin(phi_rad))
    # 所以传入 np.sinc( (2 * k * height/2 * np.sin(phi_rad)) / np.pi )?
    # 或者简单点：k * height * np.sin(phi_rad) / np.pi
    sinc_arg = (k * height * np.sin(phi_rad)) / np.pi
    
    sigma_ref = (2 * np.pi * radius * height**2 / wavelength) * \
                (np.cos(phi_rad)**2) * \
                (np.sinc(sinc_arg)**2)
    
    rcs_reference = 10 * np.log10(sigma_ref + 1e-12)
    
    # 绘图对比
    plt.figure(figsize=(10, 6))
    plt.plot(theta_deg, rcs_numerical, 'b-', label='Ribbon Solver (PO)')
    plt.plot(theta_deg, rcs_reference, 'r--', label='Textbook Reference')
    plt.grid(True)
    plt.xlabel('Theta (degrees)')
    plt.ylabel('RCS (dBsqm)')
    plt.title(f'Monostatic RCS of Cylinder (R={radius}m, H={height}m) at {freq/1e9}GHz')
    plt.legend()
    
    # 检查误差
    error = np.abs(rcs_numerical - rcs_reference)
    max_error = np.max(error[np.abs(theta_deg - 90) < 10]) # 检查主瓣附近的误差
    print(f"主瓣附近最大误差: {max_error:.4f} dB")
    
    plt.savefig('rcs_verification.png')
    print("图像已保存为 rcs_verification.png")
    
    if max_error < 0.5:
        print("验证通过！误差在 0.5 dB 以内。")
    else:
        print("验证警告：误差超过 0.5 dB。")

if __name__ == "__main__":
    main()