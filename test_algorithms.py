"""
算法对比测试脚本

在圆柱、平板、球上测试所有 PO 算法，与解析解对比。
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # 非交互式后端

import numpy as np
import time
import sys

# 添加项目路径
sys.path.insert(0, '.')

from physics.constants import C0
from physics.wave import IncidentWave
from physics.analytical_rcs import cylinder_rcs, plate_rcs, sphere_rcs, compute_error_stats
from geometry.cylinder import AnalyticCylinder
from geometry.plate import AnalyticPlate
from geometry.sphere import AnalyticSphere
from solvers.api import AVAILABLE_ALGORITHMS, get_integrator


def compute_rcs_single_angle(solver, surface, wave):
    """计算单个角度的 RCS"""
    k_mag = 2 * np.pi / wave.wavelength
    I = solver.integrate_surface(surface, wave)
    sigma = (k_mag**2 / np.pi) * np.abs(I)**2
    return 10.0 * np.log10(max(sigma, 1e-20))


def test_cylinder(algorithms, frequency=3e9, radius=0.5, height=2.0,
                  n_angles=37, samples_per_lambda=10):
    """测试圆柱体"""
    print("\n" + "="*70)
    print(f"圆柱体测试: R={radius}m, H={height}m, f={frequency/1e9:.1f}GHz")
    print("="*70)

    wavelength = C0 / frequency
    print(f"波长: {wavelength*1000:.2f}mm, 圆柱周长/λ: {2*np.pi*radius/wavelength:.1f}")

    # 角度范围：60° - 120°（侧向 ±30°）
    theta_deg = np.linspace(60, 120, n_angles)
    theta_rad = np.deg2rad(theta_deg)

    # 解析解
    rcs_analytical = cylinder_rcs(radius, height, frequency, theta_rad)

    # 创建圆柱曲面
    cylinder = AnalyticCylinder(radius, height)

    results = {}
    for algo_id, algo_info in algorithms.items():
        solver = get_integrator(algo_id, samples_per_lambda=samples_per_lambda)

        # 计算 RCS
        start_time = time.time()
        rcs_numerical = []
        for theta in theta_rad:
            wave = IncidentWave(frequency, theta, phi=np.pi/2)  # phi=90° 侧向入射
            rcs = compute_rcs_single_angle(solver, cylinder, wave)
            rcs_numerical.append(rcs)
        elapsed = time.time() - start_time

        rcs_numerical = np.array(rcs_numerical)

        # 统计误差
        stats = compute_error_stats(rcs_numerical, rcs_analytical)

        # 获取网格尺寸
        wave = IncidentWave(frequency, np.pi/2, phi=np.pi/2)
        if hasattr(solver, 'get_mesh_size'):
            mesh_size = solver.get_mesh_size(cylinder, wave, samples_per_lambda)
            n_points = mesh_size[0] * mesh_size[1]
        else:
            n_points = "N/A"

        results[algo_id] = {
            'rcs': rcs_numerical,
            'time': elapsed,
            'stats': stats,
            'n_points': n_points
        }

        print(f"\n{algo_info['name']}:")
        print(f"  网格点数: {n_points}")
        print(f"  耗时: {elapsed:.3f}s")
        print(f"  RMS误差: {stats['rms_error']:.2f} dB")
        print(f"  最大误差: {stats['max_error']:.2f} dB")
        print(f"  平均误差: {stats['mean_error']:.2f} dB")

    return theta_deg, rcs_analytical, results


def test_plate(algorithms, frequency=3e9, width=1.0, height=1.0,
               n_angles=37, samples_per_lambda=10):
    """测试平板"""
    print("\n" + "="*70)
    print(f"平板测试: {width}m × {height}m, f={frequency/1e9:.1f}GHz")
    print("="*70)

    wavelength = C0 / frequency
    print(f"波长: {wavelength*1000:.2f}mm, 平板尺寸/λ: {width/wavelength:.1f} × {height/wavelength:.1f}")

    # 角度范围：0° 到 60°（从法向到斜入射）
    # 平板法向沿 +z，theta=0 为法向入射
    theta_deg = np.linspace(0, 60, n_angles)
    theta_rad = np.deg2rad(theta_deg)

    # 解析解
    rcs_analytical = plate_rcs(width, height, frequency, theta_rad, phi_rad=0.0)

    # 创建平板曲面
    plate = AnalyticPlate(width, height)

    results = {}
    for algo_id, algo_info in algorithms.items():
        solver = get_integrator(algo_id, samples_per_lambda=samples_per_lambda)

        start_time = time.time()
        rcs_numerical = []
        for theta in theta_rad:
            # 平板法向沿 +z，theta=0 为法向入射
            # IncidentWave: theta=0 表示波从 +z 方向来（k_dir = -z）
            # 这正好照射平板正面 (n·k < 0)
            wave = IncidentWave(frequency, theta, phi=0)
            rcs = compute_rcs_single_angle(solver, plate, wave)
            rcs_numerical.append(rcs)
        elapsed = time.time() - start_time

        rcs_numerical = np.array(rcs_numerical)
        stats = compute_error_stats(rcs_numerical, rcs_analytical)

        wave = IncidentWave(frequency, np.pi, phi=0)
        if hasattr(solver, 'get_mesh_size'):
            mesh_size = solver.get_mesh_size(plate, wave, samples_per_lambda)
            n_points = mesh_size[0] * mesh_size[1]
        else:
            n_points = "N/A"

        results[algo_id] = {
            'rcs': rcs_numerical,
            'time': elapsed,
            'stats': stats,
            'n_points': n_points
        }

        print(f"\n{algo_info['name']}:")
        print(f"  网格点数: {n_points}")
        print(f"  耗时: {elapsed:.3f}s")
        print(f"  RMS误差: {stats['rms_error']:.2f} dB")
        print(f"  最大误差: {stats['max_error']:.2f} dB")
        print(f"  平均误差: {stats['mean_error']:.2f} dB")

    return theta_deg, rcs_analytical, results


def test_sphere(algorithms, frequency=10e9, radius=0.3,
                n_angles=37, samples_per_lambda=10):
    """测试球体"""
    print("\n" + "="*70)
    print(f"球体测试: R={radius}m, f={frequency/1e9:.1f}GHz")
    print("="*70)

    wavelength = C0 / frequency
    ka = 2 * np.pi * radius / wavelength
    print(f"波长: {wavelength*1000:.2f}mm, ka={ka:.1f}")

    if ka < 5:
        print(f"警告: ka={ka:.1f} < 5，可能不在光学区")

    # 球体 RCS 与角度无关（在光学区），但我们测试不同入射方向
    theta_deg = np.linspace(0, 180, n_angles)
    theta_rad = np.deg2rad(theta_deg)

    # 解析解（常数）
    rcs_analytical_val = sphere_rcs(radius, frequency)
    rcs_analytical = np.full_like(theta_rad, rcs_analytical_val)

    # 创建球体曲面
    sphere = AnalyticSphere(radius)

    results = {}
    for algo_id, algo_info in algorithms.items():
        solver = get_integrator(algo_id, samples_per_lambda=samples_per_lambda)

        start_time = time.time()
        rcs_numerical = []
        for theta in theta_rad:
            wave = IncidentWave(frequency, theta, phi=0)
            rcs = compute_rcs_single_angle(solver, sphere, wave)
            rcs_numerical.append(rcs)
        elapsed = time.time() - start_time

        rcs_numerical = np.array(rcs_numerical)
        stats = compute_error_stats(rcs_numerical, rcs_analytical)

        wave = IncidentWave(frequency, np.pi/2, phi=0)
        if hasattr(solver, 'get_mesh_size'):
            mesh_size = solver.get_mesh_size(sphere, wave, samples_per_lambda)
            n_points = mesh_size[0] * mesh_size[1]
        else:
            n_points = "N/A"

        results[algo_id] = {
            'rcs': rcs_numerical,
            'time': elapsed,
            'stats': stats,
            'n_points': n_points
        }

        print(f"\n{algo_info['name']}:")
        print(f"  网格点数: {n_points}")
        print(f"  耗时: {elapsed:.3f}s")
        print(f"  RMS误差: {stats['rms_error']:.2f} dB")
        print(f"  最大误差: {stats['max_error']:.2f} dB")
        print(f"  平均误差: {stats['mean_error']:.2f} dB")

    return theta_deg, rcs_analytical, results


def plot_results(theta_deg, rcs_analytical, results, title, algorithms):
    """绘制结果对比图"""
    try:
        import matplotlib.pyplot as plt

        # 设置中文字体
        plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：RCS 曲线
        ax1 = axes[0]
        ax1.plot(theta_deg, rcs_analytical, 'k-', linewidth=2, label='解析解')

        colors = ['r', 'g', 'b', 'm', 'c']
        for i, (algo_id, result) in enumerate(results.items()):
            color = colors[i % len(colors)]
            label = f"{algorithms[algo_id]['name']} (RMS={result['stats']['rms_error']:.1f}dB)"
            ax1.plot(theta_deg, result['rcs'], f'{color}--', linewidth=1.5, label=label)

        ax1.set_xlabel('θ (度)')
        ax1.set_ylabel('RCS (dBsm)')
        ax1.set_title(f'{title} - RCS 对比')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 右图：误差
        ax2 = axes[1]
        for i, (algo_id, result) in enumerate(results.items()):
            color = colors[i % len(colors)]
            error = result['rcs'] - rcs_analytical
            ax2.plot(theta_deg, error, f'{color}-', linewidth=1.5,
                    label=algorithms[algo_id]['name'])

        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('θ (度)')
        ax2.set_ylabel('误差 (dB)')
        ax2.set_title(f'{title} - 误差')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/{title.replace(" ", "_")}.png', dpi=150)
        plt.close()
        print(f"  图表已保存: results/{title.replace(' ', '_')}.png")

    except ImportError:
        print("matplotlib 未安装，跳过绘图")


def print_summary(all_results, algorithms):
    """打印汇总表格"""
    print("\n" + "="*70)
    print("汇总表格")
    print("="*70)

    print(f"\n{'算法':<25} {'圆柱RMS':<10} {'平板RMS':<10} {'球体RMS':<10} {'平均时间':<10}")
    print("-"*70)

    for algo_id in algorithms.keys():
        name = algorithms[algo_id]['name'][:24]

        cyl_rms = all_results['cylinder'][algo_id]['stats']['rms_error']
        plate_rms = all_results['plate'][algo_id]['stats']['rms_error']
        sphere_rms = all_results['sphere'][algo_id]['stats']['rms_error']

        avg_time = (all_results['cylinder'][algo_id]['time'] +
                   all_results['plate'][algo_id]['time'] +
                   all_results['sphere'][algo_id]['time']) / 3

        print(f"{name:<25} {cyl_rms:<10.2f} {plate_rms:<10.2f} {sphere_rms:<10.2f} {avg_time:<10.3f}s")


def main():
    print("="*70)
    print("PO 算法对比测试")
    print("="*70)

    # 确保 results 目录存在
    import os
    os.makedirs('results', exist_ok=True)

    # 选择要测试的算法
    algorithms = AVAILABLE_ALGORITHMS
    print(f"\n测试算法: {list(algorithms.keys())}")

    # 测试参数
    samples_per_lambda = 10
    n_angles = 37

    all_results = {}

    # 测试圆柱
    theta_cyl, rcs_cyl, results_cyl = test_cylinder(
        algorithms,
        frequency=3e9, radius=0.5, height=2.0,
        n_angles=n_angles, samples_per_lambda=samples_per_lambda
    )
    all_results['cylinder'] = results_cyl
    plot_results(theta_cyl, rcs_cyl, results_cyl, "圆柱体", algorithms)

    # 测试平板
    theta_plate, rcs_plate, results_plate = test_plate(
        algorithms,
        frequency=3e9, width=1.0, height=1.0,
        n_angles=n_angles, samples_per_lambda=samples_per_lambda
    )
    all_results['plate'] = results_plate
    plot_results(theta_plate, rcs_plate, results_plate, "平板", algorithms)

    # 测试球体
    theta_sphere, rcs_sphere, results_sphere = test_sphere(
        algorithms,
        frequency=10e9, radius=0.3,
        n_angles=n_angles, samples_per_lambda=samples_per_lambda
    )
    all_results['sphere'] = results_sphere
    plot_results(theta_sphere, rcs_sphere, results_sphere, "球体", algorithms)

    # 打印汇总
    print_summary(all_results, algorithms)

    print("\n测试完成！")


if __name__ == "__main__":
    main()
