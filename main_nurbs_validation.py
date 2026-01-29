import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl.visualization import VisMPL
from geomdl import NURBS

# 项目导入
from geometry.nurbs_surface import NURBSSurface
from geometry.cylinder import AnalyticCylinder
from solvers.po import DiscretePOIntegrator as RibbonIntegrator
from solvers.rcs_analyzer import RCSAnalyzer
from physics.constants import C0

def generate_nurbs_cylinder(radius, height):
    """
    手动构建 NURBS 圆柱面 (R=radius, H=height)
    轴向为 Z，中心在原点。
    """
    surf = NURBS.Surface()
    surf._dimension = 3 # HACK: Force 3D dimension to avoid auto-detect errors
    
    # 1. 设置阶数 (Degree)
    # u 方向 (圆周): 2次 (Quadratic)
    # v 方向 (高度): 1次 (Linear)
    surf.degree_u = 2
    surf.degree_v = 1
    # surf.dimension = 3 # Removed
    
    # 2. 设置控制点 (Control Points)
    # 我们需要 9 个点来构成一个完整的圆 (4个 90度弧段)
    # v 方向只需要 2 层 (底面和顶面)
    # 总共 9 * 2 = 18 个控制点
    
    # 基础圆周控制点 (R=1, z=0)
    # 顺序：东 -> 北 -> 西 -> 南 -> 东
    # 坐标 (x, y) 和 权重 w
    s = radius
    w_cor = 1.0
    w_mid = np.sqrt(2) / 2.0
    
    # 9个基础点 (x, y, w)
    base_points = [
        (s, 0, w_cor),      # 0 deg
        (s, s, w_mid),      # 45 deg
        (0, s, w_cor),      # 90 deg
        (-s, s, w_mid),     # 135 deg
        (-s, 0, w_cor),     # 180 deg
        (-s, -s, w_mid),    # 225 deg
        (0, -s, w_cor),     # 270 deg
        (s, -s, w_mid),     # 315 deg
        (s, 0, w_cor)       # 360 deg
    ]
    
    z_min = -height / 2.0
    z_max = height / 2.0
    
    # 构建 3D 控制点列表 (x, y, z, w) -> geomdl 要求 (x*w, y*w, z*w, w) 格式吗？
    # geomdl 的 ctrlpts 属性通常接受 (x, y, z) 或者是 weighted (x, y, z, w)?
    # 查阅 geomdl 文档：如果是 NURBS.Surface，ctrlpts 应该是 Cartesian coordinates? 
    # 不，NURBS 类通常存储 weighted control points 或 separate weights。
    # geomdl 中，surf.ctrlpts 是 (x, y, z)，surf.weights 是 w。
    # 我们先生成扁平化的 ctrlpts 列表和 weights 列表。
    
    ctrlpts = []
    # weights = [] # Removed
    
    # v=0 层 (Bottom)
    for x, y, w in base_points:
        ctrlpts.append([float(x*w), float(y*w), float(z_min*w), float(w)])
        
    # v=1 层 (Top)
    for x, y, w in base_points:
        ctrlpts.append([float(x*w), float(y*w), float(z_max*w), float(w)])
        
    print(f"DEBUG: Generated 4D ctrlpts, first: {ctrlpts[0]}")
    
    surf.set_ctrlpts(ctrlpts, 9, 2) 
    # surf.weights = weights # Removed
    
    # 3. 设置节点向量 (Knot Vectors)
    # u 方向: 4段，clamped. 
    # Knots: [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]
    surf.knotvector_u = [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]
    
    # v 方向: 1段，clamped.
    # Knots: [0, 0, 1, 1]
    surf.knotvector_v = [0, 0, 1, 1]
    
    surf.delta = 0.05
    print(f"DEBUG: NURBS dimension={surf.dimension}, first ctrlpt={surf.ctrlpts[0]}")
    
    return surf

def main():
    # 参数设置
    freq = 300e6 # 300 MHz
    radius = 1.0
    height = 2.0
    
    print(f"Validation Frequency: {freq/1e6} MHz")
    print(f"Geometry: Cylinder (R={radius}, H={height})")
    
    # 1. 准备几何体
    
    # A. Analytic Cylinder (基准)
    analytic_cyl = AnalyticCylinder(radius, height)
    
    # B. NURBS Cylinder (待验证)
    print("Generating NURBS cylinder...")
    nbs_cyl_raw = generate_nurbs_cylinder(radius, height)
    nurbs_cyl = NURBSSurface(nbs_cyl_raw)
    
    # 2. 准备求解器
    # 使用自适应网格，提高精度以便对比
    solver = RibbonIntegrator(samples_per_lambda=15)
    analyzer = RCSAnalyzer(solver)
    
    # 3. 扫描角度
    # theta: -90 到 90 度 (Broadside at 90, End-fire at 0/180)
    # 我们的 AnalyticCylinder 定义中，Z 轴是轴线。
    # theta=90度 (pi/2) 是垂直于 Z 轴入射 (正侧视)。
    # theta=0度 是沿 Z 轴入射 (端射)。
    # 我们扫描 theta 从 30 到 150 度 (围绕 90 度)
    scan_angles_deg = np.linspace(30, 150, 61)
    scan_angles_rad = np.radians(scan_angles_deg)
    
    # 4. 计算 RCS
    print("Computing Analytic RCS...")
    rcs_analytic = analyzer.compute_monostatic_rcs(
        analytic_cyl, 
        {'frequency': freq, 'phi': 0.0},
        scan_angles_rad
    )
    
    print("Computing NURBS RCS...")
    rcs_nurbs = analyzer.compute_monostatic_rcs(
        nurbs_cyl, 
        {'frequency': freq, 'phi': 0.0},
        scan_angles_rad
    )
    
    # 5. 分析结果
    error = np.abs(rcs_nurbs - rcs_analytic)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"\nMax Difference: {max_error:.4f} dB")
    print(f"Mean Difference: {mean_error:.4f} dB")
    
    # 6. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(scan_angles_deg, rcs_analytic, 'r-', linewidth=2, label='Analytic Surface')
    plt.plot(scan_angles_deg, rcs_nurbs, 'b--', linewidth=2, label='NURBS Surface') # 虚线以便看到重合
    plt.xlabel('Theta (deg)')
    plt.ylabel('RCS (dBsm)')
    plt.title(f'RCS Validation: NURBS vs Analytic Cylinder\n(f={freq/1e6}MHz, R={radius}m, H={height}m)')
    plt.legend()
    plt.grid(True)
    
    output_file = 'nurbs_validation.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    if max_error < 0.5:
        print("\n[SUCCESS] NURBS implementation matches Analytic result.")
    else:
        print("\n[WARNING] Discrepancy detected.")

if __name__ == "__main__":
    main()
