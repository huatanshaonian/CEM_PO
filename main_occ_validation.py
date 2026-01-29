import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# OCC 导入
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3
from OCC.Core.Geom import Geom_CylindricalSurface
from OCC.Core.GeomConvert import geomconvert

# 项目导入
from geometry.occ_surface import OCCSurface
from geometry.cylinder import AnalyticCylinder
from solvers.po import DiscretePOIntegrator as RibbonIntegrator
from solvers.rcs_analyzer import RCSAnalyzer
from physics.constants import C0

def create_occ_nurbs_cylinder(radius, height):
    """
    使用 OCC 创建一个圆柱面，并转换为 NURBS 格式。
    """
    # 定义轴系 (原点, Z方向, X方向)
    # 平移原点到 -height/2 以匹配中心在原点
    origin = gp_Pnt(0, 0, -height/2.0)
    axis = gp_Ax3(origin, gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
    
    # 1. 创建圆柱面 (它是无限长的，所以我们需要设定范围)
    cyl_surf = Geom_CylindricalSurface(axis, radius)
    
    # 2. 转换/裁剪 (注意：OCC 的表面本身可能有范围，或者在 evaluation 时控制)
    # 我们直接使用它。OCCSurface 会自动读取 Bounds。
    # 对于圆柱面，U 是 [0, 2pi], V 是 [-inf, inf]。
    # 我们通常需要将其限制在 [0, height]。
    # 在 OCC 中，我们可以使用 Geom_RectangularTrimmedSurface。
    from OCC.Core.Geom import Geom_RectangularTrimmedSurface
    trimmed_cyl = Geom_RectangularTrimmedSurface(cyl_surf, 0, 2*np.pi, 0, height, True, True)
    
    # 3. 转换为 NURBS (BSplineSurface) 以测试通用性
    nurbs_cyl = geomconvert.SurfaceToBSplineSurface(trimmed_cyl)
    
    return nurbs_cyl

def main():
    # 参数设置
    freq = 300e6 
    radius = 1.0
    height = 2.0
    
    print(f"Validation Frequency: {freq/1e6} MHz")
    print("Using PythonOCC (OpenCascade) for NURBS computation.")
    
    # 1. 准备几何体
    analytic_cyl = AnalyticCylinder(radius, height)
    
    print("Creating NURBS cylinder via PythonOCC...")
    occ_nbs_raw = create_occ_nurbs_cylinder(radius, height)
    occ_surface = OCCSurface(occ_nbs_raw)
    
    # 2. 准备求解器
    solver = RibbonIntegrator(samples_per_lambda=15)
    analyzer = RCSAnalyzer(solver)
    
    # 3. 扫描角度
    scan_angles_deg = np.linspace(30, 150, 61)
    scan_angles_rad = np.radians(scan_angles_deg)
    
    # 4. 计算 RCS
    print("Computing Analytic RCS...")
    rcs_analytic = analyzer.compute_monostatic_rcs(
        analytic_cyl, 
        {'frequency': freq, 'phi': 0.0},
        scan_angles_rad
    )
    
    print("Computing OCC-NURBS RCS...")
    rcs_occ = analyzer.compute_monostatic_rcs(
        occ_surface, 
        {'frequency': freq, 'phi': 0.0},
        scan_angles_rad
    )
    
    # 5. 分析结果
    error = np.abs(rcs_occ - rcs_analytic)
    max_error = np.max(error)
    print(f"\nMax Difference: {max_error:.4f} dB")
    
    # 6. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(scan_angles_deg, rcs_analytic, 'r-', linewidth=2, label='Analytic Surface')
    plt.plot(scan_angles_deg, rcs_occ, 'b--', linewidth=2, label='OCC NURBS Surface')
    plt.xlabel('Theta (deg)')
    plt.ylabel('RCS (dBsm)')
    plt.title(f'RCS Validation: PythonOCC NURBS vs Analytic\n(f={freq/1e6}MHz, R={radius}m, H={height}m)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('occ_validation.png')
    print("Plot saved to occ_validation.png")
    
    if max_error < 0.5:
        print("\n[SUCCESS] PythonOCC NURBS implementation matches Analytic result.")
    else:
        print("\n[WARNING] Discrepancy detected. Check normal orientation or bounds.")

if __name__ == "__main__":
    main()
