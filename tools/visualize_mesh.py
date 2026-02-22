import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry.plate import AnalyticPlate
from geometry.sphere import AnalyticSphere
from geometry.cylinder import AnalyticCylinder
from geometry.occ_surface import OCCSurface
from geometry.occ_primitives import create_occ_cylinder
from solvers.po import DiscretePOIntegrator as RibbonIntegrator
from physics.wave import IncidentWave


def plot_mesh(surface, freq, samples_per_lambda, title):
    # 初始化求解器
    solver = RibbonIntegrator()
    wave = IncidentWave(freq, 0, 0) 
    
    # 获取网格数据
    points, normals, (nu, nv) = solver.get_mesh_data(surface, wave, samples_per_lambda)
    
    print(f"[{title}]")
    print(f"  Frequency: {freq/1e6:.1f} MHz (Lambda={wave.wavelength:.2f}m)")
    print(f"  Samples/Lambda: {samples_per_lambda}")
    print(f"  Grid Size: Nu={nu}, Nv={nv} (Total: {nu*nv} points)")
    
    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X = points[..., 0]
    Y = points[..., 1]
    Z = points[..., 2]
    
    # 降采样显示以避免过卡
    stride_u = max(1, nu // 50)
    stride_v = max(1, nv // 50)
    
    ax.plot_wireframe(X, Y, Z, color='b', linewidth=0.5, rstride=stride_v, cstride=stride_u, alpha=0.6)
    
    # 法向量
    skip = max(1, min(nu, nv) // 10)
    # 处理法向量可能为0的情况
    N_x = normals[::skip, ::skip, 0]
    N_y = normals[::skip, ::skip, 1]
    N_z = normals[::skip, ::skip, 2]
    
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], Z[::skip, ::skip], 
              N_x, N_y, N_z,
              length=wave.wavelength/5, color='r', alpha=0.5, label='Normals')

    # 设置轴比例
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title(f"{title}\nGrid: {nu}x{nv}")
    plt.show()

def main():
    freq = 300e6 
    samples_per_lambda = 10 
    
    print("Visualizing Geometries...")
    
    # Case 1: Analytic Plate
    # plate = AnalyticPlate(5.0, 5.0)
    # plot_mesh(plate, freq, samples_per_lambda, "Analytic Plate (5x5m)")
    
    # Case 2: OCC Cylinder
    # 创建 OCC 几何
    print("Generating OCC Cylinder...")
    occ_geom = create_occ_cylinder(radius=1.0, height=2.0)
    occ_surface = OCCSurface(occ_geom)
    
    plot_mesh(occ_surface, freq, samples_per_lambda, "OCC Cylinder (R=1, H=2)")

if __name__ == "__main__":
    main()