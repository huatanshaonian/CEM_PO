import numpy as np

# 物理常数
C0 = 299792458.0  # 光速 (m/s)
MU0 = 4.0 * np.pi * 1e-7  # 真空磁导率
EPS0 = 1.0 / (MU0 * C0**2)  # 真空介电常数
ETA0 = np.sqrt(MU0 / EPS0)  # 自由空间阻抗 (~376.73 Ohms)
