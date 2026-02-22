import numpy as np

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3
from OCC.Core.Geom import Geom_CylindricalSurface, Geom_RectangularTrimmedSurface


def create_occ_cylinder(radius, height):
    """创建一个裁剪后的 OCC 圆柱曲面对象。"""
    origin = gp_Pnt(0, 0, -height / 2.0)
    axis = gp_Ax3(origin, gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
    cyl_surf = Geom_CylindricalSurface(axis, radius)
    trimmed_cyl = Geom_RectangularTrimmedSurface(cyl_surf, 0, 2 * np.pi, 0, height, True, True)
    return trimmed_cyl
