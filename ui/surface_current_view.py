"""
表面电流可视化视图：嵌入在主窗口右侧 view_tab_bar 的一个 tab。
独立文件以避免 gui_qt.py 进一步膨胀。
"""
from __future__ import annotations
import numpy as np
import pyvista as pv
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton, QCheckBox,
)
from pyvistaqt import QtInteractor


_DISPLAY_MODES = [
    ('|J|', 'mag'),
    ('arg(J)', 'phase'),
    ('Re(J_x)', 're_jx'),
    ('Re(J_y)', 're_jy'),
    ('Re(J_z)', 're_jz'),
    ('Im(J_x)', 'im_jx'),
    ('Im(J_y)', 'im_jy'),
    ('Im(J_z)', 'im_jz'),
]


class SurfaceCurrentView(QFrame):
    """右侧 view 区的表面电流 tab。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields = []          # list[SurfaceCurrentField]
        self._meta = {}            # 入射角/极化/频率，用于标题
        self._scalar_log_db = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        self.plotter.add_axes(color='black')
        layout.addWidget(self.plotter.interactor, 1)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('Display:'))
        self.combo_mode = QComboBox()
        for label, _ in _DISPLAY_MODES:
            self.combo_mode.addItem(label)
        self.combo_mode.currentIndexChanged.connect(self._refresh)
        ctrl.addWidget(self.combo_mode)

        self.chk_log = QCheckBox('dB (|J|)')
        self.chk_log.setToolTip('|J| 标量场以 20·log10 显示。仅对 |J| 模式生效。')
        self.chk_log.stateChanged.connect(self._refresh)
        ctrl.addWidget(self.chk_log)

        self.chk_arrows = QCheckBox('Re(J) arrows')
        self.chk_arrows.stateChanged.connect(self._refresh)
        ctrl.addWidget(self.chk_arrows)

        self.chk_dark = QCheckBox('Show dark side')
        self.chk_dark.setChecked(True)
        self.chk_dark.setToolTip('关掉则隐藏暗区面元（按 lit_mask 过滤）。')
        self.chk_dark.stateChanged.connect(self._refresh)
        ctrl.addWidget(self.chk_dark)

        ctrl.addStretch()
        self.btn_reset = QPushButton('Reset View')
        self.btn_reset.clicked.connect(lambda: self.plotter.reset_camera())
        ctrl.addWidget(self.btn_reset)
        layout.addLayout(ctrl)

        self.lbl_info = QLabel('')
        self.lbl_info.setStyleSheet('color: #555; padding: 2px 6px;')
        layout.addWidget(self.lbl_info)

    def set_fields(self, fields, meta=None):
        """更新数据并刷新渲染。"""
        self._fields = list(fields or [])
        self._meta = dict(meta or {})
        self._refresh()
        self.plotter.reset_camera()

    def _selected_scalar(self):
        return _DISPLAY_MODES[self.combo_mode.currentIndex()][1]

    def _refresh(self):
        self.plotter.clear()
        if not self._fields:
            self.lbl_info.setText('(尚未计算) 请在左侧 Surface Current 栏点击 Compute。')
            return

        mode = self._selected_scalar()
        show_dark = self.chk_dark.isChecked()
        show_arrows = self.chk_arrows.isChecked()
        as_db = self.chk_log.isChecked() and mode == 'mag'

        # 1) 汇总数据并算全局色标范围
        all_scalars = []
        per_field_scalars = []
        for f in self._fields:
            s = _extract_scalar(f, mode)
            if not show_dark:
                s = np.where(f.lit_mask, s, np.nan)
            if as_db:
                with np.errstate(divide='ignore', invalid='ignore'):
                    s = 20.0 * np.log10(np.maximum(np.abs(s), 1e-30))
            per_field_scalars.append(s)
            valid = s[np.isfinite(s)]
            if valid.size:
                all_scalars.append(valid)

        if all_scalars:
            joined = np.concatenate(all_scalars)
            clim = (float(np.nanmin(joined)), float(np.nanmax(joined)))
            if clim[1] - clim[0] < 1e-12:
                clim = (clim[0] - 1e-6, clim[1] + 1e-6)
        else:
            clim = (0.0, 1.0)

        # 2) 逐 surface 渲染
        scalar_title = _scalar_title(mode, as_db)
        diag = _global_diag([f.points for f in self._fields])

        first = True
        for f, s in zip(self._fields, per_field_scalars):
            mesh = _build_pv_mesh(f, s)
            if mesh is None:
                continue
            self.plotter.add_mesh(
                mesh,
                scalars='value',
                cmap='jet',
                clim=clim,
                show_edges=False,
                nan_opacity=0.0,
                scalar_bar_args={'title': scalar_title} if first else None,
                show_scalar_bar=first,
            )
            first = False

            if show_arrows:
                _add_current_arrows(self.plotter, f, diag)

        # 3) 信息标签
        meta = self._meta
        info_bits = []
        if 'frequency' in meta:
            info_bits.append(f"f={meta['frequency']/1e9:.3f} GHz")
        if 'theta_deg' in meta:
            info_bits.append(f"θ={meta['theta_deg']:.2f}°")
        if 'phi_deg' in meta:
            info_bits.append(f"φ={meta['phi_deg']:.2f}°")
        if 'polarization' in meta:
            info_bits.append(f"pol={meta['polarization']}")
        n_lit = sum(int(np.sum(f.lit_mask)) for f in self._fields)
        n_tot = sum(int(f.lit_mask.size) for f in self._fields)
        info_bits.append(f"lit={n_lit}/{n_tot}")
        self.lbl_info.setText('  |  '.join(info_bits))


# ----------------------------- helpers -----------------------------

def _scalar_title(mode: str, as_db: bool) -> str:
    if mode == 'mag':
        return '|J| (dB)' if as_db else '|J| (A/m)'
    if mode == 'phase':
        return 'arg(J) (rad)'
    label = {
        're_jx': 'Re(J_x)', 're_jy': 'Re(J_y)', 're_jz': 'Re(J_z)',
        'im_jx': 'Im(J_x)', 'im_jy': 'Im(J_y)', 'im_jz': 'Im(J_z)',
    }[mode]
    return f'{label} (A/m)'


def _extract_scalar(field, mode: str) -> np.ndarray:
    if mode == 'mag':
        return field.J_mag.astype(float)
    if mode == 'phase':
        # 用 |J| 选主分量（在 J 的最大幅值分量上取相位），避免零向量噪声
        idx = np.argmax(np.abs(field.J), axis=1)
        rows = np.arange(field.J.shape[0])
        return np.angle(field.J[rows, idx])
    comp = {'re_jx': (0, True), 're_jy': (1, True), 're_jz': (2, True),
            'im_jx': (0, False), 'im_jy': (1, False), 'im_jz': (2, False)}[mode]
    j = field.J[:, comp[0]]
    return j.real if comp[1] else j.imag


def _build_pv_mesh(field, scalars: np.ndarray):
    """根据 grid_shape 构造 PolyData。
    - 规则网格 (nv, nu)：cell centers 作为顶点，相邻中心连成 quad，赋 point scalar。
    - 退化网格：纯点云 + sphere glyph。
    """
    pts = np.asarray(field.points, dtype=float)
    if field.grid_shape is not None:
        nv, nu = field.grid_shape
        pts_grid = pts.reshape(nv, nu, 3)
        # 构造 quad faces 连接相邻 cell centers
        faces = []
        idx = np.arange(nv * nu).reshape(nv, nu)
        for i in range(nv - 1):
            for j in range(nu - 1):
                a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]
                faces.extend([4, a, b, c, d])
        mesh = pv.PolyData(pts_grid.reshape(-1, 3), faces=np.array(faces, dtype=np.int64))
        mesh.point_data['value'] = scalars
        return mesh
    # 退化网格：点云
    if pts.shape[0] == 0:
        return None
    cloud = pv.PolyData(pts)
    cloud.point_data['value'] = scalars
    # 用 sphere glyph 让点能被着色
    diag = _global_diag([pts])
    glyph = cloud.glyph(geom=pv.Sphere(radius=max(diag * 0.005, 1e-6)),
                        scale=False, orient=False)
    return glyph


def _add_current_arrows(plotter, field, diag):
    lit = field.lit_mask
    if not np.any(lit):
        return
    pts = field.points[lit]
    j_re = field.J[lit].real
    mag = np.linalg.norm(j_re, axis=1)
    if mag.max() <= 0:
        return
    # 全局归一化，长度按 diag 比例
    scale = (diag * 0.04) / mag.max()
    plotter.add_arrows(pts, j_re, mag=scale, color='black', opacity=0.7)


def _global_diag(points_list) -> float:
    pts = np.vstack([p.reshape(-1, 3) for p in points_list if p.size])
    if pts.size == 0:
        return 1.0
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0
