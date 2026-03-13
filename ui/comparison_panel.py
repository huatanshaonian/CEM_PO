import os

import matplotlib.ticker
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog

from physics.analytical_rcs import get_analytical_solution


class ComparisonManager:
    """Delegate class that owns all post-processing / comparison panel logic.

    Accessed via ``self.w`` (the main CEMPoQtWindow instance).
    """

    def __init__(self, window):
        self.w = window

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def add_comparison_file(self):
        paths, _ = QFileDialog.getOpenFileNames(self.w, "Select CSV Files", "", "CSV Files (*.csv)")
        for path in paths:
            try:
                df = pd.read_csv(path, comment='#')
                name = os.path.basename(path)
                self.w.comparison_data.append({
                    'name': name,
                    'data': df,
                    'path': path
                })
                self.w.comp_files_list.addItem(name)
            except Exception as e:
                self.w.log(f"Error loading {path}: {e}")

        self._refresh_dataset_combos()
        self.update_comparison_plot()

    def remove_comparison_file(self):
        selected_items = self.w.comp_files_list.selectedItems()
        if not selected_items:
            return

        rows = sorted([self.w.comp_files_list.row(item) for item in selected_items], reverse=True)
        for row in rows:
            self.w.comp_files_list.takeItem(row)
            del self.w.comparison_data[row]

        self._refresh_dataset_combos()
        self.update_comparison_plot()

    # ------------------------------------------------------------------
    # Dataset combo helpers (used by 2D heatmap mode)
    # ------------------------------------------------------------------

    def _refresh_dataset_combos(self):
        """Repopulate the three dataset combo boxes with current CSV names + sim option."""
        sim_label = "Current Simulation"
        csv_names = [item['name'] for item in self.w.comparison_data]
        options = ["(none)", sim_label] + csv_names

        for combo in (self.w.combo_ds_a, self.w.combo_ds_b, self.w.combo_ds_ref):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(options)
            idx = combo.findText(prev)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    def _resolve_dataset(self, label):
        """Return (theta_1d, phi_1d, rcs_2d, name) for the given combo label, or None."""
        if not label or label == "(none)":
            return None
        if label == "Current Simulation":
            if not self.w.last_result or self.w.last_result.get('mode') != '2d':
                return None
            return (self.w.last_result['theta_deg'],
                    self.w.last_result['phi_deg'],
                    self.w.last_result['rcs_total'],
                    "Calculated")
        for item in self.w.comparison_data:
            if item['name'] == label:
                return self._parse_csv_2d(item)
        return None

    def _resolve_dataset_1d(self, label):
        """Return (angle_1d, rcs_1d, name) for 1D/polar modes, or None.

        For 2D sim results, slices along the axis selected by combo_slice_axis at spin_slice_angle.
        Slice axis 'Phi'   → fix phi,   sweep theta  → returns (theta_deg, rcs, name)
        Slice axis 'Theta' → fix theta, sweep phi    → returns (phi_deg,   rcs, name)
        """
        if not label or label == "(none)":
            return None
        if label == "Current Simulation":
            r = self.w.last_result
            if not r:
                return None
            if r.get('mode') == '1d':
                return r['theta_deg'], r['rcs_total'], "Simulation"
            if r.get('mode') == '2d':
                axis   = self.w.combo_slice_axis.currentText()   # "Phi" or "Theta"
                angle  = self.w.spin_slice_angle.value()
                if axis == "Phi":
                    phi_arr = r['phi_deg']
                    idx = int(np.argmin(np.abs(phi_arr - angle)))
                    return r['theta_deg'], r['rcs_total'][:, idx], f"Sim (φ={phi_arr[idx]:.1f}°)"
                else:
                    theta_arr = r['theta_deg']
                    idx = int(np.argmin(np.abs(theta_arr - angle)))
                    return r['phi_deg'], r['rcs_total'][idx, :], f"Sim (θ={theta_arr[idx]:.1f}°)"
            return None
        for item in self.w.comparison_data:
            if item['name'] == label:
                return self._parse_csv_1d(item)
        return None

    # ------------------------------------------------------------------
    # Main plot entry point
    # ------------------------------------------------------------------

    def update_comparison_plot(self):
        self.w.comp_figure.clear()

        mode = self.w.combo_postproc_mode.currentText()

        if mode == "1D Line":
            self._draw_1d_lines()
        elif mode == "Polar":
            self._draw_polar_lines()
        else:
            self._draw_2d_heatmap()

        self.w.comp_canvas.draw()

    def _show_comp_message(self, msg):
        """Display a centered message in the comparison canvas."""
        ax = self.w.comp_figure.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11,
                transform=ax.transAxes, color='gray',
                bbox=dict(facecolor='lightyellow', edgecolor='gray', pad=8))
        ax.axis('off')

    # ------------------------------------------------------------------
    # Unit conversion helper
    # ------------------------------------------------------------------

    def _use_db(self):
        return self.w.combo_rcs_unit.currentText() == "dBsm"

    def _cbar_range(self, auto_vmin, auto_vmax):
        """返回 (vmin, vmax)：勾选手动时读取输入框，否则返回自动值。"""
        if not self.w.chk_cbar_manual.isChecked():
            return auto_vmin, auto_vmax
        try:
            vmin = float(self.w.cbar_vmin.text())
        except ValueError:
            vmin = auto_vmin
        try:
            vmax = float(self.w.cbar_vmax.text())
        except ValueError:
            vmax = auto_vmax
        return vmin, vmax

    def _conv(self, rcs_db_arr):
        """Convert dBsm array to display units."""
        if self._use_db():
            return rcs_db_arr
        return 10.0 ** (np.asarray(rcs_db_arr) / 10.0)

    def _unit_label(self):
        return "dBsm" if self._use_db() else "m²"

    # ------------------------------------------------------------------
    # Dataset collection (shared by 1D and Polar modes)
    # Tuple format: (angle_deg, rcs_db, name, linestyle, color)
    #   color=None → use auto color palette
    # ------------------------------------------------------------------

    def _collect_datasets(self):
        """Build list of (angle_deg, rcs_db, name, linestyle, color) from current state."""
        datasets = []
        r = self.w.last_result
        show_total = self.w.chk_show_total.isChecked()
        show_po    = self.w.chk_show_po.isChecked()
        show_ptd   = self.w.chk_show_ptd.isChecked()

        if r:
            if r.get('mode') == '1d':
                if show_total:
                    datasets.append((r['theta_deg'], r['rcs_total'], 'Sim (Total)', '-', None))
                if show_po and r.get('rcs_po') is not None:
                    datasets.append((r['theta_deg'], r['rcs_po'], 'Sim (PO)', '--', None))
                if show_ptd and r.get('rcs_ptd') is not None:
                    datasets.append((r['theta_deg'], r['rcs_ptd'], 'Sim (PTD)', ':', None))
            elif r.get('mode') == '2d':
                res = self._resolve_dataset_1d("Current Simulation")
                if res and show_total:
                    datasets.append((res[0], res[1], res[2], '-', None))

        for item in self.w.comparison_data:
            res = self._parse_csv_1d(item)
            if res:
                datasets.append((res[0], res[1], res[2], '--', None))

        if self.w.chk_analytical_comp.isChecked() and r:
            analytic = self._get_analytical_data(r)
            if analytic:
                datasets.append(analytic)

        return datasets

    # ------------------------------------------------------------------
    # 1D line plot mode
    # ------------------------------------------------------------------

    def _draw_1d_lines(self):
        ax = self.w.comp_figure.add_subplot(111)
        colors = ['#007ACC', '#E06C00', '#009900', '#CC0000', '#8800CC', '#009999']
        ci = 0

        datasets = self._collect_datasets()
        for angle_deg, rcs_db, name, ls, color in datasets:
            c = color if color else colors[ci % len(colors)]
            kw = dict(label=name, color=c)
            if ls == '-':
                kw['linewidth'] = 2
            elif ls == ':':
                kw.update(alpha=0.85, linewidth=1.5)
            else:
                kw['alpha'] = 0.85
            ax.plot(angle_deg, self._conv(rcs_db), ls, **kw)
            if not color:
                ci += 1

        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel(f"RCS ({self._unit_label()})")
        ax.set_title("RCS Comparison")
        ax.grid(True, linestyle='--', alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    # ------------------------------------------------------------------
    # Polar plot mode
    # ------------------------------------------------------------------

    def _draw_polar_lines(self):
        datasets = self._collect_datasets()
        if not datasets:
            self._show_comp_message("No data available. Run a simulation or import a CSV.")
            return

        ax = self.w.comp_figure.add_subplot(111, projection='polar')
        colors = ['#007ACC', '#E06C00', '#009900', '#CC0000', '#8800CC', '#009999']
        ci = 0

        use_db = self._use_db()
        all_vals = np.concatenate([self._conv(d[1]) for d in datasets])

        if use_db:
            db_floor = np.nanmin(all_vals) - 5.0

        for angle_deg, rcs_db, name, ls, color in datasets:
            theta_rad = np.radians(angle_deg)
            vals = self._conv(rcs_db)
            r_vals = np.clip(vals - db_floor, 0, None) if use_db else np.clip(vals, 0, None)
            c = color if color else colors[ci % len(colors)]
            ax.plot(theta_rad, r_vals, ls, label=name, color=c, linewidth=1.8)
            if not color:
                ci += 1

        # Radial tick labels
        ax.figure.canvas.draw()
        yticks = ax.get_yticks()
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(yticks))
        if use_db:
            ax.set_yticklabels([f'{v + db_floor:.0f}' for v in yticks], fontsize=7)
        else:
            ax.set_yticklabels([f'{v:.2g}' for v in yticks], fontsize=7)
        ax.set_title(f"Polar RCS Pattern ({self._unit_label()})", pad=15)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)

    # ------------------------------------------------------------------
    # 2D heatmap comparison mode (original behavior)
    # ------------------------------------------------------------------

    def _draw_2d_heatmap(self):
        da = self._resolve_dataset(self.w.combo_ds_a.currentText())
        db = self._resolve_dataset(self.w.combo_ds_b.currentText())
        dr = self._resolve_dataset(self.w.combo_ds_ref.currentText())

        if da is None:
            self._show_comp_message("Select a dataset in A.")
            return

        ta, pa, ma, na = da

        if db is None:
            # 当前仿真结果：尝试并排显示 Total / PO / PTD 子图
            if self.w.combo_ds_a.currentText() == "Current Simulation":
                r = self.w.last_result
                if r and r.get('mode') == '2d':
                    self._draw_components_2d(r, ta, pa)
                    return
            # 外部 CSV 或无分量数据：单图显示
            ax = self.w.comp_figure.add_subplot(111)
            extent = [pa.min(), pa.max(), ta.max(), ta.min()]
            phi_range = pa.max() - pa.min() if len(pa) > 1 else 1.0
            theta_range = ta.max() - ta.min() if len(ta) > 1 else 1.0
            data_aspect = phi_range / theta_range if theta_range > 0 else 1.0
            vmin, vmax = self._cbar_range(float(np.nanmin(ma)), float(np.nanmax(ma)))
            im = ax.imshow(ma, extent=extent, aspect=data_aspect, origin='upper',
                           cmap='jet', vmin=vmin, vmax=vmax)
            self.w.comp_figure.colorbar(im, ax=ax, label=f'RCS ({self._unit_label()})')
            ax.set_xlabel('Phi (deg)')
            ax.set_ylabel('Theta (deg)')
            ax.set_title(f'{na}\n(Select a second dataset in B to compare)')
            return

        tb, pb, mb, nb = db
        t_c, p_c, ma_c, mb_c = self._get_common_grid(ta, pa, ma, tb, pb, mb)
        if t_c is None:
            self._show_comp_message(
                f"No overlapping angle range between A and B.\n"
                f"A: θ[{ta.min():.1f},{ta.max():.1f}] φ[{pa.min():.1f},{pa.max():.1f}]\n"
                f"B: θ[{tb.min():.1f},{tb.max():.1f}] φ[{pb.min():.1f},{pb.max():.1f}]")
            return

        if dr is not None:
            tr, pr, mr, nr = dr
            mr_c = self._align_to_grid(mr, tr, pr, t_c, p_c)
            self._draw_6panel_2d(ma_c, na, mb_c, nb, mr_c, nr, t_c, p_c)
        else:
            self._draw_3panel_2d(ma_c, na, mb_c, nb, t_c, p_c, freq_mhz=None)

    # ------------------------------------------------------------------
    # Analytical solution helpers
    # ------------------------------------------------------------------

    def _get_analytical_data(self, result):
        """Return (angle_deg, rcs_db, label, linestyle, color) for the analytical solution, or None."""
        try:
            gtype = self.w.geo_type_combo.currentText()
            geo_params = self.w.get_geo_params()
            polarization = self.w.ptd_pol.currentText()
            theta_deg = result['theta_deg']
            theta_rad = np.radians(theta_deg)
            rcs, label = get_analytical_solution(gtype, geo_params, result['freq'],
                                                 theta_rad, polarization)
            if rcs is not None:
                return theta_deg, rcs, label, ':', 'red'
        except Exception:
            pass
        return None

    def _overlay_analytical_1d(self, ax, result):
        data = self._get_analytical_data(result)
        if data:
            theta_deg, rcs, label, ls, color = data
            ax.plot(theta_deg, rcs, ls, linewidth=2, label=label, color=color, alpha=0.9)

    # ------------------------------------------------------------------
    # CSV parsing
    # ------------------------------------------------------------------

    def _parse_csv_1d(self, item):
        """Parse CSV as 1D RCS (theta, rcs). Returns (theta_1d, rcs_1d, name) or None."""
        df = item['data']
        cols_lower = [c.lower() for c in df.columns]
        theta_col = next((df.columns[i] for i, c in enumerate(cols_lower) if 'theta' in c), None)
        rcs_col   = next((df.columns[i] for i, c in enumerate(cols_lower) if 'dbsm' in c), None)
        if rcs_col is None:
            rcs_col = next((df.columns[i] for i, c in enumerate(cols_lower)
                            if 'rcs' in c and df.columns[i] != theta_col), None)
        if not (theta_col and rcs_col):
            return None
        try:
            theta = df[theta_col].values.astype(float)
            rcs   = df[rcs_col].values.astype(float)
            return theta, rcs, item['name']
        except Exception:
            return None

    def _parse_csv_2d(self, item):
        """Parse a loaded CSV as 2D RCS data (flat/long format with Theta, Phi, RCS columns).

        Returns (theta_1d, phi_1d, rcs_2d, name) or None if not 2D.
        """
        df = item['data']
        cols_lower = [c.lower() for c in df.columns]

        theta_col = next((df.columns[i] for i, c in enumerate(cols_lower) if 'theta' in c), None)
        phi_col   = next((df.columns[i] for i, c in enumerate(cols_lower) if 'phi'   in c), None)
        rcs_col   = next((df.columns[i] for i, c in enumerate(cols_lower) if 'dbsm'  in c), None)
        if rcs_col is None:
            rcs_col = next((df.columns[i] for i, c in enumerate(cols_lower)
                            if 'rcs' in c and df.columns[i] not in (theta_col, phi_col)), None)

        if not (theta_col and phi_col and rcs_col):
            return None

        try:
            pivot    = df.pivot_table(index=theta_col, columns=phi_col, values=rcs_col, aggfunc='mean')
            theta_1d = pivot.index.values.astype(float)
            phi_1d   = pivot.columns.values.astype(float)
            rcs_2d   = pivot.values.astype(float)
            return theta_1d, phi_1d, rcs_2d, item['name']
        except Exception as e:
            self.w.log(f"2D CSV parse failed ({item['name']}): {e}")
            return None

    # ------------------------------------------------------------------
    # Grid alignment
    # ------------------------------------------------------------------

    def _get_common_grid(self, t1, p1, m1, t2, p2, m2):
        """Find overlapping theta×phi region and interpolate both datasets onto it."""
        t_min = max(t1.min(), t2.min())
        t_max = min(t1.max(), t2.max())
        p_min = max(p1.min(), p2.min())
        p_max = min(p1.max(), p2.max())
        if t_min >= t_max or p_min >= p_max:
            return None, None, None, None
        t_c = t1[(t1 >= t_min) & (t1 <= t_max)]
        p_c = p1[(p1 >= p_min) & (p1 <= p_max)]
        if len(t_c) < 2 or len(p_c) < 2:
            return None, None, None, None
        return t_c, p_c, self._align_to_grid(m1, t1, p1, t_c, p_c), self._align_to_grid(m2, t2, p2, t_c, p_c)

    def _align_to_grid(self, ref_mat, ref_theta, ref_phi, target_theta, target_phi):
        """Interpolate ref_mat (on ref_theta × ref_phi) onto target_theta × target_phi."""
        try:
            from scipy.interpolate import RegularGridInterpolator
            fn = RegularGridInterpolator((ref_theta, ref_phi), ref_mat,
                                         method='linear', bounds_error=False, fill_value=np.nan)
            Tg, Pg = np.meshgrid(target_theta, target_phi, indexing='ij')
            return fn((Tg, Pg))
        except ImportError:
            ti = np.clip(np.searchsorted(ref_theta, target_theta) - 1, 0, len(ref_theta) - 1)
            pi = np.clip(np.searchsorted(ref_phi,   target_phi)   - 1, 0, len(ref_phi)   - 1)
            return ref_mat[np.ix_(ti, pi)]

    # ------------------------------------------------------------------
    # Drawing helpers (2D heatmap)
    # ------------------------------------------------------------------

    def _draw_3panel_2d(self, data_a, name_a, data_b, name_b, theta, phi, freq_mhz):
        """Draw 3-panel 2D comparison: A | B | Diff(A−B) with RMSE/mean metrics."""
        diff = data_a - data_b
        mask = np.isfinite(data_a) & np.isfinite(data_b)
        rmse     = float(np.sqrt(np.mean(diff[mask] ** 2))) if mask.any() else float('nan')
        mean_err = float(np.mean(diff[mask]))               if mask.any() else float('nan')

        extent   = [phi.min(), phi.max(), theta.max(), theta.min()]
        vmin, vmax = self._cbar_range(
            min(np.nanmin(data_a), np.nanmin(data_b)),
            max(np.nanmax(data_a), np.nanmax(data_b)))
        diff_abs = np.nanmax(np.abs(diff[mask])) if mask.any() else 1.0
        if diff_abs < 1e-10:
            diff_abs = 1.0

        phi_range   = phi.max()   - phi.min()   if len(phi)   > 1 else 1.0
        theta_range = theta.max() - theta.min() if len(theta) > 1 else 1.0
        data_aspect = phi_range / theta_range if theta_range > 0 else 1.0

        gs   = self.w.comp_figure.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 1, 0.05], wspace=0.3)
        ax1  = self.w.comp_figure.add_subplot(gs[0])
        ax2  = self.w.comp_figure.add_subplot(gs[1])
        cax1 = self.w.comp_figure.add_subplot(gs[2])
        ax3  = self.w.comp_figure.add_subplot(gs[3])
        cax2 = self.w.comp_figure.add_subplot(gs[4])

        ax1.imshow(data_a, extent=extent, aspect=data_aspect, origin='upper',
                   cmap='jet', vmin=vmin, vmax=vmax)
        ax1.set_title(name_a, fontsize=10)
        ax1.set_xlabel('Phi (deg)')
        ax1.set_ylabel('Theta (deg)')

        im2 = ax2.imshow(data_b, extent=extent, aspect=data_aspect, origin='upper',
                         cmap='jet', vmin=vmin, vmax=vmax)
        ax2.set_title(name_b, fontsize=10)
        ax2.set_xlabel('Phi (deg)')
        self.w.comp_figure.colorbar(im2, cax=cax1, label=f'RCS ({self._unit_label()})')

        im3 = ax3.imshow(diff, extent=extent, aspect=data_aspect, origin='upper',
                         cmap='seismic', vmin=-diff_abs, vmax=diff_abs)
        ax3.set_title(f'Diff (A−B)\nRMSE={rmse:.2f} dB  Mean={mean_err:.2f} dB', fontsize=9)
        ax3.set_xlabel('Phi (deg)')
        self.w.comp_figure.colorbar(im3, cax=cax2, label='Error (dB)')

        title = f'2D RCS Comparison @ {freq_mhz:.1f} MHz — ' if freq_mhz else '2D RCS Comparison — '
        self.w.comp_figure.suptitle(f'{title}RMSE={rmse:.2f} dB  Mean={mean_err:.2f} dB', fontsize=11)

    def _draw_6panel_2d(self, rcs_a, name_a, rcs_b, name_b, rcs_ref, name_ref, theta, phi):
        """6-panel dual comparison: top row [A, B, Ref], bottom row [A−Ref, B−Ref, A−B]."""
        diff_a  = rcs_a - rcs_ref
        diff_b  = rcs_b - rcs_ref
        diff_ab = rcs_a - rcs_b

        def _stats(d):
            m = np.isfinite(d)
            if not m.any():
                return float('nan'), float('nan')
            return float(np.sqrt(np.mean(d[m] ** 2))), float(np.mean(d[m]))

        rmse_a,  mean_a  = _stats(diff_a)
        rmse_b,  mean_b  = _stats(diff_b)
        rmse_ab, mean_ab = _stats(diff_ab)

        extent   = [phi.min(), phi.max(), theta.max(), theta.min()]
        vmin, vmax = self._cbar_range(
            min(np.nanmin(rcs_a), np.nanmin(rcs_b), np.nanmin(rcs_ref)),
            max(np.nanmax(rcs_a), np.nanmax(rcs_b), np.nanmax(rcs_ref)))
        err_abs  = max(np.nanmax(np.abs(diff_a)), np.nanmax(np.abs(diff_b)), np.nanmax(np.abs(diff_ab)))
        if err_abs < 1e-10:
            err_abs = 1.0

        phi_range   = phi.max()   - phi.min()   if len(phi)   > 1 else 1.0
        theta_range = theta.max() - theta.min() if len(theta) > 1 else 1.0
        data_aspect = phi_range / theta_range if theta_range > 0 else 1.0

        gs = self.w.comp_figure.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

        row0_axes = []
        for col, (data, title) in enumerate([(rcs_a, name_a), (rcs_b, name_b), (rcs_ref, name_ref)]):
            ax = self.w.comp_figure.add_subplot(gs[0, col])
            im = ax.imshow(data, extent=extent, aspect=data_aspect, origin='upper',
                           cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('Phi (deg)')
            if col == 0:
                ax.set_ylabel('Theta (deg)')
            row0_axes.append(ax)
        cb0 = self.w.comp_figure.colorbar(im, ax=row0_axes, shrink=0.7, pad=0.02)
        cb0.set_label(f'RCS ({self._unit_label()})')

        row1_axes = []
        diff_items = [
            (diff_a,  f'A−Ref  RMSE={rmse_a:.2f} dB\nMean={mean_a:.2f} dB'),
            (diff_b,  f'B−Ref  RMSE={rmse_b:.2f} dB\nMean={mean_b:.2f} dB'),
            (diff_ab, f'A−B    RMSE={rmse_ab:.2f} dB\nMean={mean_ab:.2f} dB'),
        ]
        for col, (data, title) in enumerate(diff_items):
            ax = self.w.comp_figure.add_subplot(gs[1, col])
            im = ax.imshow(data, extent=extent, aspect=data_aspect, origin='upper',
                           cmap='seismic', vmin=-err_abs, vmax=err_abs)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('Phi (deg)')
            if col == 0:
                ax.set_ylabel('Theta (deg)')
            row1_axes.append(ax)
        cb1 = self.w.comp_figure.colorbar(im, ax=row1_axes, shrink=0.7, pad=0.02)
        cb1.set_label('Error (dB)')

        self.w.comp_figure.suptitle(f'Dual Model Comparison: {name_a} / {name_b} / {name_ref}', fontsize=11)

    def _draw_components_2d(self, result, theta, phi):
        """并排显示 Total / PO / PTD 三张 2D 热图，共用同一套 colorbar。

        根据 chk_show_total/po/ptd 勾选状态和数据可用性决定显示哪些子图。
        """
        show_total = self.w.chk_show_total.isChecked()
        show_po    = self.w.chk_show_po.isChecked()
        show_ptd   = self.w.chk_show_ptd.isChecked()

        panels = []
        if show_total:
            panels.append((self._conv(result['rcs_total']), 'Total'))
        if show_po and result.get('rcs_po') is not None:
            panels.append((self._conv(result['rcs_po']), 'PO'))
        if show_ptd and result.get('rcs_ptd') is not None:
            panels.append((self._conv(result['rcs_ptd']), 'PTD Fringe'))

        if not panels:
            self._show_comp_message("No components selected. Enable Total / PO / PTD above.")
            return

        extent = [phi.min(), phi.max(), theta.max(), theta.min()]
        phi_range   = phi.max()   - phi.min()   if len(phi)   > 1 else 1.0
        theta_range = theta.max() - theta.min() if len(theta) > 1 else 1.0
        data_aspect = phi_range / theta_range if theta_range > 0 else 1.0

        # 共用 colorbar 范围（取所有子图的 finite 值域）
        all_vals = np.concatenate([d[np.isfinite(d)].ravel() for d, _ in panels])
        vmin, vmax = self._cbar_range(float(np.nanmin(all_vals)), float(np.nanmax(all_vals)))

        n = len(panels)
        # 最后一列留给 colorbar
        width_ratios = [1] * n + [0.05]
        gs = self.w.comp_figure.add_gridspec(1, n + 1, width_ratios=width_ratios, wspace=0.25)

        axes = []
        im_last = None
        for col, (data, title) in enumerate(panels):
            ax = self.w.comp_figure.add_subplot(gs[0, col])
            im_last = ax.imshow(data, extent=extent, aspect=data_aspect,
                                origin='upper', cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Phi (deg)')
            if col == 0:
                ax.set_ylabel('Theta (deg)')
            axes.append(ax)

        cax = self.w.comp_figure.add_subplot(gs[0, n])
        self.w.comp_figure.colorbar(im_last, cax=cax, label=f'RCS ({self._unit_label()})')

        freq = result.get('freq', 0)
        freq_str = f'{freq / 1e6:.1f} MHz' if freq else ''
        self.w.comp_figure.suptitle(
            f'2D RCS Components  {freq_str}', fontsize=11)
