import os

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog


class ComparisonManager:
    """Delegate class that owns all comparison-panel logic.

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
    # Dataset combo helpers
    # ------------------------------------------------------------------

    def _refresh_dataset_combos(self):
        """Repopulate the three dataset combo boxes with current CSV names + sim option."""
        sim_label = "当前仿真结果"
        csv_names = [item['name'] for item in self.w.comparison_data]
        options = ["(未选择)", sim_label] + csv_names

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
        if not label or label == "(未选择)":
            return None
        if label == "当前仿真结果":
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

    # ------------------------------------------------------------------
    # Main plot entry point
    # ------------------------------------------------------------------

    def update_comparison_plot(self):
        self.w.comp_figure.clear()

        da = self._resolve_dataset(self.w.combo_ds_a.currentText())
        db = self._resolve_dataset(self.w.combo_ds_b.currentText())
        dr = self._resolve_dataset(self.w.combo_ds_ref.currentText())

        if da is None:
            self._show_comp_message("请在 A 中选择一个数据集。")
            self.w.comp_canvas.draw()
            return

        ta, pa, ma, na = da

        if db is None:
            # Show A alone
            ax = self.w.comp_figure.add_subplot(111)
            extent = [pa.min(), pa.max(), ta.max(), ta.min()]
            phi_range = pa.max() - pa.min() if len(pa) > 1 else 1.0
            theta_range = ta.max() - ta.min() if len(ta) > 1 else 1.0
            data_aspect = phi_range / theta_range if theta_range > 0 else 1.0
            im = ax.imshow(ma, extent=extent, aspect=data_aspect, origin='upper', cmap='jet')
            self.w.comp_figure.colorbar(im, ax=ax, label='RCS (dBsm)')
            ax.set_xlabel('Phi (deg)')
            ax.set_ylabel('Theta (deg)')
            ax.set_title(f'{na}\n(在 B 中选择第二个数据集以对比)')
            self.w.comp_canvas.draw()
            return

        tb, pb, mb, nb = db
        t_c, p_c, ma_c, mb_c = self._get_common_grid(ta, pa, ma, tb, pb, mb)
        if t_c is None:
            self._show_comp_message(
                f"A 和 B 角度范围无重叠。\n"
                f"A: θ[{ta.min():.1f},{ta.max():.1f}] φ[{pa.min():.1f},{pa.max():.1f}]\n"
                f"B: θ[{tb.min():.1f},{tb.max():.1f}] φ[{pb.min():.1f},{pb.max():.1f}]")
            self.w.comp_canvas.draw()
            return

        if dr is not None:
            tr, pr, mr, nr = dr
            mr_c = self._align_to_grid(mr, tr, pr, t_c, p_c)
            self._draw_6panel_2d(ma_c, na, mb_c, nb, mr_c, nr, t_c, p_c)
        else:
            self._draw_3panel_2d(ma_c, na, mb_c, nb, t_c, p_c, freq_mhz=None)

        self.w.comp_canvas.draw()

    def _show_comp_message(self, msg):
        """Display a centered message in the comparison canvas."""
        ax = self.w.comp_figure.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11,
                transform=ax.transAxes, color='gray',
                bbox=dict(facecolor='lightyellow', edgecolor='gray', pad=8))
        ax.axis('off')

    # ------------------------------------------------------------------
    # CSV parsing
    # ------------------------------------------------------------------

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
        """Find overlapping theta×phi region and interpolate both datasets onto it.

        Returns (t_common, p_common, m1_aligned, m2_aligned) or (None, None, None, None).
        """
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
            # Nearest-neighbour fallback when scipy is unavailable
            ti = np.clip(np.searchsorted(ref_theta, target_theta) - 1, 0, len(ref_theta) - 1)
            pi = np.clip(np.searchsorted(ref_phi,   target_phi)   - 1, 0, len(ref_phi)   - 1)
            return ref_mat[np.ix_(ti, pi)]

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_3panel_2d(self, data_a, name_a, data_b, name_b, theta, phi, freq_mhz):
        """Draw 3-panel 2D comparison: A | B | Diff(A−B) with RMSE/mean metrics."""
        diff = data_a - data_b
        mask = np.isfinite(data_a) & np.isfinite(data_b)
        rmse     = float(np.sqrt(np.mean(diff[mask] ** 2))) if mask.any() else float('nan')
        mean_err = float(np.mean(diff[mask]))               if mask.any() else float('nan')

        extent   = [phi.min(), phi.max(), theta.max(), theta.min()]
        vmin     = min(np.nanmin(data_a), np.nanmin(data_b))
        vmax     = max(np.nanmax(data_a), np.nanmax(data_b))
        diff_abs = np.nanmax(np.abs(diff[mask])) if mask.any() else 1.0
        if diff_abs < 1e-10:
            diff_abs = 1.0

        phi_range   = phi.max()   - phi.min()   if len(phi)   > 1 else 1.0
        theta_range = theta.max() - theta.min() if len(theta) > 1 else 1.0
        data_aspect = phi_range / theta_range if theta_range > 0 else 1.0

        # 5 columns: [img A] [img B] [cbar_data] [img Diff] [cbar_diff]
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
        self.w.comp_figure.colorbar(im2, cax=cax1, label='RCS (dBsm)')

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
        vmin     = min(np.nanmin(rcs_a), np.nanmin(rcs_b), np.nanmin(rcs_ref))
        vmax     = max(np.nanmax(rcs_a), np.nanmax(rcs_b), np.nanmax(rcs_ref))
        err_abs  = max(np.nanmax(np.abs(diff_a)), np.nanmax(np.abs(diff_b)), np.nanmax(np.abs(diff_ab)))
        if err_abs < 1e-10:
            err_abs = 1.0

        phi_range   = phi.max()   - phi.min()   if len(phi)   > 1 else 1.0
        theta_range = theta.max() - theta.min() if len(theta) > 1 else 1.0
        data_aspect = phi_range / theta_range if theta_range > 0 else 1.0

        gs = self.w.comp_figure.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

        # ---- Row 0: raw data ----
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
        cb0.set_label('RCS (dBsm)')

        # ---- Row 1: differences ----
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

    def _draw_comparison_1d(self):
        """1D comparison: overlay current sim result + loaded CSVs as line plots."""
        ax = self.w.comp_figure.add_subplot(111)

        if self.w.last_result:
            theta  = self.w.last_result['theta_deg']
            rcs_db = self.w.last_result['rcs_total']
            ax.plot(theta, rcs_db, label='Current Sim', linewidth=2.5, color='blue', zorder=10)

        for item in self.w.comparison_data:
            df        = item['data']
            cols      = df.columns
            theta_col = next((c for c in cols if 'theta' in c.lower()), None)
            rcs_col   = next((c for c in cols if 'dbsm'  in c.lower()), None)
            if rcs_col is None:
                rcs_col = next((c for c in cols if 'rcs' in c.lower()), None)
            if theta_col and rcs_col:
                try:
                    ax.plot(df[theta_col].values, df[rcs_col].values, '--',
                            label=item['name'], alpha=0.8)
                except Exception:
                    pass

        ax.set_xlabel('Theta (deg)')
        ax.set_ylabel('RCS (dBsm)')
        ax.grid(True, linestyle='--', alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
