"""Batch-mode result export: plot images (PNG) and CSV data files."""

import logging
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("CEM-PO.Export")


def save_plot(result, output_path, title_suffix=""):
    """Generate and save a plot for the result using matplotlib (Agg backend)."""
    try:
        mode = result.get("mode")
        freq_mhz = result.get("freq", 0) / 1e6 if "freq" in result else 0

        if mode == "freq_sweep":
            scan_mode = result.get("scan_mode")
            if scan_mode == "1d":
                fig = plt.figure(figsize=(12, 5))
                ax1 = fig.add_subplot(121)
                freqs = result["frequencies"] / 1e6
                ax1.plot(freqs, result["rcs_matrix"], "b-", linewidth=2)
                ax1.set_xlabel("Frequency (MHz)")
                ax1.set_ylabel("RCS (dBsm)")
                ax1.set_title(f"Frequency Response - {title_suffix}")
                ax1.grid(True, linestyle="--", alpha=0.6)

                ax2 = fig.add_subplot(122)
                r_ax = result["range_axis"]
                ax2.plot(r_ax, result["profile_matrix"], "r-", linewidth=2)
                ax2.set_xlabel("Down-range (m)")
                ax2.set_ylabel("Amplitude (dB)")
                ax2.set_title(f"1D Range Profile - {title_suffix}")
                ax2.grid(True, linestyle="--", alpha=0.6)
            else:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "2D Frequency Sweep Plotting not implemented yet", 
                        horizontalalignment="center", verticalalignment="center")
        elif mode == "2d":
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            rcs   = result["rcs_total"]
            theta = result["theta_deg"]
            phi   = result["phi_deg"]

            X, Y = np.meshgrid(phi, theta)
            Z = np.nan_to_num(rcs, nan=-100.0)

            c = ax.pcolormesh(X, Y, Z, cmap="jet", shading="auto")
            fig.colorbar(c, ax=ax, label="RCS (dBsm)")
            ax.set_xlabel("Phi (deg)")
            ax.set_ylabel("Theta (deg)")
            ax.set_title(f"2D RCS Pattern - {title_suffix} @ {freq_mhz:.1f} MHz")
            ax.invert_yaxis()
            ax.set_aspect("equal")

        else:  # 1d or 1d_phi
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            is_phi = (mode == "1d_phi")
            angles = result["phi_deg"] if is_phi else result["theta_deg"]
            x_label = "Phi (deg)" if is_phi else "Theta (deg)"
            
            ax.plot(angles, result["rcs_total"], "b-", linewidth=2, label="Total RCS")

            if result.get("rcs_po") is not None:
                ax.plot(angles, result["rcs_po"], "g--", linewidth=1.5, alpha=0.7, label="PO Component")
            if result.get("rcs_ptd") is not None:
                if np.max(result["rcs_ptd"]) > -150:
                    ax.plot(angles, result["rcs_ptd"], "r:", linewidth=1.5, alpha=0.7, label="PTD Component")

            ax.set_xlabel(x_label)
            ax.set_ylabel("RCS (dBsm)")
            ax.set_title(f"1D RCS Cut - {title_suffix} @ {freq_mhz:.1f} MHz")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved plot to: {output_path}")

    except Exception as e:
        logger.error(f"Plotting failed: {e}")


def save_csv(result, output_path):
    """Save results to CSV with metadata headers, matching GUI export format."""
    try:
        mode = result.get("mode")

        if mode == "freq_sweep":
            _save_freq_sweep_csv(result, output_path)
        else:
            _save_standard_csv(result, output_path)
            
        logger.info(f"Saved data to: {output_path}")

    except Exception as e:
        logger.error(f"CSV export failed: {e}")


def _save_standard_csv(res, path):
    """Save single-frequency (1D/2D) results."""
    mode = res.get('mode', '1d')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write Header Info
        p      = res.get('params', {})
        ang_p  = p.get('angles', {})
        ptd_p  = p.get('ptd', {})
        cmp_p  = p.get('compute', {})
        writer.writerow(["# CEM PO Solver Results"])
        writer.writerow(["# Mode",              mode])
        writer.writerow(["# Frequency (Hz)",    res.get('freq', '')])
        writer.writerow(["# Frequency (MHz)",   (res.get('freq') or 0) / 1e6])
        writer.writerow(["# Algorithm",         p.get('algorithm', '')])
        writer.writerow(["# Theta Start (deg)", ang_p.get('theta_start', '')])
        writer.writerow(["# Theta End (deg)",   ang_p.get('theta_end', '')])
        writer.writerow(["# N Theta",           ang_p.get('n_theta', '')])
        writer.writerow(["# Phi Start (deg)",   ang_p.get('phi_start', '')])
        writer.writerow(["# Phi End (deg)",     ang_p.get('phi_end', '')])
        writer.writerow(["# N Phi",             ang_p.get('n_phi', '')])
        writer.writerow(["# PTD Enabled",       ptd_p.get('enabled', False)])
        writer.writerow(["# PTD Edges",         ptd_p.get('edges', '')])
        writer.writerow(["# Polarization",      ptd_p.get('polarization', '')])
        writer.writerow(["# GPU",               cmp_p.get('gpu', False)])
        writer.writerow(["# Elapsed Time (s)",  f"{res.get('elapsed_time', 0):.3f}"])
        writer.writerow([])

        def _cplx_cols(key_c, idx=None):
            v = res.get(key_c)
            if v is None: return []
            c = complex(v[idx] if idx is not None else v)
            return [c.real, c.imag]

        has_po  = res.get('rcs_po')  is not None
        has_ptd = res.get('rcs_ptd') is not None
        has_I   = res.get('I_total') is not None

        if mode == '2d':
            header = ["Theta (deg)", "Phi (deg)", "RCS Total (dBsm)", "RCS Total (m^2)"]
            if has_po:  header.append("RCS PO (dBsm)")
            if has_ptd: header.append("RCS PTD (dBsm)")
            if has_I:
                header += ["I Total (Re)", "I Total (Im)", "I PO (Re)", "I PO (Im)", "I PTD (Re)", "I PTD (Im)"]
            writer.writerow(header)

            theta = res['theta_deg']
            phi   = res['phi_deg']
            for i, t in enumerate(theta):
                for j, p in enumerate(phi):
                    val_db = res['rcs_total'][i, j]
                    row = [t, p, val_db, 10 ** (val_db / 10)]
                    if has_po:  row.append(res['rcs_po'][i, j])
                    if has_ptd: row.append(res['rcs_ptd'][i, j])
                    if has_I:
                        row += _cplx_cols('I_total', (i, j))
                        row += _cplx_cols('I_po',    (i, j))
                        row += _cplx_cols('I_ptd',   (i, j))
                    writer.writerow(row)
        else:
            is_phi = (mode == '1d_phi')
            header = [("Phi (deg)" if is_phi else "Theta (deg)"), "RCS Total (dBsm)", "RCS Total (m^2)"]
            if has_po:  header.append("RCS PO (dBsm)")
            if has_ptd: header.append("RCS PTD (dBsm)")
            if has_I:
                header += ["I Total (Re)", "I Total (Im)", "I PO (Re)", "I PO (Im)", "I PTD (Re)", "I PTD (Im)"]
            writer.writerow(header)

            angles = res['phi_deg'] if is_phi else res['theta_deg']
            for i, ang in enumerate(angles):
                val_db = res['rcs_total'][i]
                row = [ang, val_db, 10 ** (val_db / 10)]
                if has_po:  row.append(res['rcs_po'][i])
                if has_ptd: row.append(res['rcs_ptd'][i])
                if has_I:
                    row += _cplx_cols('I_total', i)
                    row += _cplx_cols('I_po',    i)
                    row += _cplx_cols('I_ptd',   i)
                writer.writerow(row)


def _save_freq_sweep_csv(res, path):
    """Save frequency sweep results (RCS matrix and Range Profile)."""
    freqs     = res['frequencies']
    theta_deg = res['theta_deg']
    phi_deg   = res['phi_deg']
    scan_mode = res.get('scan_mode', '1d')
    fsp       = res.get('freq_sweep_params') or {}
    params    = res.get('params') or {}
    ptd_p     = params.get('ptd', {})
    ang_p     = params.get('angles', {})

    rcs_mat   = np.atleast_2d(res['rcs_matrix'])
    I_total   = np.atleast_2d(res['I_total_matrix'])
    I_po_raw  = res.get('I_po_matrix')
    I_ptd_raw = res.get('I_ptd_matrix')
    has_po    = I_po_raw is not None
    has_ptd   = I_ptd_raw is not None

    rcs_po_mat = rcs_ptd_mat = None
    I_po_mat = I_ptd_mat = None
    k_arr = 2.0 * np.pi * freqs / 299792458.0
    if has_po:
        I_po_mat = np.atleast_2d(I_po_raw)
        sigma_po = (k_arr[np.newaxis, :] ** 2 / np.pi) * np.abs(I_po_mat) ** 2
        rcs_po_mat = 10.0 * np.log10(np.maximum(sigma_po, 1e-30))
    if has_ptd:
        I_ptd_mat = np.atleast_2d(I_ptd_raw)
        sigma_ptd = (k_arr[np.newaxis, :] ** 2 / np.pi) * np.abs(I_ptd_mat) ** 2
        rcs_ptd_mat = 10.0 * np.log10(np.maximum(sigma_ptd, 1e-30))

    angle_list = [(th, ph) for th in theta_deg for ph in phi_deg]

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["# CEM PO Solver – Frequency Sweep RCS Results"])
        writer.writerow(["# Algorithm",          params.get('algorithm', '')])
        writer.writerow(["# Polarization",       fsp.get('polarization', '')])
        writer.writerow(["# PTD Enabled",        ptd_p.get('enabled', False)])
        writer.writerow(["# PTD Edges",          ptd_p.get('edges', '')])
        writer.writerow(["# Freq Start (MHz)",   fsp.get('f_start', '')])
        writer.writerow(["# Freq End (MHz)",     fsp.get('f_end', '')])
        writer.writerow(["# Freq Step (MHz)",    fsp.get('f_step', '')])
        writer.writerow(["# Theta Start (deg)",  ang_p.get('theta_start', '')])
        writer.writerow(["# Theta End (deg)",    ang_p.get('theta_end', '')])
        writer.writerow(["# N Theta",            ang_p.get('n_theta', '')])
        writer.writerow(["# Phi Start (deg)",    ang_p.get('phi_start', '')])
        writer.writerow(["# Phi End (deg)",      ang_p.get('phi_end', '')])
        writer.writerow(["# N Phi",              ang_p.get('n_phi', '')])
        writer.writerow(["# Scan Mode",          scan_mode])
        writer.writerow(["# Elapsed Time (s)",   f"{res.get('elapsed_time', 0):.3f}"])
        writer.writerow([])

        header = ["Theta (deg)", "Phi (deg)", "RCS Total (dBsm)"]
        if has_po:  header.append("RCS PO (dBsm)")
        if has_ptd: header.append("RCS PTD (dBsm)")
        header += ["I Total (Re)", "I Total (Im)"]
        if has_po:  header += ["I PO (Re)", "I PO (Im)"]
        if has_ptd: header += ["I PTD (Re)", "I PTD (Im)"]
        header.append("Frequency (MHz)")
        writer.writerow(header)

        for i, (th, ph) in enumerate(angle_list):
            for j, freq_hz in enumerate(freqs):
                row = [th, ph, rcs_mat[i, j]]
                if has_po:  row.append(rcs_po_mat[i, j])
                if has_ptd: row.append(rcs_ptd_mat[i, j])
                row += [I_total[i, j].real, I_total[i, j].imag]
                if has_po:  row += [I_po_mat[i, j].real,  I_po_mat[i, j].imag]
                if has_ptd: row += [I_ptd_mat[i, j].real, I_ptd_mat[i, j].imag]
                row.append(freq_hz / 1e6)
                writer.writerow(row)

    # Range Profile
    if 'range_axis' in res:
        prof_path = path.replace(".csv", "_profile.csv")
        range_axis  = res['range_axis']
        profile_mat = np.atleast_2d(res['profile_matrix'])
        stats       = res.get('stats') or {}
        with open(prof_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["# CEM PO Solver – Range Profile Results"])
            writer.writerow(["# Window",               fsp.get('window', '')])
            writer.writerow(["# Zero Pad",             fsp.get('zero_pad', '')])
            writer.writerow(["# Range Resolution (m)", stats.get('range_resolution_m', '')])
            writer.writerow(["# Max Range (m)",        stats.get('max_range_m', '')])
            writer.writerow(["# Bandwidth (MHz)",      stats.get('bandwidth_mhz', '')])
            writer.writerow([])
            
            # For simplicity, if multi-angle, we just export all profiles side by side or as long format
            # GUI seems to export only the current slice, but for batch, we'll do long format: Theta, Phi, Range, Amp
            writer.writerow(["Theta (deg)", "Phi (deg)", "DownRange (m)", "Amplitude (dB)"])
            for i, (th, ph) in enumerate(angle_list):
                for j, dist in enumerate(range_axis):
                    writer.writerow([th, ph, dist, profile_mat[i, j]])


