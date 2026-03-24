"""Batch-mode result export: plot images (PNG) and CSV data files."""

import logging

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
    """Save results to CSV."""
    try:
        mode = result.get("mode")

        if mode == "freq_sweep":
            scan_mode = result.get("scan_mode")
            if scan_mode == "1d":
                df = pd.DataFrame({
                    "Frequency_MHz": result["frequencies"] / 1e6,
                    "RCS_Total_dBsm": result["rcs_matrix"],
                    "RCS_Total_m2": 10 ** (result["rcs_matrix"] / 10)
                })
                df.to_csv(output_path, index=False)
                
                # Also save range profile
                prof_path = output_path.replace(".csv", "_profile.csv")
                df_prof = pd.DataFrame({
                    "DownRange_m": result["range_axis"],
                    "Amplitude_dB": result["profile_matrix"]
                })
                df_prof.to_csv(prof_path, index=False)
            else:
                logger.warning("2D Frequency Sweep CSV export not implemented yet")
                return

        elif mode == "2d":
            theta = result["theta_deg"]
            phi   = result["phi_deg"]
            rcs   = result["rcs_total"]

            data = []
            for i, t in enumerate(theta):
                for j, p in enumerate(phi):
                    val_db = rcs[i, j]
                    data.append({
                        "Theta":    t,
                        "Phi":      p,
                        "RCS_dBsm": val_db,
                        "RCS_m2":   10 ** (val_db / 10)
                    })
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

        else:
            is_phi = (mode == "1d_phi")
            df = pd.DataFrame({
                "Theta": result.get("theta_deg", 0.0) if is_phi else result["theta_deg"],
                "Phi":   result["phi_deg"] if is_phi else result.get("phi_deg", 0.0),
                "RCS_Total_dBsm": result["rcs_total"],
                "RCS_Total_m2":   10 ** (result["rcs_total"] / 10)
            })
            if result.get("rcs_po") is not None:
                df["RCS_PO_dBsm"] = result["rcs_po"]
            if result.get("rcs_ptd") is not None:
                df["RCS_PTD_dBsm"] = result["rcs_ptd"]

            df.to_csv(output_path, index=False)
            
        logger.info(f"Saved data to: {output_path}")

    except Exception as e:
        logger.error(f"CSV export failed: {e}")

