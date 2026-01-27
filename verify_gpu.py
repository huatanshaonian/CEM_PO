import time
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.insert(0, '.')

try:
    import cupy as cp
    HAS_CUPY = True
    print(f"Cupy detected: {cp.__version__}")
    # Print GPU device info
    try:
        dev = cp.cuda.Device(0)
        print(f"GPU Device: {dev.compute_capability}, Mem: {dev.mem_info[1] / 1024**3:.2f} GB")
    except:
        print("Could not get GPU info")
except ImportError:
    HAS_CUPY = False
    print("Cupy NOT detected. GPU tests will be skipped or simulated.")

from geometry.plate import AnalyticPlate
from solver.ribbon_solver import get_integrator, RCSAnalyzer
from physics.constants import C0

def verify_gpu_implementation():
    print("\n" + "="*60)
    print("GPU Acceleration Verification & Benchmark")
    print("="*60)

    # 1. Setup
    frequency = 3.0e9 # 3 GHz
    wavelength = C0 / frequency
    width = 1.0
    height = 1.0
    
    plate = AnalyticPlate(width, height)
    
    # Angles: -30 to 30 degrees, 101 points
    angles_deg = np.linspace(-30, 30, 101)
    angles_rad = np.deg2rad(angles_deg)
    wave_params = {'frequency': frequency, 'phi': 0.0}

    solver = get_integrator('discrete_po_sinc_dual')
    analyzer = RCSAnalyzer(solver)

    print(f"\n[Configuration]")
    print(f"  Geometry: Plate {width}x{height}m")
    print(f"  Frequency: {frequency/1e9:.1f} GHz")
    print(f"  Angles: {len(angles_deg)} steps")
    
    # ---------------------------------------------------------
    # 2. Correctness Test (Small Mesh)
    # ---------------------------------------------------------
    print("\n[Phase 1: Correctness Verification]")
    
    # Use modest sampling for correctness check
    samples_correctness = 10 
    
    # A. Run on CPU
    print("  Running CPU calculation...", end="", flush=True)
    t0 = time.time()
    res_cpu = analyzer.compute_monostatic_rcs(
        plate, wave_params, angles_rad,
        samples_per_lambda=samples_correctness,
        parallel=False, gpu=False, show_progress=False
    )
    t_cpu_corr = time.time() - t0
    print(f" Done ({t_cpu_corr:.3f}s)")

    if not HAS_CUPY:
        print("  [SKIP] GPU not available. Cannot verify correctness.")
        return

    # B. Run on GPU
    print("  Running GPU calculation...", end="", flush=True)
    t0 = time.time()
    res_gpu = analyzer.compute_monostatic_rcs(
        plate, wave_params, angles_rad,
        samples_per_lambda=samples_correctness,
        parallel=False, gpu=True, show_progress=False
    )
    t_gpu_corr = time.time() - t0
    print(f" Done ({t_gpu_corr:.3f}s)")

    # C. Compare
    diff = np.abs(res_cpu['total'] - res_gpu['total'])
    max_diff = np.max(diff)
    print(f"  Max RCS Difference (dB): {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print("  >>> CORRECTNESS CHECK: PASSED")
    else:
        print("  >>> CORRECTNESS CHECK: FAILED (Differences too high)")

    # ---------------------------------------------------------
    # 3. Performance Benchmark (Large Mesh)
    # ---------------------------------------------------------
    print("\n[Phase 2: Performance Benchmark]")
    
    # Increase load significantly
    # Plate is 1x1m, lambda is 0.1m. 
    # With samples=30, mesh is approx 300x300 = 90k points
    # With samples=50, mesh is approx 500x500 = 250k points
    samples_perf = 60
    
    print(f"  Mesh Density: {samples_perf} samples/lambda")
    # Pre-calculate approximate points to display
    nu, nv = solver._estimate_mesh_density(plate, wavelength, samples_perf)
    print(f"  Approximate Grid: {nu} x {nv} = {nu*nv} points per angle")
    print(f"  Total Integrations: {len(angles_deg)} angles")
    
    # A. Run on CPU (Parallel)
    # Usually we compare GPU vs Parallel CPU as that's the fair fight
    print("  Running CPU (Parallel, 4 workers)...", end="", flush=True)
    t0 = time.time()
    res_cpu_perf = analyzer.compute_monostatic_rcs(
        plate, wave_params, angles_rad,
        samples_per_lambda=samples_perf,
        parallel=True, n_workers=4, gpu=False, show_progress=False
    )
    t_cpu_perf = time.time() - t0
    print(f" Done ({t_cpu_perf:.3f}s)")

    # B. Run on GPU
    # Warmup first (optional, but good practice)
    # (We already ran a small case, so that acts as warmup)
    
    print("  Running GPU...", end="", flush=True)
    t0 = time.time()
    res_gpu_perf = analyzer.compute_monostatic_rcs(
        plate, wave_params, angles_rad,
        samples_per_lambda=samples_perf,
        parallel=False, gpu=True, show_progress=False
    )
    t_gpu_perf = time.time() - t0
    print(f" Done ({t_gpu_perf:.3f}s)")

    # Statistics
    speedup = t_cpu_perf / t_gpu_perf
    print(f"\n[Results]")
    print(f"  CPU Time: {t_cpu_perf:.3f} s")
    print(f"  GPU Time: {t_gpu_perf:.3f} s")
    print(f"  Speedup:  {speedup:.2f}x")

if __name__ == "__main__":
    verify_gpu_implementation()