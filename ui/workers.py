import time
import traceback

from PySide6.QtCore import QThread, Signal, QObject


class CalculationWorker(QThread):
    progress_signal = Signal(float, str)
    result_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, bridge, geo, params):
        super().__init__()
        self.bridge = bridge
        self.geo = geo
        self.params = params

    def run(self):
        def callback(current, total, msg=""):
            p = (current / total * 100) if total > 0 else 0
            self.progress_signal.emit(p, msg)

        try:
            result = self.bridge.run_simulation(self.geo, self.params, progress_callback=callback)
            self.result_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))


class LogBridge(QObject):
    new_log = Signal(str)

    def write(self, text):
        if text.strip(): self.new_log.emit(str(text))

    def flush(self): pass


class MeshStatsWorker(QThread):
    """Worker thread for generating mesh statistics"""
    progress_signal = Signal(float, str)
    result_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, bridge, geo, params):
        super().__init__()
        self.bridge = bridge
        self.geo = geo
        self.params = params

    def run(self):
        try:
            t_start = time.time()
            self.progress_signal.emit(0, "Generating mesh...")

            meshes = self.bridge.generate_mesh(self.geo, self.params)

            if meshes is None:
                self.error_signal.emit("Algorithm does not support mesh preview")
                return

            total_cells = 0
            face_stats = []

            for i, m in enumerate(meshes):
                pts = m.points
                if pts.ndim == 3:
                    nu, nv = pts.shape[1], pts.shape[0]
                    n_cells = nu * nv
                else:
                    n_cells = len(pts)
                    nu, nv = n_cells, 1
                total_cells += n_cells
                face_stats.append({'index': i, 'nu': nu, 'nv': nv, 'cells': n_cells})
                self.progress_signal.emit((i + 1) / len(meshes) * 100, f"Surface {i+1}/{len(meshes)}")

            elapsed = time.time() - t_start
            result = {
                'meshes': meshes,
                'total_cells': total_cells,
                'n_surfaces': len(meshes),
                'face_stats': face_stats,
                'elapsed': elapsed
            }
            self.result_signal.emit(result)

        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))
