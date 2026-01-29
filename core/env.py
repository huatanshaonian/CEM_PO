import os
import sys

def configure_gpu_env():
    """
    配置 CUDA 环境 (针对 Windows)
    """
    if sys.platform == 'win32' and 'CUDA_PATH' not in os.environ:
        # 尝试自动探测 CUDA 安装路径
        base_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.exists(base_path):
            versions = [d for d in os.listdir(base_path) if d.startswith('v')]
            if versions:
                # 取版本号最高的
                latest_v = sorted(versions, key=lambda x: [int(c) for c in x[1:].split('.')])[-1]
                cuda_root = os.path.join(base_path, latest_v)
                os.environ['CUDA_PATH'] = cuda_root
                
                # 将 bin 和 bin\x64 加入 PATH 以确保 DLL 能被找到
                bin_path = os.path.join(cuda_root, 'bin')
                bin_x64_path = os.path.join(cuda_root, 'bin', 'x64')
                
                if bin_path not in os.environ['PATH']:
                    os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
                if os.path.exists(bin_x64_path) and bin_x64_path not in os.environ['PATH']:
                    os.environ['PATH'] = bin_x64_path + os.pathsep + os.environ['PATH']

try:
    configure_gpu_env()
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False
