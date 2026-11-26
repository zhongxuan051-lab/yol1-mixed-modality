"""Build the supplied DCNv4 extension in the active server Python environment."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    root = Path(__file__).resolve().parents[1]
    nvcc = shutil.which("nvcc")
    if not nvcc and CUDA_HOME:
        candidate = Path(CUDA_HOME) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        if candidate.is_file():
            nvcc = str(candidate)
    compiler = shutil.which("cl" if os.name == "nt" else "c++")
    print(f"Python: {sys.executable}\nPyTorch: {torch.__version__}\nCUDA_HOME: {CUDA_HOME}\nnvcc: {nvcc}\nC++: {compiler}", flush=True)
    if not torch.cuda.is_available() or not CUDA_HOME or not nvcc or not compiler:
        raise SystemExit("Build requires a visible CUDA GPU, CUDA toolkit (nvcc), and C++ compiler. Activate the server training environment first.")
    subprocess.run([nvcc, "--version"], check=True)
    env = dict(os.environ)
    env.setdefault("MAX_JOBS", "4")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps",
                    str(root / "third_party/DCNv4_op")], env=env, check=True)
    subprocess.run([sys.executable, "-c", "from DCNv4 import ext; print('DCNv4 extension import passed')"], check=True)
    print("Build passed. Next: python tests/check_dcnv4.py --smoke-train", flush=True)


if __name__ == "__main__":
    main()
