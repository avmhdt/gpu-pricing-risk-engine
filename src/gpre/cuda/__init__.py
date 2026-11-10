"""CUDA kernels (CuPy RawKernel). Importing this package never imports CuPy."""

from pathlib import Path

KERNEL_DIR = Path(__file__).parent / "kernels"
