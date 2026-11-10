"""GPU Pricing and Risk Engine (gpre).

Monte Carlo pricing, Greeks and portfolio risk with a JAX path, a hand-written CUDA kernel
family and CPU baselines. GPU backends are optional and detected at runtime, so the package
imports and its CPU tests pass on a machine without a GPU or CuPy.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
