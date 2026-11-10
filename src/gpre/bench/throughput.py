"""First throughput measurements (M2): GBM store and fused kernels, and the JAX and NumPy
generators, on one problem size. The full speedup table, ceilings and roofline are M6.

Each record follows the benchmark schema of SPEC §7b and carries the hardware stamp. Store
kernel bytes are the path writes only (n_paths x (n_steps + 1) x itemsize), which is the
analytic lower bound of DRAM traffic; ncu measurements replace it in M6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gpre.bench.timing import Timing, jax_sync, time_fn
from gpre.models.gbm import GBM, uniform_grid
from gpre.paths.base import PathSpec
from gpre.products.european import European
from gpre.rng.streams import DType, RngSpec

HARDWARE_KEYS = (
    "gpu_name",
    "compute_capability",
    "vram_gb",
    "driver_version",
    "cuda_version",
    "jax_version",
    "cupy_version",
    "cpu_model",
    "cpu_cores",
    "os",
    "platform",
)


@dataclass(frozen=True)
class Problem:
    model: GBM
    product: European
    spec: PathSpec
    reps: int
    warmup: int
    hardware: dict[str, Any]

    @property
    def store_bytes(self) -> float:
        itemsize = np.dtype(self.spec.dtype).itemsize
        return float(self.spec.n_paths * (self.spec.n_steps + 1) * itemsize)


def record(
    prob: Problem,
    backend: str,
    variant: str,
    timing: Timing,
    *,
    bytes_moved: float | None = None,
    price: float | None = None,
    std_err: float | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """One benchmark record (§7b)."""
    spec = prob.spec
    rec: dict[str, Any] = {
        "backend": backend,
        "kernel_variant": variant,
        "model": "gbm",
        "product": "european",
        "rng": spec.rng.kind,
        "n_paths": spec.n_paths,
        "n_steps": spec.n_steps,
        "dtype": spec.dtype,
        "threads": threads,
        **timing.as_dict(),
        "path_steps_per_s": spec.n_paths * spec.n_steps / timing.time_median_s,
        "bytes_moved": bytes_moved,
        "achieved_gbps": bytes_moved / timing.time_median_s / 1e9 if bytes_moved else None,
        "flops_est": None,
        "arithmetic_intensity": None,
        "price": price,
        "std_err": std_err,
    }
    rec.update({k: prob.hardware.get(k, "") for k in HARDWARE_KEYS})
    return rec


def bench_cuda(prob: Problem) -> list[dict[str, Any]]:  # pragma: no cover - needs a GPU
    import cupy as cp

    from gpre.cuda import launch

    def sync(_x: Any) -> None:
        cp.cuda.Device().synchronize()

    def store() -> Any:
        return launch.paths_store(prob.model, prob.spec)

    t_store = time_fn(store, reps=prob.reps, warmup=prob.warmup, sync=sync)
    last: dict[str, float] = {}

    def fused() -> None:
        r = launch.price_european_cuda(prob.product, prob.model, prob.spec)
        last["price"], last["se"] = r.price, r.std_err

    t_fused = time_fn(fused, reps=prob.reps, warmup=prob.warmup)
    return [
        record(prob, "cuda", "store", t_store, bytes_moved=prob.store_bytes),
        record(prob, "cuda", "fused", t_fused, price=last["price"], std_err=last["se"]),
    ]


def bench_jax(prob: Problem) -> list[dict[str, Any]]:
    from gpre.paths.jax_paths import JaxGenerator

    gen = JaxGenerator(chunk_paths=prob.spec.n_paths)

    def simulate() -> Any:
        return gen.simulate(prob.model, prob.spec)

    t = time_fn(simulate, reps=prob.reps, warmup=prob.warmup, sync=jax_sync)
    return [record(prob, "jax", "store", t, bytes_moved=prob.store_bytes)]


def bench_numpy(prob: Problem) -> list[dict[str, Any]]:
    from gpre.paths.numpy_paths import NumpyGenerator

    gen = NumpyGenerator()

    def simulate() -> Any:
        return gen.simulate(prob.model, prob.spec)

    t = time_fn(simulate, reps=prob.reps, warmup=min(prob.warmup, 1))
    return [record(prob, "numpy", "store", t, threads=1)]


BENCHES = {"cuda": bench_cuda, "jax": bench_jax, "numpy": bench_numpy}


def run(
    backends: list[str],
    *,
    n_paths: int,
    n_steps: int,
    seed: int,
    reps: int,
    warmup: int,
    hardware: dict[str, Any],
    dtype: DType = "float32",
) -> list[dict[str, Any]]:
    """Time path generation (and the fused European pricer on CUDA) for each backend."""
    prob = Problem(
        model=GBM(spot=100.0, rate=0.03, div_yield=0.01, vol=0.2),
        product=European("call", 100.0, 1.0),
        spec=PathSpec(n_paths, uniform_grid(1.0, n_steps), RngSpec("philox", seed), dtype),
        reps=reps,
        warmup=warmup,
        hardware=hardware,
    )
    out: list[dict[str, Any]] = []
    for backend in backends:
        if backend not in BENCHES:
            raise NotImplementedError(f"backend {backend!r}: the C++ baseline arrives in M6")
        out += BENCHES[backend](prob)
    return out
