"""CuPy RawModule compile, launch and timing for the kernels in ``kernels/paths.cu``.

CuPy is imported lazily, so the package imports without it. Compiled modules are cached by
source hash and compute capability. This module needs an NVIDIA GPU and is covered by
``make test-gpu`` on the cloud instance, not by CI.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from gpre.cuda import KERNEL_DIR
from gpre.models.gbm import GBM
from gpre.paths.base import PathSpec
from gpre.products.european import European

TILE = 64
KERNELS = (
    "philox_uniforms",
    "gbm_paths_store",
    "gbm_paths_store_f64",
    "gbm_european_fused",
    "gbm_european_fused_f64",
    "gbm_asian_fused",
)
_MODULES: dict[tuple[str, str], Any] = {}


def _cupy() -> Any:
    try:
        import cupy
    except ImportError as exc:  # pragma: no cover - exercised only without the gpu group
        raise RuntimeError("CuPy is not installed (make setup-gpu)") from exc
    return cupy


def source_hash() -> str:
    h = hashlib.sha256()
    for name in ("philox4x32.cuh", "paths.cu"):
        h.update((KERNEL_DIR / name).read_bytes())
    return h.hexdigest()[:16]


def compute_capability() -> str:
    cp = _cupy()
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    return f"{props['major']}{props['minor']}"


@dataclass
class Compiled:
    module: Any
    compile_s: float


def compile_module() -> Compiled:
    """Compile (or fetch from cache) the kernel module for the current device."""
    cp = _cupy()
    key = (source_hash(), compute_capability())
    if key in _MODULES:
        return Compiled(_MODULES[key], 0.0)
    t0 = time.perf_counter()
    module = cp.RawModule(
        code=(KERNEL_DIR / "paths.cu").read_text(encoding="utf-8"),
        # CuPy picks the architecture itself and emits a cubin (no PTX JIT by the driver).
        options=("-std=c++17", f"-I{KERNEL_DIR}"),
        name_expressions=KERNELS,
    )
    for name in KERNELS:
        module.get_function(name)  # forces NVRTC compilation now, outside timed regions
    _MODULES[key] = module
    return Compiled(module, time.perf_counter() - t0)


def default_grid(n_items: int, block: int, max_blocks: int = 65535) -> int:
    return max(1, min(max_blocks, math.ceil(n_items / block)))


def philox_words_cuda(
    seed: int,
    stream: int,
    replication: int,
    n_paths: int,
    n_blocks: int,
    block: int = 256,
    grid: int | None = None,
) -> npt.NDArray[np.uint32]:
    """Raw Philox words ``(n_paths, n_blocks, 4)`` from the device."""
    cp = _cupy()
    fn = compile_module().module.get_function("philox_uniforms")
    out = cp.empty(n_paths * n_blocks * 4, dtype=cp.uint32)
    g = grid or default_grid(n_paths * n_blocks, block)
    fn(
        (g,),
        (block,),
        (
            np.uint64(seed),
            np.uint32(stream),
            np.uint32(replication),
            np.int32(n_paths),
            np.int32(n_blocks),
            out,
        ),
    )
    cp.cuda.Device().synchronize()
    words: npt.NDArray[np.uint32] = cp.asnumpy(out).reshape(n_paths, n_blocks, 4)
    return words


def _coefficients(model: GBM, spec: PathSpec) -> tuple[Any, Any, Any]:
    cp = _cupy()
    dtype = np.dtype(spec.dtype)
    drift_dt, vol_sqrt_dt = model.step_coefficients(np.asarray(spec.times, dtype=np.float64))
    return cp.asarray(drift_dt.astype(dtype)), cp.asarray(vol_sqrt_dt.astype(dtype)), dtype


def paths_store(model: GBM, spec: PathSpec, block: int = 256, grid: int | None = None) -> Any:
    """Device paths ``(n_steps + 1, n_paths)`` from ``gbm_paths_store`` (Philox only)."""
    if spec.rng.kind != "philox":
        raise ValueError("the CUDA kernels draw Philox numbers only")
    cp = _cupy()
    drift, vol, dtype = _coefficients(model, spec)
    name = "gbm_paths_store" if dtype == np.float32 else "gbm_paths_store_f64"
    fn = compile_module().module.get_function(name)
    out = cp.empty((spec.n_steps + 1) * spec.n_paths, dtype=dtype)
    r = spec.rng
    g = grid or default_grid(spec.n_paths, block)
    fn(
        (g,),
        (block,),
        (
            np.uint64(r.seed),
            np.uint32(r.stream),
            np.uint32(r.replication),
            np.int32(spec.n_paths),
            np.int32(spec.n_steps),
            drift,
            vol,
            dtype.type(model.spot),
            np.int32(r.antithetic),
            out,
        ),
    )
    return out.reshape(spec.n_steps + 1, spec.n_paths)


def european_tiles(
    product: European, model: GBM, spec: PathSpec, block: int = 256, grid: int | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-64-path tile sums of the discounted payoff and its square, from the fused kernel."""
    if spec.rng.kind != "philox":
        raise ValueError("the CUDA kernels draw Philox numbers only")
    if block % TILE:
        raise ValueError(f"block size must be a multiple of {TILE}")
    cp = _cupy()
    drift, vol, dtype = _coefficients(model, spec)
    name = "gbm_european_fused" if dtype == np.float32 else "gbm_european_fused_f64"
    fn = compile_module().module.get_function(name)
    n_tiles = -(-spec.n_paths // TILE)
    tsum = cp.zeros(n_tiles, dtype=cp.float64)
    tsq = cp.zeros(n_tiles, dtype=cp.float64)
    r = spec.rng
    g = grid or default_grid(n_tiles, block // TILE)
    fn(
        (g,),
        (block,),
        (
            np.uint64(r.seed),
            np.uint32(r.stream),
            np.uint32(r.replication),
            np.int32(spec.n_paths),
            np.int32(spec.n_steps),
            drift,
            vol,
            dtype.type(model.spot),
            dtype.type(product.strike),
            np.int32(product.is_call),
            np.float64(model.discount(product.maturity)),
            np.int32(r.antithetic),
            tsum,
            tsq,
        ),
        shared_mem=2 * block * 8,
    )
    cp.cuda.Device().synchronize()
    return cp.asnumpy(tsum), cp.asnumpy(tsq)


def merge_tiles(
    tile_sum: npt.NDArray[np.float64], tile_sumsq: npt.NDArray[np.float64], n_paths: int
) -> tuple[float, float]:
    """Mean and sample variance from per-tile sums, merged in index order.

    Each tile's M2 = sumsq - sum^2 / n_t is formed within its 64 paths, then the tiles are
    combined with the exact parallel formula M2 = sum_t [M2_t + n_t (mean_t - mean)^2], which
    avoids the cancellation of a global sum-of-squares formula (NFR-03).
    """
    counts = np.full(tile_sum.size, float(TILE))
    counts[-1] = n_paths - TILE * (tile_sum.size - 1)
    mean = float(np.sum(tile_sum)) / n_paths
    tile_mean = tile_sum / counts
    m2 = np.maximum(tile_sumsq - tile_sum * tile_mean, 0.0)
    total_m2 = float(np.sum(m2 + counts * (tile_mean - mean) ** 2))
    return mean, total_m2 / (n_paths - 1)


def price_european_cuda(
    product: European, model: GBM, spec: PathSpec, block: int = 256, grid: int | None = None
) -> Any:
    """Fused-kernel price and SE. Antithetic SE uses the pair-average variance, which the tile
    sums cannot give, so antithetic runs fall back to the store kernel for the estimator."""
    from gpre.pricing.mc import PriceResult, RunningStats

    compiled = compile_module()
    t0 = time.perf_counter()
    if spec.rng.antithetic:
        cp = _cupy()
        paths = paths_store(model, spec, block, grid)
        disc = model.discount(product.maturity)
        pay = disc * product.payoff_terminal(cp.asnumpy(paths[-1]))
        stats = RunningStats()
        stats.add(0.5 * (pay[0::2] + pay[1::2]))
        price, se, variant = stats.mean, stats.std_err, "store"
    else:
        tsum, tsq = european_tiles(product, model, spec, block, grid)
        price, var = merge_tiles(tsum, tsq, spec.n_paths)
        se, variant = math.sqrt(var / spec.n_paths), "fused"
    elapsed = time.perf_counter() - t0
    return PriceResult(
        price,
        se,
        spec.n_paths,
        elapsed,
        None,
        {"kernel_variant": variant, "compile_s": compiled.compile_s, "block": block},
    )


class CudaGenerator:
    """CUDA backend path generator (store kernel)."""

    name = "cuda"

    def iter_chunks(self, model: GBM, spec: PathSpec) -> Iterator[Any]:
        yield paths_store(model, spec)

    def simulate(self, model: GBM, spec: PathSpec) -> Any:
        return paths_store(model, spec)


def time_kernel(fn: Any, reps: int = 10, warmup: int = 3) -> list[float]:
    """Wall times (s) of ``fn()`` measured with CUDA events after warm-up runs."""
    cp = _cupy()
    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(reps):
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop) / 1000.0)
    return times
