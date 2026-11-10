"""CUDA bit-exactness and launch invariance (AC-W-06) and kernel agreement (FR-05).

Runs only on an NVIDIA GPU with CuPy (``make test-gpu`` on the cloud instance).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover
        pytest.skip("no CUDA device", allow_module_level=True)
except Exception:  # pragma: no cover
    pytest.skip("no CUDA device", allow_module_level=True)

from gpre.cuda import launch  # noqa: E402
from gpre.models.gbm import GBM, uniform_grid  # noqa: E402
from gpre.paths.base import PathSpec  # noqa: E402
from gpre.paths.numpy_paths import NumpyGenerator  # noqa: E402
from gpre.pricing.mc import price_mc  # noqa: E402
from gpre.products.european import European  # noqa: E402
from gpre.rng.streams import RngSpec, philox_words  # noqa: E402

pytestmark = pytest.mark.gpu
SEED = 20261030
MODEL = GBM(spot=100.0, rate=0.03, div_yield=0.01, vol=0.2)


def test_cuda_philox_words_bit_identical_to_numpy() -> None:
    """2**20 paths x 256 draws: 64 blocks of 4 words per path."""
    n_paths, n_blocks = 1 << 20, 64
    gpu = launch.philox_words_cuda(SEED, 5, 2, n_paths, n_blocks)
    for lo in range(0, n_paths, 1 << 17):
        p = np.arange(lo, lo + (1 << 17), dtype=np.uint64)
        ref = philox_words(SEED, 5, 2, p[:, None], np.arange(n_blocks, dtype=np.uint64)[None, :])
        np.testing.assert_array_equal(gpu[lo : lo + (1 << 17)], ref)


@pytest.mark.parametrize("block", [64, 128, 256, 512])
def test_philox_kernel_invariant_to_launch(block: int) -> None:
    ref = launch.philox_words_cuda(SEED, 0, 0, 4096, 16, block=256)
    for grid in (1, 7, None):
        np.testing.assert_array_equal(
            launch.philox_words_cuda(SEED, 0, 0, 4096, 16, block, grid), ref
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fused_tiles_invariant_to_block_and_grid(dtype: str) -> None:
    spec = PathSpec(100_003, uniform_grid(1.0, 12), RngSpec("philox", SEED), dtype)  # type: ignore[arg-type]
    opt = European("call", 100.0, 1.0)
    ref = launch.european_tiles(opt, MODEL, spec, block=256)
    for block in (64, 128, 256, 512):
        for grid in (1, 5, None):
            got = launch.european_tiles(opt, MODEL, spec, block=block, grid=grid)
            np.testing.assert_array_equal(got[0], ref[0])
            np.testing.assert_array_equal(got[1], ref[1])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_store_kernel_matches_numpy_paths(dtype: str) -> None:
    spec = PathSpec(1 << 16, uniform_grid(1.0, 16), RngSpec("philox", SEED, antithetic=True), dtype)  # type: ignore[arg-type]
    gpu = cp.asnumpy(launch.paths_store(MODEL, spec))
    cpu = NumpyGenerator().simulate(MODEL, spec)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-5 if dtype == "float32" else 1e-12)


def test_fused_store_numpy_jax_prices_agree() -> None:
    spec = PathSpec(1 << 20, uniform_grid(1.0, 64), RngSpec("philox", SEED), "float32")
    opt = European("call", 100.0, 1.0)
    prices = {b: price_mc(opt, MODEL, spec, backend=b) for b in ("cuda", "numpy", "jax")}  # type: ignore[arg-type]
    base = prices["numpy"]
    for b, r in prices.items():
        assert abs(r.price - base.price) <= 1e-5 * abs(base.price), b
        assert abs(r.price - base.price) <= 3 * base.std_err
        assert math.isclose(r.std_err, base.std_err, rel_tol=1e-3)
