"""Single-threaded NumPy GBM path generator: the CPU reference and baseline (FR-05, FR-25a).

Vectorized over paths in chunks of 2**16, with the same RNG layout and the same per-step
arithmetic as the CUDA kernels: x_k = x_{k-1} + (drift_dt_k + vol_sqrt_dt_k * z_k) in the path
precision, S_k = S_0 exp(x_k). No Python loop runs over paths.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from gpre.models.gbm import GBM
from gpre.paths.base import PathSpec, normal_chunks


def gbm_paths_from_normals(
    model: GBM, times: npt.NDArray[np.float64], z: npt.NDArray[np.floating], dtype: str
) -> npt.NDArray[np.floating]:
    """Paths ``(n_steps + 1, rows)`` from normals ``(rows, n_steps)``."""
    drift_dt, vol_sqrt_dt = model.step_coefficients(times)
    # The kernels receive the coefficients in the path precision; round them the same way.
    drift_dt = drift_dt.astype(dtype).astype(np.float64)
    vol_sqrt_dt = vol_sqrt_dt.astype(dtype).astype(np.float64)
    inc = (drift_dt[None, :] + vol_sqrt_dt[None, :] * np.asarray(z, np.float64)).astype(dtype)
    x = np.cumsum(inc, axis=1, dtype=dtype)  # sequential accumulation in the path precision
    s0 = np.asarray(model.spot, dtype=dtype)
    paths = np.empty((times.size + 1, z.shape[0]), dtype=dtype)
    paths[0] = s0
    paths[1:] = (s0 * np.exp(x)).T
    return paths


class NumpyGenerator:
    """NumPy backend."""

    name = "numpy"

    def iter_chunks(self, model: GBM, spec: PathSpec) -> Iterator[npt.NDArray[np.floating]]:
        times = np.asarray(spec.times, dtype=np.float64)
        for z in normal_chunks(spec):
            yield gbm_paths_from_normals(model, times, z.astype(spec.dtype), spec.dtype)

    def simulate(self, model: GBM, spec: PathSpec) -> npt.NDArray[np.floating]:
        """All paths ``(n_steps + 1, n_paths)``."""
        return np.concatenate(list(self.iter_chunks(model, spec)), axis=1)
