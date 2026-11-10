"""Path specification, the generator protocol and backend lookup (FR-05).

Every backend produces spot paths ``(n_steps + 1, n_paths)`` (row 0 is S_0) from the same
normals for the same :class:`~gpre.rng.streams.RngSpec`: Philox draws follow the counter layout
of docs/rng.md, Sobol draws come from the same scrambled engine.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from gpre.models.gbm import GBM
from gpre.rng.bridge import brownian_bridge, time_order_increments
from gpre.rng.sobol import sobol_normal_chunks
from gpre.rng.streams import DType, RngSpec, philox_normals

Backend = Literal["jax", "cuda", "numpy", "cpp"]
DEFAULT_CHUNK = 1 << 16


@dataclass(frozen=True)
class PathSpec:
    """What to simulate.

    Attributes:
        n_paths: number of paths (a power of two for Sobol kinds).
        times: step end times ``t_1 < ... < t_n`` in years (t_0 = 0 implied).
        rng: random-number specification.
        dtype: path precision.
        chunk_paths: paths per chunk on CPU backends (memory bound, not a result input).
    """

    n_paths: int
    times: npt.NDArray[np.float64]
    rng: RngSpec
    dtype: DType = "float32"
    chunk_paths: int = DEFAULT_CHUNK
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_paths < 1:
            raise ValueError("n_paths must be positive")
        if self.rng.antithetic and self.n_paths % 2:
            raise ValueError("antithetic sampling needs an even number of paths")
        if self.chunk_paths < 2 or self.chunk_paths % 2:
            raise ValueError("chunk_paths must be even")

    @property
    def n_steps(self) -> int:
        return int(np.asarray(self.times).size)


class PathGenerator(Protocol):
    def simulate(self, model: GBM, spec: PathSpec) -> Any: ...

    def iter_chunks(self, model: GBM, spec: PathSpec) -> Iterator[Any]: ...


def normal_chunks(spec: PathSpec) -> Iterator[npt.NDArray[np.float64]]:
    """Host normals ``(rows, n_steps)`` in path order, chunked, for any supported RNG kind.

    For ``sobol-bb`` the Sobol coordinates are mapped through the Brownian bridge and returned
    as standardized increments dW / sqrt(dt), so a path generator treats every kind alike.
    """
    kind, times = spec.rng.kind, np.asarray(spec.times)
    if kind == "philox":
        for off in range(0, spec.n_paths, spec.chunk_paths):
            rows = min(spec.chunk_paths, spec.n_paths - off)
            yield philox_normals(spec.rng, rows, spec.n_steps, spec.dtype, path_offset=off).astype(
                np.float64
            )
    elif kind in ("sobol", "sobol-bb"):
        if spec.rng.antithetic:
            raise ValueError("antithetic sampling is not combined with Sobol points")
        sd = np.sqrt(np.diff(times, prepend=0.0))
        for z in sobol_normal_chunks(
            spec.n_paths,
            spec.n_steps,
            spec.rng.seed,
            spec.rng.stream,
            spec.rng.replication,
            chunk=spec.chunk_paths,
        ):
            dw = (
                brownian_bridge(z, times) if kind == "sobol-bb" else time_order_increments(z, times)
            )
            yield dw / sd
    else:
        raise ValueError(f"rng kind {kind!r} is not supported on this backend")


def get_generator(backend: Backend) -> PathGenerator:
    """The path generator for ``backend`` (CuPy and the C++ binary are imported lazily)."""
    if backend == "numpy":
        from gpre.paths.numpy_paths import NumpyGenerator

        return NumpyGenerator()
    if backend == "jax":
        from gpre.paths.jax_paths import JaxGenerator

        return JaxGenerator()
    if backend == "cuda":
        from gpre.cuda.launch import CudaGenerator

        return CudaGenerator()
    raise NotImplementedError(f"backend {backend!r} arrives in a later milestone (cpp: M6)")
