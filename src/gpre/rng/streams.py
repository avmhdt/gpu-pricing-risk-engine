"""Reproducible Philox stream layout (FR-03): uniforms and normals as a pure function of
``(seed, stream, replication, path, step)``.

Layout (identical in NumPy, JAX, CUDA and C++):

- key = the 64-bit seed as ``(low word, high word)``;
- counter = ``(draw block j, path p, stream s, replication r)``;
- float32: one Philox call gives the uniforms for steps ``4j .. 4j+3`` of path ``p``,
  ``u = ((x >> 9) + 0.5) * 2**-23``;
- float64: one call gives steps ``2j, 2j+1``, each from two words,
  ``u = ((a << 20 | b >> 12) + 0.5) * 2**-52``;
- normals by inverse CDF; antithetic path ``p`` uses Philox path ``p // 2`` and flips the sign
  of the normal for odd ``p``.

Both uniform forms are exact in their precision and lie strictly inside (0, 1). The spec's
float32 form ``((x >> 8) + 0.5) * 2**-24`` rounds to exactly 1.0 for the largest word (an
infinite normal about once in 2**24 draws), so 23 bits are used; see docs/decisions.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.special import ndtri

from gpre.rng.philox import philox4x32_10, seed_to_key

RngKind = Literal["philox", "sobol-bb", "sobol", "threefry"]
DType = Literal["float32", "float64"]

PER_BLOCK = {"float32": 4, "float64": 2}
"""Normals per Philox call for each precision."""

F32_SCALE = 2.0**-23
F64_SCALE = 2.0**-52

# Stream ids (docs/rng.md): s = underlying_index * STREAMS_PER_UNDERLYING + purpose.
STREAMS_PER_UNDERLYING = 16
PURPOSE = {
    "paths": 0,  # spot Brownian drivers
    "variance": 1,  # second Heston factor
    "lsm_regression": 2,  # independent LSM regression paths
    "scenarios": 3,  # Monte Carlo VaR scenarios
    "validation": 4,  # one stream per validation case
}


def stream_id(underlying_index: int, purpose: str) -> int:
    """Fixed stream id for an (underlying, purpose) pair."""
    if purpose not in PURPOSE:
        raise ValueError(f"unknown purpose {purpose!r}; known: {sorted(PURPOSE)}")
    if underlying_index < 0:
        raise ValueError("underlying_index must be non-negative")
    return underlying_index * STREAMS_PER_UNDERLYING + PURPOSE[purpose]


@dataclass(frozen=True)
class RngSpec:
    """Random-number specification for a simulation.

    Attributes:
        kind: ``philox`` (counter-based MC), ``sobol-bb`` (scrambled Sobol with a Brownian
            bridge), ``sobol`` (scrambled Sobol, increments in time order) or ``threefry``
            (JAX's default generator, for comparison only).
        seed: 64-bit master seed (Philox key, Sobol scrambling seed).
        stream: stream id (see :func:`stream_id`).
        replication: replication index (RQMC scrambling or independent MC repeats).
        antithetic: pair paths ``2i, 2i+1`` with opposite normals.
    """

    kind: RngKind
    seed: int
    stream: int = 0
    replication: int = 0
    antithetic: bool = False


def philox_words(
    seed: int, stream: int, replication: int, paths: npt.ArrayLike, blocks: npt.ArrayLike
) -> npt.NDArray[np.uint32]:
    """Philox output for each ``(path, block)`` pair, broadcast together; shape ``(..., 4)``."""
    p, b = np.broadcast_arrays(np.asarray(paths, dtype=np.uint64), np.asarray(blocks, np.uint64))
    ctr = np.stack(
        [
            b.ravel(),
            p.ravel(),
            np.full(p.size, stream, np.uint64),
            np.full(p.size, replication, np.uint64),
        ],
        axis=-1,
    )
    return philox4x32_10(ctr, seed_to_key(seed)).reshape(*p.shape, 4)


def words_to_uniforms(words: npt.NDArray[np.uint32], dtype: DType) -> npt.NDArray[np.float64]:
    """Map ``(..., 4)`` Philox words to ``(..., 4)`` (float32) or ``(..., 2)`` (float64)
    uniforms, returned in float64 (the values are exact in the target precision)."""
    w = words.astype(np.uint64)
    if dtype == "float32":
        return ((w >> np.uint64(9)).astype(np.float64) + 0.5) * F32_SCALE
    hi = (w[..., 0::2] << np.uint64(20)) | (w[..., 1::2] >> np.uint64(12))
    return (hi.astype(np.float64) + 0.5) * F64_SCALE


def philox_uniforms(
    spec: RngSpec, n_paths: int, n_steps: int, dtype: DType, path_offset: int = 0
) -> npt.NDArray[np.float64]:
    """Uniforms ``(n_paths, n_steps)`` for paths ``path_offset .. path_offset + n_paths - 1``.

    With ``spec.antithetic`` the uniforms of odd paths are those of their even partner; the
    sign flip happens on the normals (:func:`philox_normals`).
    """
    per = PER_BLOCK[dtype]
    n_blocks = -(-n_steps // per)
    paths = np.arange(path_offset, path_offset + n_paths, dtype=np.uint64)
    if spec.antithetic:
        paths = paths // np.uint64(2)
    words = philox_words(
        spec.seed,
        spec.stream,
        spec.replication,
        paths[:, None],
        np.arange(n_blocks, dtype=np.uint64)[None, :],
    )
    u = words_to_uniforms(words, dtype).reshape(n_paths, n_blocks * per)
    return u[:, :n_steps]


def antithetic_signs(n_paths: int, path_offset: int = 0) -> npt.NDArray[np.float64]:
    """+1 for even global path indices, -1 for odd ones."""
    idx = np.arange(path_offset, path_offset + n_paths)
    return np.where(idx % 2 == 0, 1.0, -1.0)


def philox_normals(
    spec: RngSpec, n_paths: int, n_steps: int, dtype: DType, path_offset: int = 0
) -> npt.NDArray[np.floating]:
    """Standard normals ``(n_paths, n_steps)`` in ``dtype`` by inverse CDF (NumPy reference)."""
    z = ndtri(philox_uniforms(spec, n_paths, n_steps, dtype, path_offset))
    if spec.antithetic:
        z = z * antithetic_signs(n_paths, path_offset)[:, None]
    out: npt.NDArray[np.floating] = np.asarray(z).astype(dtype)
    return out
