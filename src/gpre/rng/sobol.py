"""Scrambled Sobol normals for quasi-Monte Carlo (FR-04).

``scipy.stats.qmc.Sobol`` (Joe-Kuo directions, Owen scrambling). Sample sizes are powers of two
only, so the balance properties hold. Each replication ``r`` uses an independent scrambling
seeded from ``(seed, stream, r)``; randomized QMC estimates and SEs come from R replications.
Points are generated in chunks from one engine (the sequence continues across calls), so host
memory stays bounded and the result does not depend on the chunk size.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from scipy.special import ndtri
from scipy.stats import qmc

U_MIN = 2.0**-53


def is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def sobol_engine(n_dims: int, seed: int, stream: int, replication: int) -> qmc.Sobol:
    """An Owen-scrambled Sobol engine for one replication."""
    rng = np.random.default_rng([seed & 0xFFFFFFFF, seed >> 32, stream, replication])
    return qmc.Sobol(d=n_dims, scramble=True, rng=rng)


def sobol_normal_chunks(
    n_paths: int,
    n_dims: int,
    seed: int,
    stream: int = 0,
    replication: int = 0,
    chunk: int = 262144,
) -> Iterator[npt.NDArray[np.float64]]:
    """Yield ``(rows, n_dims)`` float64 normals, ``n_paths`` rows in total (a power of two)."""
    if not is_power_of_two(n_paths):
        raise ValueError(f"Sobol sample sizes must be powers of two, got {n_paths}")
    engine = sobol_engine(n_dims, seed, stream, replication)
    done = 0
    while done < n_paths:
        rows = min(chunk, n_paths - done)
        u = engine.random(rows)
        yield ndtri(np.clip(u, U_MIN, 1.0 - U_MIN))
        done += rows


def sobol_normals(
    n_paths: int, n_dims: int, seed: int, stream: int = 0, replication: int = 0
) -> npt.NDArray[np.float64]:
    """All ``(n_paths, n_dims)`` normals of one replication."""
    return np.concatenate(list(sobol_normal_chunks(n_paths, n_dims, seed, stream, replication)))
