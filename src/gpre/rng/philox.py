"""Philox4x32-10 reference implementation in NumPy (FR-02).

Salmon, Moraes, Dror and Shaw (2011), "Parallel random numbers: as easy as 1, 2, 3". The
generator is a keyed bijection of a 128-bit counter: ten rounds of two 32x32->64-bit
multiplications, with a Weyl-sequence key schedule. Being counter-based, any output can be
computed from (counter, key) alone, which is what makes GPU streams independent of the launch
layout (FR-03).

The arithmetic runs in ``uint64`` so the 64-bit products are exact; the CUDA and C++ versions in
``gpre/cuda/kernels/philox4x32.cuh`` must agree bit for bit (Random123 known-answer tests in
``tests/fixtures/philox_kat.json``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

PHILOX_M0 = np.uint64(0xD2511F53)
PHILOX_M1 = np.uint64(0xCD9E8D57)
PHILOX_W0 = np.uint64(0x9E3779B9)
PHILOX_W1 = np.uint64(0xBB67AE85)
ROUNDS = 10

_MASK32 = np.uint64(0xFFFFFFFF)
_SHIFT32 = np.uint64(32)

U32Array = npt.NDArray[np.uint32]


def _as_u64(a: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    arr = np.asarray(a)
    if arr.dtype.kind not in "ui":
        raise TypeError(f"Philox inputs must be integer arrays, got {arr.dtype}")
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) > 0xFFFFFFFF):
        raise ValueError("Philox words must lie in [0, 2**32)")
    return arr.astype(np.uint64)


def philox4x32_10(counter: npt.ArrayLike, key: npt.ArrayLike, rounds: int = ROUNDS) -> U32Array:
    """Apply Philox4x32 to a batch of counters under one key.

    Args:
        counter: ``(n, 4)`` (or ``(4,)``) words ``(c0, c1, c2, c3)`` in ``[0, 2**32)``.
        key: ``(2,)`` words ``(k0, k1)``.
        rounds: number of rounds; 10 is the standard Philox4x32-10 (7 is also published).

    Returns:
        ``(n, 4)`` ``uint32`` outputs (``(4,)`` if a single counter was given).
    """
    ctr = _as_u64(counter)
    single = ctr.ndim == 1
    ctr = np.atleast_2d(ctr)
    if ctr.shape[-1] != 4:
        raise ValueError(f"counter must have 4 words per row, got shape {ctr.shape}")
    k = _as_u64(key).reshape(-1)
    if k.shape != (2,):
        raise ValueError(f"key must have 2 words, got shape {k.shape}")

    c0, c1, c2, c3 = (ctr[:, i].copy() for i in range(4))
    k0, k1 = k[0], k[1]
    for r in range(rounds):
        if r > 0:
            k0 = (k0 + PHILOX_W0) & _MASK32
            k1 = (k1 + PHILOX_W1) & _MASK32
        p0 = PHILOX_M0 * c0
        p1 = PHILOX_M1 * c2
        hi0, lo0 = p0 >> _SHIFT32, p0 & _MASK32
        hi1, lo1 = p1 >> _SHIFT32, p1 & _MASK32
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0

    out = np.stack([c0, c1, c2, c3], axis=-1).astype(np.uint32)
    return out[0] if single else out


def seed_to_key(seed: int) -> U32Array:
    """Split a 64-bit seed into the Philox key ``(k0, k1) = (low word, high word)``."""
    if not 0 <= seed < 2**64:
        raise ValueError("seed must lie in [0, 2**64)")
    return np.array([seed & 0xFFFFFFFF, seed >> 32], dtype=np.uint32)
