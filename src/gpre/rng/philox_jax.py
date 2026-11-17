"""Philox4x32-10 in JAX with 32-bit integer operations only (FR-02, Should).

JAX without x64 has no uint64, so the 32x32 -> 64-bit products are split into 16-bit limbs.
The output is bit-identical to :mod:`gpre.rng.philox` and the CUDA header; it lets the JAX path
generator draw exactly the same uniforms as the NumPy and CUDA backends.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpre.rng.streams import F32_SCALE, F64_SCALE

_M0 = 0xD2511F53
_M1 = 0xCD9E8D57
_W0 = 0x9E3779B9
_W1 = 0xBB67AE85
_LO16 = 0xFFFF


def _mulhilo(a: int, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    """High and low 32-bit words of ``a * b`` for a constant ``a`` and uint32 array ``b``."""
    a_lo, a_hi = jnp.uint32(a & _LO16), jnp.uint32(a >> 16)
    b_lo, b_hi = b & jnp.uint32(_LO16), b >> jnp.uint32(16)
    p0 = a_lo * b_lo
    p1 = a_hi * b_lo
    p2 = a_lo * b_hi
    p3 = a_hi * b_hi
    s16 = jnp.uint32(16)
    mid = (p0 >> s16) + (p1 & jnp.uint32(_LO16)) + (p2 & jnp.uint32(_LO16))
    hi = p3 + (p1 >> s16) + (p2 >> s16) + (mid >> s16)
    lo = jnp.uint32(a) * b
    return hi, lo


def philox4x32_10_jax(
    c0: jax.Array, c1: jax.Array, c2: jax.Array, c3: jax.Array, k0: int, k1: int, rounds: int = 10
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Philox4x32 on uint32 arrays (broadcast together) under the key ``(k0, k1)``."""
    key0, key1 = jnp.uint32(k0), jnp.uint32(k1)
    for r in range(rounds):
        if r > 0:
            key0 = key0 + jnp.uint32(_W0)
            key1 = key1 + jnp.uint32(_W1)
        hi0, lo0 = _mulhilo(_M0, c0)
        hi1, lo1 = _mulhilo(_M1, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ key0, lo1, hi0 ^ c3 ^ key1, lo0
    return c0, c1, c2, c3


def block_uniforms(
    seed: int, stream: int, replication: int, paths: jax.Array, block: jax.Array, dtype: str
) -> jax.Array:
    """Uniforms of one draw block for each path: ``(n, 4)`` float32 or ``(n, 2)`` float64.

    ``paths`` is the uint32 Philox path index (already halved for antithetic pairs).
    """
    shape = paths.shape
    c0 = jnp.broadcast_to(block.astype(jnp.uint32), shape)
    c2 = jnp.full(shape, stream, jnp.uint32)
    c3 = jnp.full(shape, replication, jnp.uint32)
    w = philox4x32_10_jax(c0, paths, c2, c3, seed & 0xFFFFFFFF, seed >> 32)
    if dtype == "float32":
        nine = jnp.uint32(9)
        return jnp.stack(
            [((x >> nine).astype(jnp.float32) + 0.5) * jnp.float32(F32_SCALE) for x in w], axis=-1
        )
    if not jax.config.jax_enable_x64:
        raise RuntimeError("float64 uniforms need JAX_ENABLE_X64=1")
    out = []
    for a, b in ((w[0], w[1]), (w[2], w[3])):
        k = (a.astype(jnp.uint64) << jnp.uint64(20)) | (b >> jnp.uint32(12)).astype(jnp.uint64)
        out.append((k.astype(jnp.float64) + 0.5) * F64_SCALE)
    return jnp.stack(out, axis=-1)
