"""JAX GBM path generator (FR-05): ``lax.scan`` over steps, vectorized over paths.

Philox draws are generated on the device by :mod:`gpre.rng.philox_jax` with the same counter
layout as NumPy and CUDA, so the uniforms are bit-identical. Sobol draws are generated on the
host (SciPy) and transferred, and that transfer is part of the timed work (NFR-09).
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtri

from gpre.models.gbm import GBM
from gpre.paths.base import PathSpec, normal_chunks
from gpre.rng.philox_jax import block_uniforms
from gpre.rng.streams import PER_BLOCK

DEFAULT_JAX_CHUNK = 1 << 18


def _dtype(name: str) -> jnp.dtype:
    if name == "float64" and not jax.config.jax_enable_x64:
        raise RuntimeError("float64 paths need JAX_ENABLE_X64=1")
    return jnp.dtype(name)


@partial(
    jax.jit,
    static_argnames=("n_paths", "n_steps", "dtype", "seed", "stream", "replication", "antithetic"),
)
def philox_normals_jax(
    offset: jax.Array,
    *,
    n_paths: int,
    n_steps: int,
    dtype: str,
    seed: int,
    stream: int,
    replication: int,
    antithetic: bool,
) -> jax.Array:
    """Normals ``(n_paths, n_steps)`` for global paths ``offset .. offset + n_paths - 1``."""
    per = PER_BLOCK[dtype]
    n_blocks = -(-n_steps // per)
    gidx = offset.astype(jnp.uint32) + jnp.arange(n_paths, dtype=jnp.uint32)
    pidx = gidx >> jnp.uint32(1) if antithetic else gidx
    blocks = jnp.arange(n_blocks, dtype=jnp.uint32)
    u = jax.vmap(lambda b: block_uniforms(seed, stream, replication, pidx, b, dtype), out_axes=1)(
        blocks
    )  # (n_paths, n_blocks, per)
    z = ndtri(u.reshape(n_paths, n_blocks * per)[:, :n_steps])
    if antithetic:
        sign = jnp.where(gidx % jnp.uint32(2) == 0, 1.0, -1.0).astype(z.dtype)
        z = z * sign[:, None]
    return z.astype(dtype)


@partial(jax.jit, static_argnames=("dtype",))
def gbm_paths_jax(
    z: jax.Array, drift_dt: jax.Array, vol_sqrt_dt: jax.Array, s0: jax.Array, *, dtype: str
) -> jax.Array:
    """Paths ``(n_steps + 1, rows)`` from normals ``(rows, n_steps)`` by a scan over steps."""
    dt = jnp.dtype(dtype)
    inc = (drift_dt[:, None] + vol_sqrt_dt[:, None] * z.T.astype(drift_dt.dtype)).astype(dt)

    def step(x: jax.Array, d: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = x + d
        return x, x

    _, xs = jax.lax.scan(step, jnp.zeros(z.shape[0], dt), inc)
    s0d = s0.astype(dt)
    return jnp.concatenate([jnp.full((1, z.shape[0]), s0d, dt), s0d * jnp.exp(xs)], axis=0)


class JaxGenerator:
    """JAX backend (GPU if available, else CPU)."""

    name = "jax"

    def __init__(self, chunk_paths: int = DEFAULT_JAX_CHUNK) -> None:
        self.chunk_paths = chunk_paths

    def iter_chunks(self, model: GBM, spec: PathSpec) -> Iterator[jax.Array]:
        dt = _dtype(spec.dtype)
        times = np.asarray(spec.times, dtype=np.float64)
        drift_dt, vol_sqrt_dt = model.step_coefficients(times)
        coef_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        # Coefficients rounded to the path precision, as the CUDA kernels receive them.
        drift = jnp.asarray(drift_dt.astype(spec.dtype), coef_dtype)
        vol = jnp.asarray(vol_sqrt_dt.astype(spec.dtype), coef_dtype)
        s0 = jnp.asarray(model.spot, coef_dtype)
        r = spec.rng
        if r.kind == "philox":
            chunk = min(self.chunk_paths, spec.n_paths)
            chunk += chunk % 2
            for off in range(0, spec.n_paths, chunk):
                rows = min(chunk, spec.n_paths - off)
                z = philox_normals_jax(
                    jnp.uint32(off),
                    n_paths=rows,
                    n_steps=spec.n_steps,
                    dtype=spec.dtype,
                    seed=r.seed,
                    stream=r.stream,
                    replication=r.replication,
                    antithetic=r.antithetic,
                )
                yield gbm_paths_jax(z, drift, vol, s0, dtype=spec.dtype)
        else:
            for zh in normal_chunks(spec):
                yield gbm_paths_jax(jnp.asarray(zh, dt), drift, vol, s0, dtype=spec.dtype)

    def simulate(self, model: GBM, spec: PathSpec) -> jax.Array:
        return jnp.concatenate(list(self.iter_chunks(model, spec)), axis=1)
