"""Validation suites against closed forms (§12). M2: ``european`` (AC-W-01) and ``rng``
(the CPU part of AC-W-06; the CUDA part runs in ``tests/gpu`` on the cloud instance).

Seeds, grids and tolerances come from SPEC §12 and ``configs/validation.yaml`` and are never
tuned. Each case uses its own Philox stream (``stream_id(case, "validation")``), so the cases'
errors are independent and the |z| > 2 share is meaningful.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from gpre.analytic.black_scholes import bs_price
from gpre.models.gbm import GBM, uniform_grid
from gpre.paths.base import Backend, PathSpec
from gpre.pricing.mc import price_mc
from gpre.products.european import European
from gpre.rng.streams import RngSpec, stream_id

Z_MAX = 3.0
MAX_SHARE_Z_GT_2 = 0.10


@dataclass(frozen=True)
class Case:
    case_id: str
    params: dict[str, Any]
    mc_price: float
    std_err: float
    reference: float
    elapsed_s: float

    @property
    def z_score(self) -> float:
        return (self.mc_price - self.reference) / self.std_err if self.std_err > 0 else math.inf

    @property
    def passed(self) -> bool:
        return abs(self.z_score) <= Z_MAX

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "params": self.params,
            "mc_price": self.mc_price,
            "std_err": self.std_err,
            "reference": self.reference,
            "z_score": self.z_score,
            "pass": self.passed,
            "elapsed_s": self.elapsed_s,
        }


def european_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The 36-case AC-W-01 grid from ``configs/validation.yaml``."""
    grid = itertools.product(cfg["K"], cfg["T"], cfg["vol"], cfg["types"])
    return [
        {
            "S": float(cfg["S"]),
            "K": float(k),
            "T": float(t),
            "vol": float(v),
            "r": float(cfg["r"]),
            "q": float(cfg["q"]),
            "type": typ,
        }
        for k, t, v, typ in grid
    ]


def run_european(
    cfg: dict[str, Any],
    *,
    backend: Backend,
    seed: int,
    n_paths: int,
    dtype: str = "float64",
    n_steps: int = 1,
) -> list[Case]:
    """Price every case on ``backend`` and compare with Black-Scholes.

    GBM log-Euler steps are exact, so the terminal law does not depend on ``n_steps``; one step
    is the default for the 36-case grid at 2**20 paths.
    """
    cases = []
    for i, p in enumerate(european_grid(cfg)):
        model = GBM(spot=p["S"], rate=p["r"], div_yield=p["q"], vol=p["vol"])
        product = European(p["type"], p["K"], p["T"])
        spec = PathSpec(
            n_paths=n_paths,
            times=uniform_grid(p["T"], n_steps),
            rng=RngSpec("philox", seed, stream=stream_id(i, "validation")),
            dtype=dtype,  # type: ignore[arg-type]
        )
        res = price_mc(product, model, spec, backend=backend)
        ref = bs_price(p["S"], p["K"], p["T"], p["r"], p["q"], p["vol"], p["type"])
        cid = f"{backend}-{p['type']}-K{p['K']:g}-T{p['T']:g}-v{p['vol']:g}"
        cases.append(
            Case(
                cid,
                {**p, "n_paths": n_paths, "n_steps": n_steps, "dtype": dtype, "backend": backend},
                res.price,
                res.std_err,
                ref,
                res.elapsed_s,
            )
        )
    return cases


def summarize(cases: list[Case]) -> dict[str, Any]:
    """Suite verdict: every |z| <= 3 and at most 10% of cases with |z| > 2 (§12)."""
    z = np.array([c.z_score for c in cases])
    share = float(np.mean(np.abs(z) > 2.0)) if z.size else 0.0
    return {
        "n_cases": len(cases),
        "n_pass": sum(c.passed for c in cases),
        "max_abs_z": float(np.max(np.abs(z))) if z.size else 0.0,
        "share_abs_z_gt_2": share,
        "passed": bool(cases) and all(c.passed for c in cases) and share <= MAX_SHARE_Z_GT_2,
    }


def run_rng_cpu(seed: int, n_paths: int = 4096, n_blocks: int = 64) -> dict[str, Any]:
    """CPU part of AC-W-06: Philox KAT, and JAX words and uniforms equal to the NumPy reference."""
    import jax
    import jax.numpy as jnp

    from gpre.rng.philox_jax import block_uniforms, philox4x32_10_jax
    from gpre.rng.streams import philox_uniforms, philox_words

    paths = np.arange(n_paths, dtype=np.uint32)
    ref = philox_words(seed, 7, 3, paths[:, None], np.arange(n_blocks, dtype=np.uint32)[None, :])
    jw = jax.vmap(
        lambda b: jnp.stack(
            philox4x32_10_jax(
                jnp.full(n_paths, b, jnp.uint32),
                jnp.asarray(paths),
                jnp.full(n_paths, 7, jnp.uint32),
                jnp.full(n_paths, 3, jnp.uint32),
                seed & 0xFFFFFFFF,
                seed >> 32,
            ),
            axis=-1,
        ),
        out_axes=1,
    )(jnp.arange(n_blocks, dtype=jnp.uint32))
    words_equal = bool(np.array_equal(np.asarray(jw), ref))
    spec = RngSpec("philox", seed, stream=7, replication=3)
    u_ref = philox_uniforms(spec, n_paths, 4 * n_blocks, "float32")
    u_jax = jax.vmap(
        lambda b: block_uniforms(seed, 7, 3, jnp.asarray(paths), b, "float32"), out_axes=1
    )(jnp.arange(n_blocks, dtype=jnp.uint32))
    uniforms_equal = bool(np.array_equal(np.asarray(u_jax, np.float64).reshape(n_paths, -1), u_ref))
    return {
        "n_paths": n_paths,
        "n_blocks": n_blocks,
        "jax_words_equal_numpy": words_equal,
        "jax_f32_uniforms_equal_numpy": uniforms_equal,
        "passed": words_equal and uniforms_equal,
    }
