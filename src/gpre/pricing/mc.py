"""Monte Carlo estimator with standard errors (FR-08, NFR-03).

Payoffs are discounted and accumulated in float64 whatever the path precision. Chunk results
are merged with Chan's parallel update of (count, mean, M2), so the variance never suffers the
cancellation of the sum-of-squares formula. With antithetic sampling the samples are the pair
averages; with Sobol kinds the estimate is the mean over R independently scrambled replications
and the SE is their standard deviation over sqrt(R) (randomized QMC).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from gpre.models.gbm import GBM
from gpre.paths.base import Backend, PathSpec, get_generator
from gpre.products.european import European
from gpre.rng.sobol import is_power_of_two

DEFAULT_REPLICATIONS = 32


@dataclass(frozen=True)
class PriceResult:
    price: float
    std_err: float
    n_paths: int
    elapsed_s: float
    vrf: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunningStats:
    """Numerically stable count, mean and M2 (sum of squared deviations), mergeable."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: npt.NDArray[np.float64]) -> None:
        if x.size == 0:
            return
        nb = int(x.size)
        mb = float(np.mean(x))
        m2b = float(np.sum((x - mb) ** 2))
        delta = mb - self.mean
        n = self.n + nb
        self.mean += delta * nb / n
        self.m2 += m2b + delta * delta * self.n * nb / n
        self.n = n

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else math.nan

    @property
    def std_err(self) -> float:
        return math.sqrt(self.variance / self.n) if self.n > 1 else math.nan


def discounted_payoffs(
    product: European, model: GBM, spec: PathSpec, backend: Backend
) -> list[npt.NDArray[np.float64]]:
    """Discounted float64 payoffs per chunk (antithetic pairs averaged)."""
    gen = get_generator(backend)
    disc = model.discount(product.maturity)
    out = []
    for paths in gen.iter_chunks(model, spec):
        pay = disc * product.payoff(paths)
        if spec.rng.antithetic:
            pay = 0.5 * (pay[0::2] + pay[1::2])
        out.append(pay)
    return out


def _check_grid(product: European, spec: PathSpec) -> None:
    t_last = float(np.asarray(spec.times)[-1])
    if not math.isclose(t_last, product.maturity, rel_tol=1e-12):
        raise ValueError(f"time grid ends at {t_last}, product matures at {product.maturity}")


def price_mc(
    product: European,
    model: GBM,
    spec: PathSpec,
    *,
    backend: Backend = "jax",
    control_variate: Literal["none", "geometric", "spot"] = "none",
    replications: int = DEFAULT_REPLICATIONS,
) -> PriceResult:
    """Price ``product`` by Monte Carlo on ``backend``.

    For Sobol kinds ``spec.n_paths`` is the total over ``replications`` scramblings, each of
    ``n_paths / replications`` points (a power of two).
    """
    if control_variate != "none":
        raise NotImplementedError("control variates arrive in M3 (FR-12)")
    _check_grid(product, spec)
    t0 = time.perf_counter()
    meta: dict[str, Any] = {
        "backend": backend,
        "rng": spec.rng.kind,
        "dtype": spec.dtype,
        "n_steps": spec.n_steps,
        "antithetic": spec.rng.antithetic,
    }
    if backend == "cuda":
        from gpre.cuda.launch import price_european_cuda

        res: PriceResult = price_european_cuda(product, model, spec)
        return replace(res, meta={**meta, **res.meta})
    if spec.rng.kind in ("sobol", "sobol-bb"):
        per = spec.n_paths // replications
        if per * replications != spec.n_paths or not is_power_of_two(per):
            raise ValueError("Sobol needs n_paths = replications * 2**m")
        means = []
        for r in range(replications):
            rspec = replace(spec, n_paths=per, rng=replace(spec.rng, replication=r))
            chunks = discounted_payoffs(product, model, rspec, backend)
            means.append(float(np.mean(np.concatenate(chunks))))
        m = np.asarray(means)
        price, se = float(m.mean()), float(m.std(ddof=1) / math.sqrt(replications))
        meta["replications"] = replications
    else:
        stats = RunningStats()
        for chunk in discounted_payoffs(product, model, spec, backend):
            stats.add(chunk)
        price, se = stats.mean, stats.std_err
    return PriceResult(price, se, spec.n_paths, time.perf_counter() - t0, None, meta)
