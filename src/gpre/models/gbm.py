"""Geometric Brownian motion under the risk-neutral measure.

dS/S = (r(t) - q) dt + sigma dW. On any time grid the exact log-Euler step is
ln S_k = ln S_{k-1} + (f_k - q - sigma^2/2) dt_k + sigma sqrt(dt_k) Z_k, where f_k is the
forward rate over the step (the flat rate, or from the discount curve).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from gpre.market.curve import DiscountCurve

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GBM:
    """GBM model parameters. ``rate`` is a flat continuously compounded rate or a curve."""

    spot: float
    rate: float | DiscountCurve
    div_yield: float
    vol: float

    def log_df(self, t: npt.ArrayLike) -> FloatArray:
        """ln of the discount factor at times ``t`` (years)."""
        tt = np.asarray(t, dtype=np.float64)
        if isinstance(self.rate, DiscountCurve):
            return np.log(np.asarray(self.rate.df(jnp.asarray(tt)), dtype=np.float64))
        return -self.rate * tt

    def discount(self, t: float) -> float:
        return math.exp(float(self.log_df(t)))

    def step_coefficients(self, times: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Per-step ``drift_dt`` and ``vol_sqrt_dt`` for step end times ``t_1 .. t_n``."""
        t = np.asarray(times, dtype=np.float64)
        t_prev = np.concatenate([[0.0], t[:-1]])
        dt = t - t_prev
        if np.any(dt <= 0):
            raise ValueError("times must be positive and strictly increasing")
        fwd_dt = self.log_df(t_prev) - self.log_df(t)  # integral of the forward rate
        drift_dt = fwd_dt - (self.div_yield + 0.5 * self.vol**2) * dt
        return drift_dt, self.vol * np.sqrt(dt)


def uniform_grid(maturity: float, n_steps: int) -> FloatArray:
    """Step end times ``T/n, 2T/n, ..., T``."""
    if n_steps < 1 or maturity <= 0:
        raise ValueError("need n_steps >= 1 and maturity > 0")
    return maturity * np.arange(1, n_steps + 1, dtype=np.float64) / n_steps


def steps_for(maturity: float, steps_per_year: int) -> int:
    """Number of steps for ``maturity`` at ``steps_per_year`` (at least one)."""
    return max(1, round(maturity * steps_per_year))
