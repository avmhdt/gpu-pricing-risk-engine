"""Discount curve from FRED CMT zero rates (FR-22).

Zero rates (continuously compounded decimals) are interpolated linearly in t and held flat
beyond the first and last nodes; df(t) = exp(-r(t) t). The curve is a JAX pytree whose leaves
are the node times and zero rates, so ``jax.grad`` of a price returns one rho per node.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import jax
import jax.numpy as jnp
import polars as pl

MIN_TENORS = 6


class CurveDataError(ValueError):
    """Too few tenors on the requested date (DQ-02)."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DiscountCurve:
    """Piecewise-linear zero curve.

    Attributes:
        times: node times in years, strictly increasing, shape ``(n,)``.
        zero_rates: continuously compounded zero rates at the nodes, decimal, shape ``(n,)``.
    """

    times: jax.Array
    zero_rates: jax.Array

    def zero(self, t: jax.Array | float) -> jax.Array:
        """Zero rate r(t), linear between nodes and flat outside."""
        return jnp.interp(jnp.asarray(t), self.times, self.zero_rates)

    def df(self, t: jax.Array | float) -> jax.Array:
        """Discount factor exp(-r(t) t)."""
        tt = jnp.asarray(t)
        return jnp.exp(-self.zero(tt) * tt)

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], None]:
        return (self.times, self.zero_rates), None

    @classmethod
    def tree_unflatten(cls, aux: Any, children: tuple[jax.Array, jax.Array]) -> DiscountCurve:
        del aux
        return cls(*children)

    @classmethod
    def flat(cls, rate: float) -> DiscountCurve:
        """A flat curve (single node), for runs without FRED data."""
        return cls(jnp.asarray([1.0]), jnp.asarray([rate]))

    @classmethod
    def from_yields(cls, yields: pl.DataFrame, as_of: date) -> DiscountCurve:
        """Curve on ``as_of`` from the ``yields`` table (§7b).

        Tenors with no value on that date are dropped; at least 6 must remain (DQ-02).
        """
        rows = (
            yields.filter((pl.col("date") == as_of) & pl.col("zero_cc").is_not_null())
            .sort("tenor_years")
            .select("tenor_years", "zero_cc")
        )
        if rows.height < MIN_TENORS:
            raise CurveDataError(f"{as_of}: {rows.height} tenors, need at least {MIN_TENORS}")
        return cls(
            jnp.asarray(rows["tenor_years"].to_numpy()),
            jnp.asarray(rows["zero_cc"].to_numpy()),
        )
