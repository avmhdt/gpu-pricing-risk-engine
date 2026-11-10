"""SSVI surfaces (Gatheral and Jacquier 2014) with the vsm.surface.v1 theta rule.

w(k, theta) = theta/2 * (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)), phi = phi(theta).

theta(t) follows producer rule 1 of vsm.surface.v1: linear in t between nodes, theta_1 t/t_1
below the first node and theta_n t/t_n beyond the last.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from gpre.surface.svi import RawSVI

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SSVI:
    """SSVI parameters with a piecewise-linear ATM total-variance term structure."""

    rho: float
    t_nodes: tuple[float, ...]
    theta_nodes: tuple[float, ...]
    phi: Literal["power_law", "heston_like"] = "power_law"
    eta: float = 1.0
    gamma: float = 0.5
    lam: float | None = None

    def __post_init__(self) -> None:
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")
        if len(self.t_nodes) != len(self.theta_nodes) or not self.t_nodes:
            raise ValueError("t_nodes and theta_nodes must be non-empty and the same length")
        if any(b <= a for a, b in zip(self.t_nodes, self.t_nodes[1:], strict=False)):
            raise ValueError("t_nodes must be strictly increasing")
        if self.phi == "heston_like" and not self.lam:
            raise ValueError("phi = heston_like needs lambda")

    def theta(self, t: float) -> float:
        """ATM total variance at time ``t`` (years), producer rule 1."""
        tn, th = self.t_nodes, self.theta_nodes
        if t <= tn[0]:
            return th[0] * t / tn[0]
        if t >= tn[-1]:
            return th[-1] * t / tn[-1]
        return float(np.interp(t, tn, th))

    def phi_of(self, theta: float) -> float:
        """Curvature function phi(theta)."""
        if self.phi == "power_law":
            return float(self.eta / (theta**self.gamma * (1.0 + theta) ** (1.0 - self.gamma)))
        lam = float(self.lam or 0.0)
        x = lam * theta
        return 1.0 / x * (1.0 - (1.0 - math.exp(-x)) / x)

    def slice_at(self, t: float) -> RawSVI:
        """Exact raw-SVI form of the SSVI slice at ``t`` (Gatheral-Jacquier 2014, Lemma 3.2)."""
        theta = self.theta(t)
        phi = self.phi_of(theta)
        rho = self.rho
        return RawSVI(
            a=theta / 2.0 * (1.0 - rho * rho),
            b=theta * phi / 2.0,
            rho=rho,
            m=-rho / phi,
            sigma=math.sqrt(1.0 - rho * rho) / phi,
        )

    def w(self, k: npt.ArrayLike, t: float) -> FloatArray:
        """Total implied variance at log-moneyness ``k`` and time ``t``."""
        theta = self.theta(t)
        phi = self.phi_of(theta)
        kk = np.asarray(k, dtype=np.float64)
        return (
            theta
            / 2.0
            * (1.0 + self.rho * phi * kk + np.sqrt((phi * kk + self.rho) ** 2 + 1.0 - self.rho**2))
        )


def synthetic_ssvi(
    theta_per_year: float = 0.04,
    rho: float = -0.7,
    eta: float = 1.0,
    gamma: float = 0.5,
    t_nodes: tuple[float, ...] = (1 / 12, 0.25, 0.5, 1.0),
) -> SSVI:
    """The D-05 default surface: theta(T) = theta_per_year * T, power-law phi."""
    return SSVI(
        rho=rho,
        t_nodes=t_nodes,
        theta_nodes=tuple(theta_per_year * t for t in t_nodes),
        phi="power_law",
        eta=eta,
        gamma=gamma,
    )
