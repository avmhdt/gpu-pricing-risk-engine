"""Raw SVI slices: total variance, its k-derivatives, Durrleman's g and the implied density.

Raw SVI (Gatheral 2004): w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)), with k the
log-moneyness ln(K/F) and w the total implied variance sigma_BS^2 * t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RawSVI:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def w(self, k: npt.ArrayLike) -> FloatArray:
        """Total variance at log-moneyness ``k``."""
        x = np.asarray(k, dtype=np.float64) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma**2))

    def dw(self, k: npt.ArrayLike) -> FloatArray:
        """First derivative dw/dk."""
        x = np.asarray(k, dtype=np.float64) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.sigma**2))

    def d2w(self, k: npt.ArrayLike) -> FloatArray:
        """Second derivative d2w/dk2."""
        x = np.asarray(k, dtype=np.float64) - self.m
        return self.b * self.sigma**2 / (x * x + self.sigma**2) ** 1.5


def durrleman_g(k: FloatArray, w: FloatArray, dw: FloatArray, d2w: FloatArray) -> FloatArray:
    """Durrleman's butterfly function; a slice is butterfly-free where g(k) >= 0.

    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2  (Gatheral-Jacquier 2014, eq. 2.1).
    """
    return (1.0 - k * dw / (2.0 * w)) ** 2 - dw**2 / 4.0 * (1.0 / w + 0.25) + d2w / 2.0


def implied_density(k: FloatArray, w: FloatArray, dw: FloatArray, d2w: FloatArray) -> FloatArray:
    """Risk-neutral density of x = ln(S_T/F) at x = k implied by a total-variance slice.

    p(k) = g(k) / sqrt(2 pi w) * exp(-d_minus^2 / 2), d_minus = -k/sqrt(w) - sqrt(w)/2.
    """
    g = durrleman_g(k, w, dw, d2w)
    sw = np.sqrt(w)
    d_minus = -k / sw - sw / 2.0
    out: FloatArray = g / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d_minus**2)
    return out
