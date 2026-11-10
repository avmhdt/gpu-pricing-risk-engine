"""Black-Scholes prices and Greeks with a continuous dividend yield (validation reference)."""

from __future__ import annotations

import math

from scipy.special import ndtr


def _d1_d2(S: float, K: float, T: float, r: float, q: float, vol: float) -> tuple[float, float]:
    if S <= 0 or K <= 0 or T <= 0 or vol <= 0:
        raise ValueError("S, K, T and vol must be positive")
    sd = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / sd
    return d1, d1 - sd


def bs_price(
    S: float, K: float, T: float, r: float, q: float, vol: float, option_type: str
) -> float:
    """Black-Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S, K, T, r, q, vol)
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    if option_type == "call":
        return float(S * dq * ndtr(d1) - K * dr * ndtr(d2))
    if option_type == "put":
        return float(K * dr * ndtr(-d2) - S * dq * ndtr(-d1))
    raise ValueError(f"option_type must be call or put, got {option_type!r}")


def bs_greeks(
    S: float, K: float, T: float, r: float, q: float, vol: float, option_type: str
) -> dict[str, float]:
    """Delta, gamma, vega (per unit vol), theta (per year) and rho (per unit rate)."""
    d1, d2 = _d1_d2(S, K, T, r, q, vol)
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    gamma = dq * pdf / (S * vol * math.sqrt(T))
    vega = S * dq * pdf * math.sqrt(T)
    if option_type == "call":
        delta = dq * float(ndtr(d1))
        theta = (
            -S * dq * pdf * vol / (2 * math.sqrt(T))
            - r * K * dr * float(ndtr(d2))
            + q * S * dq * float(ndtr(d1))
        )
        rho = K * T * dr * float(ndtr(d2))
    elif option_type == "put":
        delta = -dq * float(ndtr(-d1))
        theta = (
            -S * dq * pdf * vol / (2 * math.sqrt(T))
            + r * K * dr * float(ndtr(-d2))
            - q * S * dq * float(ndtr(-d1))
        )
        rho = -K * T * dr * float(ndtr(-d2))
    else:
        raise ValueError(f"option_type must be call or put, got {option_type!r}")
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
