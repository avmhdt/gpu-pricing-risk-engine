"""Brownian-bridge construction of Brownian increments (FR-04).

Quasi-Monte Carlo points are best distributed in their first coordinates. The bridge spends the
first coordinate on W(T), the next on the midpoint, then on quarter points and so on (bisection
order), so most of the path's variance sits in the best coordinates and the effective
dimension falls (Moskowitz and Caflisch 1996; Glasserman 2004, section 3.1). With plain
time-ordered increments, late coordinates matter as much as early ones.

Given W at the left and right neighbours ``l < i < r``, W(t_i) is normal with mean
``((t_r - t_i) W_l + (t_i - t_l) W_r) / (t_r - t_l)`` and variance
``(t_i - t_l)(t_r - t_i) / (t_r - t_l)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class BridgeStep:
    target: int  # index into W(t_1..t_n)
    left: int  # -1 means W(0) = 0
    right: int  # -1 means no right point (the first, terminal step)
    w_left: float
    w_right: float
    sd: float


def bridge_schedule(times: npt.ArrayLike) -> list[BridgeStep]:
    """Bisection schedule for step end times ``t_1 < ... < t_n`` (t_0 = 0 implied)."""
    t = np.asarray(times, dtype=np.float64)
    n = t.size
    if n == 0 or np.any(np.diff(t) <= 0) or t[0] <= 0:
        raise ValueError("times must be positive and strictly increasing")
    steps = [BridgeStep(n - 1, -1, -1, 0.0, 0.0, float(np.sqrt(t[-1])))]
    queue = [(-1, n - 1)]
    while queue:
        nxt = []
        for left, right in queue:
            if right - left < 2:
                continue
            mid = (left + right) // 2
            tl = 0.0 if left < 0 else t[left]
            tm, tr = t[mid], t[right]
            steps.append(
                BridgeStep(
                    mid,
                    left,
                    right,
                    (tr - tm) / (tr - tl),
                    (tm - tl) / (tr - tl),
                    float(np.sqrt((tm - tl) * (tr - tm) / (tr - tl))),
                )
            )
            nxt += [(left, mid), (mid, right)]
        queue = nxt
    return steps


def brownian_bridge(z: npt.ArrayLike, times: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Map normals ``(n_paths, n_steps)`` in bridge order to increments dW ``(n_paths, n_steps)``.

    Column 0 of ``z`` drives W(T), column 1 the midpoint, and so on in bisection order.
    """
    zz = np.asarray(z, dtype=np.float64)
    steps = bridge_schedule(times)
    if zz.shape[-1] != len(steps):
        raise ValueError(f"z has {zz.shape[-1]} columns, need {len(steps)}")
    w = np.zeros_like(zz)
    for col, s in enumerate(steps):
        mean = 0.0 if s.right < 0 else s.w_right * w[..., s.right]
        if s.left >= 0:
            mean = mean + s.w_left * w[..., s.left]
        w[..., s.target] = mean + s.sd * zz[..., col]
    return np.diff(w, axis=-1, prepend=0.0)


def time_order_increments(z: npt.ArrayLike, times: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Plain construction: dW_k = sqrt(t_k - t_{k-1}) z_k (Sobol without the bridge)."""
    dt = np.diff(np.asarray(times, dtype=np.float64), prepend=0.0)
    return np.asarray(z, dtype=np.float64) * np.sqrt(dt)
