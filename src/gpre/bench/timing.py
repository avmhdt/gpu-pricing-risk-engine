"""Timing harness (FR-26 skeleton): warm-up, synchronization, median and p10/p90.

The first call is timed separately as the compile (JIT or NVRTC) cost and excluded from the
repetitions. ``sync`` must block until device work finishes (``block_until_ready`` for JAX, a
stream synchronize for CuPy); it runs inside the timed region.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Timing:
    time_median_s: float
    time_p10_s: float
    time_p90_s: float
    n_reps: int
    compile_s: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "time_median_s": self.time_median_s,
            "time_p10_s": self.time_p10_s,
            "time_p90_s": self.time_p90_s,
            "n_reps": self.n_reps,
            "compile_s": self.compile_s,
        }


def _noop(_: Any) -> None:
    return None


def time_fn(
    fn: Callable[[], Any],
    *,
    reps: int = 10,
    warmup: int = 3,
    sync: Callable[[Any], Any] = _noop,
    clock: Callable[[], float] = time.perf_counter,
) -> Timing:
    """Time ``fn``: first call = compile time, then ``warmup`` untimed calls, then ``reps``."""
    if reps < 1:
        raise ValueError("reps must be at least 1")
    t0 = clock()
    sync(fn())
    compile_s = clock() - t0
    for _ in range(warmup):
        sync(fn())
    samples = []
    for _ in range(reps):
        t0 = clock()
        sync(fn())
        samples.append(clock() - t0)
    arr = np.asarray(samples)
    return Timing(
        float(np.median(arr)),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 90)),
        reps,
        compile_s,
    )


def jax_sync(x: Any) -> Any:
    """Block until a JAX result (or pytree) is ready."""
    import jax

    return jax.block_until_ready(x)  # type: ignore[no-untyped-call]
