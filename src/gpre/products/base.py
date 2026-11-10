"""Product protocol shared by the pricers."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class Product(Protocol):
    """A payoff evaluated on simulated paths ``(n_steps + 1, n_paths)``."""

    def payoff(self, paths: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]: ...
