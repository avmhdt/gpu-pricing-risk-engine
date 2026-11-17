"""European call and put."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class European:
    """European option paying max(S_T - K, 0) (call) or max(K - S_T, 0) (put) at ``maturity``."""

    option_type: Literal["call", "put"]
    strike: float
    maturity: float

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be call or put, got {self.option_type!r}")
        if self.strike <= 0 or self.maturity <= 0:
            raise ValueError("strike and maturity must be positive")

    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    def payoff_terminal(self, s_t: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """Undiscounted payoff from terminal spots, in float64."""
        s = np.asarray(s_t, dtype=np.float64)
        return (
            np.maximum(s - self.strike, 0.0) if self.is_call else np.maximum(self.strike - s, 0.0)
        )

    def payoff(self, paths: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """Payoff from paths ``(n_steps + 1, n_paths)``; the last row is S_T."""
        return self.payoff_terminal(paths[-1])
