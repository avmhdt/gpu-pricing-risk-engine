"""A short CUDA workload for the Nsight Compute counter check (scripts/ncu_check.sh, P-04)."""

from __future__ import annotations

from gpre.cuda.launch import european_tiles
from gpre.models.gbm import GBM, uniform_grid
from gpre.paths.base import PathSpec
from gpre.products.european import European
from gpre.rng.streams import RngSpec


def main() -> None:
    spec = PathSpec(1 << 20, uniform_grid(1.0, 64), RngSpec("philox", 20261030), "float32")
    tsum, _ = european_tiles(European("call", 100.0, 1.0), GBM(100.0, 0.03, 0.01, 0.2), spec)
    print(f"probe price {tsum.sum() / spec.n_paths:.4f}")


if __name__ == "__main__":
    main()
