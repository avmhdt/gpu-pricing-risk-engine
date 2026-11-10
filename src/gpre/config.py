"""Settings from ``.env`` and typed YAML run configs.

Two layers:

- :class:`Settings` reads environment variables and ``.env`` (pydantic-settings). It holds
  paths, keys and runtime switches. Keys are ``SecretStr`` so they never print.
- :class:`RunConfig` mirrors ``configs/default.yaml``. Unknown keys are rejected, so a typo in
  a config file fails loudly instead of being ignored.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Backend = Literal["jax", "cuda", "numpy", "cpp"]
DType = Literal["float32", "float64"]

REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repository root when running from a source checkout (src layout)."""

CONFIG_DIR = REPO_ROOT / "configs"


class Settings(BaseSettings):
    """Environment settings (see ``.env.example`` for the meaning of each variable)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        case_sensitive=False,
    )

    fred_api_key: SecretStr = SecretStr("")
    sharadar_api_key: SecretStr = SecretStr("")
    tiingo_api_key: SecretStr = SecretStr("")
    gpre_cloud_api_key: SecretStr = SecretStr("")

    gpre_cloud_provider: str = ""
    gpre_gpu_budget_usd: float = 50.0
    gpre_spend_log: Path = Path("./results/cloud_spend.csv")

    data_root: Path = Path("~/data/gpu-pricing-risk-engine")
    gpre_data_dir: Path | None = None
    gpre_results_dir: Path = Path("./results")
    gpre_reports_dir: Path = Path("./reports")
    gpre_surface_path: Path | None = None
    stooq_dir: Path = Path("/mnt/e/trading-strategy-brainstorm/data/stooq")

    gpre_backend: Backend = "jax"
    gpre_dtype: DType = "float32"
    gpre_seed: int = 20261030
    gpre_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    gpre_cpp_baseline: Path = Path("~/build/gpu-pricing-risk-engine/cpp/gpre_cpu_baseline")
    cuda_home: Path = Path("/usr/local/cuda")
    cuda_visible_devices: str = "0"

    @field_validator(
        "data_root",
        "gpre_data_dir",
        "gpre_surface_path",
        "stooq_dir",
        "gpre_cpp_baseline",
        mode="after",
    )
    @classmethod
    def _expand_user(cls, value: Path | None) -> Path | None:
        return None if value is None else value.expanduser()

    @property
    def data_dir(self) -> Path:
        """Root of ``raw/``, ``interim/`` and ``processed/`` (``GPRE_DATA_DIR`` or ``DATA_ROOT``).

        Raises:
            ValueError: if the directory sits on a Windows drive mounted in WSL (``/mnt/*``),
                which the spec forbids for data (slow I/O, §7d).
        """
        root = self.gpre_data_dir or self.data_root
        if root.as_posix().startswith("/mnt/"):
            raise ValueError(f"data directory {root} is under /mnt; use the WSL ext4 filesystem")
        return root

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def results_dir(self) -> Path:
        return self.gpre_results_dir

    @property
    def reports_dir(self) -> Path:
        return self.gpre_reports_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings singleton. Tests call ``get_settings.cache_clear()`` after changing the env."""
    return Settings()


# ---------------------------------------------------------------------------------------------
# YAML run configuration (configs/default.yaml)
# ---------------------------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MarketConfig(_Strict):
    spot: float = Field(100.0, gt=0)
    rate: float = 0.03
    div_yield: float = 0.01
    curve: str | None = None
    as_of: str | None = None


class GBMConfig(_Strict):
    vol: float = Field(0.20, gt=0)


class HestonConfig(_Strict):
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    scheme: Literal["full_truncation", "qe"] = "full_truncation"


class LVGridConfig(_Strict):
    n_k: int = 64
    n_t: int = 48
    k_min: float = -1.0
    k_max: float = 1.0
    t_max: float = 2.0


class PhiConfig(_Strict):
    type: Literal["power_law", "heston_like"] = "power_law"
    eta: float = 1.0
    gamma: float = 0.5


class SyntheticSSVIConfig(_Strict):
    theta_per_year: float = 0.04
    rho: float = -0.7
    phi: PhiConfig = PhiConfig()


class LocalVolConfig(_Strict):
    surface: str = "synthetic"
    grid: LVGridConfig = LVGridConfig()
    synthetic_ssvi: SyntheticSSVIConfig = SyntheticSSVIConfig()


class ModelsConfig(_Strict):
    gbm: GBMConfig = GBMConfig()
    heston: HestonConfig = HestonConfig()
    localvol: LocalVolConfig = LocalVolConfig()


class SobolConfig(_Strict):
    scramble: bool = True
    replications: int = 32
    chunk_paths: int = 262144


class RngConfig(_Strict):
    kind: Literal["philox", "sobol-bb", "sobol", "threefry"] = "philox"
    antithetic: bool = True
    sobol: SobolConfig = SobolConfig()


class SimulationConfig(_Strict):
    n_paths: int = Field(1048576, gt=0)
    steps_per_year: int = Field(252, gt=0)
    checkpoint_every: int = Field(16, gt=0)


class FDConfig(_Strict):
    rel_bump: float = 1.0e-4
    scheme: Literal["central", "forward"] = "central"


class GreeksConfig(_Strict):
    fd: FDConfig = FDConfig()


class CudaConfig(_Strict):
    block_size: int = 256
    variant: Literal["fused", "store"] = "fused"
    arch: str | None = None


class RunConfig(_Strict):
    """Typed mirror of ``configs/default.yaml``."""

    seed: int = 20261030
    dtype: DType = "float32"
    backend: Backend = "jax"
    market: MarketConfig = MarketConfig()
    models: ModelsConfig = ModelsConfig()
    rng: RngConfig = RngConfig()
    simulation: SimulationConfig = SimulationConfig()
    products: dict[str, dict[str, Any]] = Field(default_factory=dict)
    greeks: GreeksConfig = GreeksConfig()
    cuda: CudaConfig = CudaConfig()


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping; an empty file is an empty mapping."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return loaded


def load_run_config(path: Path | None = None) -> RunConfig:
    """Load and validate a run config (default: ``configs/default.yaml``)."""
    return RunConfig.model_validate(load_yaml(path or CONFIG_DIR / "default.yaml"))


def config_hash(obj: BaseModel | dict[str, Any]) -> str:
    """Short SHA-256 of the canonical JSON form of a config, for manifests and the log."""
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
