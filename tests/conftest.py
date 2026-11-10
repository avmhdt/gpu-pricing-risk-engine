"""Shared fixtures: an isolated data root, results dir and settings for every test."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from gpre.config import get_settings
from gpre.logging import configure_logging

FIXTURES = Path(__file__).parent / "fixtures"

_ENV_KEYS = (
    "FRED_API_KEY",
    "SHARADAR_API_KEY",
    "TIINGO_API_KEY",
    "GPRE_CLOUD_API_KEY",
    "GPRE_CLOUD_PROVIDER",
    "DATA_ROOT",
    "GPRE_DATA_DIR",
    "GPRE_RESULTS_DIR",
    "GPRE_REPORTS_DIR",
    "GPRE_SURFACE_PATH",
    "STOOQ_DIR",
    "GPRE_CPP_BASELINE",
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point every path setting into ``tmp_path`` and run from there (no stray ``.env``)."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("GPRE_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("GPRE_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("STOOQ_DIR", str(tmp_path / "stooq"))
    monkeypatch.setenv("GPRE_CPP_BASELINE", str(tmp_path / "no_such_binary"))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    configure_logging("INFO", None)
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def stooq_tree(tmp_path: Path) -> Path:
    """A miniature Stooq snapshot: SPY in a numbered ETF subfolder, two stocks elsewhere."""
    root = tmp_path / "stooq"
    lines = (FIXTURES / "stooq_small.csv").read_text().splitlines()
    header, rows = lines[0], lines[1:]
    places = {
        "SPY.US": root / "d_us_txt/data/daily/us/nyse etfs/2/spy.us.txt",
        "AAA.US": root / "d_us_txt/data/daily/us/nasdaq stocks/1/aaa.us.txt",
        "BBB.US": root / "d_us_txt/data/daily/us/nyse stocks/bbb.us.txt",
    }
    for ticker, path in places.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        body = [r for r in rows if r.startswith(ticker + ",")]
        path.write_text("\n".join([header, *body]) + "\n")
    return root


@pytest.fixture
def small_universe(tmp_path: Path) -> Path:
    path = tmp_path / "universe.yaml"
    path.write_text(
        "history: {start: 2024-01-02, end: 2024-12-31}\nfunds: [SPY]\nsymbols: [SPY, AAA, BBB]\n"
    )
    return path


def pytest_report_header() -> str:
    return f"JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64', '')}"
