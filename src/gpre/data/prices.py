"""Shared helpers for daily price tables (``prices_daily.parquet``, §7b)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from gpre.config import load_yaml

PRICE_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Universe:
    """The VaR universe (D-14): symbols and which Sharadar table each lives in."""

    symbols: tuple[str, ...]
    funds: frozenset[str]
    start: str
    end: str

    @property
    def stocks(self) -> tuple[str, ...]:
        return tuple(s for s in self.symbols if s not in self.funds)


def load_universe(path: Path) -> Universe:
    """Read ``configs/universe.yaml``."""
    raw = load_yaml(path)
    symbols = tuple(str(s).upper() for s in raw["symbols"])
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"{path}: duplicate symbols")
    funds = frozenset(str(s).upper() for s in raw.get("funds", []))
    if not funds <= set(symbols):
        raise ValueError(f"{path}: funds must be listed in symbols")
    return Universe(symbols, funds, str(raw["history"]["start"]), str(raw["history"]["end"]))


def finalize_prices(df: pl.DataFrame, source: str) -> pl.DataFrame:
    """Drop non-positive closes (DQ-03), sort, add ``log_ret`` and ``source``.

    ``log_ret`` is ln(close_t / close_{t-1}) per symbol, null on each symbol's first row.
    """
    out = (
        df.filter(pl.col("close") > 0)
        .unique(subset=["symbol", "date"], keep="first", maintain_order=True)
        .sort(["symbol", "date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol")).log().alias("log_ret"),
            pl.lit(source).alias("source"),
        )
    )
    return out
