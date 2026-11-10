"""Sharadar SEP (stocks) and SFP (funds) daily prices via Sharadar's direct API (FR-35, D-15).

The key travels only in the ``x-api-key`` header, never in a URL or a log. The table API answers
in CSV, newest row first, and truncates long answers (about 1,000 rows were seen on
2026-11-09), so each symbol is requested one calendar year at a time; a year that comes back at
the cap is treated as an error rather than silently cut.

Vendor facts (checked by equity-factor-pipeline on the Full History Bundle, 2026-11-02):
``open``/``high``/``low``/``close``/``volume`` are split-adjusted, ``closeadj`` is split- and
dividend-adjusted, ``closeunadj`` is raw. Options are written on the price, not the total
return, so ``close`` (split-adjusted) feeds ``log_ret``; ``closeadj`` is kept as ``close_adj``.

AAPL and MSFT (and SPY in SFP) answer without a valid key, so the key is checked first with a
symbol outside the free tier.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from gpre.data.prices import Universe, finalize_prices
from gpre.data.rawstore import RawStore
from gpre.logging import get_logger

log = get_logger(__name__)

BASE_URL = "https://api.sharadar.com/v1.0"
TABLE_STOCKS = "stocks"  # SEP
TABLE_FUNDS = "funds"  # SFP
KEY_CHECK = {TABLE_STOCKS: "GME", TABLE_FUNDS: "XLE"}
ROW_CAP = 1000
EXPECTED_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "closeadj",
    "closeunadj",
    "lastupdated",
]


class SharadarUnavailableError(RuntimeError):
    """No key, an invalid key, or a table the subscription does not include."""


@dataclass
class SharadarClient:
    """Throttled table-API client (at most ``max_rps`` requests per second)."""

    api_key: str
    transport: httpx.BaseTransport | None = None
    max_rps: float = 4.0
    sleep: Callable[[float], None] = time.sleep
    _last: float = 0.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise SharadarUnavailableError(
                "SHARADAR_API_KEY is not set; run `gpre ingest stooq` for the local snapshot"
            )

    def _throttle(self) -> None:
        wait = self._last + 1.0 / self.max_rps - time.monotonic()
        if wait > 0:
            self.sleep(wait)
        self._last = time.monotonic()

    def _get(self, table: str, params: dict[str, str | int]) -> httpx.Response:
        self._throttle()
        with httpx.Client(
            base_url=BASE_URL,
            headers={"x-api-key": self.api_key},
            transport=self.transport,
            timeout=60.0,
            follow_redirects=False,
        ) as client:
            for attempt in range(4):
                try:
                    resp = client.get(f"/data/{table}", params=params)
                except httpx.TransportError as exc:
                    if attempt == 3:
                        raise RuntimeError(f"Sharadar {table}: {type(exc).__name__}") from None
                    self.sleep(2.0 * 2**attempt)
                    continue
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                    self.sleep(2.0 * 2**attempt)
                    continue
                return resp
        raise AssertionError("unreachable")  # pragma: no cover

    def check_key(self, table: str) -> None:
        """Fail unless the key can read a symbol outside the free tier of ``table``."""
        resp = self._get(table, {"ticker": KEY_CHECK[table], "limit": 1})
        if resp.status_code in (401, 403):
            raise SharadarUnavailableError(
                f"Sharadar refused the key for table {table!r} (HTTP {resp.status_code}); "
                "check SHARADAR_API_KEY or run `gpre ingest stooq`"
            )
        resp.raise_for_status()

    def fetch_year(self, table: str, symbol: str, year: int) -> tuple[str, bytes]:
        """CSV body of one symbol-year; returns ``(url_without_key, body)``."""
        params: dict[str, str | int] = {
            "ticker": symbol,
            "date.gte": f"{year}-01-01",
            "date.lte": f"{year}-12-31",
        }
        resp = self._get(table, params)
        if resp.status_code in (401, 403):
            raise SharadarUnavailableError(f"{table}/{symbol}: HTTP {resp.status_code}")
        resp.raise_for_status()
        return str(resp.request.url), resp.content


def parse_csv(body: bytes, symbol: str) -> pl.DataFrame:
    """Parse one table-API CSV answer into the price columns plus ``close_adj``/``close_unadj``."""
    frame = pl.read_csv(io.BytesIO(body), infer_schema=False)
    if frame.columns != EXPECTED_COLUMNS:
        raise SharadarUnavailableError(f"{symbol}: unexpected columns {frame.columns}")
    if frame.height >= ROW_CAP:
        raise SharadarUnavailableError(f"{symbol}: answer hit the {ROW_CAP}-row cap")
    tickers = set(frame["ticker"].unique().to_list())
    if tickers - {symbol}:
        raise SharadarUnavailableError(f"{symbol}: answer holds other tickers {tickers}")
    num = ("open", "high", "low", "close", "volume", "closeadj", "closeunadj")
    return frame.select(
        pl.col("date").str.to_date("%Y-%m-%d"),
        pl.lit(symbol).alias("symbol"),
        *(pl.col(c).cast(pl.Float64) for c in num),
    ).rename({"closeadj": "close_adj", "closeunadj": "close_unadj"})


def ingest_sharadar(
    universe: Universe,
    client: SharadarClient,
    store: RawStore,
    processed_dir: Path,
    start: date,
    end: date,
    day: str,
) -> tuple[Path, dict[str, str]]:
    """Fetch the universe year by year (raw CSV kept per DQ-09) and write ``prices_daily``."""
    tables = {TABLE_FUNDS} if universe.funds else set()
    if universe.stocks:
        tables.add(TABLE_STOCKS)
    for table in sorted(tables):
        client.check_key(table)
    frames = []
    for symbol in universe.symbols:
        table = TABLE_FUNDS if symbol in universe.funds else TABLE_STOCKS
        for year in range(start.year, end.year + 1):
            path = store.path_for("sharadar", day, f"{table}_{symbol}_{year}.csv")
            if path.exists():
                body = path.read_bytes()
            else:
                url, body = client.fetch_year(table, symbol, year)
                store.write(path, body, url=url, source="sharadar", table=table, symbol=symbol)
            frames.append(parse_csv(body, symbol))
        log.info("sharadar.symbol_done", symbol=symbol, table=table)
    prices = finalize_prices(
        pl.concat(frames).filter(pl.col("date").is_between(start, end)), source="sharadar"
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "prices_daily.parquet"
    prices.write_parquet(out)
    meta = {"source": "sharadar", "close": "split-adjusted", "close_adj": "split+dividend"}
    (processed_dir / "prices_daily.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out, meta
