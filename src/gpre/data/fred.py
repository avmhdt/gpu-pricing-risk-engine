"""FRED ingest: Treasury constant-maturity yields and VIXCLS (FR-22, §7e).

With ``FRED_API_KEY`` set, the v1 ``series/observations`` API is used; FRED accepts the key only
as the ``api_key`` query parameter, so URLs are redacted before they are logged or stored.
Without a key the public ``fredgraph.csv`` endpoint is used. Requests are throttled to at most
one per second.

DQ-02: FRED's missing values ("." or empty) become null and are forward-filled for at most 3
business days; a longer gap fails the ingest. CMT par yields are treated as zero rates and
converted to continuous compounding with ``2 ln(1 + y/200)`` (documented approximation).
"""

from __future__ import annotations

import io
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from gpre.data.rawstore import RawStore
from gpre.logging import get_logger

log = get_logger(__name__)

CMT_SERIES: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}
"""The 11 CMT series and their tenors in years (1M = 1/12)."""

VIX_SERIES = "VIXCLS"
MAX_FFILL_DAYS = 3
FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
MIN_INTERVAL_S = 1.0


class FredDataError(RuntimeError):
    """A FRED series failed a data-quality rule (DQ-02)."""


def redact(url: str) -> str:
    """Remove the ``api_key`` value from a URL."""
    parsed = httpx.URL(url)
    if "api_key" not in parsed.params:
        return url
    return str(parsed.copy_set_param("api_key", "REDACTED"))


@dataclass
class FredClient:
    """Downloads one series at a time, at most one request per ``min_interval_s``."""

    api_key: str = ""
    transport: httpx.BaseTransport | None = None
    min_interval_s: float = MIN_INTERVAL_S
    sleep: Callable[[float], None] = time.sleep
    _last: float = 0.0

    def _throttle(self) -> None:
        wait = self._last + self.min_interval_s - time.monotonic()
        if wait > 0:
            self.sleep(wait)
        self._last = time.monotonic()

    def request(self, series_id: str, start: date, end: date) -> tuple[str, bytes, str]:
        """Return ``(redacted_url, body, kind)`` where kind is ``csv`` or ``json``."""
        if self.api_key:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start.isoformat(),
                "observation_end": end.isoformat(),
            }
            url, kind = FRED_API_URL, "json"
        else:
            params = {"id": series_id, "cosd": start.isoformat(), "coed": end.isoformat()}
            url, kind = FREDGRAPH_URL, "csv"
        self._throttle()
        with httpx.Client(transport=self.transport, timeout=60.0, follow_redirects=True) as c:
            for attempt in range(4):
                try:
                    resp = c.get(url, params=params)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError(
                            "retryable", request=resp.request, response=resp
                        )
                    resp.raise_for_status()
                    return redact(str(resp.request.url)), resp.content, kind
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status is not None and status not in (429, 500, 502, 503, 504):
                        raise RuntimeError(f"FRED {series_id}: HTTP {status}") from None
                    if attempt == 3:
                        raise RuntimeError(f"FRED {series_id}: {type(exc).__name__}") from None
                    self.sleep(2.0 * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover


def parse_series(body: bytes, kind: str, series_id: str) -> pl.DataFrame:
    """Parse a fredgraph CSV or API JSON body into ``(date, value)`` with nulls for missing."""
    if kind == "json":
        obs = json.loads(body)["observations"]
        dates = [o["date"] for o in obs]
        values = [o["value"] for o in obs]
    else:
        frame = pl.read_csv(io.BytesIO(body), infer_schema=False)
        date_col = next(c for c in frame.columns if c.lower() in ("date", "observation_date"))
        value_col = next(c for c in frame.columns if c != date_col)
        if value_col != series_id:
            raise FredDataError(f"{series_id}: unexpected CSV column {value_col!r}")
        dates = frame[date_col].to_list()
        values = frame[value_col].to_list()
    df = pl.DataFrame({"date": dates, "raw": values}, schema={"date": pl.Utf8, "raw": pl.Utf8})
    return df.select(
        pl.col("date").str.to_date("%Y-%m-%d"),
        pl.when(pl.col("raw").is_in([".", ""]) | pl.col("raw").is_null())
        .then(None)
        .otherwise(pl.col("raw"))
        .cast(pl.Float64)
        .alias("value"),
    ).sort("date")


def forward_fill_limited(
    df: pl.DataFrame, series_id: str, limit: int = MAX_FFILL_DAYS
) -> pl.DataFrame:
    """Forward-fill nulls for at most ``limit`` consecutive rows (business days); fail beyond.

    Leading nulls (before the first observation) are dropped, not filled.

    Raises:
        FredDataError: if any run of nulls after the first observation is longer than ``limit``.
    """
    df = df.sort("date")
    first = df.filter(pl.col("value").is_not_null()).select(pl.col("date").min()).item()
    if first is None:
        raise FredDataError(f"{series_id}: no observations")
    df = df.filter(pl.col("date") >= first)
    is_null = df["value"].is_null().to_list()
    trailing = 0
    for flag in reversed(is_null):
        if not flag:
            break
        trailing += 1
    longest = run = 0
    for flag in is_null[: len(is_null) - trailing]:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    if longest > limit:
        raise FredDataError(f"{series_id}: gap of {longest} business days exceeds {limit} (DQ-02)")
    if trailing > limit:
        df = df.head(df.height - trailing)
    return df.with_columns(
        pl.col("value").is_null().alias("filled"),
        pl.col("value").forward_fill(limit=limit),
    )


def zero_cc(yield_pct: float) -> float:
    """CMT yield in percent (semiannual bond-equivalent) to a continuously compounded decimal."""
    return 2.0 * math.log1p(yield_pct / 200.0)


def build_yields(series: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Long yields table per §7b from the per-series ``(date, value, filled)`` frames."""
    parts = []
    for sid, frame in series.items():
        parts.append(
            frame.select(
                pl.col("date"),
                pl.lit(sid).alias("series_id"),
                pl.lit(CMT_SERIES[sid]).alias("tenor_years"),
                pl.col("value").alias("yield_pct"),
                (2.0 * (pl.col("value") / 200.0).log1p()).alias("zero_cc"),
                pl.col("filled"),
            )
        )
    return pl.concat(parts).sort(["date", "tenor_years"])


def ingest_fred(
    store: RawStore,
    processed_dir: Path,
    start: date,
    end: date,
    client: FredClient,
    day: str,
) -> dict[str, Path]:
    """Download (or reuse today's raw files), validate and write yields and VIX Parquet.

    Raw files for ``day`` that already exist are reused rather than downloaded again, which
    keeps them immutable (DQ-09) and avoids repeat traffic.
    """
    frames: dict[str, pl.DataFrame] = {}
    for sid in [*CMT_SERIES, VIX_SERIES]:
        ext = "json" if client.api_key else "csv"
        path = store.path_for("fred", day, f"{sid}_{start}_{end}.{ext}")
        if path.exists():
            body, kind = path.read_bytes(), ext
            log.info("fred.reuse_raw", series=sid, path=str(path))
        else:
            url, body, kind = client.request(sid, start, end)
            store.write(path, body, url=url, source="fred", series_id=sid)
            log.info("fred.downloaded", series=sid, bytes=len(body))
        frames[sid] = forward_fill_limited(parse_series(body, kind, sid), sid)

    processed_dir.mkdir(parents=True, exist_ok=True)
    yields = build_yields({k: v for k, v in frames.items() if k in CMT_SERIES})
    vix = frames[VIX_SERIES].select("date", pl.col("value").alias("vix"), "filled")
    out = {"yields": processed_dir / "yields.parquet", "vix": processed_dir / "vix.parquet"}
    yields.write_parquet(out["yields"])
    vix.write_parquet(out["vix"])
    return out
