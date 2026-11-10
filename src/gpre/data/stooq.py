"""Stooq fallback and cross-check ingest from the owner's local snapshot (FR-23, P-07).

Nothing is downloaded. The snapshot tree under ``STOOQ_DIR`` is walked once to build a
ticker -> path index (paths are never guessed; ETFs sit in numbered subfolders such as
``nyse etfs/2/spy.us.txt``). The index is cached under ``interim/``, the universe is converted
once to Parquet under the data root, and the snapshot's last date is recorded in the manifest.

Stooq files are ``<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>``
with ``DATE`` as ``YYYYMMDD``. A header change fails loudly.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import polars as pl

from gpre.data.prices import Universe, finalize_prices
from gpre.data.rawstore import RawStore, sha256_file
from gpre.logging import get_logger

log = get_logger(__name__)

STOOQ_HEADER = [
    "<TICKER>",
    "<PER>",
    "<DATE>",
    "<TIME>",
    "<OPEN>",
    "<HIGH>",
    "<LOW>",
    "<CLOSE>",
    "<VOL>",
    "<OPENINT>",
]
INDEX_NAME = "stooq_index.json"


class StooqError(RuntimeError):
    """The snapshot is missing a ticker, has an ambiguous ticker or an unexpected format."""


def stooq_name(symbol: str) -> str:
    """Upper-case ticker (``BRK.B``) to the Stooq US file stem (``brk-b.us``)."""
    return symbol.lower().replace(".", "-") + ".us"


def build_index(stooq_dir: Path) -> dict[str, list[str]]:
    """Walk ``d_us_txt`` once and map each file stem (``spy.us``) to every path that has it."""
    root = stooq_dir / "d_us_txt"
    if not root.is_dir():
        raise StooqError(f"no d_us_txt under {stooq_dir} (set STOOQ_DIR)")
    index: dict[str, list[str]] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".txt"):
                index.setdefault(name[: -len(".txt")].lower(), []).append(
                    os.path.join(dirpath, name)
                )
    return index


def load_or_build_index(stooq_dir: Path, interim_dir: Path) -> dict[str, list[str]]:
    """Cached index keyed by the snapshot directory; rebuilt if the directory changes."""
    cache = interim_dir / INDEX_NAME
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("stooq_dir") == str(stooq_dir):
            index: dict[str, list[str]] = cached["index"]
            return index
    index = build_index(stooq_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"stooq_dir": str(stooq_dir), "index": index}), encoding="utf-8")
    return index


def resolve(index: dict[str, list[str]], symbol: str) -> Path:
    """Path of ``symbol``'s file; fails on a missing or ambiguous ticker."""
    paths = index.get(stooq_name(symbol), [])
    if not paths:
        raise StooqError(f"{symbol}: not in the Stooq snapshot ({stooq_name(symbol)}.txt)")
    if len(paths) > 1:
        raise StooqError(f"{symbol}: ambiguous, found {sorted(paths)}")
    return Path(paths[0])


def read_stooq_file(path: Path, symbol: str) -> pl.DataFrame:
    """Read one Stooq daily file into the price columns."""
    frame = pl.read_csv(path, infer_schema=False)
    if frame.columns != STOOQ_HEADER:
        raise StooqError(f"{path}: unexpected header {frame.columns}")
    if frame.height and set(frame["<PER>"].unique().to_list()) != {"D"}:
        raise StooqError(f"{path}: not a daily file")
    return frame.select(
        pl.col("<DATE>").str.to_date("%Y%m%d").alias("date"),
        pl.lit(symbol).alias("symbol"),
        *(
            pl.col(f"<{c.upper()}>").cast(pl.Float64).alias(c)
            for c in ("open", "high", "low", "close")
        ),
        pl.col("<VOL>").cast(pl.Float64).alias("volume"),
    )


def ingest_stooq(
    universe: Universe,
    stooq_dir: Path,
    store: RawStore,
    interim_dir: Path,
    processed_dir: Path,
    start: date,
    end: date,
) -> tuple[Path, dict[str, str]]:
    """Convert the universe to ``prices_daily.parquet``; return the path and per-symbol notes."""
    index = load_or_build_index(stooq_dir, interim_dir)
    frames = []
    snapshot_last = None
    for symbol in universe.symbols:
        path = resolve(index, symbol)
        full = read_stooq_file(path, symbol)
        last = full.select(pl.col("date").max()).item()
        snapshot_last = last if snapshot_last is None else max(snapshot_last, last)
        store.record(
            path,
            url=f"file://{path.as_posix()}",
            sha256=sha256_file(path),
            source="stooq",
            symbol=symbol,
            last_date=str(last),
        )
        frames.append(full.filter(pl.col("date").is_between(start, end)))
    prices = finalize_prices(pl.concat(frames), source="stooq")
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "prices_daily.parquet"
    prices.write_parquet(out)
    meta = {"source": "stooq", "snapshot_through": str(snapshot_last), "stooq_dir": str(stooq_dir)}
    (processed_dir / "prices_daily.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("stooq.converted", rows=prices.height, symbols=len(universe.symbols), **meta)
    return out, meta
