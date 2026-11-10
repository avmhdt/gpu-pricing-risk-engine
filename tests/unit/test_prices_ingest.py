"""Stooq snapshot conversion and Sharadar table-API ingest (HTTP mocked)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx

from gpre.data import sharadar, stooq
from gpre.data.prices import load_universe
from gpre.data.rawstore import RawStore

# --------------------------------------------------------------------------------- Stooq


def test_stooq_name_mapping() -> None:
    assert stooq.stooq_name("SPY") == "spy.us"
    assert stooq.stooq_name("BRK.B") == "brk-b.us"


def test_index_resolve_and_ingest(stooq_tree: Path, small_universe: Path, tmp_path: Path) -> None:
    index = stooq.build_index(stooq_tree)
    assert stooq.resolve(index, "SPY").name == "spy.us.txt"
    assert "nyse etfs/2" in stooq.resolve(index, "SPY").as_posix()
    with pytest.raises(stooq.StooqError, match="not in the Stooq"):
        stooq.resolve(index, "ZZZ")
    dup = {"aaa.us": ["/a/aaa.us.txt", "/b/aaa.us.txt"]}
    with pytest.raises(stooq.StooqError, match="ambiguous"):
        stooq.resolve(dup, "AAA")

    store = RawStore(tmp_path / "data" / "raw")
    uni = load_universe(small_universe)
    out, meta = stooq.ingest_stooq(
        uni,
        stooq_tree,
        store,
        tmp_path / "interim",
        tmp_path / "proc",
        date(2024, 1, 2),
        date(2024, 12, 31),
    )
    p = pl.read_parquet(out)
    assert sorted(p["symbol"].unique().to_list()) == ["AAA", "BBB", "SPY"]
    assert p.height == 360 and p["source"].unique().to_list() == ["stooq"]
    assert p.group_by("symbol").agg(pl.col("log_ret").null_count())["log_ret"].to_list() == [1] * 3
    assert meta["snapshot_through"] == str(p["date"].max())
    assert (tmp_path / "interim" / stooq.INDEX_NAME).exists()
    assert len(store.load_manifest()) == 3
    # The cached index is reused while the snapshot directory is unchanged.
    assert stooq.load_or_build_index(stooq_tree, tmp_path / "interim") == index


def test_stooq_rejects_header_change_and_missing_tree(tmp_path: Path) -> None:
    bad = tmp_path / "bad.us.txt"
    bad.write_text("<TICKER>,<DATE>,<CLOSE>\nBAD.US,20240102,1\n")
    with pytest.raises(stooq.StooqError, match="header"):
        stooq.read_stooq_file(bad, "BAD")
    weekly = tmp_path / "w.us.txt"
    weekly.write_text(",".join(stooq.STOOQ_HEADER) + "\nW.US,W,20240102,000000,1,1,1,1,1,0\n")
    with pytest.raises(stooq.StooqError, match="daily"):
        stooq.read_stooq_file(weekly, "W")
    with pytest.raises(stooq.StooqError, match="d_us_txt"):
        stooq.build_index(tmp_path / "nowhere")


def test_universe_validation(tmp_path: Path) -> None:
    dup = tmp_path / "u.yaml"
    dup.write_text("history: {start: 2024-01-02, end: 2024-12-31}\nsymbols: [A, A]\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_universe(dup)
    stray = tmp_path / "v.yaml"
    stray.write_text("history: {start: 2024-01-02, end: 2024-12-31}\nfunds: [X]\nsymbols: [A]\n")
    with pytest.raises(ValueError, match="funds"):
        load_universe(stray)


def test_shipped_universe_has_spy_plus_49() -> None:
    from gpre.config import CONFIG_DIR

    uni = load_universe(CONFIG_DIR / "universe.yaml")
    assert len(uni.symbols) == 50 and uni.symbols[0] == "SPY" and uni.funds == {"SPY"}
    assert len(uni.stocks) == 49


# ------------------------------------------------------------------------------ Sharadar

HEADER = "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated"


def _rows(symbol: str, year: int, n: int = 3, base: float = 100.0) -> bytes:
    lines = [HEADER]
    for i in range(n, 0, -1):  # newest first, as the API answers
        c = base + i
        lines.append(f"{symbol},{year}-01-0{i + 1},{c},{c},{c},{c},1000,{c * 0.99},{c},2026-08-10")
    return ("\n".join(lines) + "\n").encode()


def _client() -> sharadar.SharadarClient:
    c = sharadar.SharadarClient(api_key="key-xyz")
    c._throttle = lambda: None  # type: ignore[method-assign]
    return c


def test_client_needs_key() -> None:
    with pytest.raises(sharadar.SharadarUnavailableError, match="ingest stooq"):
        sharadar.SharadarClient(api_key="")


@respx.mock
def test_ingest_sharadar_year_by_year(small_universe: Path, tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["x-api-key"] == "key-xyz"
        assert "key-xyz" not in str(request.url)
        ticker = request.url.params["ticker"]
        if "limit" in request.url.params:
            return httpx.Response(200, content=HEADER.encode())
        year = int(request.url.params["date.gte"][:4])
        return httpx.Response(200, content=_rows(ticker, year))

    respx.get(url__startswith=sharadar.BASE_URL).mock(side_effect=answer)
    store = RawStore(tmp_path / "raw")
    uni = load_universe(small_universe)
    out, meta = sharadar.ingest_sharadar(
        uni,
        _client(),
        store,
        tmp_path / "proc",
        date(2024, 1, 1),
        date(2024, 12, 31),
        day="2026-11-09",
    )
    p = pl.read_parquet(out)
    assert p.height == 9 and meta["source"] == "sharadar"
    assert {"close_adj", "close_unadj", "log_ret"} <= set(p.columns)
    tables = {r.url.path.rsplit("/", 1)[-1] for r in calls}
    assert tables == {"stocks", "funds"}
    assert "key-xyz" not in store.manifest_path.read_text()
    n_calls = len(calls)
    sharadar.ingest_sharadar(
        uni,
        _client(),
        store,
        tmp_path / "proc",
        date(2024, 1, 1),
        date(2024, 12, 31),
        day="2026-11-09",
    )
    assert len(calls) == n_calls + 2  # only the two key checks; raw files are reused


@respx.mock
def test_invalid_key_is_reported() -> None:
    respx.get(url__startswith=sharadar.BASE_URL).mock(return_value=httpx.Response(403))
    with pytest.raises(sharadar.SharadarUnavailableError, match="refused"):
        _client().check_key("stocks")
    with pytest.raises(sharadar.SharadarUnavailableError, match="403"):
        _client().fetch_year("stocks", "AAPL", 2024)


@respx.mock
def test_retry_on_server_error() -> None:
    sleeps: list[float] = []
    c = _client()
    c.sleep = sleeps.append
    respx.get(url__startswith=sharadar.BASE_URL).mock(
        side_effect=[
            httpx.Response(502),
            httpx.ConnectError("x"),
            httpx.Response(200, content=_rows("AAPL", 2024)),
        ]
    )
    _url, body = c.fetch_year("stocks", "AAPL", 2024)
    assert body.startswith(HEADER.encode()) and len(sleeps) == 2


def test_parse_guards() -> None:
    with pytest.raises(sharadar.SharadarUnavailableError, match="columns"):
        sharadar.parse_csv(b"ticker,date\nA,2024-01-02\n", "A")
    with pytest.raises(sharadar.SharadarUnavailableError, match="other tickers"):
        sharadar.parse_csv(_rows("B", 2024), "A")
    big = [HEADER] + ["A,2024-01-02,1,1,1,1,1,1,1,2026-01-01"] * sharadar.ROW_CAP
    with pytest.raises(sharadar.SharadarUnavailableError, match="cap"):
        sharadar.parse_csv(("\n".join(big)).encode(), "A")
