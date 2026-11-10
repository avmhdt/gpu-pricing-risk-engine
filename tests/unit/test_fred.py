"""FRED parsing, DQ-02 limited forward fill, key redaction and the ingest (HTTP mocked)."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx

from conftest import FIXTURES
from gpre.data import fred
from gpre.data.rawstore import RawStore


def _csv(series: str, values: list[str], start: date = date(2024, 1, 2)) -> bytes:
    days = pl.date_range(start, date(2025, 12, 31), "1d", eager=True)
    days = [d for d in days.to_list() if d.weekday() < 5][: len(values)]
    rows = [f"observation_date,{series}"] + [f"{d},{v}" for d, v in zip(days, values, strict=True)]
    return ("\n".join(rows) + "\n").encode()


def test_parse_fredgraph_csv_fixture() -> None:
    df = fred.parse_series((FIXTURES / "fred_small.csv").read_bytes(), "csv", "DGS10")
    assert df.columns == ["date", "value"] and df.height == 40
    assert df["value"].null_count() == 3


def test_parse_api_json_and_dot_missing() -> None:
    body = json.dumps(
        {
            "observations": [
                {"date": "2024-01-02", "value": "4.1"},
                {"date": "2024-01-03", "value": "."},
            ]
        }
    ).encode()
    df = fred.parse_series(body, "json", "DGS10")
    assert df["value"].to_list() == [4.1, None]


def test_parse_rejects_wrong_series_column() -> None:
    with pytest.raises(fred.FredDataError, match="unexpected"):
        fred.parse_series(_csv("DGS2", ["1.0"]), "csv", "DGS10")


def test_forward_fill_limited() -> None:
    df = fred.parse_series(_csv("X", ["", "1", "", "", "", "2", "3", "", ""]), "csv", "X")
    out = fred.forward_fill_limited(df, "X")
    assert out["value"].to_list() == [1, 1, 1, 1, 2, 3, 3, 3]
    assert out["filled"].to_list() == [False, True, True, True, False, False, True, True]
    gap = fred.parse_series(_csv("X", ["1", "", "", "", "", "2"]), "csv", "X")
    with pytest.raises(fred.FredDataError, match="gap of 4"):
        fred.forward_fill_limited(gap, "X")
    tail = fred.parse_series(_csv("X", ["1", "2", "", "", "", ""]), "csv", "X")
    assert fred.forward_fill_limited(tail, "X")["value"].to_list() == [1, 2]
    with pytest.raises(fred.FredDataError, match="no observations"):
        fred.forward_fill_limited(fred.parse_series(_csv("X", ["", ""]), "csv", "X"), "X")


def test_zero_cc_and_redact() -> None:
    assert math.isclose(fred.zero_cc(4.0), 2 * math.log(1.02))
    url = "https://api.stlouisfed.org/fred/series/observations?series_id=X&api_key=SECRET1"
    assert "SECRET1" not in fred.redact(url) and "REDACTED" in fred.redact(url)
    assert fred.redact("https://x.org/a?b=1") == "https://x.org/a?b=1"


@respx.mock
def test_ingest_without_key_uses_fredgraph(tmp_path: Path) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        sid = request.url.params["id"]
        values = ["5", "", "5.1", "5.2"] if sid != "VIXCLS" else ["15", "16", "", "17"]
        return httpx.Response(200, content=_csv(sid, values))

    route = respx.get(fred.FREDGRAPH_URL).mock(side_effect=answer)
    store = RawStore(tmp_path / "raw")
    client = fred.FredClient(min_interval_s=0.0)
    out = fred.ingest_fred(
        store, tmp_path / "proc", date(2024, 1, 1), date(2024, 1, 31), client, day="2026-11-09"
    )
    assert route.call_count == 12
    y = pl.read_parquet(out["yields"])
    assert y["series_id"].n_unique() == 11 and y.height == 44
    assert set(y.columns) == {"date", "series_id", "tenor_years", "yield_pct", "zero_cc", "filled"}
    assert math.isclose(y.filter(pl.col("series_id") == "DGS1MO")["tenor_years"][0], 1 / 12)
    v = pl.read_parquet(out["vix"])
    assert v["vix"].to_list() == [15, 16, 16, 17]
    assert len(store.load_manifest()) == 12 and store.verify() == []
    # A second run on the same day reuses the immutable raw files: no new requests.
    fred.ingest_fred(
        store, tmp_path / "proc", date(2024, 1, 1), date(2024, 1, 31), client, day="2026-11-09"
    )
    assert route.call_count == 12


@respx.mock
def test_api_path_sends_key_but_never_stores_it(tmp_path: Path) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "k-123"
        obs = [{"date": "2024-01-02", "value": "4.0"}, {"date": "2024-01-03", "value": "4.1"}]
        return httpx.Response(200, json={"observations": obs})

    respx.get(fred.FRED_API_URL).mock(side_effect=answer)
    store = RawStore(tmp_path / "raw")
    client = fred.FredClient(api_key="k-123", min_interval_s=0.0)
    fred.ingest_fred(
        store, tmp_path / "proc", date(2024, 1, 1), date(2024, 1, 3), client, day="2026-11-09"
    )
    manifest_text = store.manifest_path.read_text()
    assert "k-123" not in manifest_text and "REDACTED" in manifest_text


@respx.mock
def test_client_retries_then_fails() -> None:
    sleeps: list[float] = []
    route = respx.get(fred.FREDGRAPH_URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, content=_csv("DGS10", ["1"]))]
    )
    client = fred.FredClient(min_interval_s=0.0, sleep=sleeps.append)
    _url, body, kind = client.request("DGS10", date(2024, 1, 1), date(2024, 1, 2))
    assert kind == "csv" and b"DGS10" in body and route.call_count == 2 and sleeps
    respx.get(fred.FREDGRAPH_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(RuntimeError, match="HTTP 404"):
        client.request("DGS10", date(2024, 1, 1), date(2024, 1, 2))
    respx.get(fred.FREDGRAPH_URL).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(RuntimeError, match="ConnectError"):
        client.request("DGS10", date(2024, 1, 1), date(2024, 1, 2))


def test_throttle_waits_between_requests() -> None:
    sleeps: list[float] = []
    client = fred.FredClient(min_interval_s=10.0, sleep=sleeps.append)
    client._throttle()
    client._throttle()
    assert len(sleeps) == 1 and sleeps[0] > 9.0
