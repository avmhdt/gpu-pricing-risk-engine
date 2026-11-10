"""The ``gpre`` CLI end to end on fixtures, with HTTP mocked (§11 integration)."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from conftest import FIXTURES
from gpre import __version__
from gpre.cli import app
from gpre.config import get_settings
from gpre.data import fred, sharadar

runner = CliRunner()


def _fred_csv(series: str) -> bytes:
    lines = [f"observation_date,{series}"]
    for day in range(2, 12):
        d = date(2024, 1, day)
        if d.weekday() < 5:
            lines.append(f"{d},{4.0 + day / 100}")
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture
def no_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fred.FredClient, "_throttle", lambda self: None)
    monkeypatch.setattr(sharadar.SharadarClient, "_throttle", lambda self: None)


def _log_rows(tmp_path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader((tmp_path / "results" / "experiment_log.csv").open()))


def test_version_and_help() -> None:
    assert runner.invoke(app, ["--version"]).stdout.strip() == __version__
    out = runner.invoke(app, ["--help"]).stdout
    for cmd in ("env", "ingest", "data", "surface"):
        assert cmd in out


def test_env_writes_hardware_stamp(tmp_path: Path) -> None:
    res = runner.invoke(app, ["env"])
    assert res.exit_code == 0, res.output
    files = list((tmp_path / "results" / "hardware").glob("*.json"))
    assert len(files) == 1
    stamp = json.loads(files[0].read_text())
    assert stamp["platform"] == "local" and "cpu_model" in stamp
    assert _log_rows(tmp_path)[0]["command"] == "gpre env"
    assert list((tmp_path / "results" / "logs").glob("*.jsonl"))


@pytest.mark.parametrize("req", ["gpu", "cuda", "cpp"])
def test_env_require_fails_without_backend(tmp_path: Path, req: str) -> None:
    res = runner.invoke(app, ["env", "--require", req])
    assert res.exit_code == 2
    assert "error:" in res.output


def test_env_rejects_unknown_requirement() -> None:
    assert runner.invoke(app, ["env", "--require", "tpu"]).exit_code == 2


@respx.mock
def test_ingest_fred_then_data_check(tmp_path: Path, no_throttle: None) -> None:
    respx.get(fred.FREDGRAPH_URL).mock(
        side_effect=lambda r: httpx.Response(200, content=_fred_csv(r.url.params["id"]))
    )
    res = runner.invoke(app, ["ingest", "fred", "--start", "2024-01-01", "--end", "2024-01-31"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "data" / "processed" / "yields.parquet").exists()
    assert "run_id" in res.output and "manifest" in res.output
    # Prices are missing, so a strict check fails while a plain check passes.
    assert runner.invoke(app, ["data", "check"]).exit_code == 0
    strict = runner.invoke(app, ["data", "check", "--strict"])
    assert strict.exit_code == 1
    payload = json.loads((tmp_path / "results" / "data_check.json").read_text())
    statuses = {c["rule"]: c["status"] for c in payload["checks"]}
    assert statuses["DQ-02"] == "pass" and statuses["DQ-03"] == "skip"
    assert statuses["DQ-05"] == "pass" and statuses["DQ-09"] == "pass"


@respx.mock
def test_ingest_fred_reports_http_errors(no_throttle: None) -> None:
    respx.get(fred.FREDGRAPH_URL).mock(return_value=httpx.Response(400))
    res = runner.invoke(app, ["ingest", "fred"])
    assert res.exit_code == 2 and "HTTP 400" in res.output


def test_ingest_stooq_and_strict_check(
    tmp_path: Path, stooq_tree: Path, small_universe: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    res = runner.invoke(app, ["ingest", "stooq", "--universe", str(small_universe)])
    assert res.exit_code == 0, res.output
    meta = json.loads((tmp_path / "data" / "processed" / "prices_daily.meta.json").read_text())
    assert meta["source"] == "stooq"
    missing = tmp_path / "u2.yaml"
    missing.write_text("history: {start: 2024-01-02, end: 2024-12-31}\nsymbols: [SPY, ZZZ]\n")
    bad = runner.invoke(app, ["ingest", "stooq", "--universe", str(missing)])
    assert bad.exit_code == 2 and "ZZZ" in bad.output
    # With yields as well, the strict check passes on the fixtures.
    monkeypatch.setattr(fred.FredClient, "_throttle", lambda self: None)
    with respx.mock:
        respx.get(fred.FREDGRAPH_URL).mock(
            side_effect=lambda r: httpx.Response(200, content=_fred_csv(r.url.params["id"]))
        )
        assert runner.invoke(app, ["ingest", "fred"]).exit_code == 0
    strict = runner.invoke(app, ["data", "check", "--strict"])
    assert strict.exit_code == 0, strict.output
    payload = json.loads((tmp_path / "results" / "data_check.json").read_text())
    assert payload["passed"] and payload["price_source"]["source"] == "stooq"
    assert "stooq_dir" not in payload["price_source"]


def test_ingest_sharadar_without_key_points_to_stooq(small_universe: Path) -> None:
    res = runner.invoke(app, ["ingest", "sharadar", "--universe", str(small_universe)])
    assert res.exit_code == 2 and "gpre ingest stooq" in res.output


@respx.mock
def test_ingest_sharadar_with_key(
    tmp_path: Path, small_universe: Path, monkeypatch: pytest.MonkeyPatch, no_throttle: None
) -> None:
    monkeypatch.setenv("SHARADAR_API_KEY", "k-abc")
    get_settings.cache_clear()
    header = "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated"

    def answer(request: httpx.Request) -> httpx.Response:
        t = request.url.params["ticker"]
        if "limit" in request.url.params:
            return httpx.Response(200, content=header.encode())
        y = request.url.params["date.gte"][:4]
        body = f"{header}\n{t},{y}-03-02,10,10,10,10,5,9.9,10,2026-01-01\n"
        return httpx.Response(200, content=body.encode())

    respx.get(url__startswith=sharadar.BASE_URL).mock(side_effect=answer)
    res = runner.invoke(app, ["ingest", "sharadar", "--universe", str(small_universe)])
    assert res.exit_code == 0, res.output
    assert "k-abc" not in (tmp_path / "data" / "raw" / "MANIFEST.json").read_text()
    assert "k-abc" not in (tmp_path / "results" / "experiment_log.csv").read_text()


@respx.mock
def test_ingest_all_falls_back_to_stooq(
    tmp_path: Path, stooq_tree: Path, small_universe: Path, no_throttle: None
) -> None:
    respx.get(fred.FREDGRAPH_URL).mock(
        side_effect=lambda r: httpx.Response(200, content=_fred_csv(r.url.params["id"]))
    )
    res = runner.invoke(app, ["ingest", "all", "--universe", str(small_universe)])
    assert res.exit_code == 0, res.output
    assert "using the Stooq snapshot" in res.output
    commands = [r["command"] for r in _log_rows(tmp_path)]
    assert commands[0].startswith("gpre ingest fred") and commands[1] == "gpre ingest stooq"


@respx.mock
def test_ingest_all_falls_back_when_sharadar_refuses(
    tmp_path: Path,
    stooq_tree: Path,
    small_universe: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_throttle: None,
) -> None:
    monkeypatch.setenv("SHARADAR_API_KEY", "bad")
    get_settings.cache_clear()
    respx.get(fred.FREDGRAPH_URL).mock(
        side_effect=lambda r: httpx.Response(200, content=_fred_csv(r.url.params["id"]))
    )
    respx.get(url__startswith=sharadar.BASE_URL).mock(return_value=httpx.Response(403))
    res = runner.invoke(app, ["ingest", "all", "--universe", str(small_universe)])
    assert res.exit_code == 0, res.output
    assert "falling back" in res.output


def test_ingest_surface_and_surface_commands(tmp_path: Path) -> None:
    res = runner.invoke(app, ["ingest", "surface", "--path", str(FIXTURES / "surface_ssvi.json")])
    assert res.exit_code == 0, res.output
    copies = list((tmp_path / "data" / "raw" / "surfaces").rglob("*.json"))
    assert len(copies) == 1
    again = runner.invoke(app, ["ingest", "surface", "--path", str(FIXTURES / "surface_ssvi.json")])
    assert again.exit_code == 0
    bad = runner.invoke(
        app, ["ingest", "surface", "--path", str(FIXTURES / "surface_svi_bad.json")]
    )
    assert bad.exit_code == 2 and "DQ-05" in bad.output
    junk = tmp_path / "junk.json"
    junk.write_text("{}")
    assert runner.invoke(app, ["ingest", "surface", "--path", str(junk)]).exit_code == 2

    out = tmp_path / "synthetic.json"
    assert runner.invoke(app, ["surface", "synthetic", "--out", str(out)]).exit_code == 0
    show = runner.invoke(app, ["surface", "show", "--path", str(out)])
    assert show.exit_code == 0 and "atm_vol=0.2000" in show.output
    show_bad = runner.invoke(
        app, ["surface", "show", "--path", str(FIXTURES / "surface_svi_bad.json")]
    )
    assert show_bad.exit_code == 1
    assert runner.invoke(app, ["surface", "show", "--path", str(junk)]).exit_code == 2


def test_data_check_uses_configured_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPRE_SURFACE_PATH", str(FIXTURES / "surface_svi_bad.json"))
    get_settings.cache_clear()
    res = runner.invoke(app, ["data", "check"])
    assert res.exit_code == 1
    payload = json.loads((tmp_path / "results" / "data_check.json").read_text())
    dq05 = next(c for c in payload["checks"] if c["rule"] == "DQ-05")
    assert dq05["status"] == "fail"


def test_deribit_not_yet() -> None:
    res = runner.invoke(app, ["ingest", "deribit"])
    assert res.exit_code == 2 and "M5" in res.output
