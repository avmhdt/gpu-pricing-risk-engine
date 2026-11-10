"""Data-quality rules (DQ-02, DQ-03, DQ-04, DQ-05, DQ-09) on small synthetic frames."""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from conftest import FIXTURES
from gpre.data import checks
from gpre.data.prices import finalize_prices
from gpre.data.rawstore import RawFileExistsError, RawStore
from gpre.data.surface_io import read_surface


def _days(n: int) -> list[date]:
    out, d = [], date(2024, 1, 1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _prices(closes: dict[str, list[float | None]]) -> pl.DataFrame:
    rows = []
    days = _days(max(len(v) for v in closes.values()))
    for sym, series in closes.items():
        for d, c in zip(days, series, strict=False):
            if c is not None:
                rows.append(
                    {
                        "date": d,
                        "symbol": sym,
                        "open": c,
                        "high": c,
                        "low": c,
                        "close": c,
                        "volume": 1.0,
                    }
                )
    return finalize_prices(pl.DataFrame(rows), "test")


def test_yields_check(tmp_path: Path) -> None:
    assert checks.check_yields(tmp_path / "none.parquet").status == "skip"
    d = _days(2)
    rows = [
        {
            "date": day,
            "series_id": f"S{i}",
            "tenor_years": float(i),
            "zero_cc": 0.04,
            "filled": False,
        }
        for day in d
        for i in range(11)
    ]
    good = tmp_path / "y.parquet"
    pl.DataFrame(rows).write_parquet(good)
    r = checks.check_yields(good)
    assert r.status == "pass" and r.stats["dates"] == 2
    bad = tmp_path / "b.parquet"
    pl.DataFrame([{**x, "zero_cc": None} if x["date"] == d[0] else x for x in rows]).write_parquet(
        bad
    )
    assert checks.check_yields(bad).status == "fail"


def test_prices_pass_and_skip(tmp_path: Path) -> None:
    assert [r.status for r in checks.check_prices(tmp_path / "none.parquet")] == ["skip", "skip"]
    path = tmp_path / "p.parquet"
    _prices({"SPY": [100.0 + i for i in range(20)], "AAA": [50.0] * 20}).write_parquet(path)
    dq03, dq04 = checks.check_prices(path)
    assert dq03.status == "pass" and dq04.status == "pass"
    assert dq03.stats["calendar_symbol"] == "SPY"


def test_prices_gaps_late_start_split_and_crash(tmp_path: Path) -> None:
    spy = [100.0] * 30
    gappy = [50.0] * 5 + [None] * 4 + [50.0] * 21
    late = [None] * 3 + [20.0] * 27
    split = [100.0] * 10 + [50.0] * 20  # a 2:1 split left in an "adjusted" series
    crash = [100.0] * 10 + [60.0] * 20  # log return -0.51, not a split ratio
    path = tmp_path / "p.parquet"
    _prices({"SPY": spy, "GAP": gappy, "LATE": late, "SPL": split, "CRA": crash}).write_parquet(
        path
    )
    dq03, dq04 = checks.check_prices(path)
    assert dq03.status == "warn" and "GAP" in dq03.detail and "LATE" in dq03.detail
    assert dq04.status == "fail" and "SPL" in dq04.detail and "CRA" in dq04.detail
    assert dq04.stats == {"flagged": 2, "split_like": 1, "unexplained": 1}
    only_crash = tmp_path / "c.parquet"
    _prices({"SPY": spy, "CRA": crash}).write_parquet(only_crash)
    assert checks.check_prices(only_crash)[1].status == "warn"


def test_finalize_drops_nonpositive_and_adds_log_ret() -> None:
    df = _prices({"A": [1.0, 0.0, math.e]})
    assert df["close"].to_list() == [1.0, math.e]
    assert df["log_ret"].to_list()[1] == 1.0


def test_surface_and_manifest_checks(tmp_path: Path) -> None:
    assert checks.check_surface(None, "none").status == "skip"
    ok = checks.check_surface(read_surface(FIXTURES / "surface_ssvi.json"), "fixture")
    assert ok.status == "pass" and ok.stats["surface"] == "fixture"
    bad = checks.check_surface(read_surface(FIXTURES / "surface_svi_bad.json"), "bad")
    assert bad.status == "fail" and "calendar" in bad.detail

    store = RawStore(tmp_path / "raw")
    assert checks.check_raw_manifest(store).status == "skip"
    f = store.path_for("x", "2026-11-09", "a.csv")
    store.write(f, b"abc", url="https://example.org/a.csv")
    try:
        store.write(f, b"abc", url="https://example.org/a.csv")
    except RawFileExistsError:
        pass
    else:  # pragma: no cover
        raise AssertionError("raw files must be immutable")
    external = tmp_path / "ext.txt"
    external.write_text("x")
    store.record(external, url="file://ext", sha256="0" * 64)
    assert checks.check_raw_manifest(store).status == "pass"
    f.write_bytes(b"tampered")
    r = checks.check_raw_manifest(store)
    assert r.status == "fail" and "hash changed" in r.detail
    f.unlink()
    assert "missing" in checks.check_raw_manifest(store).detail


def test_summarize_strict() -> None:
    results = [checks.CheckResult("DQ-1", "a", "pass"), checks.CheckResult("DQ-2", "b", "warn")]
    assert checks.summarize(results, strict=False)[0] is True
    ok, payload = checks.summarize(results, strict=True)
    assert ok is False and payload["passed"] is False and len(payload["checks"]) == 2
    assert checks.summarize([checks.CheckResult("x", "y", "fail")], strict=False)[0] is False
