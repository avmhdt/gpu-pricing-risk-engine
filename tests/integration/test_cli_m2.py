"""CLI commands added in M2: price, validate, bench throughput, spend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpre.cli import app
from gpre.config import get_settings

runner = CliRunner()


def test_price_numpy_and_errors() -> None:
    res = runner.invoke(app, ["price", "--backend", "numpy", "--paths", "4096", "--steps", "4"])
    assert res.exit_code == 0, res.output
    assert "black-scholes" in res.output and "z " in res.output
    for args in (
        ["--product", "asian"],
        ["--backend", "tpu"],
        ["--dtype", "float16"],
        ["--cv", "spot"],
    ):
        assert runner.invoke(app, ["price", *args]).exit_code == 2


def test_validate_european_small(tmp_path: Path) -> None:
    res = runner.invoke(
        app, ["validate", "--suite", "european", "--backends", "numpy", "--paths", "16384"]
    )
    payload = json.loads((tmp_path / "results" / "validation" / "european.json").read_text())
    assert payload["summary"]["n_cases"] == 36
    assert res.exit_code == (0 if payload["summary"]["passed"] else 1)
    assert {c["params"]["backend"] for c in payload["cases"]} == {"numpy"}


def test_validate_rng_and_unknown_suite(tmp_path: Path) -> None:
    res = runner.invoke(app, ["validate", "--suite", "rng"])
    assert res.exit_code == 0, res.output
    assert json.loads((tmp_path / "results" / "validation" / "rng.json").read_text())["summary"][
        "passed"
    ]
    assert runner.invoke(app, ["validate", "--suite", "heston"]).exit_code == 2


def test_bench_throughput(tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        [
            "bench",
            "throughput",
            "--backends",
            "numpy",
            "--paths",
            "1024",
            "--steps",
            "4",
            "--reps",
            "2",
            "--warmup",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    files = list((tmp_path / "results" / "bench").glob("throughput_*.json"))
    rec = json.loads(files[0].read_text())[0]
    assert rec["backend"] == "numpy" and rec["run_id"] and rec["cpu_model"] is not None


def test_spend_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPRE_SPEND_LOG", str(tmp_path / "results" / "cloud_spend.csv"))
    monkeypatch.setenv("GPRE_CLOUD_PROVIDER", "hyperstack")
    get_settings.cache_clear()
    assert "spent 0.00 of 50.00" in runner.invoke(app, ["spend", "show"]).output
    assert runner.invoke(app, ["spend", "check"]).exit_code == 0
    add = [
        "spend",
        "add",
        "--instance-type",
        "n3-RTX-A4000x1",
        "--gpu-model",
        "NVIDIA RTX A4000",
        "--start",
        "2026-11-10T12:00:00",
        "--stop",
        "2026-11-10T13:30:00",
    ]
    ok = runner.invoke(app, [*add, "--cost-usd", "0.23", "--end-state", "hibernated"])
    assert ok.exit_code == 0, ok.output
    assert "hyperstack" in (tmp_path / "results" / "cloud_spend.csv").read_text()
    assert runner.invoke(app, [*add, "--cost-usd", "0.1", "--end-state", "shutoff"]).exit_code == 2
    assert runner.invoke(app, [*add, "--cost-usd", "39.9", "--end-state", "deleted"]).exit_code == 0
    assert runner.invoke(app, ["spend", "check"]).exit_code == 2
    assert "PAUSED" in runner.invoke(app, ["spend", "show"]).output
