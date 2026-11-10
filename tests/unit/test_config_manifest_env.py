"""Settings, run configs, run manifests, the experiment log and the hardware stamp."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from gpre import env as genv
from gpre import manifest
from gpre.config import (
    CONFIG_DIR,
    RunConfig,
    Settings,
    config_hash,
    get_settings,
    load_run_config,
    load_yaml,
)


def test_settings_defaults_and_paths(tmp_path: Path) -> None:
    s = get_settings()
    assert s.data_dir == tmp_path / "data"
    assert s.raw_dir == tmp_path / "data" / "raw"
    assert s.interim_dir.name == "interim" and s.processed_dir.name == "processed"
    assert s.results_dir == tmp_path / "results" and s.reports_dir == tmp_path / "reports"
    assert s.gpre_seed == 20261030 and s.gpre_gpu_budget_usd == 50
    assert s.sharadar_api_key.get_secret_value() == ""
    assert "**" in repr(Settings(sharadar_api_key="abc"))  # type: ignore[arg-type]


def test_settings_reads_dotenv_and_ignores_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATA_ROOT")
    (tmp_path / ".env").write_text("DATA_ROOT=~/somewhere\nGPRE_SURFACE_PATH=\nGPRE_SEED=7\n")
    s = Settings()
    assert s.data_root == Path("~/somewhere").expanduser()
    assert s.gpre_surface_path is None and s.gpre_seed == 7


def test_data_dir_rejects_windows_mounts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPRE_DATA_DIR", "/mnt/d/data")
    with pytest.raises(ValueError, match="/mnt"):
        _ = Settings().data_dir


def test_default_yaml_loads() -> None:
    cfg = load_run_config()
    assert cfg.seed == 20261030 and cfg.simulation.n_paths == 2**20
    assert cfg.models.localvol.grid.n_k == 64 and cfg.models.heston.rho == -0.7
    assert cfg.products["asian"]["n_fixings"] == 12
    for name in ("bench", "book", "risk", "validation", "universe"):
        assert load_yaml(CONFIG_DIR / f"{name}.yaml")


def test_config_rejects_typos(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\nsimulaton: {n_paths: 4}\n")
    with pytest.raises(ValidationError):
        load_run_config(bad)
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(lst)
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert load_yaml(empty) == {}


def test_config_hash_is_stable_and_sensitive() -> None:
    a, b = RunConfig(), RunConfig(seed=1)
    assert config_hash(a) == config_hash(RunConfig())
    assert config_hash(a) != config_hash(b)
    assert config_hash({"x": 1, "y": 2}) == config_hash({"y": 2, "x": 1})


def test_run_id_and_git_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    rid = manifest.new_run_id()
    assert len(rid) == len("20261109T231500Z-3fa2c1") and rid[8] == "T"
    assert manifest.git_hash() != ""

    def boom(*_a: object, **_k: object) -> None:
        raise subprocess.SubprocessError

    monkeypatch.setattr(manifest.subprocess, "run", boom)
    assert manifest.git_hash() == "unknown"


def test_experiment_log_and_manifest(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = manifest.Run(
        command="gpre test",
        results_dir=results,
        config={"a": 1},
        seed=3,
        hardware={"gpu_name": "none"},
    )
    path = run.finish(outputs=["x.json"], key_metric_name="m", key_metric_value=1.5)
    data = json.loads(path.read_text())
    assert data["seed"] == 3 and data["outputs"] == ["x.json"] and data["config_hash"]
    manifest.Run(command="gpre test2", results_dir=results).finish(outputs=[])
    rows = list(csv.DictReader((results / "experiment_log.csv").open()))
    assert [r["command"] for r in rows] == ["gpre test", "gpre test2"]
    assert rows[0]["key_metric_value"] == "1.5" and rows[0]["gpu_name"] == "none"
    assert list(rows[0]) == list(manifest.EXPERIMENT_LOG_COLUMNS)
    with pytest.raises(ValueError, match="unknown"):
        manifest.append_experiment_log(results / "experiment_log.csv", {"bogus": 1})


def test_collect_on_cpu_host(tmp_path: Path) -> None:
    stamp = genv.collect(tmp_path / "missing", "")
    for key in ("host", "cpu_model", "cpu_cores", "ram_gb", "os", "jax_version", "gpu_name"):
        assert key in stamp
    assert stamp["platform"] == "local" and stamp["cpp_baseline_present"] is False
    assert stamp["jax_gpu"] is False
    with pytest.raises(genv.BackendUnavailableError, match="JAX GPU"):
        genv.check_requirement(stamp, "gpu")
    with pytest.raises(genv.BackendUnavailableError):
        genv.check_requirement(stamp, "cuda")
    with pytest.raises(genv.BackendUnavailableError, match="C\\+\\+ baseline"):
        genv.check_requirement(stamp, "cpp")


def test_requirements_pass_when_present(tmp_path: Path) -> None:
    exe = tmp_path / "gpre_cpu_baseline"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    stamp = genv.collect(exe, "provider/gpu.1x")
    assert stamp["platform"] == "provider/gpu.1x" and stamp["cpp_baseline_present"]
    genv.check_requirement(stamp, "cpp")
    genv.check_requirement({"jax_gpu": True}, "gpu")
    genv.check_requirement({"cupy_version": "14", "compute_capability": "8.9"}, "cuda")
    with pytest.raises(genv.BackendUnavailableError, match=r"7\.0"):
        genv.check_requirement({"cupy_version": "14", "compute_capability": "6.1"}, "cuda")


def test_cpu_info_parses_lscpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    text = "Model name: Test CPU 9000\nSocket(s): 2\nCore(s) per socket: 8\n"
    monkeypatch.setattr(genv, "_run", lambda cmd, timeout=20.0: text)
    info = genv.cpu_info()
    assert info == {
        "cpu_model": "Test CPU 9000",
        "cpu_threads": info["cpu_threads"],
        "cpu_cores": 16,
    }
    monkeypatch.setattr(genv, "_run", lambda cmd, timeout=20.0: "")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor: 0\nmodel name\t: Fallback CPU\n")
    assert genv.cpu_info(cpuinfo)["cpu_model"] == "Fallback CPU"
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       65011712 kB\n")
    assert genv.memory_gb(meminfo) == 62.0
    assert genv.memory_gb(tmp_path / "nope") == 0.0


def test_nvidia_smi_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(genv.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        genv, "_run", lambda cmd, timeout=20.0: "NVIDIA L4, 570.86.15, 23034, 8.9\n"
    )
    assert genv.nvidia_smi() == {
        "gpu_name": "NVIDIA L4",
        "driver_version": "570.86.15",
        "vram_gb": 22.5,
        "compute_capability": "8.9",
    }
    monkeypatch.setattr(genv, "_run", lambda cmd, timeout=20.0: "")
    assert genv.nvidia_smi() == {}
