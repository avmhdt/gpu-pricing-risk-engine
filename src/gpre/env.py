"""Backend detection and the hardware stamp (FR-01, P-02, P-13).

:func:`collect` returns a flat mapping with the GPU, driver, CUDA, library, CPU, memory and OS
facts that every benchmark row carries (§7b). Each probe is optional: a missing GPU, CuPy or
``nvidia-smi`` yields empty fields, never an exception, so the CPU path always works.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Literal

Requirement = Literal["gpu", "cuda", "cpp"]

MIN_COMPUTE_CAPABILITY = 7.0


class BackendUnavailableError(RuntimeError):
    """A backend requested with ``--require`` is not usable on this host."""


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _run(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout


def cpu_info(proc_cpuinfo: Path = Path("/proc/cpuinfo")) -> dict[str, Any]:
    """CPU model, logical CPUs and physical cores (Linux ``lscpu``; platform fallback)."""
    model = ""
    sockets = cores_per_socket = 0
    for line in _run(["lscpu"]).splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "Model name" and not model:
            model = value
        elif key == "Socket(s)" and value.isdigit():
            sockets = int(value)
        elif key == "Core(s) per socket" and value.isdigit():
            cores_per_socket = int(value)
    if not model and proc_cpuinfo.exists():
        for line in proc_cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                model = line.partition(":")[2].strip()
                break
    logical = os.cpu_count() or 0
    physical = sockets * cores_per_socket if sockets and cores_per_socket else 0
    return {
        "cpu_model": model or platform.processor(),
        "cpu_threads": logical,
        "cpu_cores": physical or logical,
    }


def memory_gb(meminfo: Path = Path("/proc/meminfo")) -> float:
    """Total RAM in GiB from ``/proc/meminfo`` (0.0 where unavailable)."""
    if not meminfo.exists():
        return 0.0
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 1024**2, 1)
    return 0.0


def os_info() -> dict[str, Any]:
    """OS name, kernel and whether this is WSL."""
    pretty = ""
    os_release = Path("/etc/os-release")
    if os_release.exists():
        for line in os_release.read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                pretty = line.partition("=")[2].strip().strip('"')
    kernel = platform.release()
    return {
        "os": pretty or platform.platform(),
        "kernel": kernel,
        "wsl": "microsoft" in kernel.lower(),
    }


def jax_info() -> dict[str, Any]:
    """JAX version, default backend and devices; a broken install yields an error string."""
    try:
        import jax

        devices = jax.devices()
        return {
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in devices],
            "jax_gpu": any(d.platform == "gpu" for d in devices),
        }
    except Exception as exc:  # any import or plugin failure means "no JAX here"
        return {"jax_version": _version("jax"), "jax_error": repr(exc), "jax_gpu": False}


def nvidia_smi() -> dict[str, Any]:
    """GPU name, driver and VRAM from ``nvidia-smi`` (first GPU), if present."""
    if shutil.which("nvidia-smi") is None:
        return {}
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    if not out:
        return {}
    name, driver, mem_mib, cc = [part.strip() for part in out.splitlines()[0].split(",")]
    return {
        "gpu_name": name,
        "driver_version": driver,
        "vram_gb": round(float(mem_mib) / 1024, 1),
        "compute_capability": cc,
    }


def cupy_info() -> dict[str, Any]:
    """CuPy device facts: name, compute capability, VRAM, driver and runtime versions."""
    try:
        import cupy
    except Exception:
        return {"cupy_version": ""}
    try:
        runtime = cupy.cuda.runtime
        props = runtime.getDeviceProperties(0)
        name = props["name"]
        name = name.decode() if isinstance(name, bytes) else str(name)
        drv = int(runtime.driverGetVersion())
        rt = int(runtime.runtimeGetVersion())
        return {
            "cupy_version": cupy.__version__,
            "cupy_gpu_name": name,
            "cupy_compute_capability": f"{props['major']}.{props['minor']}",
            "cupy_vram_gb": round(int(props["totalGlobalMem"]) / 1024**3, 1),
            "cuda_driver_api": f"{drv // 1000}.{(drv % 1000) // 10}",
            "cuda_version": f"{rt // 1000}.{(rt % 1000) // 10}",
        }
    except Exception as exc:
        return {"cupy_version": cupy.__version__, "cupy_error": repr(exc)}


def collect(cpp_binary: Path | None = None, platform_label: str = "") -> dict[str, Any]:
    """Full hardware and software stamp for this host.

    Args:
        cpp_binary: path to ``gpre_cpu_baseline``; reported as present or not.
        platform_label: cloud provider plus instance type, or empty for ``local``.
    """
    stamp: dict[str, Any] = {
        "host": socket.gethostname(),
        "platform": platform_label or "local",
        "python_version": platform.python_version(),
        "numpy_version": _version("numpy"),
        "scipy_version": _version("scipy"),
        "gpre_version": _version("gpre"),
        "ram_gb": memory_gb(),
    }
    stamp.update(cpu_info())
    stamp.update(os_info())
    stamp.update(jax_info())
    smi = nvidia_smi()
    cp = cupy_info()
    stamp["gpu_name"] = smi.get("gpu_name") or cp.get("cupy_gpu_name", "")
    stamp["compute_capability"] = smi.get("compute_capability") or cp.get(
        "cupy_compute_capability", ""
    )
    stamp["vram_gb"] = smi.get("vram_gb") or cp.get("cupy_vram_gb", 0.0)
    stamp["driver_version"] = smi.get("driver_version", "")
    stamp["cuda_version"] = cp.get("cuda_version", "")
    stamp["cupy_version"] = cp.get("cupy_version", "")
    if "cupy_error" in cp:
        stamp["cupy_error"] = cp["cupy_error"]
    stamp["cpp_baseline"] = str(cpp_binary) if cpp_binary else ""
    stamp["cpp_baseline_present"] = bool(
        cpp_binary and cpp_binary.is_file() and os.access(cpp_binary, os.X_OK)
    )
    return stamp


def check_requirement(stamp: dict[str, Any], requirement: Requirement) -> None:
    """Raise :class:`BackendUnavailableError` with a clear message if ``requirement`` fails."""
    if requirement == "gpu":
        if not stamp.get("jax_gpu"):
            raise BackendUnavailableError(
                "no JAX GPU device: install the gpu group (make setup-gpu) on an NVIDIA host; "
                f"JAX sees {stamp.get('jax_devices') or stamp.get('jax_error')}"
            )
    elif requirement == "cuda":
        if not stamp.get("cupy_version"):
            raise BackendUnavailableError("CuPy is not installed (make setup-gpu)")
        cc = stamp.get("compute_capability") or ""
        if not cc:
            raise BackendUnavailableError(
                f"CuPy finds no CUDA device: {stamp.get('cupy_error', 'no device')}"
            )
        if float(cc) < MIN_COMPUTE_CAPABILITY:
            raise BackendUnavailableError(f"compute capability {cc} < {MIN_COMPUTE_CAPABILITY}")
    elif requirement == "cpp":
        if not stamp.get("cpp_baseline_present"):
            raise BackendUnavailableError(
                f"C++ baseline not found or not executable at {stamp.get('cpp_baseline')!r}; "
                "build it with cmake (cpp/) or set GPRE_CPP_BASELINE"
            )
    else:  # pragma: no cover - guarded by the Literal type and the CLI choices
        raise ValueError(f"unknown requirement {requirement!r}")
