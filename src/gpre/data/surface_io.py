"""Read, validate and write ``vsm.surface.v1`` surface files (§7b, DQ-05).

vol-surface-modeling owns the format. Its JSON Schema is vendored in ``data/schemas/`` and
every file is validated against it; a pydantic model mirrors the subset this engine uses.
Files this project writes (the D-05 synthetic surface) are vsm.surface.v1 too.

DQ-05 repeats the producer's arbitrage checks independently on k in [-1, 1] (step 0.01) and
every maturity in the file: total variance non-decreasing in T and Durrleman g(k) >= -1e-10.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import jsonschema
import numpy as np
from pydantic import BaseModel, ConfigDict

from gpre.surface.ssvi import SSVI, synthetic_ssvi
from gpre.surface.svi import RawSVI, durrleman_g, implied_density

SCHEMA_ID = "vsm.surface.v1"
SCHEMA_PATH = Path(__file__).parent / "schemas" / "vsm.surface.v1.schema.json"
SECONDS_PER_YEAR = 365.0 * 86400.0
G_TOL = -1e-10
CAL_TOL = -1e-12
K_GRID = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.01), 10)


class SurfaceFormatError(ValueError):
    """The file is not valid vsm.surface.v1."""


class SurfaceArbitrageError(ValueError):
    """The surface fails DQ-05."""


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class SviRawModel(_Loose):
    a: float
    b: float
    rho: float
    m: float
    sigma: float


class ExpiryModel(_Loose):
    expiry: str
    expiry_ns: int
    t: float
    forward: float
    discount_factor: float
    basis_annualized: float
    dividend_yield: float | None
    svi_raw: SviRawModel
    slice_source: str


class SsviModel(_Loose):
    phi: Literal["power_law", "heston_like"]
    rho: float
    eta: float
    gamma: float
    t_nodes: list[float]
    theta_nodes: list[float]
    # "lambda" is a Python keyword; read it from the extra fields.


class SpotModel(_Loose):
    index_name: str
    value: float


class SurfaceModel(_Loose):
    """Subset of vsm.surface.v1 used by this engine."""

    schema_: str
    underlying: str
    venue: str
    as_of: str
    as_of_ns: int
    spot: SpotModel
    conventions: dict[str, str]
    canonical: Literal["ssvi", "svi"]
    ssvi: SsviModel | None
    expiries: list[ExpiryModel]
    provenance: dict[str, Any]


@lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


@dataclass(frozen=True)
class Surface:
    """A parsed surface with the evaluation rules of vsm.surface.v1."""

    model: SurfaceModel
    raw: dict[str, Any] = field(repr=False)

    @property
    def ssvi(self) -> SSVI | None:
        s = self.model.ssvi
        if s is None:
            return None
        lam = (self.raw.get("ssvi") or {}).get("lambda")
        return SSVI(
            rho=s.rho,
            t_nodes=tuple(s.t_nodes),
            theta_nodes=tuple(s.theta_nodes),
            phi=s.phi,
            eta=s.eta,
            gamma=s.gamma,
            lam=lam,
        )

    @property
    def expiry_times(self) -> list[float]:
        return [e.t for e in self.model.expiries]

    def slices(self) -> list[RawSVI]:
        return [RawSVI(**e.svi_raw.model_dump()) for e in self.model.expiries]

    def w(self, k: np.ndarray[Any, Any], t: float) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Total variance: SSVI continuum (rule 1) or, for canonical = svi, rule 3."""
        ssvi = self.ssvi
        if self.model.canonical == "ssvi" and ssvi is not None:
            return ssvi.w(k, t)
        ts, sl = self.expiry_times, self.slices()
        if t <= ts[0]:
            return sl[0].w(k) * t / ts[0]
        if t >= ts[-1]:
            return sl[-1].w(k) * t / ts[-1]
        i = int(np.searchsorted(ts, t)) - 1
        lam = (t - ts[i]) / (ts[i + 1] - ts[i])
        return (1.0 - lam) * sl[i].w(k) + lam * sl[i + 1].w(k)

    def forward(self, t: float) -> float:
        """F(t): ln F linear in t between expiries, flat annualized basis beyond (rule 2)."""
        exps = self.model.expiries
        spot = self.model.spot.value
        if t <= exps[0].t:
            return spot * math.exp(exps[0].basis_annualized * t)
        if t >= exps[-1].t:
            return spot * math.exp(exps[-1].basis_annualized * t)
        ts = [e.t for e in exps]
        lnf = [math.log(e.forward) for e in exps]
        return math.exp(float(np.interp(t, ts, lnf)))


def parse_surface(data: dict[str, Any]) -> Surface:
    """Validate a decoded vsm.surface.v1 document and parse it."""
    if data.get("schema") != SCHEMA_ID:
        raise SurfaceFormatError(f"schema must be {SCHEMA_ID!r}, got {data.get('schema')!r}")
    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.path) or "<root>"
        raise SurfaceFormatError(f"{where}: {first.message} ({len(errors)} schema errors)")
    payload = dict(data)
    payload["schema_"] = payload.pop("schema")
    model = SurfaceModel.model_validate(payload)
    ts = [e.t for e in model.expiries]
    if any(b <= a for a, b in itertools.pairwise(ts)):
        raise SurfaceFormatError("expiries[].t must be strictly increasing")
    if model.canonical == "ssvi" and model.ssvi is None:
        raise SurfaceFormatError("canonical = ssvi needs an ssvi block")
    if model.canonical == "svi" and model.ssvi is not None:
        raise SurfaceFormatError("canonical = svi is allowed only without an ssvi block")
    return Surface(model=model, raw=data)


def read_surface(path: Path) -> Surface:
    """Read and validate a surface file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SurfaceFormatError(f"{path}: not JSON ({exc})") from None
    return parse_surface(data)


@dataclass(frozen=True)
class ArbitrageReport:
    """Outcome of DQ-05 on one surface."""

    passed: bool
    min_g: float
    min_calendar_dw: float
    density_mass_max_err: float | None
    n_maturities: int
    failures: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "min_g": self.min_g,
            "min_calendar_dw": self.min_calendar_dw,
            "density_mass_max_err": self.density_mass_max_err,
            "n_maturities": self.n_maturities,
            "failures": self.failures,
        }


def _maturities(surface: Surface) -> list[float]:
    times = set(surface.expiry_times)
    if surface.ssvi is not None:
        times.update(surface.ssvi.t_nodes)
    return sorted(times)


def _slice_derivs(
    surface: Surface, t: float, k: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    ssvi = surface.ssvi
    if surface.model.canonical == "ssvi" and ssvi is not None:
        sl = ssvi.slice_at(t)
        return sl.w(k), sl.dw(k), sl.d2w(k)
    ts = surface.expiry_times
    i = ts.index(t)
    sl = surface.slices()[i]
    return sl.w(k), sl.dw(k), sl.d2w(k)


def density_mass_error(surface: Surface, n: int = 8001, z: float = 12.0) -> float:
    """Max over maturities of |1 - integral of the implied density| on k in +-z sqrt(w_atm)."""
    worst = 0.0
    for t in _maturities(surface):
        atm = float(surface.w(np.array([0.0]), t)[0])
        k = np.linspace(-z * math.sqrt(atm), z * math.sqrt(atm), n)
        w, dw, d2w = _slice_derivs(surface, t, k)
        mass = float(np.trapezoid(implied_density(k, w, dw, d2w), k))
        worst = max(worst, abs(1.0 - mass))
    return worst


def check_arbitrage(surface: Surface, k_grid: np.ndarray[Any, Any] = K_GRID) -> ArbitrageReport:
    """DQ-05: calendar and butterfly checks on ``k_grid`` x every maturity in the file."""
    failures: list[str] = []
    times = _maturities(surface)
    min_g = math.inf
    for t in times:
        w, dw, d2w = _slice_derivs(surface, t, k_grid)
        if np.any(w <= 0):
            failures.append(f"t={t:.6g}: non-positive total variance")
            continue
        g = durrleman_g(k_grid, w, dw, d2w)
        g_min = float(g.min())
        min_g = min(min_g, g_min)
        if g_min < G_TOL:
            k_bad = float(k_grid[int(np.argmin(g))])
            failures.append(f"t={t:.6g}: butterfly g={g_min:.3e} at k={k_bad:+.2f}")
    min_dw = 0.0 if len(times) < 2 else math.inf
    for t0, t1 in itertools.pairwise(times):
        dw_cal = surface.w(k_grid, t1) - surface.w(k_grid, t0)
        d_min = float(dw_cal.min())
        min_dw = min(min_dw, d_min)
        if d_min < CAL_TOL:
            k_bad = float(k_grid[int(np.argmin(dw_cal))])
            failures.append(f"t={t0:.6g}->{t1:.6g}: calendar dw={d_min:.3e} at k={k_bad:+.2f}")
    mass_err = density_mass_error(surface) if not failures else None
    return ArbitrageReport(
        passed=not failures,
        min_g=min_g,
        min_calendar_dw=min_dw,
        density_mass_max_err=mass_err,
        n_maturities=len(times),
        failures=failures,
    )


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ns(dt: datetime) -> int:
    return int(dt.timestamp()) * 1_000_000_000


SYNTHETIC_AS_OF = datetime(2025, 12, 31, 21, 0, 0, tzinfo=UTC)


def synthetic_surface_doc(
    ssvi: SSVI | None = None,
    spot: float = 100.0,
    rate: float = 0.03,
    div_yield: float = 0.01,
    as_of: datetime = SYNTHETIC_AS_OF,
    git_sha: str = "",
    config_hash: str = "",
) -> dict[str, Any]:
    """The D-05 synthetic SSVI surface as a vsm.surface.v1 document.

    canonical = ssvi with theta nodes at the book tenors and one expiries[] entry per node,
    whose svi_raw is the exact raw-SVI form of that SSVI slice (slice_source = ssvi_fallback,
    fit.n_quotes = 0). heston, sabr and local_vol are empty. Expiry times are whole seconds,
    so t is exact on ACT/365 seconds.
    """
    ssvi = ssvi or synthetic_ssvi()
    expiries: list[dict[str, Any]] = []
    times: list[float] = []
    for t in ssvi.t_nodes:
        seconds = round(t * SECONDS_PER_YEAR)
        exp = as_of + timedelta(seconds=seconds)
        tt = seconds / SECONDS_PER_YEAR
        times.append(tt)
        sl = ssvi.slice_at(tt)
        expiries.append(
            {
                "expiry": _iso(exp),
                "expiry_ns": _ns(exp),
                "t": tt,
                "forward": spot * math.exp((rate - div_yield) * tt),
                "forward_method": "future",
                "forward_se": None,
                "discount_factor": math.exp(-rate * tt),
                "basis_annualized": rate - div_yield,
                "dividend_yield": div_yield,
                "svi_raw": {"a": sl.a, "b": sl.b, "rho": sl.rho, "m": sl.m, "sigma": sl.sigma},
                "slice_source": "ssvi_fallback",
                "k_range_quoted": [-1.0, 1.0],
                "fit": {
                    "n_quotes": 0,
                    "n_liquid": 0,
                    "rmse_iv_bp": 0.0,
                    "within_spread_frac": None,
                },
            }
        )
    doc: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "underlying": "SYNTH",
        "venue": "synthetic",
        "as_of": _iso(as_of),
        "as_of_ns": _ns(as_of),
        "spot": {"index_name": "synthetic", "value": spot},
        "conventions": {
            "option_style": "european",
            "settlement": "cash_usd",
            "premium_currency": "USD",
            "pricing": "Black-76 on the expiry forward, discounted with discount_factor",
            "year_fraction": "ACT/365 seconds",
            "expiry_time_utc": as_of.strftime("%H:%M:%S"),
            "moneyness": "k = ln(K/F)",
            "total_variance": "w = sigma^2 * t",
            "coin_rate": "implied",
        },
        "canonical": "ssvi",
        "ssvi": {
            "phi": ssvi.phi,
            "rho": ssvi.rho,
            "eta": ssvi.eta,
            "gamma": ssvi.gamma,
            "lambda": ssvi.lam,
            "theta_interpolation": "linear_in_t",
            "t_nodes": times,
            "theta_nodes": [ssvi.theta(t) for t in times],
        },
        "expiries": expiries,
        "arbitrage": {
            "passed": True,
            "min_g": 0.0,
            "min_calendar_dw": 0.0,
            "density_mass_max_err": 0.0,
            "grid": {"z_min": -1.0, "z_max": 1.0, "n": len(K_GRID)},
        },
        "heston": None,
        "sabr": [],
        "local_vol": None,
        "provenance": {
            "vsm_version": "n/a",
            "git_sha": git_sha,
            "config_hash": config_hash,
            "snapshot_id": "synthetic-ssvi-d05",
            "created_at": _iso(as_of),
            "producer": "synthetic",
        },
    }
    report = check_arbitrage(parse_surface(doc))
    if not report.passed:
        raise SurfaceArbitrageError(f"synthetic surface fails DQ-05: {report.failures}")
    doc["arbitrage"].update(
        min_g=report.min_g,
        min_calendar_dw=report.min_calendar_dw,
        density_mass_max_err=report.density_mass_max_err,
    )
    return doc


def write_surface(doc: dict[str, Any], path: Path) -> Path:
    """Write a surface document as indented JSON (validated first)."""
    parse_surface(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path
