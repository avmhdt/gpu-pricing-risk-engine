"""The ``gpre`` command line (§10). Commands for later milestones are added as they land."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from gpre import __version__
from gpre.config import CONFIG_DIR, Settings, get_settings
from gpre.logging import configure_logging, get_logger
from gpre.manifest import Run, new_run_id, utc_now

app = typer.Typer(help="GPU pricing and risk engine.", no_args_is_help=True, add_completion=False)
ingest_app = typer.Typer(help="Download or convert input data (§7e).", no_args_is_help=True)
data_app = typer.Typer(help="Data-quality checks (§7c).", no_args_is_help=True)
surface_app = typer.Typer(help="Volatility surface files (vsm.surface.v1).", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(data_app, name="data")
app.add_typer(surface_app, name="surface")

log = get_logger("gpre.cli")

DEFAULT_UNIVERSE = CONFIG_DIR / "universe.yaml"


def _start(command: str, settings: Settings, **run_kwargs: Any) -> Run:
    run_id = new_run_id()
    configure_logging(settings.gpre_log_level, settings.results_dir / "logs" / f"{run_id}.jsonl")
    run = Run(command=command, results_dir=settings.results_dir, run_id=run_id, **run_kwargs)
    typer.echo(f"run_id {run.run_id}")
    return run


def _finish(run: Run, outputs: list[Path], **kwargs: Any) -> None:
    manifest = run.finish(outputs=[str(p) for p in outputs], **kwargs)
    for path in outputs:
        typer.echo(f"wrote {path}")
    typer.echo(f"manifest {manifest}")


def _fail(message: str, code: int = 2) -> NoReturn:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit(0)


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", help="Print the version.", callback=_version, is_eager=True)
    ] = False,
) -> None:
    """GPU pricing and risk engine."""


# ---------------------------------------------------------------------------------------------
# gpre env
# ---------------------------------------------------------------------------------------------


@app.command("env")
def env_cmd(
    require: Annotated[
        str | None, typer.Option(help="Fail unless this backend works: gpu, cuda or cpp.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Output file (default results/hardware/<host>.json).")
    ] = None,
) -> None:
    """Detect backends and write the hardware stamp (FR-01)."""
    from gpre.env import BackendUnavailableError, check_requirement, collect

    settings = get_settings()
    if require not in (None, "gpu", "cuda", "cpp"):
        _fail(f"--require must be gpu, cuda or cpp, not {require!r}")
    run = _start(f"gpre env --require {require}" if require else "gpre env", settings)
    stamp = collect(settings.gpre_cpp_baseline, settings.gpre_cloud_provider)
    target = out or settings.results_dir / "hardware" / f"{stamp['host']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run.hardware = stamp
    for key in (
        "host",
        "platform",
        "cpu_model",
        "cpu_cores",
        "cpu_threads",
        "ram_gb",
        "os",
        "jax_version",
        "jax_backend",
        "gpu_name",
        "compute_capability",
        "cuda_version",
        "driver_version",
        "cupy_version",
        "cpp_baseline_present",
    ):
        typer.echo(f"{key:22s} {stamp.get(key, '')}")
    if require:
        try:
            check_requirement(stamp, require)  # type: ignore[arg-type]
        except BackendUnavailableError as exc:
            _finish(
                run,
                [target],
                key_metric_name=f"require_{require}",
                key_metric_value=0,
                notes=str(exc),
            )
            _fail(str(exc))
    _finish(run, [target], key_metric_name="jax_backend", key_metric_value=stamp.get("jax_backend"))


# ---------------------------------------------------------------------------------------------
# gpre ingest
# ---------------------------------------------------------------------------------------------


def _store(settings: Settings) -> Any:
    from gpre.data.rawstore import RawStore

    return RawStore(settings.raw_dir)


@ingest_app.command("fred")
def ingest_fred_cmd(
    start: Annotated[str, typer.Option(help="First date, YYYY-MM-DD.")] = "2012-01-01",
    end: Annotated[str, typer.Option(help="Last date, YYYY-MM-DD.")] = "2025-12-31",
) -> None:
    """Treasury CMT yields and VIXCLS from FRED (API if FRED_API_KEY is set, else fredgraph)."""
    from gpre.data.fred import FredClient, FredDataError, ingest_fred

    settings = get_settings()
    run = _start(
        f"gpre ingest fred --start {start} --end {end}",
        settings,
        config={"start": start, "end": end},
    )
    client = FredClient(api_key=settings.fred_api_key.get_secret_value())
    try:
        out = ingest_fred(
            _store(settings),
            settings.processed_dir,
            date.fromisoformat(start),
            date.fromisoformat(end),
            client,
            day=utc_now().date().isoformat(),
        )
    except (FredDataError, RuntimeError) as exc:
        _fail(str(exc))
    _finish(
        run,
        list(out.values()),
        key_metric_name="source",
        key_metric_value="fred_api" if client.api_key else "fredgraph_csv",
    )


def _price_dates(universe_path: Path, start: str | None, end: str | None) -> tuple[Any, date, date]:
    from gpre.data.prices import load_universe

    universe = load_universe(universe_path)
    return (
        universe,
        date.fromisoformat(start or universe.start),
        date.fromisoformat(end or universe.end),
    )


@ingest_app.command("sharadar")
def ingest_sharadar_cmd(
    universe: Annotated[Path, typer.Option(help="Universe YAML.")] = DEFAULT_UNIVERSE,
    start: Annotated[str | None, typer.Option(help="First date (default: universe).")] = None,
    end: Annotated[str | None, typer.Option(help="Last date (default: universe).")] = None,
) -> None:
    """Daily prices from Sharadar SEP/SFP (default source, D-15)."""
    from gpre.data.sharadar import SharadarClient, SharadarUnavailableError, ingest_sharadar

    settings = get_settings()
    uni, d0, d1 = _price_dates(universe, start, end)
    run = _start(
        "gpre ingest sharadar",
        settings,
        config_path=str(universe),
        config={"symbols": list(uni.symbols), "start": str(d0), "end": str(d1)},
    )
    try:
        client = SharadarClient(api_key=settings.sharadar_api_key.get_secret_value())
        out, meta = ingest_sharadar(
            uni,
            client,
            _store(settings),
            settings.processed_dir,
            d0,
            d1,
            day=utc_now().date().isoformat(),
        )
    except SharadarUnavailableError as exc:
        _fail(f"{exc}. Fallback: gpre ingest stooq")
    _finish(run, [out], key_metric_name="source", key_metric_value=meta["source"])


@ingest_app.command("stooq")
def ingest_stooq_cmd(
    universe: Annotated[Path, typer.Option(help="Universe YAML.")] = DEFAULT_UNIVERSE,
    stooq_dir: Annotated[
        Path | None, typer.Option(help="Snapshot root (default STOOQ_DIR).")
    ] = None,
    start: Annotated[str | None, typer.Option(help="First date (default: universe).")] = None,
    end: Annotated[str | None, typer.Option(help="Last date (default: universe).")] = None,
) -> None:
    """Daily prices from the owner's local Stooq snapshot (fallback and cross-check, FR-23)."""
    from gpre.data.stooq import StooqError, ingest_stooq

    settings = get_settings()
    uni, d0, d1 = _price_dates(universe, start, end)
    root = (stooq_dir or settings.stooq_dir).expanduser()
    run = _start(
        "gpre ingest stooq",
        settings,
        config_path=str(universe),
        config={
            "symbols": list(uni.symbols),
            "start": str(d0),
            "end": str(d1),
            "stooq_dir": str(root),
        },
    )
    try:
        out, meta = ingest_stooq(
            uni, root, _store(settings), settings.interim_dir, settings.processed_dir, d0, d1
        )
    except StooqError as exc:
        _fail(str(exc))
    _finish(
        run, [out], key_metric_name="snapshot_through", key_metric_value=meta["snapshot_through"]
    )


@ingest_app.command("surface")
def ingest_surface_cmd(
    path: Annotated[Path, typer.Option(help="vsm.surface.v1 file to validate and copy.")],
) -> None:
    """Validate a surface file (DQ-05) and copy it into raw/surfaces/<date>/."""
    from gpre.data.rawstore import RawFileExistsError, sha256_file
    from gpre.data.surface_io import SurfaceFormatError, check_arbitrage, read_surface

    settings = get_settings()
    run = _start(f"gpre ingest surface --path {path}", settings)
    try:
        surface = read_surface(path)
    except SurfaceFormatError as exc:
        _fail(str(exc))
    report = check_arbitrage(surface)
    if not report.passed:
        _fail(f"DQ-05 rejects {path}: {'; '.join(report.failures)}")
    store = _store(settings)
    dest = store.path_for(
        "surfaces",
        utc_now().date().isoformat(),
        f"{surface.model.underlying}_{surface.model.as_of_ns}.json",
    )
    try:
        store.write(
            dest,
            path.read_bytes(),
            url=f"file://{path.resolve().as_posix()}",
            source="surface",
            producer=surface.model.provenance.get("producer", ""),
        )
    except RawFileExistsError:
        if sha256_file(dest) != sha256_file(path):
            _fail(f"{dest} exists with different content")
    _finish(run, [dest], key_metric_name="min_g", key_metric_value=report.min_g)


@ingest_app.command("deribit")
def ingest_deribit_cmd(currency: str = "BTC") -> None:
    """Deribit option snapshot (FR-24, arrives in M5)."""
    del currency
    _fail("gpre ingest deribit arrives in M5 (FR-24)")


@ingest_app.command("all")
def ingest_all_cmd(
    universe: Annotated[Path, typer.Option(help="Universe YAML.")] = DEFAULT_UNIVERSE,
) -> None:
    """FRED, then Sharadar (or the Stooq snapshot if Sharadar is unavailable)."""
    ingest_fred_cmd()
    settings = get_settings()
    if settings.sharadar_api_key.get_secret_value():
        try:
            ingest_sharadar_cmd(universe=universe)
            return
        except typer.Exit as exc:
            if exc.exit_code == 0:
                return
            typer.echo("Sharadar failed; falling back to the Stooq snapshot")
    else:
        typer.echo("SHARADAR_API_KEY not set; using the Stooq snapshot")
    ingest_stooq_cmd(universe=universe)


# ---------------------------------------------------------------------------------------------
# gpre surface
# ---------------------------------------------------------------------------------------------


@surface_app.command("synthetic")
def surface_synthetic_cmd(
    out: Annotated[Path, typer.Option(help="Output path.")],
) -> None:
    """Write the D-05 synthetic SSVI surface as vsm.surface.v1."""
    from gpre.config import config_hash, load_run_config
    from gpre.data.surface_io import synthetic_surface_doc, write_surface
    from gpre.manifest import git_hash
    from gpre.surface.ssvi import synthetic_ssvi

    cfg = load_run_config()
    s = cfg.models.localvol.synthetic_ssvi
    doc = synthetic_surface_doc(
        synthetic_ssvi(s.theta_per_year, s.rho, s.phi.eta, s.phi.gamma),
        spot=cfg.market.spot,
        rate=cfg.market.rate,
        div_yield=cfg.market.div_yield,
        git_sha=git_hash(),
        config_hash=config_hash(s),
    )
    write_surface(doc, out)
    typer.echo(f"wrote {out}")


@surface_app.command("show")
def surface_show_cmd(
    path: Annotated[Path, typer.Option(help="vsm.surface.v1 file.")],
) -> None:
    """Summarize a surface file and its DQ-05 result."""
    import numpy as np

    from gpre.data.surface_io import SurfaceFormatError, check_arbitrage, read_surface

    try:
        surface = read_surface(path)
    except SurfaceFormatError as exc:
        _fail(str(exc))
    m = surface.model
    typer.echo(f"{m.underlying}/{m.venue} as_of {m.as_of} canonical {m.canonical}")
    for e in m.expiries:
        atm = float(surface.w(np.array([0.0]), e.t)[0])
        typer.echo(f"  t={e.t:.4f} F={e.forward:.4f} atm_vol={(atm / e.t) ** 0.5:.4f}")
    report = check_arbitrage(surface)
    typer.echo(json.dumps(report.as_dict(), indent=2))
    if not report.passed:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------------------------
# gpre data check
# ---------------------------------------------------------------------------------------------


@data_app.command("check")
def data_check_cmd(
    strict: Annotated[bool, typer.Option("--strict", help="Warnings and skips fail too.")] = False,
) -> None:
    """Run the data-quality rules and write results/data_check.json."""
    from gpre.data import checks
    from gpre.data.surface_io import parse_surface, read_surface, synthetic_surface_doc

    settings = get_settings()
    run = _start("gpre data check" + (" --strict" if strict else ""), settings)
    if settings.gpre_surface_path:
        surface, label = read_surface(settings.gpre_surface_path), str(settings.gpre_surface_path)
    else:
        surface, label = parse_surface(synthetic_surface_doc()), "synthetic (D-05)"
    results = [
        checks.check_yields(settings.processed_dir / "yields.parquet"),
        *checks.check_prices(settings.processed_dir / "prices_daily.parquet"),
        checks.check_surface(surface, label),
        checks.check_raw_manifest(_store(settings)),
    ]
    ok, payload = checks.summarize(results, strict)
    meta_path = settings.processed_dir / "prices_daily.meta.json"
    if meta_path.exists():
        payload["price_source"] = json.loads(meta_path.read_text(encoding="utf-8"))
        payload["price_source"].pop("stooq_dir", None)
    out = settings.results_dir / "data_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for r in results:
        colour = {"pass": "green", "warn": "yellow", "fail": "red", "skip": "blue"}[r.status]
        typer.secho(f"{r.rule:6s} {r.name:18s} {r.status:5s} {r.detail}", fg=colour)
    _finish(run, [out], key_metric_name="passed", key_metric_value=int(ok))
    if not ok:
        raise typer.Exit(1)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
