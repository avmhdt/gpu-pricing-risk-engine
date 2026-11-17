"""The ``gpre`` command line (§10). Commands for later milestones are added as they land."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, cast

import typer

from gpre import __version__
from gpre.config import CONFIG_DIR, Backend, DType, Settings, get_settings
from gpre.logging import configure_logging, get_logger
from gpre.manifest import Run, new_run_id, utc_now

OptionType = Literal["call", "put"]
RngKind = Literal["philox", "sobol-bb", "sobol", "threefry"]

app = typer.Typer(help="GPU pricing and risk engine.", no_args_is_help=True, add_completion=False)
ingest_app = typer.Typer(help="Download or convert input data (§7e).", no_args_is_help=True)
data_app = typer.Typer(help="Data-quality checks (§7c).", no_args_is_help=True)
surface_app = typer.Typer(help="Volatility surface files (vsm.surface.v1).", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(data_app, name="data")
app.add_typer(surface_app, name="surface")
bench_app = typer.Typer(help="Benchmarks (FR-26 to FR-32).", no_args_is_help=True)
spend_app = typer.Typer(help="Cloud GPU spend log (NFR-12).", no_args_is_help=True)
app.add_typer(bench_app, name="bench")
app.add_typer(spend_app, name="spend")

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


# ---------------------------------------------------------------------------------------------
# gpre price / validate
# ---------------------------------------------------------------------------------------------


@app.command("price")
def price_cmd(
    product: Annotated[
        str, typer.Option(help="european (more products arrive in M3).")
    ] = "european",
    model: Annotated[str, typer.Option(help="gbm (heston and localvol arrive in M3).")] = "gbm",
    backend: Annotated[str, typer.Option(help="jax, cuda or numpy.")] = "jax",
    paths: Annotated[int, typer.Option(help="Number of paths.")] = 1 << 20,
    steps: Annotated[int, typer.Option(help="Time steps to maturity.")] = 252,
    rng: Annotated[str, typer.Option(help="philox, sobol-bb or sobol.")] = "philox",
    antithetic: Annotated[bool, typer.Option("--antithetic/--no-antithetic")] = False,
    cv: Annotated[str, typer.Option(help="Control variate (M3).")] = "none",
    dtype: Annotated[str, typer.Option(help="float32 or float64.")] = "float32",
    config: Annotated[Path | None, typer.Option(help="Run config YAML.")] = None,
    seed: Annotated[int | None, typer.Option(help="Seed (default: config).")] = None,
) -> None:
    """Price one product by Monte Carlo and print price, SE and the closed form."""
    from gpre.analytic.black_scholes import bs_price
    from gpre.config import load_run_config
    from gpre.models.gbm import GBM, uniform_grid
    from gpre.paths.base import PathSpec
    from gpre.pricing.mc import price_mc
    from gpre.products.european import European
    from gpre.rng.streams import RngSpec

    if product != "european" or model != "gbm":
        _fail("M2 prices the European option under GBM; other products and models arrive in M3")
    if backend not in ("jax", "cuda", "numpy") or rng not in ("philox", "sobol-bb", "sobol"):
        _fail("backend must be jax, cuda or numpy and rng philox, sobol-bb or sobol")
    if dtype not in ("float32", "float64"):
        _fail("dtype must be float32 or float64")
    if cv != "none":
        _fail("control variates arrive in M3 (FR-12)")
    settings = get_settings()
    cfg = load_run_config(config)
    seed = cfg.seed if seed is None else seed
    p = cfg.products.get("european", {})
    opt = European(
        cast(OptionType, str(p.get("type", "call"))),
        float(p.get("strike", 100.0)),
        float(p.get("maturity", 1.0)),
    )
    gbm = GBM(cfg.market.spot, cfg.market.rate, cfg.market.div_yield, cfg.models.gbm.vol)
    spec = PathSpec(
        paths,
        uniform_grid(opt.maturity, steps),
        RngSpec(cast(RngKind, rng), seed, antithetic=antithetic),
        cast(DType, dtype),
    )
    run = _start(
        f"gpre price --product {product} --model {model} --backend {backend} "
        f"--paths {paths} --steps {steps} --rng {rng} --dtype {dtype}"
        + (" --antithetic" if antithetic else ""),
        settings,
        config=cfg.model_dump(mode="json"),
        config_path=str(config or ""),
        seed=seed,
        backend=backend,
        dtype=dtype,
    )
    res = price_mc(opt, gbm, spec, backend=cast(Backend, backend))
    ref = bs_price(
        gbm.spot, opt.strike, opt.maturity, cfg.market.rate, gbm.div_yield, gbm.vol, opt.option_type
    )
    z = (res.price - ref) / res.std_err
    typer.echo(
        f"price {res.price:.6f}  se {res.std_err:.6f}  black-scholes {ref:.6f}  z {z:+.2f}  "
        f"elapsed {res.elapsed_s:.3f}s"
    )
    _finish(
        run,
        [],
        key_metric_name="price",
        key_metric_value=res.price,
        notes=f"se={res.std_err:.6g} bs={ref:.6g} z={z:+.3f}",
    )


@app.command("validate")
def validate_cmd(
    suite: Annotated[str, typer.Option(help="european or rng (more suites arrive in M3/M4).")],
    backends: Annotated[str, typer.Option(help="Comma-separated backends.")] = "numpy,jax",
    paths: Annotated[int | None, typer.Option(help="Override n_paths (smoke runs only).")] = None,
    out: Annotated[Path | None, typer.Option(help="Output directory.")] = None,
) -> None:
    """Run a validation suite and write results/validation/<suite>.json."""
    from gpre import validation
    from gpre.config import CONFIG_DIR, load_yaml

    settings = get_settings()
    vcfg = load_yaml(CONFIG_DIR / "validation.yaml")
    seed = int(vcfg["seed"])
    out_dir = out or settings.results_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    run = _start(
        f"gpre validate --suite {suite} --backends {backends}"
        + (f" --paths {paths}" if paths else ""),
        settings,
        config=vcfg,
        config_path=str(CONFIG_DIR / "validation.yaml"),
        seed=seed,
        dtype=str(vcfg["dtype"]),
    )
    if suite == "european":
        n = paths or int(vcfg["european"]["n_paths"])
        cases = []
        for b in backends.split(","):
            cases += validation.run_european(
                vcfg["european"],
                backend=cast(Backend, b.strip()),
                seed=seed,
                n_paths=n,
                dtype=str(vcfg["dtype"]),
            )
        summary = validation.summarize(cases)
        payload: dict[str, Any] = {
            "suite": suite,
            "run_id": run.run_id,
            "seed": seed,
            "summary": summary,
            "cases": [c.as_dict() for c in cases],
        }
    elif suite == "rng":
        summary = validation.run_rng_cpu(seed)
        payload = {
            "suite": suite,
            "run_id": run.run_id,
            "seed": seed,
            "summary": summary,
            "note": "CPU part; the CUDA bit-exactness checks run in tests/gpu",
        }
    else:
        _fail(f"suite {suite!r} arrives in a later milestone")
    target = out_dir / f"{suite}.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verdict = "PASS" if summary["passed"] else "FAIL"
    typer.echo(f"{suite}: {verdict} {json.dumps(summary)}")
    _finish(run, [target], key_metric_name="passed", key_metric_value=int(summary["passed"]))
    if not summary["passed"]:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------------------------
# gpre bench
# ---------------------------------------------------------------------------------------------


@bench_app.command("throughput")
def bench_throughput_cmd(
    backends: Annotated[str, typer.Option(help="Comma-separated: cuda, jax, numpy.")] = "jax,numpy",
    paths: Annotated[int, typer.Option(help="Number of paths.")] = 1 << 20,
    steps: Annotated[int, typer.Option(help="Time steps.")] = 252,
    reps: Annotated[int, typer.Option(help="Timed repetitions.")] = 10,
    warmup: Annotated[int, typer.Option(help="Untimed warm-up runs.")] = 3,
) -> None:
    """First throughput numbers for the path generators (M2); results/bench/throughput_<id>.json."""
    from gpre.bench.throughput import run as run_throughput
    from gpre.env import collect

    settings = get_settings()
    hardware = collect(settings.gpre_cpp_baseline, settings.gpre_cloud_provider)
    run = _start(
        f"gpre bench throughput --backends {backends} --paths {paths} --steps {steps}",
        settings,
        seed=settings.gpre_seed,
        hardware=hardware,
        dtype="float32",
    )
    records = run_throughput(
        [b.strip() for b in backends.split(",")],
        n_paths=paths,
        n_steps=steps,
        seed=settings.gpre_seed,
        reps=reps,
        warmup=warmup,
        hardware=hardware,
    )
    for rec in records:
        rec.update(
            run_id=run.run_id,
            timestamp_utc=run.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            git_hash=run.git,
            config_hash="",
        )
        typer.echo(
            f"{rec['backend']:6s} {rec['kernel_variant']:6s} "
            f"{rec['time_median_s'] * 1e3:10.2f} ms  "
            f"{rec['path_steps_per_s'] / 1e9:8.3f} G path-steps/s  "
            f"{rec['achieved_gbps'] or 0:8.1f} GB/s"
        )
    target = settings.results_dir / "bench" / f"throughput_{run.run_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    best = max(r["path_steps_per_s"] for r in records)
    _finish(
        run,
        [target],
        key_metric_name="max_path_steps_per_s",
        key_metric_value=best,
        notes="preliminary (M2); published benchmarks follow the shared-PC rules",
    )


# ---------------------------------------------------------------------------------------------
# gpre spend
# ---------------------------------------------------------------------------------------------


@spend_app.command("show")
def spend_show_cmd() -> None:
    """Print logged cloud spend against the cap."""
    from gpre.cloud import status

    settings = get_settings()
    st = status(settings.gpre_spend_log, settings.gpre_gpu_budget_usd)
    typer.echo(
        f"spent {st.cumulative_usd:.2f} of {st.cap_usd:.2f} USD; pause at "
        f"{st.pause_at_usd:.2f}; remaining {st.remaining_usd:.2f}"
        + ("  PAUSED: owner must confirm" if st.paused else "")
    )


@spend_app.command("check")
def spend_check_cmd() -> None:
    """Exit non-zero if a new GPU session may not start (spend at or above 80% of the cap)."""
    from gpre.cloud import BudgetError, check_can_start

    settings = get_settings()
    try:
        st = check_can_start(settings.gpre_spend_log, settings.gpre_gpu_budget_usd)
    except BudgetError as exc:
        _fail(str(exc))
    typer.echo(f"ok: {st.remaining_usd:.2f} USD left under the cap")


@spend_app.command("add")
def spend_add_cmd(
    instance_type: Annotated[str, typer.Option(help="Provider flavor, e.g. n3-RTX-A4000x1.")],
    gpu_model: Annotated[str, typer.Option(help="GPU model as nvidia-smi names it.")],
    start: Annotated[str, typer.Option(help="Start, UTC ISO time (YYYY-MM-DDTHH:MM:SS).")],
    stop: Annotated[str, typer.Option(help="Stop, UTC ISO time.")],
    cost_usd: Annotated[float, typer.Option(help="Cost from the provider's billing page.")],
    end_state: Annotated[str, typer.Option(help="hibernated or deleted.")],
    notes: Annotated[str, typer.Option(help="Free text.")] = "",
) -> None:
    """Append one cloud session to the spend log (before the session ends)."""
    from datetime import UTC, datetime

    from gpre.cloud import BudgetError, append_session

    settings = get_settings()
    try:
        st = append_session(
            settings.gpre_spend_log,
            provider=settings.gpre_cloud_provider or "unknown",
            instance_type=instance_type,
            gpu_model=gpu_model,
            start=datetime.fromisoformat(start).replace(tzinfo=UTC),
            stop=datetime.fromisoformat(stop).replace(tzinfo=UTC),
            cost_usd=cost_usd,
            end_state=end_state,
            cap_usd=settings.gpre_gpu_budget_usd,
            notes=notes,
        )
    except (BudgetError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(f"logged; cumulative {st.cumulative_usd:.2f} of {st.cap_usd:.2f} USD")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
