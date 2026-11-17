"""GBM coefficients, path generators, the MC estimator, Black-Scholes and the benchmark parts."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpre import cloud, validation
from gpre.analytic.black_scholes import bs_greeks, bs_price
from gpre.bench import throughput
from gpre.bench.timing import jax_sync, time_fn
from gpre.cuda.launch import merge_tiles
from gpre.market.curve import DiscountCurve
from gpre.models.gbm import GBM, steps_for, uniform_grid
from gpre.paths.base import PathSpec, get_generator, normal_chunks
from gpre.paths.jax_paths import JaxGenerator
from gpre.paths.numpy_paths import NumpyGenerator
from gpre.pricing.mc import PriceResult, RunningStats, price_mc
from gpre.products.european import European
from gpre.rng.streams import RngSpec

SEED = 20261030
MODEL = GBM(spot=100.0, rate=0.03, div_yield=0.01, vol=0.2)


def _spec(
    n: int = 4096,
    steps: int = 12,
    kind: str = "philox",
    dtype: str = "float64",
    antithetic: bool = False,
    T: float = 1.0,
) -> PathSpec:
    return PathSpec(
        n,
        uniform_grid(T, steps),
        RngSpec(kind, SEED, antithetic=antithetic),  # type: ignore[arg-type]
        dtype,
        chunk_paths=1024,
    )  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- model, BS


def test_gbm_coefficients_flat_and_curve() -> None:
    times = uniform_grid(1.0, 4)
    drift, vol = MODEL.step_coefficients(times)
    np.testing.assert_allclose(drift, (0.03 - 0.01 - 0.02) * 0.25, atol=1e-15)
    np.testing.assert_allclose(vol, 0.2 * 0.5)
    flat_curve = GBM(100, DiscountCurve.flat(0.03), 0.01, 0.2)
    np.testing.assert_allclose(flat_curve.step_coefficients(times)[0], drift, atol=1e-7)
    assert math.isclose(flat_curve.discount(1.0), math.exp(-0.03), rel_tol=1e-6)
    with pytest.raises(ValueError):
        MODEL.step_coefficients([0.5, 0.5])
    with pytest.raises(ValueError):
        uniform_grid(1.0, 0)
    assert steps_for(0.25, 252) == 63 and steps_for(0.001, 252) == 1


def test_black_scholes_reference_values_and_parity() -> None:
    # Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, vol=20%, T=0.5.
    assert math.isclose(bs_price(42, 40, 0.5, 0.1, 0.0, 0.2, "call"), 4.759, abs_tol=5e-4)
    assert math.isclose(bs_price(42, 40, 0.5, 0.1, 0.0, 0.2, "put"), 0.8086, abs_tol=5e-4)
    c = bs_price(100, 110, 2.0, 0.03, 0.01, 0.3, "call")
    p = bs_price(100, 110, 2.0, 0.03, 0.01, 0.3, "put")
    assert math.isclose(c - p, 100 * math.exp(-0.02) - 110 * math.exp(-0.06), rel_tol=1e-12)
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.0, 0.0, 0.2, "straddle")
    with pytest.raises(ValueError):
        bs_price(100, 100, 0, 0.0, 0.0, 0.2, "call")


@pytest.mark.parametrize("typ", ["call", "put"])
def test_bs_greeks_match_finite_differences(typ: str) -> None:
    args = dict(S=100.0, K=95.0, T=0.75, r=0.03, q=0.01, vol=0.25)
    g = bs_greeks(*args.values(), typ)  # type: ignore[arg-type]
    h = 1e-4

    def bump(name: str, d: float) -> float:
        a = dict(args)
        a[name] += d
        return bs_price(*a.values(), typ)  # type: ignore[arg-type]

    assert math.isclose(g["delta"], (bump("S", h) - bump("S", -h)) / (2 * h), rel_tol=1e-6)
    assert math.isclose(g["vega"], (bump("vol", h) - bump("vol", -h)) / (2 * h), rel_tol=1e-6)
    assert math.isclose(g["rho"], (bump("r", h) - bump("r", -h)) / (2 * h), rel_tol=1e-6)
    assert math.isclose(g["theta"], -(bump("T", h) - bump("T", -h)) / (2 * h), rel_tol=1e-5)
    fd_gamma = (bump("S", 1e-2) - 2 * bs_price(*args.values(), typ) + bump("S", -1e-2)) / 1e-4  # type: ignore[arg-type]
    assert math.isclose(g["gamma"], fd_gamma, rel_tol=1e-4)
    with pytest.raises(ValueError):
        bs_greeks(*args.values(), "digital")  # type: ignore[arg-type]


def test_european_product_validation() -> None:
    with pytest.raises(ValueError):
        European("straddle", 100, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        European("call", -1, 1)
    put = European("put", 100, 1)
    np.testing.assert_array_equal(put.payoff_terminal(np.array([90.0, 110.0])), [10.0, 0.0])


# ---------------------------------------------------------------------------- generators


@pytest.mark.parametrize("kind", ["philox", "sobol", "sobol-bb"])
def test_numpy_and_jax_paths_agree(kind: str) -> None:
    spec = _spec(n=1024, steps=9, kind=kind, dtype="float32")
    a = NumpyGenerator().simulate(MODEL, spec)
    b = np.asarray(JaxGenerator(chunk_paths=512).simulate(MODEL, spec))
    assert a.shape == b.shape == (10, 1024) and a.dtype == np.float32
    np.testing.assert_allclose(b, a, rtol=2e-6)
    assert np.all(a[0] == 100.0)


def test_paths_independent_of_chunking() -> None:
    spec = _spec(n=1000, steps=5)
    a = NumpyGenerator().simulate(MODEL, spec)
    b = NumpyGenerator().simulate(MODEL, replace(spec, chunk_paths=128))
    np.testing.assert_array_equal(a, b)
    if jax.config.jax_enable_x64:
        c = np.asarray(JaxGenerator(chunk_paths=256).simulate(MODEL, spec))
        np.testing.assert_allclose(c, a, rtol=1e-13)


def test_pathspec_and_generator_errors() -> None:
    with pytest.raises(ValueError):
        PathSpec(0, uniform_grid(1, 2), RngSpec("philox", SEED))
    with pytest.raises(ValueError):
        PathSpec(3, uniform_grid(1, 2), RngSpec("philox", SEED, antithetic=True))
    with pytest.raises(ValueError):
        PathSpec(4, uniform_grid(1, 2), RngSpec("philox", SEED), chunk_paths=3)
    with pytest.raises(ValueError, match="antithetic"):
        next(
            normal_chunks(PathSpec(4, uniform_grid(1, 2), RngSpec("sobol", SEED, antithetic=True)))
        )
    with pytest.raises(ValueError, match="not supported"):
        next(normal_chunks(PathSpec(4, uniform_grid(1, 2), RngSpec("threefry", SEED))))
    with pytest.raises(NotImplementedError):
        get_generator("cpp")
    assert get_generator("numpy").name == "numpy"  # type: ignore[attr-defined]
    assert get_generator("jax").name == "jax"  # type: ignore[attr-defined]


def test_log_path_is_exact_gbm() -> None:
    spec = _spec(n=200_000, steps=4)
    st = NumpyGenerator().simulate(MODEL, spec)[-1]
    lr = np.log(st / 100.0)
    assert abs(lr.mean() - (0.02 - 0.02)) < 0.002 and abs(lr.std() - 0.2) < 0.002


# ------------------------------------------------------------------------------ estimator


def test_running_stats_matches_numpy() -> None:
    x = np.random.default_rng(3).normal(5.0, 2.0, 10_001)
    s = RunningStats()
    for part in np.array_split(x, 7):
        s.add(part)
    s.add(np.array([]))
    assert s.n == x.size and math.isclose(s.mean, x.mean(), rel_tol=1e-12)
    assert math.isclose(s.variance, x.var(ddof=1), rel_tol=1e-10)
    assert math.isnan(RunningStats().std_err)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_price_mc_close_to_black_scholes(backend: str) -> None:
    opt = European("call", 100.0, 1.0)
    ref = bs_price(100, 100, 1, 0.03, 0.01, 0.2, "call")
    res = price_mc(opt, MODEL, _spec(n=1 << 15, steps=1, dtype="float32"), backend=backend)  # type: ignore[arg-type]
    assert isinstance(res, PriceResult) and abs(res.price - ref) < 4 * res.std_err
    anti = price_mc(
        opt, MODEL, _spec(n=1 << 15, steps=1, antithetic=True, dtype="float32"), backend=backend
    )  # type: ignore[arg-type]
    assert abs(anti.price - ref) < 4 * anti.std_err and anti.meta["antithetic"]
    qmc = price_mc(
        opt,
        MODEL,
        _spec(n=1 << 14, steps=4, kind="sobol-bb", dtype="float32"),
        backend=backend,  # type: ignore[arg-type]
        replications=16,
    )
    assert abs(qmc.price - ref) < 4 * qmc.std_err and qmc.std_err < res.std_err / 10
    assert qmc.meta["replications"] == 16


def test_put_call_parity_path_by_path() -> None:
    spec = _spec(n=2048, steps=3)
    call = price_mc(European("call", 105, 1.0), MODEL, spec, backend="numpy")
    put = price_mc(European("put", 105, 1.0), MODEL, spec, backend="numpy")
    st = NumpyGenerator().simulate(MODEL, spec)[-1]
    expected = math.exp(-0.03) * (st.mean() - 105)
    assert math.isclose(call.price - put.price, expected, rel_tol=1e-12)


def test_price_mc_errors() -> None:
    opt = European("call", 100, 1.0)
    with pytest.raises(NotImplementedError):
        price_mc(opt, MODEL, _spec(), backend="numpy", control_variate="spot")
    with pytest.raises(ValueError, match="grid ends"):
        price_mc(European("call", 100, 2.0), MODEL, _spec(), backend="numpy")
    with pytest.raises(ValueError, match="replications"):
        price_mc(opt, MODEL, _spec(n=1000, kind="sobol"), backend="numpy", replications=8)


def test_merge_tiles_matches_direct_moments() -> None:
    x = np.random.default_rng(5).exponential(3.0, 64 * 50 + 17)
    n = x.size
    tiles = [x[i : i + 64] for i in range(0, n, 64)]
    tsum = np.array([t.sum() for t in tiles])
    tsq = np.array([(t * t).sum() for t in tiles])
    mean, var = merge_tiles(tsum, tsq, n)
    assert math.isclose(mean, x.mean(), rel_tol=1e-13)
    assert math.isclose(var, x.var(ddof=1), rel_tol=1e-10)


# ------------------------------------------------------------------- validation, bench, spend


def test_european_validation_small() -> None:
    cfg = {
        "S": 100,
        "K": [100],
        "T": [0.5],
        "vol": [0.2],
        "r": 0.03,
        "q": 0.01,
        "types": ["call", "put"],
    }
    cases = validation.run_european(cfg, backend="numpy", seed=SEED, n_paths=1 << 14)
    summary = validation.summarize(cases)
    assert summary["n_cases"] == 2 and summary["passed"]
    assert cases[0].as_dict()["pass"] is True and "numpy-call" in cases[0].case_id
    assert (
        len(
            validation.european_grid(
                {
                    "S": 100,
                    "K": [80, 100, 120],
                    "T": [0.25, 1, 2],
                    "vol": [0.1, 0.3],
                    "r": 0.03,
                    "q": 0.01,
                    "types": ["call", "put"],
                }
            )
        )
        == 36
    )
    assert validation.summarize([])["passed"] is False


def test_rng_validation_cpu() -> None:
    out = validation.run_rng_cpu(SEED, n_paths=64, n_blocks=4)
    assert out["passed"]


def test_timing_harness() -> None:
    ticks = iter(range(100))
    t = time_fn(lambda: None, reps=4, warmup=2, clock=lambda: float(next(ticks)))
    assert t.n_reps == 4 and t.compile_s == 1.0 and t.time_median_s == 1.0
    assert set(t.as_dict()) == {"time_median_s", "time_p10_s", "time_p90_s", "n_reps", "compile_s"}
    with pytest.raises(ValueError):
        time_fn(lambda: None, reps=0)
    assert float(jax_sync(jnp.ones(2)).sum()) == 2.0


def test_throughput_records() -> None:
    hw = {"cpu_model": "test", "platform": "local"}
    recs = throughput.run(
        ["numpy", "jax"], n_paths=2048, n_steps=8, seed=SEED, reps=2, warmup=1, hardware=hw
    )
    assert [r["backend"] for r in recs] == ["numpy", "jax"]
    assert recs[1]["achieved_gbps"] > 0 and recs[0]["threads"] == 1
    assert recs[0]["cpu_model"] == "test" and recs[0]["path_steps_per_s"] > 0
    with pytest.raises(NotImplementedError):
        throughput.run(["cpp"], n_paths=2, n_steps=1, seed=1, reps=1, warmup=0, hardware={})


def test_spend_log(tmp_path: Path) -> None:
    log = tmp_path / "cloud_spend.csv"
    t0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
    assert cloud.check_can_start(log, 50).remaining_usd == 50
    st = cloud.append_session(
        log,
        provider="hyperstack",
        instance_type="n3-RTX-A4000x1",
        gpu_model="NVIDIA RTX A4000",
        start=t0,
        stop=t0 + timedelta(hours=2),
        cost_usd=0.30,
        end_state="hibernated",
        cap_usd=50,
    )
    assert math.isclose(st.cumulative_usd, 0.30) and not st.paused
    rows = cloud.read_spend(log)
    assert rows[0]["billed_hours"] == "2.0" and rows[0]["end_state"] == "hibernated"
    with pytest.raises(ValueError, match="hibernate"):
        cloud.append_session(
            log,
            provider="h",
            instance_type="x",
            gpu_model="g",
            start=t0,
            stop=t0 + timedelta(hours=1),
            cost_usd=0.1,
            end_state="shutoff",
            cap_usd=50,
        )
    with pytest.raises(ValueError, match="after start"):
        cloud.append_session(
            log,
            provider="h",
            instance_type="x",
            gpu_model="g",
            start=t0,
            stop=t0,
            cost_usd=0.1,
            end_state="deleted",
            cap_usd=50,
        )
    with pytest.raises(cloud.BudgetError, match="exceed"):
        cloud.append_session(
            log,
            provider="h",
            instance_type="x",
            gpu_model="g",
            start=t0,
            stop=t0 + timedelta(hours=1),
            cost_usd=60,
            end_state="deleted",
            cap_usd=50,
        )
    cloud.append_session(
        log,
        provider="h",
        instance_type="x",
        gpu_model="g",
        start=t0,
        stop=t0 + timedelta(hours=1),
        cost_usd=39.8,
        end_state="deleted",
        cap_usd=50,
    )
    with pytest.raises(cloud.BudgetError, match="80%"):
        cloud.check_can_start(log, 50)
    assert cloud.status(log, 50).paused
