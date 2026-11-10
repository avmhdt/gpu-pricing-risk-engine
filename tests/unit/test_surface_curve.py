"""SSVI and raw SVI evaluation, vsm.surface.v1 reading, DQ-05, and the discount curve."""

from __future__ import annotations

import copy
import json
import math
from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from conftest import FIXTURES
from gpre.data import surface_io as sio
from gpre.market.curve import CurveDataError, DiscountCurve
from gpre.surface.ssvi import SSVI, synthetic_ssvi
from gpre.surface.svi import RawSVI, durrleman_g

K = np.linspace(-1, 1, 41)


def test_ssvi_slice_equals_raw_svi() -> None:
    s = synthetic_ssvi()
    for t in (0.05, 0.25, 0.7, 1.5):
        np.testing.assert_allclose(s.slice_at(t).w(K), s.w(K, t), rtol=1e-13, atol=1e-15)
        # ATM total variance is theta(t), and theta = 0.04 t (D-05) everywhere.
        assert math.isclose(float(s.w(np.array([0.0]), t)[0]), 0.04 * t, rel_tol=1e-13)


def test_theta_rule_1() -> None:
    s = SSVI(rho=-0.5, t_nodes=(0.5, 1.0), theta_nodes=(0.02, 0.05))
    assert math.isclose(s.theta(0.25), 0.01)
    assert math.isclose(s.theta(0.75), 0.035)
    assert math.isclose(s.theta(2.0), 0.1)


def test_ssvi_validation_and_heston_phi() -> None:
    with pytest.raises(ValueError):
        SSVI(rho=1.0, t_nodes=(1.0,), theta_nodes=(0.04,))
    with pytest.raises(ValueError):
        SSVI(rho=0.0, t_nodes=(1.0, 0.5), theta_nodes=(0.04, 0.05))
    with pytest.raises(ValueError):
        SSVI(rho=0.0, t_nodes=(1.0,), theta_nodes=(0.04, 0.05))
    with pytest.raises(ValueError):
        SSVI(rho=0.0, t_nodes=(1.0,), theta_nodes=(0.04,), phi="heston_like")
    h = SSVI(rho=-0.3, t_nodes=(1.0,), theta_nodes=(0.04,), phi="heston_like", lam=2.0)
    x = 2.0 * 0.04
    assert math.isclose(h.phi_of(0.04), 1 / x * (1 - (1 - math.exp(-x)) / x))


def test_svi_derivatives_match_finite_differences() -> None:
    sl = RawSVI(a=0.01, b=0.1, rho=-0.4, m=0.05, sigma=0.2)
    h = 1e-5
    np.testing.assert_allclose(sl.dw(K), (sl.w(K + h) - sl.w(K - h)) / (2 * h), rtol=1e-7)
    np.testing.assert_allclose(sl.d2w(K), (sl.dw(K + h) - sl.dw(K - h)) / (2 * h), rtol=1e-6)
    flat = np.full_like(K, 0.04)
    # A flat smile has g = (1)^2 - 0 + 0 = 1 everywhere.
    np.testing.assert_allclose(durrleman_g(K, flat, 0 * K, 0 * K), 1.0)


def test_synthetic_fixture_matches_generator_and_passes_dq05() -> None:
    on_disk = json.loads((FIXTURES / "surface_ssvi.json").read_text())
    fresh = sio.synthetic_surface_doc()
    for d in (on_disk, fresh):
        d["provenance"].pop("git_sha"), d["provenance"].pop("config_hash")
    assert on_disk == json.loads(json.dumps(fresh))
    surf = sio.read_surface(FIXTURES / "surface_ssvi.json")
    report = sio.check_arbitrage(surf)
    assert report.passed and report.min_g > 0 and report.min_calendar_dw > 0
    assert report.density_mass_max_err is not None and report.density_mass_max_err < 1e-3
    assert surf.model.underlying == "SYNTH" and surf.ssvi is not None


def test_bad_surface_is_rejected() -> None:
    surf = sio.read_surface(FIXTURES / "surface_svi_bad.json")
    report = sio.check_arbitrage(surf)
    assert not report.passed and any("calendar" in f for f in report.failures)
    assert report.density_mass_max_err is None


def test_svi_canonical_interpolation_rule_3() -> None:
    doc = copy.deepcopy(json.loads((FIXTURES / "surface_ssvi.json").read_text()))
    doc["canonical"], doc["ssvi"] = "svi", None
    surf = sio.parse_surface(doc)
    sl = surf.slices()
    ts = surf.expiry_times
    mid = 0.5 * (ts[1] + ts[2])
    np.testing.assert_allclose(surf.w(K, mid), 0.5 * (sl[1].w(K) + sl[2].w(K)))
    np.testing.assert_allclose(surf.w(K, ts[0] / 2), sl[0].w(K) / 2)
    np.testing.assert_allclose(surf.w(K, 2 * ts[-1]), 2 * sl[-1].w(K))
    assert sio.check_arbitrage(surf).passed


def test_forward_rule_2() -> None:
    surf = sio.read_surface(FIXTURES / "surface_ssvi.json")
    e = surf.model.expiries
    assert math.isclose(surf.forward(e[1].t), e[1].forward, rel_tol=1e-12)
    assert math.isclose(surf.forward(2.0), 100 * math.exp(0.02 * 2.0), rel_tol=1e-12)
    assert math.isclose(surf.forward(0.01), 100 * math.exp(0.02 * 0.01), rel_tol=1e-12)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.update(schema="vsm.surface.v2"), "schema must be"),
        (lambda d: d.pop("spot"), "spot"),
        (lambda d: d["expiries"].reverse(), "strictly increasing"),
        (lambda d: d.update(canonical="svi"), "only without"),
        (lambda d: d["arbitrage"].update(passed=False), "passed"),
    ],
)
def test_format_errors(mutate: object, match: str) -> None:
    doc = json.loads((FIXTURES / "surface_ssvi.json").read_text())
    mutate(doc)  # type: ignore[operator]
    with pytest.raises(sio.SurfaceFormatError, match=match):
        sio.parse_surface(doc)


def test_read_surface_rejects_non_json(tmp_path: object) -> None:
    from pathlib import Path

    p = Path(str(tmp_path)) / "x.json"
    p.write_text("{nope")
    with pytest.raises(sio.SurfaceFormatError, match="not JSON"):
        sio.read_surface(p)


def test_synthetic_with_arbitrage_raises() -> None:
    bad = SSVI(rho=-0.9, t_nodes=(1.0,), theta_nodes=(0.04,), eta=5.0, gamma=0.5)
    with pytest.raises(sio.SurfaceArbitrageError):
        sio.synthetic_surface_doc(bad)


# ----------------------------------------------------------------------------------- curve


def test_flat_curve() -> None:
    c = DiscountCurve.flat(0.03)
    assert math.isclose(float(c.df(2.0)), math.exp(-0.06), rel_tol=1e-6)


def test_curve_interpolation_and_extrapolation() -> None:
    c = DiscountCurve(jnp.asarray([1.0, 2.0, 5.0]), jnp.asarray([0.01, 0.02, 0.05]))
    assert math.isclose(float(c.zero(1.5)), 0.015, rel_tol=1e-6)
    assert math.isclose(float(c.zero(0.25)), 0.01, rel_tol=1e-6)
    assert math.isclose(float(c.zero(30.0)), 0.05, rel_tol=1e-6)


def test_curve_is_a_pytree_with_per_node_rho() -> None:
    c = DiscountCurve(jnp.asarray([1.0, 2.0, 5.0]), jnp.asarray([0.01, 0.02, 0.05]))
    grad = jax.grad(lambda curve: curve.df(1.5))(c)
    # d df / d r_i at t=1.5 = -t df * weight_i with weights (0.5, 0.5, 0).
    df = math.exp(-0.015 * 1.5)
    np.testing.assert_allclose(
        np.asarray(grad.zero_rates), [-0.75 * df, -0.75 * df, 0.0], rtol=1e-5
    )
    leaves = jax.tree_util.tree_leaves(c)
    assert len(leaves) == 2


def test_curve_from_yields() -> None:
    d = date(2024, 1, 2)
    tenors = [1 / 12, 0.25, 0.5, 1, 2, 3, 5]
    y = pl.DataFrame(
        {
            "date": [d] * len(tenors),
            "tenor_years": tenors,
            "zero_cc": [0.05, 0.05, 0.049, None, 0.045, 0.043, 0.042],
        }
    )
    c = DiscountCurve.from_yields(y, d)
    assert c.times.shape == (6,)
    with pytest.raises(CurveDataError, match="need at least 6"):
        DiscountCurve.from_yields(y.filter(pl.col("tenor_years") < 2), d)
