"""Regenerate the small synthetic fixtures in tests/fixtures/ (deterministic, seed 20261030).

Run: uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from gpre.data.surface_io import synthetic_surface_doc

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
SEED = 20261030


def business_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def surfaces() -> None:
    doc = synthetic_surface_doc()
    (FIX / "surface_ssvi.json").write_text(json.dumps(doc, indent=2) + "\n")
    bad = copy.deepcopy(doc)
    bad["canonical"] = "svi"
    bad["ssvi"] = None
    bad["provenance"]["snapshot_id"] = "synthetic-calendar-arbitrage"
    # Shrink the second slice's level so total variance falls between T1 and T2.
    bad["expiries"][1]["svi_raw"]["a"] = bad["expiries"][0]["svi_raw"]["a"] * 0.5
    bad["expiries"][1]["svi_raw"]["b"] = bad["expiries"][0]["svi_raw"]["b"] * 0.5
    (FIX / "surface_svi_bad.json").write_text(json.dumps(bad, indent=2) + "\n")


def fred_small() -> None:
    rng = np.random.default_rng(SEED)
    days = business_days(date(2024, 1, 2), 40)
    lines = ["observation_date,DGS10"]
    level = 4.0
    for i, d in enumerate(days):
        level += float(rng.normal(0, 0.03))
        value = "" if i in (5, 17, 18) else f"{level:.2f}"
        lines.append(f"{d.isoformat()},{value}")
    (FIX / "fred_small.csv").write_text("\n".join(lines) + "\n")


def stooq_small() -> None:
    rng = np.random.default_rng(SEED + 1)
    days = business_days(date(2024, 1, 2), 120)
    header = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"
    lines = [header]
    for ticker, s0 in (("SPY.US", 470.0), ("AAA.US", 50.0), ("BBB.US", 120.0)):
        s = s0
        for d in days:
            s *= float(np.exp(rng.normal(0.0002, 0.012)))
            o = s * float(np.exp(rng.normal(0, 0.003)))
            hi, lo = max(o, s) * 1.004, min(o, s) * 0.996
            vol = int(rng.integers(1_000_000, 5_000_000))
            lines.append(f"{ticker},D,{d:%Y%m%d},000000,{o:.4f},{hi:.4f},{lo:.4f},{s:.4f},{vol},0")
    (FIX / "stooq_small.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    surfaces()
    fred_small()
    stooq_small()
