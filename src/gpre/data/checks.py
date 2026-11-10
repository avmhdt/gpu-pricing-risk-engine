"""Data-quality rules run by ``gpre data check`` (§7c).

Each rule returns a :class:`CheckResult` with status ``pass``, ``warn``, ``fail`` or ``skip``.
The output (``results/data_check.json``) holds counts and dates only, never prices or VIX
values, so it can be committed. ``--strict`` turns warnings into failures; a skipped rule
(input not ingested) fails under ``--strict`` too.

Rules in M1: DQ-02 (yields), DQ-03 and DQ-04 (prices), DQ-05 (surface), DQ-09 (raw manifest).
DQ-06 and DQ-08 arrive with local vol (M3) and the Deribit ingest (M5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from gpre.data.rawstore import RawStore
from gpre.data.surface_io import Surface, check_arbitrage

Status = Literal["pass", "warn", "fail", "skip"]

LARGE_MOVE = 0.4
MAX_MISSING_RUN = 3
SPLIT_FACTORS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 1.5, 4 / 3)
SPLIT_TOL = 0.03
CALENDAR_SYMBOL = "SPY"


@dataclass
class CheckResult:
    rule: str
    name: str
    status: Status
    detail: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


def check_yields(path: Path) -> CheckResult:
    """DQ-02: no gaps left after the limited fill, and at least 6 tenors on every date."""
    if not path.exists():
        return CheckResult("DQ-02", "yields", "skip", f"{path.name} not ingested")
    y = pl.read_parquet(path)
    per_date = y.group_by("date").agg(pl.col("zero_cc").is_not_null().sum().alias("n"))
    short = per_date.filter(pl.col("n") < 6)
    stats = {
        "dates": per_date.height,
        "first": str(y["date"].min()),
        "last": str(y["date"].max()),
        "series": y["series_id"].n_unique(),
        "filled_values": int(y["filled"].sum()),
        "dates_with_lt_6_tenors": short.height,
    }
    status: Status = "pass" if short.height == 0 and stats["series"] == 11 else "fail"
    detail = "" if status == "pass" else f"{short.height} dates with fewer than 6 tenors"
    return CheckResult("DQ-02", "yields", status, detail, stats)


def _max_run(flags: list[bool]) -> int:
    longest = run = 0
    for f in flags:
        run = run + 1 if f else 0
        longest = max(longest, run)
    return longest


def check_prices(path: Path) -> list[CheckResult]:
    """DQ-03 (positive closes, calendar gaps against SPY) and DQ-04 (large moves vs splits)."""
    if not path.exists():
        return [
            CheckResult("DQ-03", "prices", "skip", f"{path.name} not ingested"),
            CheckResult("DQ-04", "corporate_actions", "skip", f"{path.name} not ingested"),
        ]
    p = pl.read_parquet(path)
    nonpos = p.filter(pl.col("close") <= 0).height
    symbols = sorted(p["symbol"].unique().to_list())
    cal_symbol = CALENDAR_SYMBOL if CALENDAR_SYMBOL in symbols else symbols[0]
    calendar = p.filter(pl.col("symbol") == cal_symbol).select("date").sort("date")
    gaps: dict[str, int] = {}
    short_names = []
    for sym in symbols:
        dates = set(p.filter(pl.col("symbol") == sym)["date"].to_list())
        flags = [d not in dates for d in calendar["date"].to_list()]
        # Only gaps inside the symbol's own history count as missing days.
        inside = flags[flags.index(False) :] if False in flags else flags
        while inside and inside[-1]:
            inside.pop()
        run = _max_run(inside)
        if run > MAX_MISSING_RUN:
            gaps[sym] = run
        if flags and flags[0]:
            short_names.append(sym)
    dq03_status: Status = "fail" if nonpos else ("warn" if gaps or short_names else "pass")
    dq03 = CheckResult(
        "DQ-03",
        "prices",
        dq03_status,
        "; ".join(
            [f"{nonpos} non-positive closes"] * bool(nonpos)
            + [f"gaps > {MAX_MISSING_RUN} days: {gaps}"] * bool(gaps)
            + [f"start after the {cal_symbol} calendar: {short_names}"] * bool(short_names)
        ),
        {
            "symbols": len(symbols),
            "rows": p.height,
            "calendar_symbol": cal_symbol,
            "calendar_days": calendar.height,
            "first": str(p["date"].min()),
            "last": str(p["date"].max()),
            "names_with_gaps": len(gaps),
            "names_starting_late": len(short_names),
        },
    )

    big = p.filter(pl.col("log_ret").abs() > LARGE_MOVE).select("symbol", "date", "log_ret")
    explained: list[str] = []
    unexplained: list[str] = []
    for sym, day, lr in big.iter_rows():
        ratio = float(np.exp(-lr))
        near = [f for f in SPLIT_FACTORS for r in (ratio, 1 / ratio) if abs(r / f - 1) < SPLIT_TOL]
        (explained if near else unexplained).append(f"{sym} {day}")
    # A move that matches a split ratio in a series that should be split-adjusted is a
    # failure; other large moves are reported, not dropped.
    dq04_status: Status = "fail" if explained else ("warn" if unexplained else "pass")
    dq04 = CheckResult(
        "DQ-04",
        "corporate_actions",
        dq04_status,
        "; ".join(
            [f"split-like moves (series not adjusted?): {explained}"] * bool(explained)
            + [f"unexplained |log ret| > {LARGE_MOVE}: {unexplained}"] * bool(unexplained)
        ),
        {"flagged": big.height, "split_like": len(explained), "unexplained": len(unexplained)},
    )
    return [dq03, dq04]


def check_surface(surface: Surface | None, label: str) -> CheckResult:
    """DQ-05 on the configured surface."""
    if surface is None:
        return CheckResult("DQ-05", "surface", "skip", "no surface")
    report = check_arbitrage(surface)
    stats = report.as_dict()
    stats.pop("failures")
    stats["surface"] = label
    return CheckResult(
        "DQ-05",
        "surface",
        "pass" if report.passed else "fail",
        "; ".join(report.failures),
        stats,
    )


def check_raw_manifest(store: RawStore) -> CheckResult:
    """DQ-09: every raw file in the manifest exists with its recorded SHA-256."""
    entries = store.load_manifest()
    if not entries:
        return CheckResult("DQ-09", "raw_manifest", "skip", "no raw manifest yet")
    problems = store.verify()
    return CheckResult(
        "DQ-09",
        "raw_manifest",
        "fail" if problems else "pass",
        "; ".join(problems[:20]),
        {"entries": len(entries), "problems": len(problems)},
    )


def summarize(results: list[CheckResult], strict: bool) -> tuple[bool, dict[str, Any]]:
    """Overall pass flag and the JSON payload for ``results/data_check.json``."""
    bad: set[str] = {"fail", "warn", "skip"} if strict else {"fail"}
    ok = all(r.status not in bad for r in results)
    return ok, {"strict": strict, "passed": ok, "checks": [asdict(r) for r in results]}
