"""Cloud GPU spend log (NFR-12, D-02).

``results/cloud_spend.csv`` gets one row per cloud session, written before the session ends,
with the cost read from the provider's billing page. GPU work pauses once logged spend reaches
80% of the approved cap until the owner confirms; the cap itself is never exceeded.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SPEND_COLUMNS = (
    "date",
    "provider",
    "instance_type",
    "gpu_model",
    "start_utc",
    "stop_utc",
    "billed_hours",
    "cost_usd",
    "cumulative_usd",
    "end_state",
    "notes",
)
PAUSE_FRACTION = 0.8


class BudgetError(RuntimeError):
    """Logged spend would pass the pause threshold or the cap."""


@dataclass(frozen=True)
class SpendStatus:
    cumulative_usd: float
    cap_usd: float

    @property
    def pause_at_usd(self) -> float:
        return PAUSE_FRACTION * self.cap_usd

    @property
    def paused(self) -> bool:
        return self.cumulative_usd >= self.pause_at_usd

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.cumulative_usd


def read_spend(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def status(path: Path, cap_usd: float) -> SpendStatus:
    rows = read_spend(path)
    total = sum(float(r["cost_usd"]) for r in rows)
    return SpendStatus(round(total, 6), cap_usd)


def check_can_start(path: Path, cap_usd: float) -> SpendStatus:
    """Raise unless a new GPU session may start (below 80% of the cap)."""
    st = status(path, cap_usd)
    if st.paused:
        raise BudgetError(
            f"logged spend {st.cumulative_usd:.2f} USD has reached 80% of the {cap_usd:.0f} USD "
            "cap; GPU work pauses until the owner confirms"
        )
    return st


def append_session(
    path: Path,
    *,
    provider: str,
    instance_type: str,
    gpu_model: str,
    start: datetime,
    stop: datetime,
    cost_usd: float,
    end_state: str,
    cap_usd: float,
    notes: str = "",
) -> SpendStatus:
    """Append one session row; refuse a row that would take spend past the cap."""
    if stop <= start:
        raise ValueError("stop must be after start")
    if end_state not in ("hibernated", "deleted"):
        raise ValueError("a session must end hibernated or deleted (never plain shut-off)")
    before = status(path, cap_usd)
    total = round(before.cumulative_usd + cost_usd, 6)
    if total > cap_usd:
        raise BudgetError(f"spend {total:.2f} USD would exceed the {cap_usd:.0f} USD cap")
    new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPEND_COLUMNS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(
            {
                "date": start.date().isoformat(),
                "provider": provider,
                "instance_type": instance_type,
                "gpu_model": gpu_model,
                "start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "stop_utc": stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "billed_hours": round((stop - start).total_seconds() / 3600, 4),
                "cost_usd": f"{cost_usd:.4f}",
                "cumulative_usd": f"{total:.4f}",
                "end_state": end_state,
                "notes": notes,
            }
        )
    return SpendStatus(total, cap_usd)
