"""UAE retail calendar: the events that move demand.

Islamic dates follow the lunar calendar and shift ~11 days earlier each year, so a
model that only knows "month of year" gets Ramadan wrong. Dates below are approximate
(the exact start depends on moon sighting) and are easy to update.
"""
from __future__ import annotations

import pandas as pd

EVENTS = {
    # name: list of (start, end) inclusive
    "ramadan": [("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-29"), ("2026-02-18", "2026-03-19")],
    "eid_al_fitr": [("2024-04-10", "2024-04-12"), ("2025-03-30", "2025-04-01"), ("2026-03-20", "2026-03-22")],
    "eid_al_adha": [("2024-06-16", "2024-06-18"), ("2025-06-06", "2025-06-08"), ("2026-05-27", "2026-05-29")],
    "dsf": [("2023-12-08", "2024-01-14"), ("2024-12-06", "2025-01-12"), ("2025-12-05", "2026-01-11")],  # Dubai Shopping Festival
    "white_friday": [("2024-11-27", "2024-12-01"), ("2025-11-26", "2025-11-30")],
    "national_day": [("2024-12-01", "2024-12-03"), ("2025-12-01", "2025-12-03")],
    "back_to_school": [("2024-08-15", "2024-09-01"), ("2025-08-14", "2025-08-31")],
}


def _flag(dates: pd.DatetimeIndex, ranges) -> pd.Series:
    out = pd.Series(0, index=dates)
    for a, b in ranges:
        out[(dates >= a) & (dates <= b)] = 1
    return out


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    df = pd.DataFrame(index=dates)
    df["dow"] = dates.dayofweek                       # UAE weekend: Saturday (5) and Sunday (6)
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)    # half working day
    df["month"] = dates.month
    df["day_of_month"] = dates.day
    df["is_payday_week"] = (dates.day >= 25).astype(int) | (dates.day <= 2).astype(int)
    df["is_summer"] = dates.month.isin([7, 8]).astype(int)   # residents travel, footfall drops
    for name, ranges in EVENTS.items():
        df[name] = _flag(dates, ranges).values
    # Days until Eid al-Fitr: pre-Eid shopping peaks in the last 10 days of Ramadan
    eid_starts = pd.to_datetime([a for a, _ in EVENTS["eid_al_fitr"]])
    days_to = []
    for d in dates:
        future = eid_starts[eid_starts >= d]
        days_to.append(min(int((future[0] - d).days), 60) if len(future) else 60)
    df["days_to_eid_fitr"] = days_to
    df["last_10_ramadan"] = ((df["ramadan"] == 1) & (df["days_to_eid_fitr"] <= 10)).astype(int)
    return df


EVENT_COLS = list(EVENTS) + ["last_10_ramadan"]
