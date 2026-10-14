import pandas as pd

from retailfc.calendar import calendar_features


def test_ramadan_moves_earlier_each_year():
    cal = calendar_features(pd.date_range("2025-01-01", "2026-12-31"))
    start_2025 = cal.index[cal["ramadan"] == 1][0]
    start_2026 = cal.index[(cal["ramadan"] == 1) & (cal.index.year == 2026)][0]
    assert 9 <= (start_2025.dayofyear - start_2026.dayofyear) <= 13


def test_uae_weekend_is_saturday_sunday():
    cal = calendar_features(pd.date_range("2026-06-01", periods=7))
    assert cal.loc[cal["is_weekend"] == 1].index.dayofweek.tolist() == [5, 6]


def test_last_10_days_of_ramadan():
    cal = calendar_features(pd.date_range("2026-02-15", "2026-03-25"))
    last10 = cal.index[cal["last_10_ramadan"] == 1]
    assert len(last10) == 10 and last10[-1] == pd.Timestamp("2026-03-19")
