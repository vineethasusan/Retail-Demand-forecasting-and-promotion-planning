"""Feature engineering for direct 1-28 day ahead forecasting.

Every lag is >= 28 days, so one model can forecast the next 4 weeks without feeding
its own predictions back in (no error compounding), and features for the forecast
window are always known at forecast time. Planned promotions and the event calendar
are known in advance, so they are legitimate future features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import calendar_features

HORIZON = 28
LAGS = [28, 35, 42, 49, 56, 364]

CAL_COLS = ["dow", "is_weekend", "is_friday", "month", "day_of_month", "is_payday_week", "is_summer",
            "ramadan", "eid_al_fitr", "eid_al_adha", "dsf", "white_friday", "national_day", "back_to_school",
            "days_to_eid_fitr", "last_10_ramadan"]
FEATURES = (["store_code", "sku_code", "category_code", "format_code", "regular_price", "discount_pct", "on_promo"]
            + CAL_COLS + [f"lag_{l}" for l in LAGS]
            + ["roll_mean_7_l28", "roll_mean_28_l28", "roll_std_28_l28", "same_dow_mean_4w_l28",
               "promo_days_last_28"])


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store", "sku", "date"]).reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    cal = calendar_features(pd.DatetimeIndex(df["date"].unique()))
    df = df.merge(cal, left_on="date", right_index=True, how="left")
    for col, name in [("store", "store_code"), ("sku", "sku_code"), ("category", "category_code"),
                      ("store_format", "format_code")]:
        df[name] = df[col].astype("category").cat.codes
    df["on_promo"] = (df["discount_pct"] > 0).astype(int)

    g = df.groupby(["store", "sku"], sort=False)["units"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    l28 = g.shift(28)
    k = df.groupby(["store", "sku"], sort=False).ngroup()
    df["roll_mean_7_l28"] = l28.groupby(k).transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["roll_mean_28_l28"] = l28.groupby(k).transform(lambda s: s.rolling(28, min_periods=1).mean())
    df["roll_std_28_l28"] = l28.groupby(k).transform(lambda s: s.rolling(28, min_periods=2).std())
    df["same_dow_mean_4w_l28"] = (g.shift(28) + g.shift(35) + g.shift(42) + g.shift(49)) / 4
    df["promo_days_last_28"] = df.groupby(["store", "sku"], sort=False)["on_promo"].transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).sum())
    return df


def X(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].astype(float).to_numpy()
