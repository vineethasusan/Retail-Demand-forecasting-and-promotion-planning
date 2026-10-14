import numpy as np
import pandas as pd

from retailfc.features import FEATURES, HORIZON, build


def test_dates_sell_more_in_ramadan(small_sales):
    d = small_sales[small_sales["category"] == "Dates & Sweets"]
    ram = d[(d.date >= "2026-02-18") & (d.date <= "2026-03-19")].groupby("date").units.sum().mean()
    normal = d[(d.date >= "2026-04-15") & (d.date <= "2026-05-15")].groupby("date").units.sum().mean()
    assert ram > 1.8 * normal


def test_promotions_lift_sales(small_sales):
    g = small_sales.groupby("on_promo" if "on_promo" in small_sales else small_sales["discount_pct"] > 0).units.mean()
    assert g.loc[True] > g.loc[False]


def test_all_lags_respect_the_horizon(small_sales):
    feat = build(small_sales)
    one = feat[(feat.store == "Al Barsha") & (feat.sku == "SKU001")].set_index("date")
    d = pd.Timestamp("2026-03-01")
    assert one.loc[d, "lag_28"] == one.loc[d - pd.Timedelta(days=28), "units"]
    lag_cols = [c for c in FEATURES if c.startswith("lag_")]
    assert min(int(c.split("_")[1]) for c in lag_cols) >= HORIZON


def test_rolling_features_only_use_data_older_than_horizon(small_sales):
    feat = build(small_sales)
    one = feat[(feat.store == "Al Barsha") & (feat.sku == "SKU001")].set_index("date")
    d = pd.Timestamp("2026-03-01")
    window = one.loc[d - pd.Timedelta(days=34): d - pd.Timedelta(days=28), "units"]
    assert np.isclose(one.loc[d, "roll_mean_7_l28"], window.mean())
