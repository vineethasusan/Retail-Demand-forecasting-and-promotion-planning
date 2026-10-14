"""Promotion effectiveness: what did each promotion really add?

For every promoted day the model predicts demand twice: with the actual discount and
with no discount (the counterfactual). The difference is incremental units; profit
compares margin earned at the promo price against margin that would have been earned
anyway at full price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import X
from .data import MARGINS


def promo_effect(models: dict, feat: pd.DataFrame) -> pd.DataFrame:
    promo = feat[feat["on_promo"] == 1].copy()
    base = promo.copy()
    base["discount_pct"], base["on_promo"] = 0.0, 0
    m = models["p50"]
    promo["pred_with_promo"] = np.clip(m.predict(X(promo)), 0, None)
    promo["pred_without_promo"] = np.clip(m.predict(X(base)), 0, None)
    cost = promo["regular_price"] * (1 - promo["category"].map(MARGINS))
    promo_price = promo["regular_price"] * (1 - promo["discount_pct"])
    promo["incremental_units"] = promo["pred_with_promo"] - promo["pred_without_promo"]
    promo["profit_with_promo"] = promo["pred_with_promo"] * (promo_price - cost)
    promo["profit_without_promo"] = promo["pred_without_promo"] * (promo["regular_price"] - cost)
    promo["incremental_profit_aed"] = promo["profit_with_promo"] - promo["profit_without_promo"]
    return promo


def by_category(effect: pd.DataFrame) -> pd.DataFrame:
    g = effect.groupby("category").agg(
        promo_days=("units", "size"), avg_discount=("discount_pct", "mean"),
        incremental_units=("incremental_units", "sum"), base_units=("pred_without_promo", "sum"),
        incremental_profit_aed=("incremental_profit_aed", "sum"))
    g["uplift_pct"] = g["incremental_units"] / g["base_units"]
    g["verdict"] = np.where(g["incremental_profit_aed"] > 0, "profitable", "loses margin")
    return g.sort_values("incremental_profit_aed", ascending=False)


def what_if(models: dict, feat_rows: pd.DataFrame, discount: float) -> dict:
    """Predict units and profit for given future rows at a chosen discount (used by the dashboard)."""
    rows = feat_rows.copy()
    rows["discount_pct"], rows["on_promo"] = discount, int(discount > 0)
    units = np.clip(models["p50"].predict(X(rows)), 0, None)
    price = rows["regular_price"] * (1 - discount)
    cost = rows["regular_price"] * (1 - rows["category"].map(MARGINS))
    return {"units": float(units.sum()), "revenue_aed": float((units * price).sum()),
            "profit_aed": float((units * (price - cost)).sum())}
