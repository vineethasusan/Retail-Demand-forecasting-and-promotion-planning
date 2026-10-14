"""Turn forecasts into replenishment orders and measure the business impact.

Simulates daily ordering for every store x product over each 28-day backtest window:
  * lead time: 1 day for fresh and dairy, 3 days for everything else
  * order-up-to policy: order enough to cover forecast demand until the next delivery
  * lost sales when shelves are empty; spoilage for perishables

Two policies are compared on the same actual demand:
  * current practice: recent-weeks average + 20% safety stock
  * forecast-driven: model median + safety stock sized from the P90 forecast
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORY_EFFECTS, MARGINS

HOLDING_RATE = 0.0008    # cost of holding stock per day, share of cost price
PERISHABLE_SPOIL = 0.12  # share of leftover stock spoiled per day for short-life items


def _simulate(demand, forecast, lead, price, perishable, margin):
    n = len(demand)
    on_hand = forecast[: lead + 1].sum()
    pipeline = [0.0] * (n + lead + 2)
    lost = sold = spoiled = holding = 0.0
    for t in range(n):
        on_hand += pipeline[t]
        s = min(on_hand, demand[t])
        sold += s
        lost += demand[t] - s
        on_hand -= s
        if perishable:
            sp = on_hand * PERISHABLE_SPOIL
            spoiled += sp
            on_hand -= sp
        holding += on_hand
        target = forecast[t + 1: t + 1 + lead].sum() if t + 1 < n else 0.0  # cover demand until this order lands
        inbound = sum(pipeline[t + 1: t + 1 + lead])
        order = max(0.0, target - on_hand - inbound)
        if t + lead < len(pipeline):
            pipeline[t + lead] += order
    cost = price * (1 - margin)
    return {"demand": demand.sum(), "sold": sold, "lost_units": lost, "lost_margin_aed": lost * price * margin,
            "spoiled_units": spoiled, "spoilage_aed": spoiled * cost, "holding_aed": holding * cost * HOLDING_RATE}


def simulate_policies(preds: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (origin, store, sku), g in preds.sort_values("date").groupby(["origin", "store", "sku"]):
        cat = g["category"].iloc[0]
        shelf = CATEGORY_EFFECTS[cat][-1]
        perishable = shelf <= 10
        lead = 1 if perishable else 3
        demand = g["units"].to_numpy(float)
        price = float(g["regular_price"].iloc[0])
        policies = {
            "current_practice": g["b1"].to_numpy(float) * 1.2,
            # median + a share of the gap to P90: buffer only where uncertainty is high, and less
            # for perishables, where an unsold unit is wasted (category-specific service levels)
            "forecast_driven": g["pred"].to_numpy(float)
            + (0.15 if perishable else 0.6) * np.clip(g["p90"] - g["pred"], 0, None).to_numpy(float),
        }
        for name, fc in policies.items():
            r = _simulate(demand, fc, lead, price, perishable, MARGINS[cat])
            out.append({"origin": origin, "store": store, "sku": sku, "category": cat, "policy": name, **r})
    return pd.DataFrame(out)


def summarize(sim: pd.DataFrame) -> pd.DataFrame:
    s = sim.groupby("policy")[["demand", "sold", "lost_units", "lost_margin_aed", "spoilage_aed", "holding_aed"]].sum()
    s["fill_rate"] = s["sold"] / s["demand"]
    s["total_cost_aed"] = s["lost_margin_aed"] + s["spoilage_aed"] + s["holding_aed"]
    return s
