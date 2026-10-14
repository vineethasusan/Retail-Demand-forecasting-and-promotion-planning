"""Synthetic daily sales for a UAE grocery & general-merchandise chain.

12 stores x 40 products x 2 years. Demand responds the way UAE retail actually does:
weekends, paydays, summer slowdown, Ramadan (category-specific), the pre-Eid rush,
Eid, Dubai Shopping Festival, White Friday, back-to-school, plus promotions with
price elasticity and a post-promotion dip from pantry loading.

    python -m retailfc.data --out data/sales.csv.gz
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import calendar_features

STORES = [
    # name, format, emirate, traffic multiplier
    ("Dubai Marina", "supermarket", "Dubai", 1.0), ("Al Barsha", "hypermarket", "Dubai", 1.8),
    ("Deira", "hypermarket", "Dubai", 1.7), ("Jumeirah", "supermarket", "Dubai", 0.9),
    ("Silicon Oasis", "supermarket", "Dubai", 0.8), ("Business Bay", "express", "Dubai", 0.5),
    ("Al Khan", "hypermarket", "Sharjah", 1.5), ("Al Nahda", "supermarket", "Sharjah", 1.0),
    ("Khalifa City", "hypermarket", "Abu Dhabi", 1.6), ("Al Reem", "supermarket", "Abu Dhabi", 0.9),
    ("Al Ain Central", "supermarket", "Abu Dhabi", 0.8), ("Ajman Corniche", "express", "Ajman", 0.5),
]

# category: (ramadan, last10, eid, dsf, white_friday, back_to_school, summer, elasticity, shelf_life_days)
CATEGORY_EFFECTS = {
    "Dates & Sweets":       (2.6, 1.3, 1.6, 1.1, 1.0, 1.0, 0.85, 1.6, 60),
    "Beverages":            (1.5, 1.1, 1.3, 1.1, 1.0, 1.0, 1.15, 2.8, 180),
    "Fresh Produce":        (1.35, 1.1, 1.2, 1.0, 1.0, 1.0, 0.85, 1.2, 5),
    "Dairy":                (1.25, 1.05, 1.1, 1.0, 1.0, 1.0, 0.85, 1.3, 10),
    "Rice & Staples":       (1.5, 1.2, 1.1, 1.0, 1.0, 1.0, 0.9, 2.2, 365),
    "Personal Care":        (0.95, 1.05, 1.1, 1.3, 1.4, 1.0, 0.85, 3.2, 720),
    "Perfume & Gifts":      (1.1, 1.8, 2.0, 1.7, 1.6, 1.0, 0.7, 2.6, 720),
    "Electronics Accessories": (0.9, 1.1, 1.2, 1.6, 2.2, 1.4, 0.9, 3.5, 720),
    "Stationery":           (0.8, 1.0, 1.0, 1.0, 1.1, 3.2, 0.8, 1.4, 720),
    "Home & Summer":        (0.9, 1.0, 1.0, 1.3, 1.5, 1.0, 1.5, 1.8, 720),
}

# Gross margin on regular price, by category (grocery is thin, gifts and electronics are rich)
MARGINS = {"Dates & Sweets": 0.30, "Beverages": 0.25, "Fresh Produce": 0.20, "Dairy": 0.18, "Rice & Staples": 0.15,
           "Personal Care": 0.35, "Perfume & Gifts": 0.50, "Electronics Accessories": 0.40, "Stationery": 0.40,
           "Home & Summer": 0.35}

PRODUCTS = {
    "Dates & Sweets": [("Medjool Dates 1kg", 45), ("Khalas Dates 500g", 18), ("Maamoul Assorted 400g", 22), ("Luqaimat Mix 500g", 12)],
    "Beverages": [("Laban Up 1L", 5), ("Vimto Cordial 710ml", 14), ("Mineral Water 6x1.5L", 9), ("Orange Juice 1L", 8)],
    "Fresh Produce": [("Bananas 1kg", 6), ("Tomatoes 1kg", 5), ("Onions 1kg", 4), ("Cucumber 1kg", 6)],
    "Dairy": [("Fresh Milk 2L", 11), ("Labneh 400g", 10), ("Halloumi 250g", 14), ("Greek Yogurt 500g", 9)],
    "Rice & Staples": [("Basmati Rice 5kg", 42), ("Sunflower Oil 1.8L", 17), ("Chickpeas 1kg", 8), ("Sugar 2kg", 9)],
    "Personal Care": [("Shampoo 400ml", 19), ("Body Wash 500ml", 16), ("Toothpaste 120g", 8), ("Diapers Size 4", 55)],
    "Perfume & Gifts": [("Oud Perfume 100ml", 180), ("Bakhoor Box", 65), ("Gift Hamper", 150), ("Chocolate Box 500g", 48)],
    "Electronics Accessories": [("Power Bank 10000mAh", 79), ("USB-C Cable", 25), ("Wireless Earbuds", 149), ("Phone Case", 39)],
    "Stationery": [("School Bag", 89), ("Notebook 5-pack", 15), ("Pencil Case Set", 22), ("Lunch Box", 29)],
    "Home & Summer": [("Pedestal Fan", 129), ("Cooler Box 25L", 99), ("Sunscreen SPF50", 45), ("Beach Towel", 35)],
}
BASE_UNITS = {"Fresh Produce": 40, "Dairy": 35, "Beverages": 30, "Rice & Staples": 14, "Dates & Sweets": 8,
              "Personal Care": 9, "Perfume & Gifts": 2.5, "Electronics Accessories": 3, "Stationery": 3,
              "Home & Summer": 2.5}


def generate(start: str = "2024-07-01", end: str = "2026-06-30", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    cal = calendar_features(dates)
    n = len(dates)
    trend = np.linspace(1.0, 1.08, n)                      # modest like-for-like growth
    dow_mult = np.array([0.95, 0.92, 0.95, 1.0, 1.08, 1.22, 1.15])[cal["dow"].to_numpy()]
    payday = 1 + 0.08 * cal["is_payday_week"].to_numpy()

    rows = []
    sku_id = 0
    products = []
    for cat, items in PRODUCTS.items():
        for name, price in items:
            sku_id += 1
            products.append((f"SKU{sku_id:03d}", name, cat, price))

    for s_idx, (store, fmt, emirate, traffic) in enumerate(STORES):
        for sku, name, cat, price in products:
            ram, last10, eid, dsf, wf, bts, summer, elast, _ = CATEGORY_EFFECTS[cat]
            base = BASE_UNITS[cat] * traffic * rng.uniform(0.6, 1.4)
            if fmt == "express" and cat in ("Electronics Accessories", "Home & Summer", "Stationery"):
                base *= 0.3
            mult = (trend * dow_mult * payday
                    * np.where(cal["ramadan"], ram, 1.0)
                    * np.where(cal["last_10_ramadan"], last10, 1.0)
                    * np.where(cal["eid_al_fitr"] | cal["eid_al_adha"], eid, 1.0)
                    * np.where(cal["dsf"], dsf, 1.0)
                    * np.where(cal["white_friday"], wf, 1.0)
                    * np.where(cal["back_to_school"], bts, 1.0)
                    * np.where(cal["is_summer"], summer, 1.0)
                    * np.where(cal["national_day"], 1.1, 1.0))

            # Promotions: ~8% of days, in 7-14 day windows, 10-30% off
            discount = np.zeros(n)
            d = int(rng.integers(0, 40))
            while d < n:
                d += int(rng.integers(45, 120))
                if d >= n:
                    break
                length = int(rng.integers(7, 15))
                discount[d:d + length] = rng.choice([0.1, 0.15, 0.2, 0.25, 0.3])
                d += length
            promo_mult = (1 - discount) ** (-elast)
            # post-promo dip for storable goods (customers stocked up)
            dip = np.ones(n)
            ends = np.where((discount[:-1] > 0) & (discount[1:] == 0))[0] + 1
            if cat in ("Rice & Staples", "Personal Care", "Beverages"):
                for e in ends:
                    dip[e:e + 7] *= 0.85

            mu = base * mult * promo_mult * dip
            # negative binomial noise (overdispersed counts, like real store sales)
            k = 6.0
            units = rng.negative_binomial(k, k / (k + mu))
            rows.append(pd.DataFrame({
                "date": dates, "store": store, "store_format": fmt, "emirate": emirate, "sku": sku,
                "product": name, "category": cat, "regular_price": float(price),
                "discount_pct": discount, "units": units,
            }))
    df = pd.concat(rows, ignore_index=True)
    df["price"] = (df["regular_price"] * (1 - df["discount_pct"])).round(2)
    df["revenue"] = (df["price"] * df["units"]).round(2)
    return df


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sales.csv.gz")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = generate(seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df):,} rows | {df.store.nunique()} stores | {df.sku.nunique()} SKUs | "
          f"{df.date.min().date()} to {df.date.max().date()} | AED {df.revenue.sum() / 1e6:,.1f}M revenue")
