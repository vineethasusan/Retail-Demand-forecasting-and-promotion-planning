"""Models, baselines and rolling-origin backtesting.

    python -m retailfc.forecast            # backtest 3 origins, compare with baselines, save report

Three 28-day backtest windows are chosen on purpose:
  * the start of Ramadan 2026 (Ramadan moved ~11 days earlier than in 2025),
  * the last 10 days of Ramadan + Eid al-Fitr (the biggest peak of the year),
  * a normal June period.
Baselines that rely on "same time last year" or "recent weeks" fail exactly when
the business needs the forecast most.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURES, HORIZON, X, build

ORIGINS = ["2026-02-10", "2026-03-06", "2026-06-02"]


def make_model(loss: str = "poisson", quantile: float | None = None, seed: int = 42):
    kw = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=60,
              l2_regularization=1.0, random_state=seed,
              categorical_features=[FEATURES.index(c) for c in ("store_code", "sku_code", "category_code",
                                                                  "format_code")])
    if quantile is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)
    return HistGradientBoostingRegressor(loss=loss, **kw)


@dataclass
class Fold:
    origin: pd.Timestamp
    train: pd.DataFrame
    test: pd.DataFrame


def split(feat: pd.DataFrame, origin: str) -> Fold:
    o = pd.Timestamp(origin)
    train = feat[(feat["date"] <= o) & feat["lag_56"].notna()]
    test = feat[(feat["date"] > o) & (feat["date"] <= o + pd.Timedelta(days=HORIZON))]
    return Fold(o, train, test)


# ------------------------------------------------------------------ baselines
def baseline_recent_weeks(feat: pd.DataFrame, fold: Fold) -> np.ndarray:
    """Average of the same weekday over the 4 weeks before the forecast origin."""
    hist = feat[(feat["date"] > fold.origin - pd.Timedelta(days=28)) & (feat["date"] <= fold.origin)]
    prof = hist.groupby(["store", "sku", "dow"])["units"].mean().rename("bl")
    return fold.test.join(prof, on=["store", "sku", "dow"])["bl"].fillna(0).to_numpy()


def baseline_last_year(feat: pd.DataFrame, fold: Fold) -> np.ndarray:
    """Same day last year (364 days, keeps the weekday), scaled by year-on-year growth."""
    hist = feat[(feat["date"] > fold.origin - pd.Timedelta(days=56)) & (feat["date"] <= fold.origin)]
    growth = (hist.groupby(["store", "sku"])["units"].sum() /
              hist.groupby(["store", "sku"])["lag_364"].sum().replace(0, np.nan)).clip(0.5, 2).rename("g")
    t = fold.test.join(growth, on=["store", "sku"])
    return (t["lag_364"] * t["g"].fillna(1)).fillna(0).to_numpy()


# -------------------------------------------------------------------- metrics
def wape(y, p) -> float:
    return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9))


def bias(y, p) -> float:
    return float((p.sum() - y.sum()) / max(y.sum(), 1e-9))


def coverage(y, lo, hi) -> float:
    return float(((y >= lo) & (y <= hi)).mean())


def backtest(sales: pd.DataFrame, origins=ORIGINS, out_dir: str = "outputs") -> dict:
    feat = build(sales)
    report = {"folds": [], "by_category": {}}
    cat_rows = []
    for origin in origins:
        f = split(feat, origin)
        y = f.test["units"].to_numpy()
        model = make_model().fit(X(f.train), f.train["units"])
        p = np.clip(model.predict(X(f.test)), 0, None)
        q10 = make_model(quantile=0.1).fit(X(f.train), f.train["units"]).predict(X(f.test))
        q90 = make_model(quantile=0.9).fit(X(f.train), f.train["units"]).predict(X(f.test))
        b1, b2 = baseline_recent_weeks(feat, f), baseline_last_year(feat, f)
        label = {"2026-02-10": "Ramadan start", "2026-03-06": "Last 10 days of Ramadan + Eid",
                 "2026-06-02": "Normal period (June)"}.get(origin, origin)
        report["folds"].append({
            "origin": origin, "period": label, "rows": int(len(y)),
            "model_wape": round(wape(y, p), 4), "model_bias": round(bias(y, p), 4),
            "baseline_recent_weeks_wape": round(wape(y, b1), 4), "baseline_last_year_wape": round(wape(y, b2), 4),
            "p10_p90_coverage": round(coverage(y, q10, q90), 3),
        })
        t = f.test.assign(pred=p, p10=q10, p90=q90, b1=b1, b2=b2, origin=origin)
        cat_rows.append(t)

    allc = pd.concat(cat_rows)
    for cat, g in allc.groupby("category"):
        report["by_category"][cat] = {"model_wape": round(wape(g.units.values, g.pred.values), 3),
                                      "baseline_recent_weeks_wape": round(wape(g.units.values, g.b1.values), 3),
                                      "baseline_last_year_wape": round(wape(g.units.values, g.b2.values), 3)}
    y, p = allc.units.values, allc.pred.values
    report["overall"] = {
        "model_wape": round(wape(y, p), 4),
        "baseline_recent_weeks_wape": round(wape(y, allc.b1.values), 4),
        "baseline_last_year_wape": round(wape(y, allc.b2.values), 4),
    }
    best_bl = min(report["overall"]["baseline_recent_weeks_wape"], report["overall"]["baseline_last_year_wape"])
    report["overall"]["error_reduction_vs_best_baseline"] = round(1 - report["overall"]["model_wape"] / best_bl, 3)
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, "backtest_report.json").write_text(json.dumps(report, indent=2))
    keep = ["origin", "date", "store", "sku", "product", "category", "regular_price", "discount_pct", "units",
            "pred", "p10", "p90", "b1", "b2"]
    allc[keep].to_csv(Path(out_dir) / "backtest_predictions.csv.gz", index=False)
    return report


def fit_production(sales: pd.DataFrame, out_dir: str = "outputs"):
    """Train on all history and save the point + quantile models for forecasting and the dashboard."""
    import joblib

    feat = build(sales)
    train = feat[feat["lag_56"].notna()]
    models = {"p50": make_model().fit(X(train), train["units"]),
              "p10": make_model(quantile=0.1).fit(X(train), train["units"]),
              "p90": make_model(quantile=0.9).fit(X(train), train["units"])}
    Path(out_dir).mkdir(exist_ok=True)
    joblib.dump(models, Path(out_dir) / "models.joblib")
    return models


if __name__ == "__main__":
    sales = pd.read_csv("data/sales.csv.gz", parse_dates=["date"])
    r = backtest(sales)
    print(f"{'Period':<32}{'Model':>8}{'Recent wks':>12}{'Last year':>11}{'P10-P90 cov.':>14}")
    for f in r["folds"]:
        print(f"{f['period']:<32}{f['model_wape']:>8.1%}{f['baseline_recent_weeks_wape']:>12.1%}"
              f"{f['baseline_last_year_wape']:>11.1%}{f['p10_p90_coverage']:>14.0%}")
    o = r["overall"]
    print(f"{'Overall WAPE':<32}{o['model_wape']:>8.1%}{o['baseline_recent_weeks_wape']:>12.1%}"
          f"{o['baseline_last_year_wape']:>11.1%}")
    print(f"Error reduction vs best baseline: {o['error_reduction_vs_best_baseline']:.0%}")


def future_frame(sales: pd.DataFrame, days: int = HORIZON, discounts: dict | None = None) -> pd.DataFrame:
    """Feature rows for the next `days` after the last date in `sales`.

    `discounts` maps (store, sku) -> planned discount for the whole window (default: no promo).
    """
    last = sales["date"].max()
    future_dates = pd.date_range(last + pd.Timedelta(days=1), periods=days, freq="D")
    keys = sales[["store", "store_format", "emirate", "sku", "product", "category", "regular_price"]].drop_duplicates(
        ["store", "sku"])
    fut = keys.merge(pd.DataFrame({"date": future_dates}), how="cross")
    fut["discount_pct"] = [(discounts or {}).get((s, k), 0.0) for s, k in zip(fut["store"], fut["sku"])]
    fut["units"] = np.nan
    hist = sales[sales["date"] > last - pd.Timedelta(days=400)]
    feat = build(pd.concat([hist, fut], ignore_index=True))
    return feat[feat["date"] > last]


def forecast_future(models: dict, sales: pd.DataFrame, days: int = HORIZON, discounts: dict | None = None):
    fut = future_frame(sales, days, discounts)
    Xf = X(fut)
    fut = fut.assign(p50=np.clip(models["p50"].predict(Xf), 0, None),
                     p10=np.clip(models["p10"].predict(Xf), 0, None),
                     p90=np.clip(models["p90"].predict(Xf), 0, None))
    return fut[["date", "store", "sku", "product", "category", "regular_price", "discount_pct", "p10", "p50", "p90"]]
