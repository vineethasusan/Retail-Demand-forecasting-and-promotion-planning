"""End-to-end run: data -> backtest -> production models -> inventory simulation -> promo analysis.

    python -m retailfc.pipeline
Writes everything to outputs/ (report.md is the executive summary).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import data, forecast, inventory, promo
from .features import build

OUT = Path("outputs")


def main():
    t0 = time.time()
    path = Path("data/sales.csv.gz")
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        data.generate().to_csv(path, index=False)
    sales = pd.read_csv(path, parse_dates=["date"])

    print("1/4 Backtesting (3 x 28-day windows)…")
    report = forecast.backtest(sales, out_dir=str(OUT))
    preds = pd.read_csv(OUT / "backtest_predictions.csv.gz", parse_dates=["date"])

    print("2/4 Simulating replenishment…")
    sim = inventory.simulate_policies(preds)
    inv = inventory.summarize(sim)
    inv.to_csv(OUT / "inventory_policy_comparison.csv")

    print("3/4 Training production models on full history…")
    models = forecast.fit_production(sales, out_dir=str(OUT))

    print("4/4 Measuring promotion effectiveness…")
    feat = build(sales)
    eff = promo.promo_effect(models, feat[feat["lag_56"].notna()])
    cat = promo.by_category(eff)
    cat.to_csv(OUT / "promo_effectiveness.csv")

    cur, new = inv.loc["current_practice"], inv.loc["forecast_driven"]
    windows = preds["origin"].nunique()
    saving_per_window = (cur["total_cost_aed"] - new["total_cost_aed"]) / windows
    summary = {
        "forecast_wape": report["overall"]["model_wape"],
        "best_baseline_wape": min(report["overall"]["baseline_recent_weeks_wape"],
                                  report["overall"]["baseline_last_year_wape"]),
        "error_reduction": report["overall"]["error_reduction_vs_best_baseline"],
        "fill_rate_current": round(float(cur["fill_rate"]), 4), "fill_rate_forecast": round(float(new["fill_rate"]), 4),
        "lost_margin_current_aed": round(float(cur["lost_margin_aed"])), "lost_margin_forecast_aed": round(float(new["lost_margin_aed"])),
        "spoilage_current_aed": round(float(cur["spoilage_aed"])), "spoilage_forecast_aed": round(float(new["spoilage_aed"])),
        "cost_saving_per_28_days_aed": round(float(saving_per_window)),
        "annualised_saving_aed": round(float(saving_per_window * 365 / 28)),
        "promo_categories_losing_margin": cat.index[cat["verdict"] == "loses margin"].tolist(),
        "runtime_s": round(time.time() - t0),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# UAE Retail Demand Forecasting: results", "",
             "## Forecast accuracy (WAPE, store x product x day, 28 days ahead)", "",
             "| Period | Model | Recent-weeks baseline | Last-year baseline | P10–P90 coverage |", "|---|---|---|---|---|"]
    lines += [f"| {f['period']} | {f['model_wape']:.1%} | {f['baseline_recent_weeks_wape']:.1%} | "
              f"{f['baseline_last_year_wape']:.1%} | {f['p10_p90_coverage']:.0%} |" for f in report["folds"]]
    lines += ["", "## Accuracy by category", "", "| Category | Model | Recent weeks | Last year |", "|---|---|---|---|"]
    lines += [f"| {c} | {v['model_wape']:.1%} | {v['baseline_recent_weeks_wape']:.1%} | {v['baseline_last_year_wape']:.1%} |"
              for c, v in report["by_category"].items()]
    lines += ["", "## Replenishment impact (12 stores, 40 products, 3 x 28 days)", "",
              "| Policy | Fill rate | Lost margin (AED) | Spoilage (AED) | Holding (AED) | Total cost (AED) |",
              "|---|---|---|---|---|---|"]
    for name, r in inv.iterrows():
        lines.append(f"| {name.replace('_', ' ')} | {r.fill_rate:.1%} | {r.lost_margin_aed:,.0f} | {r.spoilage_aed:,.0f} | "
                     f"{r.holding_aed:,.0f} | {r.total_cost_aed:,.0f} |")
    lines += ["", f"**Estimated saving: AED {summary['cost_saving_per_28_days_aed']:,} per 28 days "
              f"(~AED {summary['annualised_saving_aed']:,} a year).**", "",
              "## Promotion effectiveness", "", "| Category | Avg discount | Uplift | Incremental profit (AED) | Verdict |",
              "|---|---|---|---|---|"]
    lines += [f"| {c} | {r.avg_discount:.0%} | {r.uplift_pct:+.0%} | {r.incremental_profit_aed:,.0f} | {r.verdict} |"
              for c, r in cat.iterrows()]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\nDone in {summary['runtime_s']}s. Executive summary: outputs/report.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
