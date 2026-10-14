import numpy as np
import pandas as pd

from retailfc import forecast, inventory


def test_model_beats_baselines(small_sales, tmp_path):
    rep = forecast.backtest(small_sales, origins=["2026-03-06"], out_dir=str(tmp_path))
    f = rep["folds"][0]
    assert f["model_wape"] < f["baseline_recent_weeks_wape"]
    assert f["model_wape"] < f["baseline_last_year_wape"]
    assert 0.6 <= f["p10_p90_coverage"] <= 0.95


def test_perfect_forecast_never_loses_sales():
    demand = np.array([10, 12, 8, 15, 9, 11, 10, 13] * 4, dtype=float)
    r = inventory._simulate(demand, demand.copy(), lead=1, price=10, perishable=False, margin=0.3)
    assert r["lost_units"] < 1e-9


def test_under_forecasting_loses_sales():
    demand = np.full(28, 20.0)
    r = inventory._simulate(demand, demand * 0.5, lead=1, price=10, perishable=False, margin=0.3)
    assert r["lost_units"] > 100


def test_wape():
    assert forecast.wape(np.array([10, 10]), np.array([12, 8])) == 0.2
