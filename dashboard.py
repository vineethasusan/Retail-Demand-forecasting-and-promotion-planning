"""Retail planning dashboard: forecasts, promotion what-if and business impact.

    streamlit run dashboard.py      (run `make pipeline` first)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from retailfc.data import MARGINS  # noqa: E402
from retailfc.forecast import forecast_future  # noqa: E402

st.set_page_config(page_title="UAE Retail Demand Planner", page_icon="🛒", layout="wide")
OUT = Path("outputs")


@st.cache_data
def load_sales():
    return pd.read_csv("data/sales.csv.gz", parse_dates=["date"])


@st.cache_resource
def load_models():
    return joblib.load(OUT / "models.joblib")


@st.cache_data
def load_backtest():
    return pd.read_csv(OUT / "backtest_predictions.csv.gz", parse_dates=["date"])


if not (OUT / "models.joblib").exists():
    st.error("Run `make pipeline` first to train the models.")
    st.stop()

sales, models, bt = load_sales(), load_models(), load_backtest()
summary = json.loads((OUT / "summary.json").read_text())

st.title("🛒 UAE Retail Demand Planner")
st.caption("12 stores · 40 products · forecasts that understand Ramadan, Eid, DSF and the summer slowdown")

k = st.columns(4)
k[0].metric("Forecast error (WAPE)", f"{summary['forecast_wape']:.1%}",
            f"-{summary['error_reduction']:.0%} vs best baseline", delta_color="inverse")
k[1].metric("Shelf availability", f"{summary['fill_rate_forecast']:.1%}",
            f"+{(summary['fill_rate_forecast'] - summary['fill_rate_current']) * 100:.1f} pts")
k[2].metric("Lost margin avoided / 28 days",
            f"AED {(summary['lost_margin_current_aed'] - summary['lost_margin_forecast_aed']) / 3:,.0f}")
k[3].metric("Est. annual saving", f"AED {summary['annualised_saving_aed'] / 1e6:,.2f}M")

tab1, tab2, tab3 = st.tabs(["📈 Forecast explorer", "🏷️ Promotion what-if", "📊 Business impact"])

with tab1:
    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", sorted(sales["store"].unique()), index=1)
    product = c2.selectbox("Product", sorted(sales["product"].unique()), index=0)
    hist = sales[(sales.store == store) & (sales["product"] == product)].set_index("date")["units"]
    b = bt[(bt.store == store) & (bt["product"] == product)].set_index("date")
    st.markdown("**Backtest: forecast made 1–28 days ahead vs what actually sold**")
    window = hist[hist.index >= "2026-01-15"].rename("actual units").to_frame()
    window = window.join(b[["pred", "p10", "p90", "b1"]].rename(columns={
        "pred": "forecast", "p10": "low (P10)", "p90": "high (P90)", "b1": "recent-weeks baseline"}))
    st.line_chart(window, height=340)
    st.caption("Ramadan 2026: 18 Feb – 19 Mar · Eid al-Fitr: 20–22 Mar. Gaps = periods outside the backtest windows.")

with tab2:
    st.markdown("Plan a promotion for the **next 28 days** and see the predicted effect on units and profit.")
    c1, c2, c3 = st.columns(3)
    p_store = c1.selectbox("Store ", sorted(sales["store"].unique()), index=1, key="ps")
    p_product = c2.selectbox("Product ", sorted(sales["product"].unique()), index=28, key="pp")
    disc = c3.slider("Discount", 0, 40, 20, step=5) / 100
    sku = sales.loc[sales["product"] == p_product, "sku"].iloc[0]
    cat = sales.loc[sales["product"] == p_product, "category"].iloc[0]
    base = forecast_future(models, sales)
    promo = forecast_future(models, sales, discounts={(p_store, sku): disc})
    sel = lambda f: f[(f.store == p_store) & (f.sku == sku)]  # noqa: E731
    b0, b1 = sel(base), sel(promo)
    price = float(b0["regular_price"].iloc[0])
    m = MARGINS[cat]

    def profit(units, d):
        return float((units * (price * (1 - d) - price * (1 - m))).sum())

    r = st.columns(4)
    r[0].metric("Units (no promo)", f"{b0.p50.sum():,.0f}")
    r[1].metric("Units (with promo)", f"{b1.p50.sum():,.0f}", f"{b1.p50.sum() / max(b0.p50.sum(), 1) - 1:+.0%}")
    gain = profit(b1.p50, disc) - profit(b0.p50, 0)
    r[2].metric("Incremental profit", f"AED {gain:,.0f}", "profitable" if gain > 0 else "loses margin",
                delta_color="normal" if gain > 0 else "inverse")
    r[3].metric("Category margin", f"{m:.0%}")
    st.line_chart(pd.DataFrame({"no promo": b0.set_index("date").p50, "with promo": b1.set_index("date").p50}), height=260)
    st.caption("Item-level view: excludes basket and footfall effects, which a full promo review would add.")

with tab3:
    st.markdown((OUT / "report.md").read_text())
