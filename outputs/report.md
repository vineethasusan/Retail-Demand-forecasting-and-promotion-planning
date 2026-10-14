# UAE Retail Demand Forecasting: results

## Forecast accuracy (WAPE, store x product x day, 28 days ahead)

| Period | Model | Recent-weeks baseline | Last-year baseline | P10–P90 coverage |
|---|---|---|---|---|
| Ramadan start | 36.4% | 46.0% | 54.4% | 75% |
| Last 10 days of Ramadan + Eid | 37.5% | 46.0% | 66.6% | 75% |
| Normal period (June) | 37.6% | 45.6% | 60.0% | 76% |

## Accuracy by category

| Category | Model | Recent weeks | Last year |
|---|---|---|---|
| Beverages | 36.1% | 49.1% | 64.4% |
| Dairy | 34.9% | 39.8% | 51.6% |
| Dates & Sweets | 39.0% | 58.3% | 72.5% |
| Electronics Accessories | 52.2% | 65.6% | 89.5% |
| Fresh Produce | 34.8% | 39.8% | 53.0% |
| Home & Summer | 55.2% | 63.3% | 79.4% |
| Perfume & Gifts | 51.9% | 61.7% | 82.7% |
| Personal Care | 40.8% | 51.5% | 67.5% |
| Rice & Staples | 36.9% | 44.8% | 63.4% |
| Stationery | 52.0% | 59.5% | 74.0% |

## Replenishment impact (12 stores, 40 products, 3 x 28 days)

| Policy | Fill rate | Lost margin (AED) | Spoilage (AED) | Holding (AED) | Total cost (AED) |
|---|---|---|---|---|---|
| current practice | 82.9% | 787,213 | 112,998 | 6,935 | 907,146 |
| forecast driven | 90.7% | 282,864 | 86,234 | 9,369 | 378,468 |

**Estimated saving: AED 176,226 per 28 days (~AED 2,297,233 a year).**

## Promotion effectiveness

| Category | Avg discount | Uplift | Incremental profit (AED) | Verdict |
|---|---|---|---|---|
| Perfume & Gifts | 19% | +79% | 26,073 | profitable |
| Electronics Accessories | 20% | +133% | 16,947 | profitable |
| Personal Care | 20% | +135% | -47,041 | loses margin |
| Stationery | 20% | +45% | -62,671 | loses margin |
| Home & Summer | 21% | +59% | -133,894 | loses margin |
| Dates & Sweets | 20% | +53% | -156,292 | loses margin |
| Beverages | 20% | +103% | -212,553 | loses margin |
| Fresh Produce | 21% | +34% | -213,005 | loses margin |
| Rice & Staples | 20% | +76% | -287,043 | loses margin |
| Dairy | 20% | +40% | -381,102 | loses margin |
