export PYTHONPATH := src

.PHONY: install data pipeline backtest dashboard test lint docker

install:    ## install dependencies
	pip install -r requirements-dev.txt
data:       ## generate 2 years of daily sales (12 stores x 40 products)
	python -m retailfc.data
backtest:   ## forecast accuracy vs baselines on Ramadan, Eid and normal periods
	python -m retailfc.forecast
pipeline:   ## everything: backtest, production models, replenishment and promo analysis -> outputs/report.md
	python -m retailfc.pipeline
dashboard:  ## planner UI on :8501
	streamlit run dashboard.py
test:
	pytest -q
lint:
	ruff check .
docker:
	docker build -t uae-retail-forecasting . && docker run -p 8501:8501 uae-retail-forecasting
