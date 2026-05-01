.PHONY: install install-dev test test-fast data-download data-convert \
        run-dashboard run-benchmark figures help

PYTHON ?= python
BUNDLE  = models/delay_predictor_v4_score_best_online_safe_bundle.joblib

# ── Setup ─────────────────────────────────────────────────────────────────────

## Install all runtime dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt

## Install runtime + test dependencies
install-dev: install
	$(PYTHON) -m pip install pytest pytest-cov

# ── Data ──────────────────────────────────────────────────────────────────────

## Download all MBTA datasets (~20 GB; resume-safe)
data-download:
	$(PYTHON) src/data/download_data.py

## Check download status without re-downloading
data-status:
	$(PYTHON) src/data/download_data.py --status

## Convert raw CSVs to a single Parquet file (much faster for analysis)
data-convert:
	$(PYTHON) src/data/convert_all_to_parquet.py

# ── Analysis ──────────────────────────────────────────────────────────────────

## Run full Q1-Q7 base analysis on 2020-2024 training data
analysis:
	$(PYTHON) src/run_analysis.py

## Quick analysis on 2024 data only (no full dataset needed)
analysis-quick:
	$(PYTHON) src/run_analysis.py --quick

## Validation analysis on 2025-2026 holdout data
analysis-validate:
	$(PYTHON) src/run_analysis.py --validate

# ── Model training ────────────────────────────────────────────────────────────

## Train V3 GRU + time-series features (requires full dataset)
train-v3:
	$(PYTHON) src/models/train_delay_predictor_v3_fixed.py

## Train V5 NeuronSpark SNN and V6 Transformer (GPU recommended)
train-v5-v6:
	$(PYTHON) src/models/train_v5_v6_quick.py

## Run V4 LightGBM sweep and build the deployable bundle
train-v4:
	$(PYTHON) -m src.models.sweep_delay_predictor_v4 \
	  --max-train-rows 50000 \
	  --max-validation-rows 50000 \
	  --max-test-rows 10000 \
	  --candidates dummy,ridge,hist_gradient_boosting \
	  --feature-profiles v2_core \
	  --include-validation-in-final

# ── Inference dashboard ───────────────────────────────────────────────────────

## Start the realtime inference dashboard on http://127.0.0.1:8000
run-dashboard:
	$(PYTHON) -m src.inference.serve \
	  --bundle $(BUNDLE) \
	  --host 127.0.0.1 \
	  --port 8000

## Benchmark inference latency and generate report
run-benchmark:
	$(PYTHON) -m src.inference.benchmark_latency

## Generate the prediction-over-time trace figure
figures-realtime:
	$(PYTHON) -m src.inference.plot_predictions_over_time

## Regenerate V4 optimization story figures
figures-v4:
	$(PYTHON) -m src.visualization.create_v4_optimization_figures

# ── Tests ─────────────────────────────────────────────────────────────────────

## Run the full test suite
test:
	$(PYTHON) -m pytest tests/ -v

## Run only the fast unit tests (no model loading)
test-fast:
	$(PYTHON) -m pytest tests/ -v -m "not slow" --ignore=tests/test_deployment.py

## Run tests with coverage report
test-cov:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Boston Bus Equity — available make targets"
	@echo "==========================================="
	@grep -E '^##' Makefile | sed 's/## /  /'
	@echo ""
	@echo "  Override Python:  make <target> PYTHON=/path/to/python"
	@echo "  Override bundle:  make run-dashboard BUNDLE=models/your_bundle.joblib"
	@echo ""
