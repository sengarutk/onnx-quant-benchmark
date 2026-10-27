PY ?= python3

.PHONY: help setup data train baselines export quantize validate benchmark report experiments paper pipeline smoke-test test lint clean

help:
	@echo "make setup        Install Python dependencies (requirements.txt)"
	@echo "make data         Generate synthetic test + calibration splits"
	@echo "make train        Train the detector and autoencoder from scratch"
	@echo "make baselines    PyTorch FP32 reference metrics"
	@echo "make export       ONNX export, simplification, parity audit"
	@echo "make quantize     FP16 conversion + static INT8 PTQ + quality gates"
	@echo "make validate     Runtime backend validation (docs/runtimes.md)"
	@echo "make benchmark    One full benchmark session (results/raw/session_*)"
	@echo "make report       runs.csv, tables, figures, decision matrix"
	@echo "make experiments  Q-Aware NMS ablation + scalability sweep"
	@echo "make paper        Paper tables/macros, PDF (if pdflatex) and overleaf_paper.zip"
	@echo "make pipeline     Everything above, in order"
	@echo "make smoke-test   Unit tests + quick benchmark into a temp dir (no tracked files modified)"
	@echo "make test         pytest with coverage"

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/prepare_sample_data.py
	$(PY) scripts/prepare_calibration_data.py

train:
	$(PY) scripts/train_models.py

baselines:
	$(PY) scripts/run_pytorch_baselines.py

export:
	$(PY) scripts/export_and_validate.py

quantize:
	$(PY) scripts/quantize_and_validate.py

validate:
	$(PY) scripts/validate_runtimes.py

benchmark:
	$(PY) scripts/benchmark_all.py

report:
	$(PY) scripts/generate_report.py
	$(PY) scripts/update_readme.py

experiments:
	$(PY) src/experiments/run_q_aware_ablation.py
	$(PY) src/experiments/run_scalability_sweep.py

paper:
	$(PY) scripts/build_paper.py

pipeline:
	$(PY) scripts/run_full_pipeline.py

smoke-test:
	bash scripts/smoke_test.sh

test:
	$(PY) -m pytest tests/ --cov=src --cov-report=term-missing

lint:
	$(PY) -m compileall -q src scripts tests

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov
