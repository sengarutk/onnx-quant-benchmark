.PHONY: all pipeline setup manifest baselines export quantize benchmark report smoke-test test lint clean help

all: pipeline

pipeline:
	python scripts/run_full_pipeline.py

setup:
	bash scripts/setup_env.sh

manifest:
	python scripts/generate_env_manifest.py

baselines:
	python scripts/prepare_sample_data.py
	python scripts/run_pytorch_baselines.py

export:
	python scripts/export_and_validate.py

quantize:
	python scripts/prepare_calibration_data.py
	python scripts/quantize_and_validate.py

benchmark:
	python scripts/benchmark_all.py

report:
	python scripts/generate_report.py

smoke-test:
	bash scripts/smoke_test.sh

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	python -m py_compile src/**/*.py tests/*.py scripts/*.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .coverage htmlcov

help:
	@echo "Available Makefile Targets:"
	@echo "  make pipeline    - Run full 9-stage end-to-end benchmark pipeline"
	@echo "  make setup       - Configure virtual environment and dependencies"
	@echo "  make manifest    - Introspect hardware and write environment manifest"
	@echo "  make baselines   - Generate datasets and PyTorch reference baselines"
	@echo "  make export      - Export PyTorch models to ONNX and simplify graphs"
	@echo "  make quantize    - Run FP16 conversion and static INT8 PTQ"
	@echo "  make benchmark   - Run multi-session MAD stability benchmarking suite"
	@echo "  make report      - Aggregate results, generate tables and 300-DPI plots"
	@echo "  make smoke-test  - Run end-to-end pipeline smoke test"
	@echo "  make test        - Run complete pytest suite with code coverage"
	@echo "  make clean       - Remove temporary artifacts and caches"
