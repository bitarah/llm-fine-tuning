.PHONY: venv install install-mlx lint lint-fix typecheck test \
        test-phase1 test-phase2 test-phase3 test-phase4 test-phase5 \
        download-data process-data finetune finetune-pytorch \
        evaluate serve dashboard docker-build docker-up clean

VENV   := .venv
PYTHON := $(shell [ -f $(VENV)/bin/python ] && echo $(VENV)/bin/python || which python3.12 || which python3.11 || which python3.10)

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e .
	$(VENV)/bin/pip install -r requirements.txt
	@if [ "$$(uname -m)" = "arm64" ]; then \
		echo "Apple Silicon detected — installing mlx..."; \
		$(VENV)/bin/pip install -r requirements-mlx.txt; \
	fi
	@echo ""
	@echo "venv ready. Activate with:  source .venv/bin/activate"

install:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements.txt

install-mlx:
	$(PYTHON) -m pip install -r requirements-mlx.txt

lint:
	$(PYTHON) -m ruff check src/ tests/

lint-fix:
	$(PYTHON) -m ruff check --fix src/ tests/

typecheck:
	$(PYTHON) -m mypy src/ --ignore-missing-imports

test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

test-phase1:
	$(PYTHON) -m pytest tests/test_smoke.py -v

test-phase2:
	$(PYTHON) -m pytest tests/test_data.py -v

test-phase3:
	$(PYTHON) -m pytest tests/test_training.py -v

test-phase4:
	$(PYTHON) -m pytest tests/test_evaluation.py -v

test-phase5:
	$(PYTHON) -m pytest tests/test_serving.py -v

download-data:
	$(PYTHON) -m src.data.downloader

process-data:
	$(PYTHON) -m src.data.preprocessor
	$(PYTHON) -m src.data.splitter
	$(PYTHON) -m src.data.formatter

finetune:
	$(PYTHON) -m src.training.trainer_mlx

finetune-pytorch:
	$(PYTHON) -m src.training.trainer_pytorch

evaluate:
	$(PYTHON) -m src.evaluation.evaluator

serve:
	$(VENV)/bin/uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	PYTHONPATH=. $(VENV)/bin/streamlit run src/dashboard/app.py --server.port 8501

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
