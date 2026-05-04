.PHONY: install benchmark figures clean

install:
	pip install -e ".[dev]"

benchmark: install
	@echo "=== Running WiFi-TTA-Bench (full suite) ==="
	python scripts/run_widar_full_tta.py
	python scripts/run_all_neurips_experiments.py
	python scripts/generate_figures.py
	@echo "=== Done. Results in outputs/ ==="

figures:
	python scripts/generate_figures.py

test:
	pytest --tb=short -q

lint:
	ruff check . && ruff format --check . && mypy pinn4csi/

clean:
	rm -rf outputs/widar_full_tta outputs/neurips_experiments outputs/db_ablations
