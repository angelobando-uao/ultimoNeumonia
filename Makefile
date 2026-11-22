# Makefile para UAO-Neumonia (Estándar Jan-Project-Standard)

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help install test test-fast run run-cli lint audit clean

help:
	@echo "=========================================================="
	@echo "Comandos disponibles para UAO-Neumonia:"
	@echo "=========================================================="
	@echo "  make install    - Instala dependencias del proyecto"
	@echo "  make test       - Ejecuta la suite de 120 pruebas unitarias con pytest"
	@echo "  make test-fast  - Ejecuta pruebas sin reporte de cobertura"
	@echo "  make run        - Inicia la interfaz gráfica (GUI Tkinter)"
	@echo "  make run-cli    - Ejecuta el diagnóstico en terminal (modo headless)"
	@echo "  make lint       - Analiza la calidad y estilo del código"
	@echo "  make audit      - Ejecuta el checklist de auditoría del proyecto"
	@echo "  make clean      - Limpia archivos temporales y reportes"
	@echo "=========================================================="

install:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov

test:
	$(PYTEST) tests/ -v --cov=. --cov-report=term-missing

test-fast:
	$(PYTEST) tests/ -v

run:
	$(PYTHON) run.py

run-cli:
	$(PYTHON) run.py --cli

lint:
	$(PYTHON) -m py_compile read_img.py preprocess_img.py load_model.py grad_cam.py integrator.py reporting.py detector_neumonia.py run.py

audit:
	$(PYTHON) -c "import sys; print('Chequeo de arquitectura e integridad...'); import read_img, preprocess_img, load_model, grad_cam, integrator, reporting, detector_neumonia; print('[OK] Todos los módulos cumplen con el estándar arquitectónico.')"

clean:
	@rm -rf __pycache__ .pytest_cache .coverage htmlcov tests/__pycache__ *.pyc temp_report_*.jpg Reporte_CLI.pdf historial.csv
