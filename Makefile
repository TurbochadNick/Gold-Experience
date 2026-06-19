PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)

.PHONY: test generate-small generate-suite evaluate-synthetic run smoke-app

test:
	$(PYTHON) -m pytest

generate-small:
	$(PYTHON) scripts/smoke_generate.py --overwrite

generate-suite:
	$(PYTHON) scripts/generate_synthetic.py --overwrite

evaluate-synthetic:
	$(PYTHON) scripts/evaluate_synthetic_robustness.py

run:
	$(PYTHON) scripts/serve_api.py --host 127.0.0.1 --port 8000

smoke-app:
	$(PYTHON) scripts/smoke_app.py
