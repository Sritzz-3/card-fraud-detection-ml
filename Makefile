.PHONY: setup data run test all
setup:
	python3.11 -m venv .venv
	.venv/bin/pip install -r requirements.txt
data:
	.venv/bin/python -m src.data_generator --rows 1000000
run:
	.venv/bin/python -m src.run_pipeline
test:
	.venv/bin/python -m pytest -q
all: data run test
