.PHONY: setup train run test lint type docker

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	python scripts/train.py

run:
	uvicorn app.main:app --reload

test:
	pytest -q

lint:
	ruff check .

type:
	mypy app

docker:
	docker build -t edps-mlops-demo:latest .
