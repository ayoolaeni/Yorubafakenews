.PHONY: setup corpus preprocess train evaluate app api test lint

setup:
	python -m venv venv
	venv/bin/pip install -r requirements.txt || venv/Scripts/pip install -r requirements.txt

corpus:
	python -m src.scraping.run_scrapers --source all --class genuine --limit 500
	python -m src.scraping.run_scrapers --source all --class fake
	python -m src.annotation.validate_corpus

preprocess:
	python -m src.preprocessing.pipeline --build-all-variants
	python -m src.preprocessing.pipeline --make-splits

train:
	python -m src.models.train --model all --features all

evaluate:
	python -m src.evaluation.report
	python -m src.evaluation.error_analysis --model best

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api:app --reload --port 8000

test:
	pytest -v
