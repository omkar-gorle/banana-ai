install:
	python -m pip install -r requirements.txt

dataset-report:
	python -m banana_ai.ml.dataset_report --dataset-root ..\banana_classification

train-v1:
	python -m banana_ai.ml.train --dataset-root ..\banana_classification

train-v2:
	python -m banana_ai.ml.train_v2 --dataset-root ..\banana_classification

evaluate-v1:
	python -m banana_ai.ml.evaluate_extended --checkpoint models/banana_cnn_best.pt --dataset-root ..\banana_classification --output-dir reports/v1 --version-label v1

evaluate-v2:
	python -m banana_ai.ml.evaluate_extended --checkpoint models/banana_cnn_v2.pt --dataset-root ..\banana_classification --output-dir reports/v2 --version-label v2

compare:
	python -m banana_ai.ml.analysis --v1-dir reports/v1 --v2-dir reports/v2

db-up:
	docker context use default && docker compose up -d db

db-init:
	python -m banana_ai.db.init_db

test:
	pytest -q

api:
	uvicorn banana_ai.api.main:app --reload

app:
	streamlit run src/banana_ai/app.py

device:
	python -m banana_ai.ml.device

compile-check:
	python -m compileall -q src

integration-test:
	python scripts/integration_test.py
