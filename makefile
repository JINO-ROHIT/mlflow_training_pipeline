ABSOLUTE_PATH := ${shell pwd}

.PHONY: dev
dev:
	pip install -r requirements.txt

.PHONY: train
train:
	mlflow run . --env-manager=local

.PHONY: ui
ui:
	mlflow ui