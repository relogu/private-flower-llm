#!/bin/bash

echo "Formatting started"
echo "Run isort"
poetry run python -m isort pollen_worker/
echo "Run black"
poetry run python -m black -q pollen_worker/
echo "Run yamlfix"
poetry run yamlfix pollen_worker/conf/
echo "Run docformatter"
poetry run python -m docformatter -i -r pollen_worker/
echo "Run ruff"
poetry run python -m ruff check --fix pollen_worker/
echo "Formatting done"
