#!/bin/bash


echo "Formatting started"
echo "Run isort"
poetry run python -m isort src/
echo "Run black"
poetry run python -m black -q src/
echo "Run yamlfix"
poetry run yamlfix src/conf/
poetry run yamlfix sweepers/
echo "Run docformatter"
poetry run python -m docformatter -i -r src/
echo "Run ruff"
poetry run python -m ruff check --fix src/
echo "Formatting done"
