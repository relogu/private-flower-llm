#!/bin/bash


echo "=== test.sh ==="

echo "- Start Python checks"

echo "- isort: start"
poetry run python -m isort --check-only pollen_worker/
echo "- isort: done"

echo "- black: start"
poetry run python -m black --check pollen_worker/
echo "- black: done"

echo "- docformatter: start"
poetry run python -m docformatter -c -r pollen_worker/
echo "- docformatter:  done"

echo "- ruff: start"
poetry run python -m ruff check pollen_worker/
echo "- ruff: done"

echo "- mypy: start"
poetry run python -m mypy --incremental pollen_worker/
echo "- mypy: done"


