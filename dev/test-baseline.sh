#!/bin/bash


echo "=== test.sh ==="

echo "- Start Python checks"

echo "- isort: start"
poetry python -m isort --check-only src/
echo "- isort: done"

echo "- black: start"
poetry python -m black --check src/
echo "- black: done"

echo "- docformatter: start"
poetry python -m docformatter -c -r src/
echo "- docformatter:  done"

echo "- ruff: start"
poetry python -m ruff check src/
echo "- ruff: done"


